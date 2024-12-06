import einops
import torch
import torch as th
import torch.nn as nn
import warnings

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import os 
import sys 

# packages i added :
from ldm.util import default
try :
    from STEERER.steerer_inference import CounterWrapper
except :

    project_root = os.path.abspath(os.path.dirname(__file__))
    steerer_path = os.path.join(project_root, 'STEERER')
    if steerer_path not in sys.path:
        sys.path.append(steerer_path)
    from STEERER.steerer_inference import CounterWrapper
    '''
    import sys 
    steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
    if steerer_loc not in sys.path :
        sys.path.append(steerer_loc)'''

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, InterpolationMode
from typing import List, Optional  

#for counting DDIM sampling
from ldm.modules.diffusionmodules.util import extract_into_tensor
from torch.nn.functional import interpolate
from tqdm import tqdm
import os 
from torch.nn.functional import mse_loss 

#plots
from tools.CGsampling_plotter import sample as CG_plot_sample

# Set STEERER here 
try :
    from STEERER.lib.models.build_counter import freeze_model
except : 

    project_root = os.path.abspath(os.path.dirname(__file__))
    steerer_path = os.path.join(project_root, 'STEERER')
    if steerer_path not in sys.path:
        sys.path.append(steerer_path)
    from STEERER.lib.models.build_counter import freeze_model

counter = None 
counter_device = "cpu" 
class STEERER_memory_alloc(nn.Module):
    def __init__(self):
        super(STEERER_memory_alloc, self).__init__()

    def forward(self, x):
        return x

# DivergenceLosses :
from tools.divergence_loss import DivergenceLoss, to_dmap

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

class ControlLDM(LatentDiffusion):

    def __init__(
        self, 
        control_stage_config, 
        control_key, 
        only_mid_control, 
        control_eval,           # the type of loss to use "MSE"/ "W2-count" / "w2-TV" / "count TV"  
        loss_downscaling_factor, 
        counter_path,
        scale_mse,
        scale_w2,
        scale_tv,
        scale_count,
        *args, **kwargs
    ) :

        super().__init__(*args, **kwargs)
        
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        self.counter = STEERER_memory_alloc().to(self.device)
        self.counter_dict = counter_path

        self.scale_mse = scale_mse
        self.scale_w2 = scale_w2
        self.scale_tv = scale_tv
        self.scale_count = scale_count

        self.m_key = 'mean'
        self.g_key = 'gaussian'

        if not control_eval in ["mse", "w2-count", "w2-tv", "count-tv"] :
            raise ValueError(f'Invalid {control_eval=} passed in ControlLDM; valid input are "MSE", "W2-count", "w2-TV", "count TV". ')
        if control_eval != "MSE" :
            self.loss_downscaling_factor = loss_downscaling_factor
            self.DivLoss = DivergenceLoss(
                downscaling_factor = self.loss_downscaling_factor, 
                extract_type = 'ggm-em', 
                device = self.device, 
                eval_on_low_dim = False
                )
        self.control_eval = control_eval

    @torch.no_grad()
    def on_validation_epoch_end(self):
        
        #perform count guidance sampling with same image for reproducibility
        id = '0055870'
        control_map = torch.load(f'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/map/{id}.pt',map_location = self.device)
        control_map = torch.unsqueeze(control_map,0)
        sampling_data_csv = 'temp_sampling_process.csv'
        denoising_steps = 1000
        '''
        with torch.enable_grad():
            sample = self.count_guided_sampling(control_map,
                prompt = ['a photograph of a crowd of people holding flags'], 
                denoising_steps = denoising_steps, 
                progress_track = sampling_data_csv)
        
        sample = enhance_tensor(sample)
        sample_loc = f'./saves/CG_sample-rk={self.global_rank}-ep={self.current_epoch}-step={self.global_step}.png'
        graph_loc = f'./saves/CG_graph-rk={self.global_rank}-ep={self.current_epoch}-step={self.global_step}.png'
        CG_plot_sample(loc=graph_loc, temp_file=sampling_data_csv)

        fig, axis = plt.subplots(1,2, figsize=(12,6))
        axis[0].imshow(control_map.squeeze(0).permute(2,1,0).cpu())
        axis[0].axis('off')
        axis[1].imshow(sample.squeeze(0).permute(2,1,0).cpu())
        axis[1].axis('off')
        plt.savefig(sample_loc, bbox_inches='tight', pad_inches=0)
        plt.close()'''
        
    # Initialize STEERER when training
    def on_train_start(self) :
        
        self.DivLoss.device = self.device
        
        counter_device = self.device
        if isinstance(self.counter, STEERER_memory_alloc) :
            print(f'loaded {self.counter_dict} on STEERER on device={counter_device}')
            self.counter = CounterWrapper(path = self.counter_dict).to(counter_device)
            freeze_model(self.counter)
        
    def on_validation_start(self) :
        self.on_train_start()
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, dropout = True, *args, **kwargs):
        #Modified to original get_input() to handle different control inputs

        # 20% Dropout rate for promptless conditioning 
        if dropout :
            for i in range(len(batch['txt'])):
                if torch.rand(1) < 0.2 :
                    batch['txt'][i] = ''
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        
        if bs is not None:
            control = control[:bs]            

        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        gaussian = control.detach()
        gaussian = Resize(size=(1536, 2048), 
                            interpolation=InterpolationMode.BICUBIC)(gaussian)
        gaussian = gaussian * (512**2/(1536*2048))

        if self.control_eval in ["w2-count", "w2-tv", "count-tv"] :
            mean = batch[self.m_key]
            if bs is not None:
                mean = mean[:bs] 
            return x, {'c_crossattn' : [c], 'c_concat' : [control], f'{self.m_key}' : mean, f'{self.g_key}' : gaussian }
        return x, {'c_crossattn' : [c], 'c_concat' : [control], f'{self.g_key}' : gaussian }

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
    
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N, dropout = False)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0                 
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def decode_first_stage_train(self, z, predict_cids=False, force_not_quantize=False):
        z = 1. / self.scale_factor * z
        '''try :
            if self.trainer.validating :
                return self.first_stage_model.decode(z)
            elif self.trainer.training :
                return torch.utils.checkpoint.checkpoint(self.first_stage_model.decode, z, use_reentrant=True)
            else :
                return self.first_stage_model.decode(z)
        except RuntimeError :
            return torch.utils.checkpoint.checkpoint(self.first_stage_model.decode, z, use_reentrant=True)'''
        return torch.utils.checkpoint.checkpoint(self.first_stage_model.decode, z, use_reentrant=True)
    
    def compute_loss(self, x_start, cond, t, noise=None, mode='train', *args, **kwargs) :

        def time_scale(t,T=400, alpha=.1) :
            '''
            assigns high weight for t close to 0 and 1 to close to 400
            '''
            if t >= T :
                return 1.
            return alpha*(T-t)/T + 1.

        #cloning to grad() as in setting y -> f(x) + g(x). Otherwise dict() is mutable (and tensor too ?), 
        # consequently we might end up in setting y -> f(x) + g(x') where x' is modified.
        x_0 = x_start.clone()

        if noise is None :
            noise = default(noise, lambda: torch.randn_like(x_start))
        Lc, loss_dict, eps_t, x_t = super().p_losses(x_0, cond, t ,noise, return_model_output = True)

        # Fetch t < 400
        mask = t < 400
        if all(not x for x in mask) :
            return Lc, loss_dict
        indices = mask.nonzero(as_tuple = True)[0]

        noise400 = noise[indices]
        x_start400 = x_start[indices]
        x_t400 = x_t[indices]
        eps_t400 = eps_t[indices]
        t400 = t[indices]
        
        if self.control_eval != 'mse' :        
            contr_cond = cond.pop(self.m_key)
            control_means = [contr_cond[i] for i in indices]
        control_gaussians = cond.pop(self.g_key).to(self.device)[indices]

        # reconstruct images
        l_reconstructed = self.predict_reconstructed_from_noise(x_t=x_t400, t=t400, noise = eps_t400)
        #check for gradient tracking (remove in future)
        if self.trainer.training :
            assert l_reconstructed.requires_grad, '1'
            #print(f'{l_reconstructed.grad_fn=}')
        raw_reconstructed = self.decode_first_stage_train(l_reconstructed)
        #check for gradient tracking (remove in future)
        if self.trainer.training :
            assert raw_reconstructed.requires_grad, '2'
            #print(f'{raw_reconstructed.grad_fn=}')
        final_reconstructed = enhance_tensor(raw_reconstructed)
        #check for gradient tracking (remove in future)
        if self.trainer.training :
            assert final_reconstructed.requires_grad, '3'
            #print(f'{final_reconstructed.grad_fn=}')
        # get densities 
        #if self.device != self.counter.device :
        #    reconstructed = reconstructed.to(self.counter.device)
        densities = self.counter.get_count(final_reconstructed, mode = mode).to(self.device)
        #check for gradient tracking (remove in future)
        if self.trainer.training :
            assert densities.requires_grad
            #print(f'{densities.grad_fn=}')
        
        #compute loss
        if self.control_eval == 'mse' :    #GAUSSIAN GT
            control_loss = mse_loss(
                densities, 
                control_gaussians, 
                reduction ='none'
                ).mean(dim=[1, 2, 3])
            aux = self.DivLoss.TV_norm(dmap = densities, gt_dmap = control_gaussians)
            l_scale = self.scale_mse
            aux_scale = self.scale_tv * 0.1
        elif self.control_eval == 'w2-count' : #MEAN GT
            control_loss = self.DivLoss.wasserstein2(
                b_dmap = densities, 
                b_gt = control_means, 
                include_spread_loss = False,
                space_scaler = None
                )
            aux = abs(control_gaussians.sum(dim=(1,2,3)) - densities.sum(dim=(1,2,3))) #torch.tensor([tensor_points.shape[0] for tensor_points in control_means], device = self.device)
            l_scale = self.scale_w2
            aux_scale = self.scale_count * 0.1
        elif self.control_eval == 'w2-tv' : 
            control_loss = self.DivLoss.wasserstein2(
                b_dmap = densities, 
                b_gt = control_means, 
                include_spread_loss = False,
                space_scaler = None
                )
            aux = self.DivLoss.TV_norm(dmap = densities, gt_dmap = control_gaussians)
            l_scale = self.scale_w2
            aux_scale = self.scale_tv * 0.1
        elif self.control_eval == 'count-tv' :
            control_loss =  abs(control_gaussians.sum(dim=(1,2,3)) - densities.sum(dim=(1,2,3)))
            aux = self.DivLoss.TV_norm(dmap = densities, gt_dmap = control_gaussians)
            l_scale = self.scale_count
            aux_scale = self.scale_tv * 0.1
        else :
            raise ValueError(f'None existing {self.control_eval=}. ')
        
        if self.trainer.training :
            assert control_loss.requires_grad
            #print(f'{control_loss.grad_fn=}')

        # computing Lc + s*Lcontrol + s*Laux and time-scaling 
        time_scaling = torch.tensor(list(map(time_scale,t))).to(self.device)
        time_scaled_loss = time_scaling * (
            l_scale * control_loss + aux_scale * aux
            )
        loss = time_scaled_loss.mean() + Lc

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/L_DM': Lc.clone().detach().item()})
        loss_dict.update({f'{log_prefix}/L_contr:': control_loss.clone().detach().mean().item() *l_scale})
        loss_dict.update({f'{log_prefix}/L_aux:': aux.clone().detach().mean().item() *aux_scale})
        loss_dict.update({f'{log_prefix}/L_{self.control_eval}:': time_scaled_loss.clone().detach().mean().item()})
        
        #check for gradient tracking (remove in future)
        if self.trainer.training :
            #assert L_count.requires_grad
            assert loss.requires_grad
            #print(f'{loss.grad_fn=}, {aux.grad_fn=}, {control_loss.grad_fn=}')
        
        x_0_ = x_start400[0].clone().detach().permute(1,2,0)
        x_t_ = x_t400[0].clone().detach().permute(1,2,0)
        noise_ = noise400[0].clone().detach().permute(1,2,0)
        eps_t_ = eps_t400[0].clone().detach().permute(1,2,0)
        t_ = t400[0].item()
        true_map = control_gaussians[0].clone().detach()
        x_map = densities[0].clone().detach()
        x_denoised = final_reconstructed[0].clone().detach().permute(1,2,0)

        b_plot = dict(
            x0 = x_0_,
            xt = x_t_,
            z = noise_,
            epst = eps_t_,
            t = t_,
            y = true_map,
            y_hat = x_map,
            x_hat = x_denoised
        )
        self.plot_per_steps(b_results=b_plot)
        
        return loss, loss_dict
    
    def p_losses(self, x_start, cond, t, noise=None,*args, **kwargs) :
        
        return self.compute_loss(x_start, cond, t, noise=None, mode = 'train', *args, **kwargs)

    def plot_per_steps(self, b_results : dict, per_global_step : int = 100) :
        
        if self.global_step % per_global_step == 0 and self.trainer.training :
            
            x_0, x_t, noise, eps_t, t, true_map, x_map, x_denoised = b_results.values()
            true_count = int(true_map.sum().item())
            approx_count = int(x_map.sum().item())
            
            x_map = Resize(size=(512,512), interpolation = InterpolationMode.NEAREST_EXACT)(x_map)  
           
            true_map = Resize(size=(512,512), interpolation = InterpolationMode.NEAREST_EXACT)(true_map).permute(1,2,0)
            
            name = f'./saves/grid-rk={self.global_rank}-ep={self.current_epoch}-step={self.global_step}.png'
            fig, axis = plt.subplots(2,4, figsize=(12,6))

            try : 
                axis[0,0].imshow(x_0.cpu())
                axis[0,0].axis('off')
                axis[0,0].set_title(f'x_0')

                axis[0,1].imshow(x_denoised.cpu())
                axis[0,1].axis('off')
                axis[0,1].set_title(f'x_denoised')

                axis[0,2].imshow(true_map.cpu())
                axis[0,2].axis('off')
                axis[0,2].set_title(f'true_map')

                axis[0,3].imshow(x_map.permute(1,2,0).cpu())
                axis[0,3].axis('off')
                axis[0,3].set_title(f'x_map')

                axis[1,0].imshow(x_t.cpu())
                axis[1,0].axis('off')
                axis[1,0].set_title(f'x_{t}')

                axis[1,1].imshow(eps_t.cpu())
                axis[1,1].axis('off')
                axis[1,1].set_title(f'eps_{t}')

                axis[1,2].imshow(noise.cpu())
                axis[1,2].axis('off')
                axis[1,2].set_title(f'z')

                fig.suptitle(f'true_count = {true_count}, approx_count={approx_count}', fontsize=14)
                plt.savefig(name, bbox_inches='tight')

            except Exception as e :
                print(f'Exception at plot_per_steps() --> Err={e}')
                pass

    def count_guided_sampling(self, 
     dmap: torch.tensor,
     prompt : List[str],
     gradient_scale : float = .1, 
     denoising_steps : int = 1000,
     ddim_discretize = 'uniform',
     unconditional_guidance_scale = .1,
     progress_track : Optional[str]= None ) -> torch.Tensor :
        '''
        progress_track : savefilename as .csv file; will be saved in working dir.
        '''
        self.register_schedule(timesteps=denoising_steps, set_device=self.device)

        '''if ddim_discretize == 'uniform':
            c = 1000 // denoising_steps
            time_steps = np.asarray(list(range(0, 1000, c))) + 1
        elif ddim_discretize == 'quad':
            time_steps = ((np.linspace(0, np.sqrt(1000 * .8), denoising_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)
        '''

        timesteps = reversed([t for t in range(1,denoising_steps)])#reversed([t for t in range(1,denoising_steps)])
        #last_n_steps, timesteps = timesteps[-2:][0], timesteps[:-2]
        #timesteps = timesteps + [i for i in reversed(range(1,last_n_steps))]
        
        if isinstance(self.counter,STEERER_memory_alloc) :
            self.on_train_start()
        assert self.device == self.counter.device, f'This method requires both models to be on the same device for grad computation. '
        assert isinstance(dmap, torch.Tensor), f'requires dmap to be a (3,512,512) torch tensor. You passed a {type(dmap)} ! '
        if len(dmap.shape) == 3 :
            dmap = dmap.unsqueeze(0)
            b=1
        else :
            assert dmap.shape[0] == len(prompt), f'Not enough prompts for dmaps given. Got {len(prompt)}, expected {dmap.shape[0]}. '
            b = dmap.shape[0]

        if progress_track :
            epsmin, epsmax, tepsmin, tepsmax, xmin, xmax, sc_min, sc_max, alp = [], [], [], [], [], [], [], [], [] 

        gaussian = dmap.detach()
        gaussian = Resize(size=(1536, 2048), 
                          interpolation=InterpolationMode.BICUBIC)(gaussian)  
        gaussian = gaussian * (512**2/(1536*2048))

        batch = {'jpg' : torch.zeros((b,512,512,3), device=self.device), 'txt' : prompt, 'hint' : dmap}
        _, txt_encoded = super().get_input(batch, self.first_stage_key)

        #dmap = einops.rearrange(dmap, 'b h w c -> b c h w')
        dmap = dmap.to(memory_format=torch.contiguous_format).float()
        cond = dict(c_crossattn=[txt_encoded], c_concat=[dmap])
        unconditional_cond = dict(c_crossattn=[txt_encoded], c_concat=None)

        x_t = default(None, lambda: torch.randn_like(torch.zeros(b,4,64,64), device = self.device))

        bar = tqdm(timesteps,
            desc='steps',
            total=denoising_steps)
        #bar_format="{l_bar}{bar} {n}/{total} steps - [{e_min}; {e_max}], {score} | {Closs} | {ycount}/{true_count}, [{te_min};{te_max}]")

        for ts in timesteps :
            
            t = torch.full((b,), ts, device=self.device, dtype=torch.long)

            with torch.no_grad() :
                model_t = self.apply_model(x_t, t, cond)
                model_uncond = self.apply_model(x_t, t, unconditional_cond)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
                if self.parameterization == "v":
                    eps_t = self.predict_eps_from_z_and_v(x_t, t, model_output)
                else:
                    eps_t = model_output

            if progress_track :
                epsmin.append(eps_t.min().item())
                epsmax.append(eps_t.max().item())
            xt = x_t.clone().detach().requires_grad_(True)
            xt.retain_grad()

            xt_reconstructed = self.predict_reconstructed_from_noise(x_t=xt, t=t, noise = eps_t) #noise = eps_t
            xt_512 = self.decode_first_stage_train(xt_reconstructed)
            xt_pretty = enhance_tensor(xt_512)

            #plt.imshow(xt_pretty.clone().detach().cpu().squeeze(0).permute(2,1,0))
            #plt.savefig(f'IMAGE-{ts}.png', bbox_inches='tight', pad_inches=0 )
            #plt.close()
        
            ymap = self.counter.get_count(interpolate(xt_pretty, size=(1536,2048), mode = 'nearest'), mode='train')
            #norm = torch.linalg.norm(gaussian - ymap, ord = 'fro', dim = (2,3))**2
            norm = mse_loss(gaussian,ymap, reduction ='none').mean(dim=[1, 2, 3])#.mean() no mean in case we want batch wise sampling
            
            #L_count = torch.linalg.norm( gaussian - ymap, ord = 'fro', dim = (2,3) )**2
            #L_count = L_count.squeeze(1)
            
            #L_count.backward()
            norm.backward()
            score = -xt.grad #TODO: multiply with optimal regularizer hoping it will give better results when using less denoising steps

            with torch.no_grad() :
                #experimental : trying to scale alpha with magnitude regularizer
                alpha = gradient_scale*(denoising_steps-ts)/denoising_steps
                eps_tilde = eps_t - alpha*extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, score.shape)*score
                x = self.predict_reconstructed_from_noise(x_t=x_t, t=t, noise = eps_tilde)
                x_t = extract_into_tensor( torch.sqrt(self.alphas_cumprod), t-1, x.shape )* x + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t-1, eps_tilde.shape) * eps_tilde #try with exchanged x and eps_tilde
                
            if progress_track :
                tepsmin.append(eps_tilde.min().item())
                tepsmax.append(eps_tilde.max().item())
                xmin.append(x_t.min().item())
                xmax.append(x_t.max().item())
                sc_min.append(score.min().item())
                sc_max.append(score.max().item())
                alp.append(alpha)


            bar.set_postfix(
                time = ts,
                eps_range = (eps_t.min().item(), eps_t.max().item()), 
                score = (score.min().item(), score.max().item()), 
                ycount = f'{ymap.sum().item()}/',
                true_count = gaussian.sum().item(),
                tilde_eps_range = (eps_tilde.min().item(), eps_tilde.max().item())
            )
            bar.update(1)
        
            if progress_track :
                import csv
                if '.csv' not in progress_track :
                    progress_track = progress_track+'.csv'
                with open(progress_track, 'w', newline='') as data:
                    writer = csv.writer(data)
                    writer.writerow(['eps_min', 'eps_max', 'teps_min', 'teps_max', 'x_min', 'x_max', 'score_min', 'score_max', 'alpha_'])  # Write header
                    for eps_min, eps_max, teps_min, teps_max, x_min, x_max, score_min, score_max, alpha_ in zip(epsmin, epsmax, tepsmin, tepsmax, xmin, xmax,sc_min,sc_max,alp):
                        writer.writerow([eps_min, eps_max,  teps_min, teps_max, x_min, x_max, score_min, score_max, alpha_])
        image = self.decode_first_stage(x_t)
        return image





def gradient_skip(function) :
    '''
    the purpose of this wrapper is to save the current gradient state of the input of function,
    compute function without updating computational graph.
    Only works if non-default arguments of function are torch.Tensor and these will have their gradient saved.
    function needs to return a tensor.
    '''
    def wrapper(*args, **kwargs):  

        cloned_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                cloned_args.append(arg.clone())
            else:
                cloned_args.append(arg)

        with torch.no_grad():
            output = function(*cloned_args, **kwargs)
        
        if isinstance(output, torch.Tensor):
            args[0].data = output.data
            return args[0]
        else:
            raise ValueError("The function must return a torch.Tensor.")
    
    return wrapper

#@gradient_skip
def enhance_tensor(img : torch.Tensor) -> torch.Tensor :
    img = torch.clamp(img, -1., 1.)
    img = (img + 1.0) / 2.0
    return img
