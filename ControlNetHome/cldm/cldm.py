import einops
import torch
import torch as th
import torch.nn as nn

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

# packages i added :
from ldm.util import default
try :
    from STEERER.inference import CounterWrapper
except :
    import sys 
    steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
    if steerer_loc not in sys.path :
        sys.path.append(steerer_loc)

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# sampler needed to reconstruct images from noise 
from cldm.ddim_hacked import DDIMSampler
from tqdm import tqdm

# Set STEERER here 
counter = None 
counter_device = 0

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

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):

        #steerer_stage_config = kwargs.pop('counter_stage_config')
        # Added STEERER to counter model  to ControlNet
        #self.counter = instantiate_from_config(steerer_stage_config)  #CounterWrapper(device = self.device)

        super().__init__(*args, **kwargs)
        
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13


        # for hyperparameter tuning
        self.register_buffer('magnitude_regularizer', torch.tensor(0.01)) #start value
        self.smooth_magnitude_tuning_start = 4 #start epoch
        self.magnitude_reg_previous_importance = 0.99    # 0 means the parameter will be updated to a completely new value at the end of each epoch.
                                                        # a value of 1 means the parameter remains constant throughout training.
        assert self.magnitude_reg_previous_importance > 0 and self.magnitude_reg_previous_importance < 1, 'Invalid importance scaled passed for magnitude regularizer parameter. Range should be (0,1).'
        #self.magnitude_every_x_epochs = 1  

    ######################################################################################################
    
    @torch.no_grad()
    def tune_magnitude_regularizer(self,beta):
        '''
        This is done by going through all 50 validation batches. See README.md for details.
        This computes the training loss so a step is faster than a regular validation step (ddim sampling).
        '''
        val_dataloader = self.trainer.val_dataloaders

        Lc_over_Lcount = []
        Lcount_over_Lc = []
        progress = tqdm(total = len(val_dataloader))
        
        #kept for debugging : removing  after
        error_save  =[[],[]]
        #####################################
        for _,batch in enumerate(val_dataloader):
                
            x,c = self.get_input(batch, k='jpg',bs=None, dropout = False)
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
            if all( time < 400 for time in t) : 
                continue                 
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

            Lc,Lcount = self.compute_loss( x, c, t, noise=None)[2:]
            Lc_over_Lcount.append(Lc / (Lcount))
            Lcount_over_Lc.append((Lcount)/Lc)
            error_save[0].append(Lc)  #remove after debugging 
            error_save[1].append(Lcount)#remove after debugging 

            progress.update(1)

            #debugging section to find culprit batch :
            condition1 = Lc.item() >=  1e30 
            try : 
                condition2 = Lcount.item() <= 1e-34 
            except Exception as e :

                exc = (f'exception : {str(e)}\n'
                f'Lcount does not have .item() attribute --> is not a tensor --> type : {type(Lcount)}\n'
                'Setting the condition to False ... \n')
                with open('tuning_config.txt', 'a') as file :
                        file.write(f'{exc}\n')

                condition2 = Lcount <= 1e-34 
                loss2 = Lcount
            condition = condition1 or condition2
            if condition :
                try :         
                    if condition1 and condition2 :
                        losses = f'Lc = {Lc.item()} / Lcount = {loss2}'
                    if condition1 : 
                        losses = f'Lc = {Lc.item()}'
                    elif condition2 :
                        losses = f'{loss2}'
                    else:
                        losses = 'No abnormal losses detected'
                    batch_infos = (f'txt : {batch["txt"]}\n'
                                    f'count : {batch["count"]}')
                    culprit = ( '\n' 
                                '\n'
                                'Found the culprit : \n'
                                f'losses = {losses} \n'
                                f'batch_idx = {_} \n'
                                f'lambda : {self.magnitude_regularizer} / lambda^-1 = {1/self.magnitude_regularizer}\n'
                                f' batch : \n '
                                f'  {batch_infos} \n'
                                f' epoch : {self.current_epoch}\n'
                                f'Lc/Lcount = {Lc_over_Lcount}\n'
                                f'Lcount/Lc = {Lcount_over_Lc}'
                                '\n'
                                '\n')
                    with open('tuning_config.txt', 'a') as file :
                        file.write(f'{culprit}\n')
                    try :
                        file_path = f'./culprit_dict-{self.current_epoch}-{self.global_step}.ckpt'
                        torch.save(self.state_dict(), file_path)
                        print(f"Model weights saved to {file_path}")
                    except Exception as e :
                        print(f"Could not save Model weights to {file_path}. Error : {str(e)}")
                except Exception as e :
                    with open('tuning_config.txt', 'a') as file :
                        file.write(f'Could not write culprit: {str(e)}\n')

        mean1 = (1/self.magnitude_regularizer) * torch.mean(torch.tensor(Lc_over_Lcount))
        mean2 = self.magnitude_regularizer * torch.mean(torch.tensor(Lcount_over_Lc))

        new_reg = 0.5*(mean1 + mean2)
        self.magnitude_regularizer = beta*self.magnitude_regularizer + (1-beta)*new_reg

        infos = (f'[Updated] magnitude_regularizer={self.magnitude_regularizer} / ratios = ({mean1}|{mean2})\n')
        print(infos)
        with open('tuning_config.txt', 'a') as file :
                file.write(f'{infos}\n'
                           f'minLc = {min(error_save[0])} / minLcount = {min(error_save[0])} / maxLc = {max(error_save[0])} / maxLcount = {max(error_save[0])}')
        progress.close()

    @torch.no_grad()
    def on_validation_epoch_end(self):
        '''
        perform the regularizer step for the hyperparameter that scales the L_count loss to
        the same magnitude as the usual DM loss.
        '''
        if self.current_epoch >= self.smooth_magnitude_tuning_start :
            infos = (f'[tuning step]   config : \n'
                  f'        start_epoch : {self.smooth_magnitude_tuning_start} \n'
                  f'        transition_factor :  {self.magnitude_reg_previous_importance} \n'
                  f'        current_magnitude_regularizer : {self.magnitude_regularizer}\n'
                  f'        current_epoch : {self.current_epoch}\n'
                  f'        lenght data loader : {self.trainer.val_dataloaders.__len__()}\n'
                   )
            print(infos)
            with open('tuning_config.txt', 'a') as file :
                file.write(infos)
            
            self.tune_magnitude_regularizer(beta = self.magnitude_reg_previous_importance)

    # Done to avoid double initialization from lightning.trainer and overloading cpu 
    def on_train_start(self) :
        global counter 
        if counter is None :
            counter = CounterWrapper().to(counter_device)
    
    def on_validation_start(self) :
        global counter 
        if counter is None :
            counter = CounterWrapper().to(counter_device)
    
    #modified function
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, dropout = True, *args, **kwargs):
        
        if dropout :    
            # 20% Dropout rate for promptless conditioning 
            for i in range(len(batch['txt'])):
                if torch.rand(1) < 0.2 :
                    batch['txt'][i] = ''
            
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]

        gaussian = batch['gaussian']

        if bs is not None:
            control = control[:bs]
            gaussian = gaussian[:bs]

        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        return x, dict(c_crossattn=[c], c_concat=[control], gaussian = gaussian)

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
        return torch.utils.checkpoint.checkpoint(self.first_stage_model.decode, z, use_reentrant=True)

    def compute_loss(self, x_start, cond, t, noise=None, *args, **kwargs) :

        #cloning to grad() as in setting y -> f(x) + g(x). Otherwise dict() is mutable (and tensor too ?), 
        # consequently we might end up in setting y -> f(x) + g(x') where x' is modified.
        x_ = x_start.clone()

        if noise is None :
            noise = default(noise, lambda: torch.randn_like(x_start))
        Lc, loss_dict, eps_t, x_t = super().p_losses(x_, cond, t ,noise)

        # fetch t>400 for counting loss
        mask = t > 400
        if all(not x for x in mask) :
            return Lc, loss_dict, Lc, torch.tensor(0)
        indices = mask.nonzero(as_tuple = True)[0]

        noise = noise[indices]
        x_start = x_start[indices]
        eps_t = eps_t[indices]
        t = t[indices]
        gaussian = cond.pop('gaussian').to(self.device)[indices]

        # reconstruct the images :
        reconstructed = self.predict_reconstructed_from_noise(x_t=eps_t, t=t, noise = noise)
        reconstructed = self.decode_first_stage_train(reconstructed)
        reconstructed = enhance_tensor(reconstructed)

        if len(reconstructed.shape) < 4 :
            print(reconstructed.shape)

        reconstructed = reconstructed.to(counter.device)
        densities = counter.get_count(reconstructed)[1].to(self.device)

        L_count = torch.mean(torch.linalg.norm( densities - gaussian, ord = 'fro', dim = (2,3) )**2)
        loss = Lc + self.magnitude_regularizer*L_count

        if self.global_step % 100 == 0 and self.current_epoch != 0:
            recon = reconstructed.detach()
            if len(recon.shape) == 3 :
                recon = recon.cpu().permute(1,2,0)
            elif len(recon.shape) == 4 :
                recon = recon[0].squeeze(0).cpu().permute(1,2,0)

            self.train_plot(recon,
                            gaussian[0].detach().squeeze(0).cpu(),
                            densities[0].detach().squeeze(0).cpu(),
                            true_count=densities[0].sum().item(),
                            t=t[0].detach())

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_dict.update({f'{log_prefix}/Lc': Lc.mean(),
                           'lambda' : round(self.magnitude_regularizer.item(),2), 
                           f'{log_prefix}/L_count': L_count.mean()})
        
        return loss, loss_dict, Lc, L_count
    
    def p_losses(self, x_start, cond, t, noise=None,*args, **kwargs) :
        
        return self.compute_loss(x_start, cond, t, noise=None, *args, **kwargs)[:2]

    def train_plot(self, recon, gt, dens, true_count, t) :

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(recon)
        axes[0].axis('off')
        axes[0].set_title(f'Reconstructed Image - time_step={t}')

        axes[1].imshow(gt)
        axes[1].axis('off')
        axes[1].set_title('Original Density')

        axes[2].imshow(dens)
        axes[2].axis('off')
        axes[2].set_title('Reconstructed Density')

        fig.text(0.36, 0.02, f'true_count={true_count}(gt)|{round(gt.sum().item())}(comp.)', ha='center', fontsize=12)
        fig.text(0.73, 0.02, f'approximated_count={dens.sum().item()}', ha='center', fontsize=12)

        plt.savefig(f'./saves/plot_combined_grid-{self.global_rank}-{self.current_epoch}-{self.global_step}.png', bbox_inches='tight')
        plt.show()
        plt.close('fig')
    

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

@gradient_skip
def enhance_tensor(img : torch.Tensor) -> torch.Tensor :
    img = torch.clamp(img, -1., 1.)
    img = (img + 1.0) / 2.0
    return img
