'''
import torch 
model = model.to(torch.device(1))
model.counter = model.counter.to(torch.device(1))
model.cond_stage_model.device = torch.device(1)

import einops as ei 
map = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/map/0017.pt',map_location= torch.device(1))
image = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/img/0017.pt',map_location=torch.device(1))
count = torch.tensor([13], device = torch.device(1))
prompt = 'a photo of a crowd of people running, no weather degradation'

map = torch.unsqueeze(ei.rearrange(map, 'c h w -> h w c' ), 0)
image = torch.unsqueeze(ei.rearrange(image, 'c h w -> h w c' ), 0)

gaussians = [torch.rand_like(map).to(map.device) for i in range(5)]

batch = {'jpg' : map, 'txt' : [prompt], 'hint' : map, 'gaussians' : gaussians, 'count' : count}

model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
images = model.log_images(batch, sample = True) #unconditional_guidance_scale=9.00 by default

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
for key in list(images.keys()) :
    img = torch.clamp(images[key].squeeze(0), -1., 1.).cpu()
    img = ( img + 1.0 ) /2.0
    img = img.transpose(0,1).transpose(1,2).squeeze(-1)
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.rotate(-90, expand = True)
    print(f'key:{key}')
    
    plt.imshow(img)
    plt.axis('off')
'''
#%%
import sys
import torch 
import os.path as path
import matplotlib.pyplot as plt 
import  PIL.Image as Image 
import torchvision.transforms as T 
import einops as ei
from typing import Optional, List, Union
from tqdm import tqdm
import numpy as np 
import PIL.Image as Image

gp = path.dirname(path.dirname(__file__))
if gp not in sys.path :
    sys.path.append(gp)

steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

control_loc = '/home/luk02485/development/ControlNet/ControlNetHome'
if control_loc not in sys.path :
    sys.path.append(control_loc)
    
from ControlNetHome.cldm.cldm import ControlLDM
from ControlNetHome.ldm.util import default
from ControlNetHome.cldm.logger import ImageLogger
from ControlNetHome.cldm.model import create_model, load_state_dict
from ControlNetHome.cldm.ddim_hacked import DDIMSampler

from inference import ckpt_search

#%%
#TODO: pass in original density size --> use 'gaussian' as input instead if density 


class TrainedControlNet(ControlLDM) :

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):

        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.ddim_sampler = DDIMSampler(self)
        
    @torch.no_grad()
    def sample(self, prompt : Union[str, List[str]] , density : torch.Tensor, method = 'DDPM') :

        assert len(density.shape) > 2 and len(density.shape) <= 4, 'usage : [B,1,H,W] or [1,H,W]'
        original_dim = tuple(density.shape[-2:])
        if density.shape[-2:] != (512,512) :
            prev_area = density.shape[-1]*density.shape[-2]
            new_area = 512**2
            scaler = prev_area / new_area
            density = T.Resize((512,512), interpolation=T.InterpolationMode.NEAREST_EXACT)(density) * scaler
        if len(density.shape) == 4 and density.shape[0] > 1 :
            densities = list(torch.tensor_split(density,density.shape[0]))
            for k,dens in enumerate(densities) :
                densities[k] = self.sample(dens, method )
        elif len(density.shape) == 4 :
            densities = self.sample(densities, method)
        else :
            densities = density.unsqueeze(0)
        if method == 'DDPM' :
            densities = self.DDPM_generate(dict(density=densities, txt=prompt))
        try :
            densities = T.Resize((original_dim), interpolation=T.InterpolationMode.NEAREST_EXACT)(densities)
            densities = self.process(densities)
        except :
            for dens in densities : 
                dens = T.Resize((original_dim), interpolation=T.InterpolationMode.NEAREST_EXACT)(dens)
                dens = self.process(dens)
        return densities

    def process(self,img) :
        img = torch.clamp(img.squeeze(0), -1., 1.).cpu()
        img = ( img + 1.0 ) /2.0
        img = img.transpose(0,1).transpose(1,2).squeeze(-1)
        img = img.numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        #img = img.rotate(-90, expand = True)
        return img

    @torch.no_grad()
    def DDPM_generate(self, input : dict ) -> torch.Tensor :
        # input : {'density' : map, 'txt' : prompt, 'hint' : map }

        density = input['density']
        txt = [input['txt']] if type(input['txt'])==str else input['txt']
        #assuming densityalready in correct dimension (latent)
        '''if density.shape[0] == 1 :
            density = torch.unsqueeze(ei.rearrange(density, 'b c h w -> b h w c'), 0)
        '''
        shadow_x = 2 * torch.randn(1,3,512,512) - 1
        density = ei.rearrange(density, 'b c h w -> b h w c')
        shadow_x = ei.rearrange(shadow_x, 'b c h w -> b h w c')
        
        #form batch and get input here
        batch = {
            'jpg' : shadow_x, 
            'txt' : txt, 
            'hint' : density, 
            'gaussian' : density, 
            'count' : torch.tensor([density.sum().item()])

        }
        print(batch.keys())
        x, cond = self.get_input(batch, k=self.first_stage_key, dropout = False) #passes through encoder
        # encode into latent dimension here
        x_start = x#2 * torch.rand((1,4,64,64), device = self.device) -1
        
        # use get_input for this
        progress = tqdm(total=1000)
        step = 10
        for t in reversed(range(0,1000,step)) :
            time = torch.tensor([t], device = self.device)
            #noise = default(noise, lambda: torch.randn_like(x_start)) if t > 1 else torch.zeros(x_start.shape)
            noise = torch.randn_like(x_start) if t > 1 else torch.zeros(x_start.shape, device = self.device)
            #x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) This is forward process
            model_output = self.apply_model(x_start, time, cond)

            x_start = self.predict_reconstructed_from_noise(x_t=model_output, t=time, noise = noise) + torch.sqrt(self.betas[t]) * torch.randn_like(noise)
            progress.update(1*step)

        reconstructed = self.decode_first_stage(x_start)
        progress.close()

        return reconstructed

    @torch.no_grad()
    def DDIM_generate(self, input : dict, ddim_steps = 50, ddim_eta=0.0, plot_denoise_rows=False,  unconditional_guidance_scale=9.0) :
        log = dict()
        use_ddim =  ddim_steps is not None 
        z, c = self.get_input(input, self.first_stage_key, dropout = False)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = z.shape[0]
        samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
        x_samples = self.decode_first_stage(samples)
        log["samples"] = self.process(x_samples)
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
            
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
#%%

try:
        resume_path = ckpt_search()
except : 
    resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'#'./saves/checkpoints/crowdnet_dict-epoch-60.ckpt'#crowdnet_dict12epochs.ckpt'#'./models/control_sd15_ini.ckpt'

STEERER_path = '/home/luk02485/development/ControlNet/STEERER/JHU_mae_54.5_mse_240.6.pth'

batch_size = 16
logger_freq = 300
learning_rate = 2e-5#1e-5   SET TO PAPER VALUES
sd_locked = True
only_mid_control = False
AVAILABLE_GPU = [1,2]

model = create_model('./ControlNetHome/models/trained_cldm.yaml',location=None).cpu()

interm = load_state_dict(resume_path, STEERER_path=STEERER_path,location=None)

model.load_state_dict(interm,strict = False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# set all parts to same device
DEVICE = 1
model = model.to(torch.device(DEVICE))
model.counter = model.counter.to(torch.device(DEVICE))
model.first_stage_model.device = model.first_stage_model.to(torch.device(DEVICE))
model.cond_stage_model.device = torch.device(DEVICE)

#%%

def testing(nb = 0, gs = 1) :

    map = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/map/0017.pt',map_location= torch.device(1))
    image = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/img/0017.pt',map_location=torch.device(1))
    gaussian = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/gaussians/0017-2048.pt',map_location=torch.device(1))
    count = torch.tensor([gaussian.sum().item()], device = torch.device(1))
    prompt = 'a group of people standing on a sidewalk'

    map_ = torch.unsqueeze(ei.rearrange(map, 'c h w -> h w c' ), 0)
    image_ = torch.unsqueeze(ei.rearrange(image, 'c h w -> h w c' ), 0)


    batch = {'jpg' : image_, 'txt' : [prompt], 'hint' : map_, 'gaussian' : gaussian, 'count' : count}

    #model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
    samples = model.DDIM_generate(input=batch, unconditional_guidance_scale=gs)['samples']

    import matplotlib.pyplot as plt 

    fig, axes = plt.subplots(1,3,figsize=(15,3))
    axes[0].imshow(image.cpu().squeeze(0).permute(1,2,0))
    axes[0].axis('off')
    axes[0].set_title(f'txt : {prompt}')

    axes[1].imshow(gaussian.squeeze(0).cpu())
    axes[1].axis('off')
    axes[1].set_title(f'count : {count}')

    axes[2].imshow(samples)
    axes[2].axis('off')
    axes[2].set_title('samples')

    plt.savefig(f'./saves/20-8-24-runv1-9460steps-magn-regularizer/testing/nb={nb}-gs={gs}')
    plt.show()
    plt.close('fig')

testing(1,1)
testing(1,0)
testing(1,0)
testing(1,5)
testing(1,5)
testing(1,10)
testing(1,10)
testing(1,15)
testing(1,20)
testing(1,50)
testing(1,50)
testing(1,100)

# %%
