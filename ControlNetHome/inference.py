#%%
from share import *

from cldm.model import create_model, load_state_dict
import cv2
import sys
from annotator.util import resize_image
import numpy as np
import torch
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import torch
from torchvision import transforms as T
from cldm.cldm import ControlLDM

steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

#%%
class crowd_sampler():

    def __init__(self,resume_path = 'ControlNet/checkpoints/crowdnet_dict12epochs.ckpt'):

        super().__init__()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(resume_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def sample(self, N=1, control = None, ddim_steps=50):
        if control is None :
            # sample random gaussian
            pass 

        if isinstance(control,torch.Tensor):
            if control.shape != (3,512,512):
                control = T.reshape(512,512)(control)

        if not ((control >= -1) & (control <= 1)).all() :
            control = T.Normalize([0.5],[0.5])(control)

        c = self.model.get_unconditional_conditioning(N)
        uc_cross = self.model.get_unconditional_conditioning(N)
        c_cat = control.cuda()
        uc_cat = c_cat
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        cond={"c_concat": [c_cat], "c_crossattn": [c]}
        b, c, h, w = cond["c_concat"][0].shape
        shape = (4, h // 8, w // 8)
    	
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, N, 
                                             shape, cond, verbose=False, eta=0.0, 
                                             unconditional_guidance_scale=9.0,
                                             unconditional_conditioning=uc_full
                                             )
        x_samples = self.model.decode_first_stage(samples)
        x_samples = x_samples.squeeze(0)
        x_samples = (x_samples + 1.0) / 2.0
        x_samples = x_samples.transpose(0, 1).transpose(1, 2)
        x_samples = x_samples.cpu().numpy()
        x_samples = (x_samples * 255).astype(np.uint8)

        return samples
#%%

'''
resume_path = '/ControlNet/lightning_logs/version_6/checkpoints/last.ckpt' # your checkpoint path
N = 1
ddim_steps = 50


model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

img_path = 'your image path'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = resize_image(img, 512)


control = torch.from_numpy(img.copy()).float().cuda() / 255.0
control = torch.stack([control for _ in range(N)], dim=0)
control = einops.rearrange(control, 'b h w c -> b c h w').clone()
c_cat = control.cuda()
c = model.get_unconditional_conditioning(N)
uc_cross = model.get_unconditional_conditioning(N)
uc_cat = c_cat
uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
cond={"c_concat": [c_cat], "c_crossattn": [c]}
b, c, h, w = cond["c_concat"][0].shape
shape = (4, h // 8, w // 8)

samples, intermediates = ddim_sampler.sample(ddim_steps, N, 
                                             shape, cond, verbose=False, eta=0.0, 
                                             unconditional_guidance_scale=9.0,
                                             unconditional_conditioning=uc_full
                                             )
x_samples = model.decode_first_stage(samples)
x_samples = x_samples.squeeze(0)
x_samples = (x_samples + 1.0) / 2.0
x_samples = x_samples.transpose(0, 1).transpose(1, 2)
x_samples = x_samples.cpu().numpy()
x_samples = (x_samples * 255).astype(np.uint8)

image_name = img_path.split('/')[-1]
Image.fromarray(x_samples).save('./outputs/' + image_name)
'''