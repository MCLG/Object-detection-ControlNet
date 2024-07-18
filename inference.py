#%%
import sys
control_loc = '/home/luk02485/development/ControlNet/ControlNetHome'
if control_loc not in sys.path :
    sys.path.append(control_loc)

from ControlNetHome.share import *
from data_processor import density_map, MapConfig
import tempfile
from ControlNetHome.cldm.model import create_model, load_state_dict
import cv2
from ControlNetHome.annotator.util import resize_image
import numpy as np
import torch
import einops
from ControlNetHome.cldm.ddim_hacked import DDIMSampler
from PIL import Image
import torch
import pytorch_lightning as pl 
from torchvision import transforms as T
from ControlNetHome.cldm.cldm import ControlLDM
import glob
import os 
from typing import Tuple, Optional
import random as rd 

steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

def ckpt_search() -> str :
    '''
    searches through lightning_logs for the models dict of the latest code version. 
    '''
    path = os.curdir + '/lightning_logs/version_*'
    versions = glob.glob(path) 
    checkpoints = []
    while len(checkpoints) == 0:
        try:
            ver = versions.pop()
        except Exception as e:
            print(f'Error : {e} occured. No existing ckpt of previous model version exists !')
            sys.exit(1)

        checkpoints = glob.glob(ver + '/checkpoints/epoch=*')
    return checkpoints[0]

#%%
class crowd_sampler(pl.LightningModule):
    '''
    WIP - A wrapper class for the trained crowd generating control net. Includes :
        - a rd gaussian map generator.
        - sample method.
    '''
    def __init__(self,resume_path = None, device = 2 ):
        super().__init__()

        if resume_path is None :
            resume_path = ckpt_search()

        
        
        self.model = create_model('./ControlNetHome/models/cldm_v15.yaml',location=device)
        self.model.load_state_dict(load_state_dict(resume_path, location=2),strict=False)
        #self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.map_config = MapConfig()

        self.device = self.to(self.model.device)
        self.model.cond_stage_model.device = torch.device(self.device)

    def get_rd_map(self, n = None ) -> torch.Tensor :
         '''
         generates a random gaussian map following self.map_config parameters. So far only generates map with constant box size 4,4
         n : number of gaussian points
         '''
         with tempfile.NamedTemporaryFile(suffix='.txt', delete=True, mode='w') as positions:
                if n is None :
                    num = rd.randint(1, 2000)
                else :
                    assert isinstance(n,int), 'TypeError : specified "n" must be an integer.'
                    num : int = n

                # create file with lines x,y,width,height
                dim = (512,512)
                for n in range(1,num):
                    line = f'{rd.randint(1,dim[0])} {rd.randint(1,dim[1])} 4 4\n'
                    positions.write(line)
                
                return density_map(config=self.map_config, gt_path=positions.name, img_dim = dim)

    def sample(self, control : Tuple[Optional[int],Optional[torch.Tensor]] = (None,None), prompt = '', N=1, ddim_steps=50) -> torch.Tensor:

        #assert isinstance(control, Tuple[Optional[int],Optional[torch.Tensor]]), f'Wrong type of control is passed in {self.__qualname__}'
        assert (control[0] is None or isinstance(control[0], int)), f'First element of control must be None or int in {self}'
        assert (control[1] is None or isinstance(control[1], torch.Tensor)), f'Second element of control must be None or torch.Tensor in {self}'

        if control[1] is None :
           control = [self.get_rd_map(n=control[0]) for _ in range(N)]

        else :

            if len(control[1].shape) == 4 :
                control = torch.stack(torch.tensor_split(control[1],control[1].shape[0]))
                control = [T.reshape(512,512)(c.squeeze(0)) for c in control]
            elif control[1].shape != (3,512,512) :
                control[1] = T.reshape(512,512)(control)
            control = [control[1]]
        
        for pos,c in enumerate(control) :    
            if not ((c >= -1) & (c <= 1)).all() :
                c = T.Normalize([0.5],[0.5])(c)
                control.pop(pos)
                control.insert(pos,c)

        with torch.no_grad():
            c_prompt = [self.model.get_learned_conditioning(prompt) for _ in range(len(control))]
            # {'c_concat':[torch.stack(cond400['c_concat'])], 'c_crossattn':[torch.stack(cond400['c_crossattn'])]}
            
            #try :
            #    cond = {'c_concat': [torch.stack(control)].to(self.device), 'c_crossattn' : [torch.stack(c_prompt) ]}
            #except :

            # cond400["c_concat"][0].shape,cond400["c_crossattn"][0].shape =  torch.Size([11, 3, 512, 512]) torch.Size([11, 77, 768])

            #print('handling cond as 3,512,512 tensors')
            cond = {'c_concat': [control[0].unsqueeze(0).to(self.device)], 'c_crossattn' : c_prompt}
            #print('cond400["c_concat"][0].shape,cond400["c_crossattn"][0].shape = ',cond['c_concat'][0].shape,cond['c_crossattn'][0].shape)
            
            #else :
            #    print('handling cond as 1,3,512,512 tensors')
            #    cond = {'c_concat': [control[0].unsqueeze(0)].to(self.device), 'c_crossattn' : [c_prompt[0].unsqueeze(0)]}
            
        
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, N, 
                                                              shape=(4,64,64),
                                                              conditioning = cond)
            samples = self.model.decode_first_stage(samples)

            return samples
#%%
#s = crowd_sampler()
#out = s.sample()
#print(out)
'''
with torch.no_grad():

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
'''
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