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
from ControlNetHome.ldm.util import rotate_clockwise
import matplotlib.pyplot as plt
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
         
    @torch.no_grad()
    def sample(self, control : Optional[torch.Tensor] = None, prompt = '', N=1, ddim_steps=50, return_control = False) -> torch.Tensor:
        '''
        control : either None or a tensor of size (3,512,512) or (N,3,512,512)
        N : number of images to sample
        prompt : a text prompt for cross-attention condition

        if N does not match control.shape[0], it will duplicate the control for each sample, provided control.shape[0]==1.
        '''
        assert (control is None or isinstance(control, torch.Tensor)), f'Control must be None or torch.Tensor in {self}'
        
        if control is None :
            control = [self.get_rd_map() for _ in range(N)]
            control = torch.stack(control)
            return self.sample(control=control, prompt= prompt, N=N, ddim_steps=ddim_steps, return_control=return_control)
            
        if N > 1 :
            samples = []
            if len(control.shape) == 4:
                if control.shape[0] != N :
                    assert control.shape[0] == 1, f'Ambiguous amount of control tensors ({control.shape[0]}) passed for {N} desired samples.'
                    print(f'Ambiguous amount of control tensors ({control.shape[0]}) passed for {N} desired samples. Duplicating control map for {N} samples.')
                    c_ = control
                    for k in range(N-1) :
                        control = torch.cat((control,c_))

                control_tuple = torch.tensor_split(control,control.shape[0])
                for c in control_tuple:
                    samples.append(self.sample(c,prompt=prompt,N=1,ddim_steps=ddim_steps,return_control=return_control))
                return samples 
            
            elif len(control.shape) == 3 :
                control = control.unsqueeze(0)
                c_ = control
                for k in range(N-1) :
                    control = torch.cat((control,c_))
                return self.sample(control=control, prompt= prompt, N=N, ddim_steps=ddim_steps, return_control=return_control)
            
            else :
                print(f'Unsupported dimension of control tensor passed : {control.shape}')
                sys.exit(1)
                
        elif N == 1 :
            assert control.shape[0] == 1 and len(control.shape) == 4, f'Error : received control of shape {control.shape} which is unsupported for DDIM sampling. '

        if control.shape[2:] != (512, 512) :
            control = T.Resize((512,512))(control)
        if not ((control >= -1) & (control <= 1)).all() :
            control = T.Normalize([0.5],[0.5])(control)
        
        c_prompt = [self.model.get_learned_conditioning(prompt)]
        cond = {'c_concat': [control.to(self.device)], 'c_crossattn' : c_prompt}            
    
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, N, 
                                                            shape=(4,64,64),
                                                            conditioning = cond,verbose=False)
        samples = (self.model.decode_first_stage(samples)+1)/2
        if return_control :
            return samples, (control +1)/2
        
        return samples

#%%
if __name__ == "__main__":
    s = crowd_sampler()
    path = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/test/map/0006.pt'
    map = rotate_clockwise(torch.load(path))

    sample = s.sample(control=map.unsqueeze(0),
                    prompt='a photo of a crowd of people in a concert, no weather degradation',
                    N=1,
                    return_control=True)

    sample_ = sample[0].squeeze(0).cpu()
    map_ = sample[1].squeeze(0).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the map tensor
    axes[0].imshow(sample_.permute(2, 1, 0))
    axes[0].axis('off')  # Remove axes

    # Display the sample tensor
    axes[1].imshow(map_.permute(2, 1, 0))
    axes[1].axis('off')  # Remove axes

    plt.show()

#%%
#from STEERER.inference import CounterWrapper


#model = CounterWrapper(device=3)

# %%
