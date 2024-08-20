#%%
import torch
import sys
control_loc = '/home/luk02485/development/ControlNet/ControlNetHome'
if control_loc not in sys.path :
    sys.path.append(control_loc)

#from ControlNetHome.share import *
from data_jhu import density_map, MapConfig

import tempfile
from ControlNetHome.cldm.model import create_model, load_state_dict
from ControlNetHome.annotator.util import resize_image

import numpy as np
from ControlNetHome.cldm.ddim_hacked import DDIMSampler

from PIL import Image
import pytorch_lightning as pl 

from torchvision import transforms as T

from ControlNetHome.cldm.cldm import ControlLDM

import glob
import os 
from typing import Tuple, Optional
import random as rd 

from ControlNetHome.ldm.util import rotate_clockwise

import matplotlib.pyplot as plt

from STEERER.inference import CounterWrapper

from mmengine.config.config import Config


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




class crowd_sampler(pl.LightningModule):
    '''
    WIP - A wrapper class for the trained crowd generating control net. Includes :
        - a rd gaussian map generator.
        - sample method.
    '''
    def __init__(self,resume_path = None, device = 2, activate_counter = False ):
        super().__init__()

        if activate_counter :
            self.counter = CounterWrapper(device= torch.device(2))#.to(torch.device(self.device))
            #self.counter = self.counter.to(self.device)
            #config = Config.fromfile('configs/SHHB_final.py')
            #model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)

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

        RETURNS
            torch.Tensor of dim 1,3,512,512 or a list of such tensors.
            if return_control == True, then will return a tuple of tensor ([1,3,512,512], [1,3,512,512]) or a tuple of two lists of such tensors.
        '''
        assert (control is None or isinstance(control, torch.Tensor)), f'Control must be None or torch.Tensor in {self}'
        
        if control is None :
            control = [self.get_rd_map() for _ in range(N)]
            control = torch.stack(control)
            return self.sample(control=control, prompt= prompt, N=N, ddim_steps=ddim_steps, return_control=return_control)
            
        if N > 1 :
            samples = []
            if return_control :
                control_return = []

            if len(control.shape) == 4:
                if control.shape[0] != N :
                    assert control.shape[0] == 1, f'Ambiguous amount of control tensors ({control.shape[0]}) passed for {N} desired samples.'
                    print(f'Ambiguous amount of control tensors ({control.shape[0]}) passed for {N} desired samples. Duplicating control map for {N} samples.')
                    c_ = control
                    for k in range(N-1) :
                        control = torch.cat((control,c_))

                control_tuple = torch.tensor_split(control,control.shape[0])
                for c in control_tuple:
                    sc = self.sample(c,prompt=prompt,N=1,ddim_steps=ddim_steps,return_control=return_control)
                    try:
                        samples.append(sc[0])
                        control_return.append(sc[1])
                    except:
                        samples.append(sc)

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
    
    def confidence_sample(self, p : float, control : Optional[torch.Tensor] = None, true_count : Optional[list[int]] = None, prompt = '', N=1, ddim_steps=50) -> torch.Tensor:

        assert hasattr(self, 'counter'), 'confidence sampling requires crowd_sampler to be initialized with activate_counter=True. Default is False.'
        assert p >= 0 and p <= 100 , 'confidence "p" needs to be in range (0,100)'

        def conf(x : float, y : float ) -> float:
            assert x >= 0 and y >= 0, 'invalid input passed in confidence_sample.conf() '
            return max(0, 100 - abs(x-y)/np.floor((x+y)/2)*100)
        
        output = []
        den_output = []
        while N != 0 :
            
            print(N)
            samples = self.sample(control=control, prompt=prompt,N=N, return_control= True)
            
            #samples = samples.to(self.counter.device)
            if isinstance(samples,list):
                remove = []
                for k in range(len(samples)) :
                    c,den = self.counter.get_count(samples[0])
                    if true_count is None :
                        count = control[k].squeeze(0)[0].sum().item()       #only supported if the map is a true gaussian
                    else : 
                        count = true_count[k]
                    print(f'c={c}/count={count}/conf={conf(count,c)}')
                    if conf(count,c) >= p :    
                        output.append(samples[k])
                        den_output.append(den)
                        remove.append(k)
                        N -= 1
                if not remove :
                    break
                remove.sort(reverse=True)
                control = [tens for tens in torch.tensor_split(control, control.shape[0])]
                for k in remove :
                    control.pop(k)
                    true_count.pop(k)
                control = torch.cat(control)
            else :
                c,den = self.counter.get_count(samples[0])
                if true_count is None:
                    count = control.squeeze(0)[0].sum().item()          #only supported if the map is a true gaussian
                else : 
                    count = true_count[0]
                print(f'c={c}/count={count}/conf={conf(count,c)}')
                if conf(count, c ) >= p :
                    output.append(samples)
                    den_output.append(den)
                    N -= 1
                
        return output,den_output
        

def main_single() :

    # Load the model
    s = crowd_sampler(activate_counter=True)
    s.counter = s.counter.to(s.device)      # to properly load steerer on the same device

    #load map
    path = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/map/0017.pt'
    map = torch.load(path) #rotate_clockwise(torch.load(path))

    # sample and density maps
    sample,dens = s.confidence_sample(p = 50, control=map.unsqueeze(0), true_count= [13],
                    prompt='a photo of a crowd of people running, no weather degradation',
                    N=1)
    try :
        print(len(sample),len(sample[0]),type(sample),type(sample[0]))
        print(sample[0][0].shape)
    except :
        pass   
    sample = sample[0]     
    # reshape the samples 
    resolution = (1536, 2048)
    sample_ = T.ToPILImage()(sample[0].squeeze(0).cpu())
    map_ = T.ToPILImage()(map.squeeze(0).cpu())
    
    
    resize_transform = T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC)
    sample_ = resize_transform(sample_)
    map_ = resize_transform(map_)
    
    #convert back to tensor
    sample_ = T.ToTensor()(sample_)
    map_ = T.ToTensor()(map_)
    den = dens[0]

    # plot 
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Display the sample tensor
    axes[0,0].imshow(sample_.permute(1,2,0))
    axes[0,0].axis('off')  # Remove axes

    # Display the map tensor
    axes[0,1].imshow(map_.permute(1,2,0))
    axes[0,1].axis('off')  # Remove axes

    # Display the density tensor
    axes[0,2].imshow(den.squeeze(0).squeeze(0).cpu().numpy())
    axes[0,2].axis('off')

    # Display the true image tensor
    true_img = (torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/img/0017.pt')+1)/2
    _, approx_density = s.counter.get_count(true_img)

    axes[1,0].imshow(resize_transform(true_img).squeeze(0).cpu().permute(1,2,0))
    axes[1,0].axis('off')

    # Display the true density tensor
    axes[1,1].imshow(resize_transform(approx_density).squeeze(0).squeeze(0).cpu().numpy())
    axes[1,1].axis('off')

    axes[1,2].imshow(torch.randn(resolution))
    axes[1,2].axis('off')


    fig.savefig(os.path.join('./imgs_dump', 'sample_img.png'))
    plt.show()

def main_plural() :
    # Load the model
    s = crowd_sampler(activate_counter=True)
    s.counter = s.counter.to(s.device)      # to properly load steerer on the same device

    #load map
    paths = ['/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/map/0017.pt',
             '/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/map/0018.pt',
             '/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/map/0019.pt']
    
    maps = [torch.load(path).unsqueeze(0) for path in paths]#rotate_clockwise(torch.load(path))

    # sample and density maps
    sample,dens = s.confidence_sample(p = 50, control=torch.cat(maps), true_count= [13,80,27],
                    prompt='a photo of a crowd of people in the street, no weather degradation',
                    N=len(paths))
    
    # reshape the samples 
    resolution = (1536, 2048)
    sample_ = T.ToPILImage()(sample[0].squeeze(0).cpu())
    map_ = T.ToPILImage()(maps[0].squeeze(0).cpu())
    
    
    resize_transform = T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC)
    sample_ = resize_transform(sample_)
    map_ = resize_transform(map_)
    
    #convert back to tensor
    sample_ = T.ToTensor()(sample_)
    map_ = T.ToTensor()(map_)
    den = dens[0]

    # plot 
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Display the sample tensor
    axes[0,0].imshow(sample_.permute(1,2,0))
    axes[0,0].axis('off')  # Remove axes

    # Display the map tensor
    axes[0,1].imshow(map_.permute(1,2,0))
    axes[0,1].axis('off')  # Remove axes

    # Display the density tensor
    axes[0,2].imshow(den.squeeze(0).squeeze(0).cpu().numpy())
    axes[0,2].axis('off')

    # Display the true image tensor
    true_img = (torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/train/img/0017.pt')+1)/2
    _, approx_density = s.counter.get_count(true_img)

    axes[1,0].imshow(resize_transform(true_img).squeeze(0).cpu().permute(1,2,0))
    axes[1,0].axis('off')

    # Display the true density tensor
    axes[1,1].imshow(resize_transform(approx_density).squeeze(0).squeeze(0).cpu().numpy())
    axes[1,1].axis('off')

    axes[1,2].imshow(torch.randn(resolution))
    axes[1,2].axis('off')


    fig.savefig(os.path.join('./imgs_dump', 'sample_img.png'))
    plt.show()
    


if __name__ == "__main__": 
    #main_single()
    main_plural()


