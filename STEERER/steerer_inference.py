import os
import pprint

from . lib.core.Counter import Counter
from . lib.utils.utils import create_logger, random_seed_setting
from . lib.utils.modelsummary import get_model_summary
from . lib.core.cc_function import test_cc

import torch

from . lib.models.build_counter import Baseline_Counter
from . lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmengine.config.config import Config #from mmcv import Config, DictAction

import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
import torch
from typing import Optional, Tuple
from torch.nn.functional import interpolate

'''
    Contains Wrapper class for STEERER
'''

# pi3p install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
# install mmengine as described on the git #
#Changes in doc include : 
    #Steerer.lib.models.heads.base_head.py : line 5, from mmengine.mmengine.model import base_module.py
#pip3 install dict_recursive_update
#pip3 install yacs

class CounterWrapper(Baseline_Counter):

    def __init__(self,
                 freeze =  True, 
                 path = None, 
                 path_to_config = './STEERER/configs/JHU_final.py', 
                 device = 'cpu',
                 *args, **kwargs): #self, config=None,weight=200, route_size=(64,64),device=None):
        
        self.instantiate_from_config(kwargs)
        self.device = torch.device(device)
        config = Config.fromfile(path_to_config)
        super().__init__(config.network, config.dataset.den_factor, config.train.route_size, torch.device(device))

        if path is None :
            pretrained_dict = torch.load('./nwpu_pre_trained.pth', map_location=self.device)
        elif os.path.exists(path):
            pretrained_dict = torch.load(path, map_location= self.device)
        else :
            pprint(f'path:{path} to weights of counter not found. Check for path again or select None for base JHU-weights.')

        self.load_state_dict(pretrained_dict,strict=False)
        
    def instantiate_from_config(self, config):
        #symbolic method to replicate the loading of STEERER like stable diffusion
        pass
        
    '''def freeze(self):
        for param in  self.parameters():
            param.requires_grad =  False
        self.freeze_weight = True''' 

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward_wrapper(self, input):
        img,labels,mode = input  
        return self(img,labels = labels, mode = mode)

    def get_count(self, img : torch.Tensor, mode = 'val') -> torch.Tensor :
        # assumes format B,C,H,W of input image. Computes the Gaussian density map of the tensor(s)
        # if mode = 'train' then the model uses gradient checkpointing if img.requires_grad = True
        
        assert len(img.shape) > 2, f'input {img.shape=} but requires RGB 3,H,W or MAP 1,H,W shape of input tensor. '

        if len(img.shape) == 3 :
            img = img.unsqueeze(0).to(self.device)

        if img.shape[-2:] != torch.Size([1536,2048]):
            img = interpolate(img, size = (1536,2048), mode='bicubic')
        
        if mode == 'val' :
            with torch.no_grad():
                return self(img, labels=None, mode='val')
        elif mode == 'train' :
            return self(img, labels=None, mode='val')
        else :
            print(f'unknown {mode=} passed to get_count. Options are "val" or "train". ')
            sys.exit(0)
    
def main() :
    noise = torch.randn(3,1536,2048)
    counter = CounterWrapper(device=torch.device(0))

    dmap = counter.get_count(noise)
    print(f'{dmap.sum().item()}')
    
if __name__ == '__main__' :
    main()