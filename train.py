#%%
# resolver new env xcontrol after https://github.com/lllyasviel/ControlNet/issues/612
import sys
import os.path as path

gp = path.dirname(path.dirname(__file__))
if gp not in sys.path :
    sys.path.append(gp)

steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

control_loc = '/home/luk02485/development/ControlNet/ControlNetHome'
if control_loc not in sys.path :
    sys.path.append(control_loc)

from data_processor import CrowdDataSetv2, MapConfig
import torch 
from dataclasses import dataclass

import sys 
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from ControlNetHome.cldm.logger import ImageLogger
from ControlNetHome.cldm.model import create_model, load_state_dict

from inference import ckpt_search

# REQUIREMENTS AND LATEST PACKAGES FILE CHANGES
'''
INSTALLATION OF  CONDA FOR CONTROL NET
xcontrol : https://github.com/lllyasviel/ControlNet/issues/612
conda create xcontrol python=3.9

pip3 install -U xformers torchvision --index-url https://download.pytorch.org/whl/cu118

conda env update xcontrol environment_X.yaml


Running 'python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt'

    The following changes have been done to create and load the model on GPU directly rather than CPU.
    Process kept getting killed because ran out of RAM. 
    Followed merge pull request : https://github.com/andreemic/ControlNet/commit/d04147f037b3da6d50d594876c0bfffecbe9ed13#diff-e349da9602c94a21a14bd5bee4f90d52c843e66a0c94ecf723ac06d77652e257
        - cldm.model.py         at line 12 : changed default value 'location='cuda'.'
                                at  line 26 : replaced with 'model = instantiate_from_config(config.model).cuda()'
        - tool_add_control.py   at line 48 : removed 'location' variable -> runs with default 'cuda' 

UPDATING PACKAGES FOR STEERER
pi3p install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
follow installation isntructions of mmengine on the git #
Changes in doc include : 
    #Steerer.lib.models.heads.base_head.py : line 5, from mmengine.mmengine.model import base_module.py
pip3 install dict_recursive_update
pip3 install yacs


9/7/2024 If upon  initializing the model, the warning :
    "Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used when initializing CLIPTextModel:..."
persists then run "pip install --upgrade transformers"

If the warning "UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed "
SOLVED:
    1) conda env export > NAME.yaml (Here control2.yaml)
    2) re-install pytorch with the correct cuda version. For this build :
        "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    3) This will not run unless you you "conda env update -f NAME.yaml" after

    everything should run without warning/error.
'''
# %%



# Configs
try:
    resume_path = ckpt_search()
except : 
    resume_path = './saves/checkpoints/crowdnet_dict-epoch-60.ckpt'#crowdnet_dict12epochs.ckpt'#'./models/control_sd15_ini.ckpt'
    # 163+47 = 210
#resume_path = './saves/checkpoints/crowdnet_dict-epoch-136_.ckpt'

batch_size = 16
logger_freq = 300
learning_rate = 2e-5#1e-5   SET TO PAPER VALUES
sd_locked = True
only_mid_control = False
AVAILABLE_GPU = 2
#%%
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./ControlNetHome/models/cldm_v15.yaml',location=AVAILABLE_GPU)
#print('OUTSIDE created modeöl. prior to loading dict location is ',model.device)
interm = load_state_dict(resume_path, location=AVAILABLE_GPU)

model.load_state_dict(interm,strict = False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

#print(f'finished instantiation of models, locations are :{model.device} for main model and {model.counter.device} for counter ' )
#model.counter.move_to(model.device)
#%%
# GPU fix to have counter and controlLDM on same device
model.counter = model.counter.to(model.device)


#%%
    # Data
@dataclass
class MapConfig :
    include_box_size : bool = False
    scale_gaussian : bool = False   #for type = 'RGB' this will return a map that seems black but still has the gt info
    save_as : str = 'Tensor'  #or Tensor
    type : str = 'RGB'    #'RGB' or 'HeatMap' -> if Tensor then has dim 1,512,512 instead of 3,512,512
    save_dir : str = "/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/"  #location to save processed maps
    save_for : str = 'train' # or 'val' or 'test'
    load_dir : str = "train"
    CSV_include_count : bool = True 
config = MapConfig

train_set = CrowdDataSetv2(config)
train_loader = DataLoader(train_set, num_workers=16, batch_size=batch_size, shuffle=True)

config.load_dir = 'val'
val_set = CrowdDataSetv2(config)
val_loader = DataLoader(val_set, num_workers=16, batch_size=batch_size, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(devices=[AVAILABLE_GPU],accelerator="gpu", 
                     precision=32, 
                     profiler='simple',
                     limit_train_batches=200,
                     limit_val_batches=50,
                     )
#%%

# Train! 
trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

# %%
