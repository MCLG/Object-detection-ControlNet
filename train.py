from torch import set_float32_matmul_precision, device
from torch.cuda import set_device, empty_cache
import sys
import os.path as path
import os 

gp = path.dirname(path.dirname(__file__))
if gp not in sys.path :
    sys.path.append(gp)

project_root = os.path.abspath(os.path.dirname(__file__))
steerer_loc = os.path.join(project_root, 'STEERER')
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

project_root = os.path.abspath(os.path.dirname(__file__))
control_loc = os.path.join(project_root, 'ControlNetHome')
if control_loc not in sys.path :
    sys.path.append(control_loc)

#from data_jhu import CrowdDataSetv2, MapConfig
#from data_nwpu import CrowdDataSet
from data import CCNetSet, MapConfig
import sys 
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta

from torch.utils.data import DataLoader, random_split#ConcatDataset
from ControlNetHome.cldm.logger import ImageLogger
from ControlNetHome.cldm.model import create_model, load_state_dict, create_model_og, load_state_dict_og
from ControlNetHome.tools.utils import ckpt_search

from pytorch_lightning.callbacks import DeviceStatsMonitor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    

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
    '''
    [New] Creating of empty dictionarry for cldm_v15_2.yaml :
        in the ControlLDM(LatentDiffusion).__init__() in ControlNetHome.cldm.cldm, you need to first comment out the follwoing lines :
            steerer_stage_config = kwargs.pop('counter_stage_config')   line 339
            self.counter = instantiate_from_config(steerer_stage_config)   line 350
        additionally define self.counter = None in the __init__().

        in the cldm_v15_2.yaml, you will also need to comment out 
        counter_stage_config:
            target: STEERER.inference.CounterWrapper

        The reason for all this is because we do not want to add the dict of STEERER to the entire model, this leaves us the freedom of setting it on any device        
    Finally, run :  "python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini_v2.ckpt"
    '''

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #choose here the GPUs you wish to work with
    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"
    AVAILABLE_GPU = [3]

    if len(AVAILABLE_GPU) > 1 :
        # setting the DDP env variables according to doc https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html 
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        # the next 3 variables increase significantly the memory usage of each GPU
        os.environ["NCCL_NSOCKS_PERTHREAD "] = "4" #try 4 after base value is 2 (i think)
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # Start with 8, can increase to 16 if needed
        os.environ["NCCL_MIN_NCHANNELS"] = "2" # if more than2 can cause issues : https://github.com/NVIDIA/nccl/issues/438 
        #Check the PCI devices
        # for InfiniBand or RDMA-capable Ethernet devices: 
        #Use the following command to list all PCI devices and look for keywords like "InfiniBand", "Mellanox", or "RDMA":

        #lspci | grep -i 'infiniband\|rdma\|mellanox'
        #If the command returns nothing, it’s another indication that the system doesn’t have InfiniBand or RoCE hardware.
        # --> returned nothing 
        #os.environ["NCCL_IB_DISABLE"] = "1" #NCCL_IB_DISABLE=1: This disables InfiniBand (IB) and RoCE. When this variable is set, NCCL will fall back to using TCP/IP sockets for communication. This is generally slower and less efficient but can be used if InfiniBand is either unavailable or causing issues.
                                            #NCCL_IB_DISABLE=0 (or not set): This allows NCCL to use InfiniBand/RDMA if available. This is the default behavior because InfiniBand and RoCE offer better performance compared to IP-based sockets.
        #os.environ["NCCL_NET_GDR_LEVEL"] = "LOC" #LOC:Description: Disables GPU Direct RDMA entirely, meaning the NIC will never directly access GPU memory.
                                                 #Usage: Suitable when RDMA should not be used or if it provides no significant benefit.
        #os.environ["NCCL_SOCKET_FAMILY "] = "AF_INET6" 
        if dist.get_rank() == 0 :
                
            store = dist.TCPStore(
                host_name='172.20.100.10',
                port=49152
            )
        else :
            store = None 

        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(seconds=7200000),
            init_method='tcp://172.20.100.10:49152',
            world_size=len(AVAILABLE_GPU), #may have to include DataLoader workers on cpu. Used for if store is specified
            store = store,
            rank=int(os.environ['RANK']) 
        )
        set_device(dist.get_rank())

    try:
        resume_path = ckpt_search()
    except : 
        resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'#'./saves/checkpoints/crowdnet_dict-epoch-60.ckpt'#crowdnet_dict12epochs.ckpt'#'./models/control_sd15_ini.ckpt'

    #STEERER_path = '/home/luk02485/development/ControlNet/STEERER/JHU_mae_54.5_mse_240.6.pth'
    #resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'
    batch_size = 2#16
    logger_freq = 300
    learning_rate = 2e-5#   SET TO PAPER VALUES
    sd_locked = True
    only_mid_control = False

    # load data
    Data = CCNetSet('/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/img',
            '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/map',
            '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/label.csv')
    train, val, test = random_split(Data, [0.7,0.2,0.1])
    train_loader = DataLoader(train, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, num_workers=8, persistent_workers=False ,batch_size=batch_size, shuffle=False)

    #for gpu in AVAILABLE_GPU :    
    #    set_device(device(gpu))
    #empty_cache()

    #changed persistent_workers to False and reduced workers to 4
    #TODO: read https://pytorch.org/docs/stable/data.html to solve timeout issue

    #training strategy :
    accumulate_grad_batches = 32#12 for 2 devices
    nb_training_data = train.__len__()
    nb_val_data = val.__len__()
    max_train_per_epoch = 3200#int(nb_training_data/ (accumulate_grad_batches))
    max_val_per_epoch = 64
    check_val_every_n_epoch = 4
    accumulation_steps = 10000
    smooth_magnitude_tuning_start = 9 #start epoch (validation epoch). consider epoch count starts at 0 !
    magnitude_reg_previous_importance = 0.8 
    #load model
    model = create_model_og('./ControlNetHome/models/cldm_v15_2.yaml').cpu()
    interm = load_state_dict_og(resume_path)
    model.load_state_dict(interm,strict = False)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    if len(AVAILABLE_GPU)>1 :
        '''# setting the DDP env variables according to doc https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html 
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        # the next 3 variables increase significantly the memory usage of each GPU
        os.environ["NCCL_NSOCKS_PERTHREAD "] = "4" #try 4 after base value is 2 (i think)
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # Start with 8, can increase to 16 if needed
        os.environ["NCCL_MIN_NCHANNELS"] = "2" # if more than2 can cause issues : https://github.com/NVIDIA/nccl/issues/438 

        #Check the PCI devices
        # for InfiniBand or RDMA-capable Ethernet devices: 
        #Use the following command to list all PCI devices and look for keywords like "InfiniBand", "Mellanox", or "RDMA":

        #lspci | grep -i 'infiniband\|rdma\|mellanox'
        #If the command returns nothing, it’s another indication that the system doesn’t have InfiniBand or RoCE hardware.
        # --> returned nothing 
        #os.environ["NCCL_IB_DISABLE"] = "1" #NCCL_IB_DISABLE=1: This disables InfiniBand (IB) and RoCE. When this variable is set, NCCL will fall back to using TCP/IP sockets for communication. This is generally slower and less efficient but can be used if InfiniBand is either unavailable or causing issues.
                                            #NCCL_IB_DISABLE=0 (or not set): This allows NCCL to use InfiniBand/RDMA if available. This is the default behavior because InfiniBand and RoCE offer better performance compared to IP-based sockets.
        #os.environ["NCCL_NET_GDR_LEVEL"] = "LOC" #LOC:Description: Disables GPU Direct RDMA entirely, meaning the NIC will never directly access GPU memory.
                                                 #Usage: Suitable when RDMA should not be used or if it provides no significant benefit.
        #os.environ["NCCL_SOCKET_FAMILY "] = "AF_INET6"'''

        model = DDP(model, device_ids=[dist.get_rank()])

        strat = DDPStrategy(
            process_group_backend='nccl',         # Set the backend to NCCL
            timeout=timedelta(seconds=1800),           # Set the timeout to 1 hour
            find_unused_parameters=True,    # was True initially     
            gradient_as_bucket_view=True,
            #static_graph=True      
        )
    else :
        strat = "auto"
    
    model.smooth_magnitude_tuning_start = smooth_magnitude_tuning_start
    model.magnitude_reg_previous_importance = magnitude_reg_previous_importance

    print(f'[TRAINING CONFIG] \n'
          '     loss balancing :\n'
          f'        model.magnitude_regularizer={model.magnitude_regularizer.item()}\n'
          f'        {model.smooth_magnitude_tuning_start=}\n'
          f'        {model.magnitude_reg_previous_importance=}\n'
          '     trainer setting: \n'
          f'        {batch_size=}\n'
          f'        {accumulate_grad_batches=}\n'
          f'        {check_val_every_n_epoch=}\n'
          f'        {max_train_per_epoch=}\n'
          f'        {max_val_per_epoch=}\n'
          f'        {accumulation_steps=}\n'
          f'        effective batch_size = {int(batch_size*accumulate_grad_batches*len(AVAILABLE_GPU))}'
        )
    
    set_float32_matmul_precision('medium')
    trainer = pl.Trainer(devices=AVAILABLE_GPU,accelerator="gpu", 
                        precision=32, 
                        profiler='simple',
                        num_nodes=1,
                        strategy=strat,#'ddp_find_unused_parameters_true',
                        limit_train_batches=max_train_per_epoch,
                        limit_val_batches=max_val_per_epoch,
                        max_steps=accumulation_steps,
                        check_val_every_n_epoch = check_val_every_n_epoch,
                        accumulate_grad_batches = accumulate_grad_batches,

                        enable_checkpointing = True 
                        )

    # Train! 
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    
if __name__ == '__main__':
    main()

#https://github.com/Lightning-AI/pytorch-lightning/pull/3514/files

#%%

# sample procedure :
'''
from torch import set_float32_matmul_precision, device
from torch.cuda import set_device, empty_cache
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

#from data_jhu import CrowdDataSetv2, MapConfig
#from data_nwpu import CrowdDataSet
from data import CCNetSet, MapConfig
import sys 
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta

from torch.utils.data import DataLoader, random_split#ConcatDataset
from ControlNetHome.cldm.logger import ImageLogger
from ControlNetHome.cldm.model import create_model, load_state_dict, create_model_og, load_state_dict_og

from inference import ckpt_search

from pytorch_lightning.callbacks import DeviceStatsMonitor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

resume_path = ckpt_search()
model = create_model_og('./ControlNetHome/models/cldm_v15_2.yaml').cpu()
interm = load_state_dict_og(resume_path)
model.load_state_dict(interm,strict = False)
model = model.to(0)

import einops as ei
import torch 
map = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet/train/map/0105217.pt',map_location= torch.device(0))
image = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet/train/img/0105217.pt',map_location=torch.device(0))

map = torch.unsqueeze(ei.rearrange(map, 'c h w -> h w c' ), 0)
image = torch.unsqueeze(ei.rearrange(image, 'c h w -> h w c' ), 0)
batch = {'jpg' : image, 'txt' : ['a photograph of a crowd of people at a restaurant'], 'hint' : map}

images = model.log_images(batch, sample = True)
reconstr, control, conditioning, samples, samples_cfg_scale_900 = images.values()

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,4,figsize=(20,10))

ax[0].imshow(reconstr.cpu().squeeze(0).permute(1,2,0))
ax[0].axis('off')
ax[1].imshow(control.cpu().squeeze(0).permute(1,2,0))
ax[1].axis('off')
ax[2].imshow(samples.cpu().squeeze(0).permute(1,2,0))
ax[2].axis('off')
ax[3].imshow(samples_cfg_scale_900.cpu().squeeze(0).permute(1,2,0))
ax[3].axis('off')

plt.show()
'''
#%%
