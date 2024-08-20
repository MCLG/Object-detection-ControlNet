#%%
def main():
    
    # To allow training with multiple GPU's from 'vid-moseisley', we need to set the CUDA_VISIBLE_DEVICES to the desired devices prior to importing pytorch.
    # This excludes the device:0 as processes are always active on it and this prevents the pl.Trainer to star new processes, even when all models 
    # and its submodules are initialized on the CPU.

    import os
    #for multi-devices training, uncomment this :
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3' #choose here the GPUs you wish to work with
    import torch

    '''print(f'torch.cuda.device_count() : {torch.cuda.device_count()}\n')
    for i in range(torch.cuda.device_count()):
        print(f"Device-index {i}: {torch.cuda.get_device_name(i)}/ Device-name: {i+1}")
        # Check no active process is running.
        assert 'no processes are running' in torch.cuda.list_gpu_processes(i)
        , f'there is an active process {torch.cuda.list_gpu_processes(i)} on {torch.cuda.get_device_name(i)}/Device-name:{i+1}'
    '''

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

    from data_jhu import CrowdDataSetv2, MapConfig
    from data_nwpu import CrowdDataSet
    import sys 
    import pytorch_lightning as pl


    from torch.utils.data import DataLoader, ConcatDataset
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

    # Configs
    try:
        resume_path = ckpt_search()
    except : 
        resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'#'./saves/checkpoints/crowdnet_dict-epoch-60.ckpt'#crowdnet_dict12epochs.ckpt'#'./models/control_sd15_ini.ckpt'

    STEERER_path = '/home/luk02485/development/ControlNet/STEERER/JHU_mae_54.5_mse_240.6.pth'
    resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'
    batch_size = 16
    logger_freq = 300
    learning_rate = 2e-5#1e-5   SET TO PAPER VALUES
    sd_locked = True
    only_mid_control = False
    AVAILABLE_GPU = [1,2]

    model = create_model('./ControlNetHome/models/cldm_v15_2.yaml',location=None).cpu()

    interm = load_state_dict(resume_path, STEERER_path=STEERER_path,location=None)

    model.load_state_dict(interm,strict = False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    
    # GPU fix to have counter and controlLDM on same device
    #model.counter = model.counter.to(model.device)



    # Data        
    config = MapConfig

    train_set = CrowdDataSetv2(config)

    confignwpu = MapConfig()
    confignwpu.save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/'
    nwpuset = CrowdDataSet(confignwpu)

    train_set = ConcatDataset([train_set,nwpuset])
    train_loader = DataLoader(train_set, num_workers=16, batch_size=batch_size, shuffle=True)

    config.load_dir = 'val'
    val_set = CrowdDataSetv2(config)
    val_loader = DataLoader(val_set, num_workers=16, persistent_workers=True ,batch_size=batch_size, shuffle=False)

    #logger = ImageLogger(batch_frequency=logger_freq, batch_size= batch_size, log_first_step= True)
    
    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(devices=[0,2],accelerator="gpu", 
                        precision=32, 
                        profiler='simple',
                        num_nodes=1,
                        strategy='ddp_find_unused_parameters_true',
                        limit_train_batches=200,
                        limit_val_batches=50,
                        max_steps=7000,
                        check_val_every_n_epoch = 5,
                        #callbacks=[logger],
                        accumulate_grad_batches= 4
                        )

    # Train! 
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    
if __name__ == '__main__':
    main()

#https://github.com/Lightning-AI/pytorch-lightning/pull/3514/files

#%%
# sample procedure :
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
# %%
