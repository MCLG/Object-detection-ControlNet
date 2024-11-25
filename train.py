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

from data import CCNetSet, MapConfig
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split#ConcatDataset
from ControlNetHome.cldm.model import create_model, load_state_dict, create_model_og, load_state_dict_og
from ControlNetHome.tools.utils import ckpt_search

import torch.distributed as dist

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #choose here the GPUs you wish to work with
    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    AVAILABLE_GPU = [3]

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
