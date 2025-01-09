from torch import set_float32_matmul_precision, device
from torch.cuda import set_device, empty_cache
import sys
import os.path as path
import os 
import numpy as np

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

from data_tools.utils import CCNetSet, custom_collate
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split#ConcatDataset
from ControlNetHome.cldm.model import create_model, load_state_dict, create_model_og, load_state_dict_og
import torch.distributed as dist

#deprecation warning removed :
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

def set_training(total_steps : int, data_len : int, accumulation_steps = 32, bs : int) -> tuple: 
    iterations_needed = accumulation_steps * total_steps
    batches = round(data_len/bs)
    epochs = np.ceil(iterations_needed / batches)
    return epochs, batches

def main():
    
    ###########################################################################################################################
    # CHANGE HERE TO YOUR SETTING 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # available GPUs
    AVAILABLE_GPU = [2]                            # Choose one GPU
    resume_path = 'epoch=0-step=50-v2.ckpt'#       # weight_dict or './ControlNetHome/models/control_sd15_ini_v2.ckpt' for untrained dict
    batch_size = 2                                 # Achieved maximum of 2
    learning_rate = 2e-5                           # set to paper values
    sd_locked = True                               # Keep True
    only_mid_control = False                       # keep True
    num_workers = 15                               # DataLoader workers

    #training strategy :
    accumulate_grad_batches = 32                   # should be set higher for smaller batch-size
    max_val_per_epoch = 64
    check_val_every_n_epoch = 4
    accumulation_steps = 10000
    ###########################################################################################################################
    
    # load data 
    Data = CCNetSet('/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/img',      # img folder
            '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/map',              # map folder
            '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/label.csv',        # label.csv path
            '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/mean')             # means folder

    train, val = random_split(Data, [0.81,0.19])
    train_loader = DataLoader(
                    train, 
                    num_workers=num_workers, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    pin_memory=True,
                    collate_fn = custom_collate,
                    persistent_workers = True
                    )
    val_loader = DataLoader(
                    val, 
                    num_workers=1, 
                    persistent_workers=False ,
                    batch_size=batch_size, 
                    shuffle=False,
                    collate_fn = custom_collate
                    )

    nb_training_data = train.__len__()
    nb_val_data = val.__len__()

    epochs, batch_iterations = set_training(
        total_steps = accumulation_steps, 
        data_len = nb_training_data, 
        accumulation_steps = accumulate_grad_batches, 
        bs = batch_size)
     
    # Loading model
    model = create_model_og('./ControlNetHome/models/cldm_v15_2.yaml').cpu()
    interm = load_state_dict_og(resume_path)
    model.load_state_dict(interm,strict = False)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    strat = "auto"
    print(f'[TRAINING CONFIG] \n'
          '     trainer setting: \n'
          f'        {batch_size=}\n'
          f'        {accumulate_grad_batches=}\n'
          f'        {check_val_every_n_epoch=}\n'
          f'        {batch_iterations=}\n'
          f'        {max_val_per_epoch=}\n'
          f'        {accumulation_steps=}\n'
          f'        effective batch_size = {int(batch_size*accumulate_grad_batches*len(AVAILABLE_GPU))}\n'
          f'        loss_type = {model.control_eval}'
        )
    
    #import torch
    #torch.autograd.set_detect_anomaly(True)
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./training_dict/',
        every_n_train_steps = 50
    )
    empty_cache()
    set_float32_matmul_precision('medium')
    trainer = pl.Trainer(devices=AVAILABLE_GPU,accelerator="gpu", 
                        precision=32,
                        num_nodes=1,
                        strategy=strat,#'ddp_find_unused_parameters_true',
                        limit_train_batches=batch_iterations,
                        limit_val_batches=max_val_per_epoch,
                        max_steps=accumulation_steps,
                        check_val_every_n_epoch = check_val_every_n_epoch,
                        accumulate_grad_batches = accumulate_grad_batches,
                        enable_checkpointing = True,
                        detect_anomaly=False ,
                        callbacks=[checkpoint_callback]
                        )

    # Train! 
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    #started : mse_tv 13181steps
    # 1 run :  w2-count 600steps
    # starting new at mse_tv 13181steps w/ loss w2-tv --> crashed at 13% epoch 0
    # 26/1224 started new w2-tv with SD-dict
    #27/2024 5ep 900steps --> works fine --> relaunched with better img logging
    #stopped at ep1-200steps to check timestep issue
    # trianing was seeded after first run of GMM. Need to start training again from 0
    # crashed at 100steps --> relaunched from 100steps 27/12 (19h38) 
    # same crash --> in DivergenceLoss non_z_componenets.shape[0] was sometime smaller than the number of centroids to construct which result in assert error in GMM.
    #   This was due to loss of pixel information during dimension reduction.
    #   relaunched at steps 200 27/12/2024 (23h34) --> total steps already 300steps
    #crashed at ep11/1795steps (total_Steps=2095) -->relaunched with save 'epoch=11-step=1750.ckpt'
    #
    #
    # FINISHED TRAINING :
    # total_steps =12095 --> starting training from those steps 
    # crashed at 100steps
    # crashed again at 100steps --> relaunchd (total=12295)
    # +1 12395
    # crashed at ep7-1200steps --> total = 12395+1200 = 13595
    # relaunched
    # crashed at 50steps -> relaunched -> total steps = 13645
if __name__ == '__main__':
    main()
