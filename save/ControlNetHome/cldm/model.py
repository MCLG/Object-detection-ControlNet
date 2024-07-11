import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

# modified
def load_state_dict(ckpt_path, steerer_on = True ,location=0):
    
    if not steerer_on :
            
        location = f'cuda:{location}'
        _, extension = os.path.splitext(ckpt_path)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(ckpt_path, device=location)
        else:
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
        state_dict = get_state_dict(state_dict)
        print(f'Loaded state_dict from [{ckpt_path}]')
        return state_dict
    
    steerer_path = '/home/luk02485/development/ControlNet/STEERER/JHU_mae_54.5_mse_240.6.pth'

    location = f'cuda:{location}'
    _, extension = os.path.splitext(ckpt_path)
    
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    state_dict_steerer = get_state_dict(torch.load(steerer_path,map_location=torch.device(location)))

    for key in list(state_dict_steerer.keys()):
        new_key = f"counter.{key}"
        state_dict_steerer[new_key] = state_dict_steerer.pop(key)

    state_dict_steerer =get_state_dict(state_dict_steerer)
    print(f'Loaded state_dict from [{steerer_path}]')

    concat_state_dicts = dict(state_dict)
    concat_state_dicts.update(state_dict_steerer) 
    print('Successfully concatenated Steerer and ControlNet state_dict')
    
    return concat_state_dicts

# modified
def create_model(config_path,location=0):
    location = f'cuda:{location}'
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).to(torch.device(location))
    print(f'INSIDE create_model() --- location of newly created model IS [{model.device}]')
    print(f'Loaded model config from [{config_path}]')
    return model
