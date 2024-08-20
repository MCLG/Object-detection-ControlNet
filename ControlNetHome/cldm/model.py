import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

# modified
def load_state_dict(ckpt_path, STEERER_path = None ,location = None):

    if location is not None :
        assert isinstance(location,int)
        location = f'cuda:{location}'
    else :
        location = 'cpu'


    if not STEERER_path :
            
        
        _, extension = os.path.splitext(ckpt_path)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(ckpt_path, device=location)
        else:
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
        state_dict = get_state_dict(state_dict)
        print(f'Loaded state_dict from [{ckpt_path}]')
        return state_dict

    assert os.path.exists(STEERER_path), 'Invalid path to STEERER weights !'
    _, extension = os.path.splitext(ckpt_path)
    
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    state_dict_steerer = get_state_dict(torch.load(STEERER_path,map_location=torch.device(location)))

    for key in list(state_dict_steerer.keys()):
        new_key = f"counter.{key}"
        state_dict_steerer[new_key] = state_dict_steerer.pop(key)

    state_dict_steerer =get_state_dict(state_dict_steerer)
    print(f'Loaded state_dict from [{STEERER_path}]')

    concat_state_dicts = dict(state_dict)
    concat_state_dicts.update(state_dict_steerer) 
    print('Successfully concatenated Steerer and ControlNet state_dict')
    
    return concat_state_dicts

# modified
def create_model(config_path,location=None):
    '''
    specify location to load on device, the pl.Trainer will then not auto handle the devices available during training. If not specified (location=None),
    the original function from https://github.com/lllyasviel/ControlNet/blob/main/cldm/model.py will be executed which loads the model on the cpu.
    '''
    if location is None :
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model).cpu()
        print(f'Loaded model config from [{config_path}] on device {model.device}')
        return model
    
    assert isinstance(location, int)

    location = f'cuda:{location}'
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).to(torch.device(location))
    print(f'Loaded model config from [{config_path}] on device {model.device}')
    return model

# original functions
''' 
def load_state_dict_og(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict'''


def create_model_og(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model