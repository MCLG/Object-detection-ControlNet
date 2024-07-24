
## Installation and setting up the ControlNet: 
Clone the Git rep and follow the following instructions :

1) create a conda venv `conda create xcontrol python=3.9`.Then activate the env. This followed the steps from https://github.com/lllyasviel/ControlNet/issues/612. 

    First install this package 
    `pip3 install -U xformers torchvision --index-url https://download.pytorch.org/whl/cu118`

    Then install the rest of the packages from the .yaml file `conda env update xcontrol environment_X.yaml`

2) Follow the first steps of Step 3 from https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md .

    a) download the file "v1-5-pruned.ckpt" from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main . This file **needs** to be located in  "./ControlNetHome/models/v1-5-pruned.ckpt". 

    b) Navigate inside "./ControlNetHome" and execute
    
     ```
     python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
     ```

     The newly created file "./ControlNetHome/models/control_sd15_ini.ckpt" should appear.

3) Installing STEERER : This code of STEERER has been written with an obsolete version of mmcv. We now require mmengine aswell and had to modify imports inside the code. Do 
```
pi3p install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
```

4) Strictly follow the installation instructions from https://github.com/open-mmlab/mmengine to install mmengine.

5) Make sure the following line is correctly written in the file "./Steerer./lib.models/heads.base_head.py" at line 5 :
`from mmengine.mmengine.model import base_module.py`

6) Finish installing the other packages 
```
pip3 install dict_recursive_update
pip3 install yacs
```

7) For STEERER to work, you need to download the weights _"JHU_mae_54.5_mse_40.6"_ from the git https://github.com/taohan10200/STEERER/tree/main

### Additional warnings that may occur:
 If upon  initializing the model
 
 (include graphic here), 
 
1) the warning :
_"Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used when initializing CLIPTextModel:..."_
arises then run 
```
pip install --upgrade transformers
```

2) the warning _"UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed "_
follow the steps :
    1) Create backup of current env `conda env export > NAME.yaml` (e.g. _control2.yaml_)
    2) re-install pytorch with the correct cuda version. For this build :
        `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    3) Then install the backup env again `conda env update -f NAME.yaml`.

    Everything should run without warning/error now.

## Changes to the original ControlNet Git

The following changes have been done to create and load the model on GPU directly rather than CPU.
    Process kept getting killed because ran out of RAM. 
    Followed merge pull request : https://github.com/andreemic/ControlNet/commit/d04147f037b3da6d50d594876c0bfffecbe9ed13#diff-e349da9602c94a21a14bd5bee4f90d52c843e66a0c94ecf723ac06d77652e257
        - cldm.model.py         at line 12 : changed default value 'location='cuda'.'
                                at  line 26 : replaced with 'model = instantiate_from_config(config.model).cuda()'
        - tool_add_control.py   at line 48 : removed 'location' variable -> runs with default 'cuda' 
