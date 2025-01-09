import torch
import os 
import sys 
import torch.nn.functional as F
import matplotlib.pyplot as plt 

project_root = os.path.abspath(os.path.dirname(__file__))
control_loc = os.path.join(project_root, 'ControlNetHome')
if control_loc not in sys.path :
    sys.path.append(control_loc)

from ControlNetHome.cldm.model import create_model, load_state_dict, create_model_og, load_state_dict_og
from ControlNetHome.ldm.models.diffusion.ddim import DDIMSampler
from ControlNetHome.cldm.cldm import enhance_tensor
from STEERER.lib.models.build_counter import freeze_model
from STEERER.steerer_inference import CounterWrapper
from ControlNetHome.tools.CGsampling_plotter import sample as CG_plot_sample

import argparse

def main() :
    
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("sampling_method", help=" 'ddim', 'count_guidance_ddpm' or 'ddim_guided' ")
    parser.add_argument("nb_sample", help = 'number of samples to produce')
    args = parser.parse_args()

    ###########################################################################################################################
    # CHANGE HERE TO YOUR SETTING /lightning_logs/version_4/checkpoints/epoch=82-step=13188.ckpt   ./epoch=7-step=1200.ckpt
    resume_path = '/home/luk02485/development/ControlNet_2/control_mse_tv/lightning_logs/version_4/checkpoints/epoch=82-step=13188.ckpt'    # weights_path
    gpu = torch.device(0)                       # device
    path_to_img = '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/img/029002.pt'   #image_path
    path_to_map = '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/map/029002.pt'   #map_path
    prompt = ['a group of people sitting on chairs in a room']  #text prompt or [''] for promptless sampling

    # specific to count_guidance_ddpm 
    control_eval = 'MSE'    # for count guidance sampling specify the control loss ("MSE"/ "W2-count" / "w2-TV" / "count TV")
    path_to_means = '/net/vid-raxus/storage/deeplearning/users/luk02485/CC_filter_data/train/mean/029002.pt' # centroids_path
    ############################################################################################################################

    #loading the model
    model = create_model_og('./ControlNetHome/models/cldm_v15_2.yaml').cpu()
    interm = load_state_dict_og(resume_path)
    model.load_state_dict(interm,strict = False)
    model = model.to(gpu)
    model.counter = CounterWrapper(path = model.counter_dict).to(gpu)
    freeze_model(model.counter)

    #ddim parameters
    guess_mode = True
    strength = 1. #does not work outside of interval [0,1]
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    steps = 250

    #loading images
    dmap = torch.load(path_to_map).to(model.device)
    img = torch.load(path_to_img).to(model.device).unsqueeze(0)
    batch = {
        'jpg' : torch.randn(1,512,512,3),
        'hint' : dmap.permute(1,2,0).unsqueeze(0),
        'txt' : prompt,
        'mean' : [torch.randn(1,2)*512]
    }
    
    nb_samples = int(args.nb_sample)
    
    if args.sampling_method == 'ddim' :
        folder_name=create_folder(args.sampling_method, nb_samples, steps = steps)
        for k in range(nb_samples) : 
            ddim_sample(model=model, batch=batch, gpu=gpu, img=img, dmap=dmap.unsqueeze(0), guided = False, folder_name=folder_name)
    elif args.sampling_method == 'ddim_guided' :
        folder_name=create_folder(args.sampling_method, nb_samples, steps = steps)
        for k in range(nb_samples) : 
            ddim_sample(model=model, batch=batch, gpu=gpu, img=img, dmap=dmap.unsqueeze(0), guided = True, folder_name=folder_name)
    elif args.sampling_method == 'count_guidance_ddpm' :
        folder_name=create_folder(args.sampling_method, nb_samples, steps = None)
        means = torch.load(path_to_means, map_location = gpu)            
        model.DivLoss.device = model.device
        for k in range(nb_samples) : 
            count_guidance_ddpm(model=model, gpu=gpu, dmap=dmap.unsqueeze(0), img=img, means = means, prompt= prompt,folder_name=folder_name)
    else :
        raise ValueError(f'Invalid sampling method {args.sampling_method} passed. ')

def ddim_sample(
    model,
    batch,
    gpu,
    img,
    dmap,
    guided,
    folder_name,
    steps=50
) :
        
    #sending batch to latent space
    z, c = model.get_input(batch, model.first_stage_key, dropout = False)
    c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

    #ddim parameters
    ddim_sampler = DDIMSampler(model)
    ddim_steps = steps
    unconditional_guidance_scale = 1.
    eta = 0.
    temperature = .1
    noise_dropout = 0.1

    #ddim sampling
    b, c, h, w = cond["c_concat"][0].shape
    shape = (model.channels, h // 8, w // 8)
    batch_size=1
    samples, _ = ddim_sampler.sample(
        ddim_steps, 
        dmap,
        batch_size, 
        shape, 
        cond,
        x0 = torch.randn(1,4,64,64),
        unconditional_guidance_scale = unconditional_guidance_scale,
        verbose=False,
        eta = eta,
        temperature=temperature, 
        noise_dropout = noise_dropout, #prob between 0 and 1,
        guided=guided
        )

    #decode 
    samples = model.decode_first_stage(samples)
    samples = enhance_tensor(samples)

    # get density map
    reconstructed_density_map = model.counter.get_count(samples)
    dmap = batch['hint'].permute(0,3,1,2)

    # increase dimension and enhance imgs
    samples = F.interpolate(samples, size=(1200, 1200), mode='bicubic', align_corners=False)
    dmap = F.interpolate(dmap, size=(1200, 1200), mode='bicubic', align_corners=False)
    img = F.interpolate(img, size=(1200, 1200), mode='bicubic', align_corners=False)
    reconstructed_density_map = F.interpolate(reconstructed_density_map, size=(1200, 1200), mode='bicubic', align_corners=False)

    #saving results
    fig, axis = plt.subplots(2,2, figsize=(12,12))
    axis[0,0].imshow(img.squeeze(0).permute(1,2,0).cpu())
    axis[0,0].axis('off')

    axis[0,1].imshow(samples.squeeze(0).permute(1,2,0).cpu())
    axis[0,1].axis('off')
    axis[1,0].imshow(dmap.squeeze(0).permute(1,2,0).cpu())
    axis[1,0].axis('off')
    axis[1,1].imshow(reconstructed_density_map.squeeze(0).permute(1,2,0).cpu())
    axis[1,1].axis('off')
    fig.tight_layout()

    filename = os.path.join(folder_name, 'ddim-sample_0.png')
    counter = 0
    while os.path.exists(filename):
        counter += 1
        filename = os.path.join(folder_name, f'ddim-sample_{counter}.png')
    
    plt.savefig(
        filename,
        bbox_inches='tight', pad_inches=0)
    plt.close()

def count_guidance_ddpm(
    model,
    gpu,
    dmap,
    img,
    means,
    prompt,
    folder_name
) :

    sampling_data_csv = 'temp_sampling_process.csv'
    denoising_steps = 1000
            
    with torch.enable_grad():
        print(f'{model.device=}, {dmap.device=}, {means.device=}')
        sample = model.count_guided_sampling(dmap.to(gpu),
            prompt = prompt,
            mean = [means.to(gpu)],
            denoising_steps = denoising_steps, 
            progress_track = sampling_data_csv)
    
    sample = enhance_tensor(sample)
    reconstructed_density_map = model.counter.get_count(sample)

    sample = F.interpolate(sample, size=(1200, 1200), mode='bicubic', align_corners=False)
    dmap = F.interpolate(dmap, size=(1200, 1200), mode='bicubic', align_corners=False)
    img = F.interpolate(img, size=(1200, 1200), mode='bicubic', align_corners=False)
    reconstructed_density_map = F.interpolate(reconstructed_density_map, size=(1200, 1200), mode='bicubic', align_corners=False)

    graph_loc = './ddpm_guidance_graph_0.png'
    counter = 0
    while os.path.exists(graph_loc):
        counter += 1
        graph_loc = f'./ddpm_guidance_graph_{counter}.png'

    CG_plot_sample(loc=graph_loc, temp_file=sampling_data_csv)

    #saving results
    fig, axis = plt.subplots(2,2, figsize=(12,12))
    axis[0,0].imshow(img.squeeze(0).permute(1,2,0).cpu())
    axis[0,0].axis('off')

    axis[0,1].imshow(sample.squeeze(0).permute(1,2,0).cpu())
    axis[0,1].axis('off')
    axis[1,0].imshow(dmap.squeeze(0).permute(1,2,0).cpu())
    axis[1,0].axis('off')
    axis[1,1].imshow(reconstructed_density_map.squeeze(0).permute(1,2,0).cpu())
    axis[1,1].axis('off')
    fig.tight_layout()


    filename = os.path.join(folder_name, 'ddpm_guidance-sample_0.png')
    counter = 0
    while os.path.exists(filename):
        counter += 1
        filename = os.path.join(folder_name, f'ddpm_guidance-sample_{counter}.png')

    plt.savefig(
        filename,
        bbox_inches='tight', pad_inches=0)
    plt.close()

def create_folder(type,n, steps = None):
    if not steps :
        folder_name=f'./samples-{type}-{n}'
        if os.path.exists(folder_name) :
            folder_name=f'./samples-{type}-{n}_0'
        counter=0
        while os.path.exists(folder_name) :
            counter += 1
            folder_name=f'./samples-{type}-{n}_{counter}'
        os.mkdir(folder_name)
        return folder_name
    folder_name=f'./samples-{type}_{steps}-steps-{n}'
    if os.path.exists(folder_name) :
        folder_name=f'./samples-{type}_{steps}-steps-{n}_0'
    counter=0
    while os.path.exists(folder_name) :
        counter += 1
        folder_name=f'./samples-{type}_{steps}-steps-{n}_{counter}'
    os.mkdir(folder_name)
    return folder_name

if __name__ == '__main__' :
    main()