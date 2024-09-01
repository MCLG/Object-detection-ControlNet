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
#%%
import sys
import torch 
import os
import os.path as path
import matplotlib.pyplot as plt 
import  PIL.Image as Image 
import torchvision.transforms as T 
import einops as ei
from collections.abc import Optional, List, Union, Tuple
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import torch.nn.functional as F      
import numpy as np 
import PIL.Image as Image
import matplotlib.pyplot as plt 
import json 

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

gp = path.dirname(path.dirname(__file__))
if gp not in sys.path :
    sys.path.append(gp)

steerer_loc = '/home/luk02485/development/ControlNet/STEERER'
if steerer_loc not in sys.path :
    sys.path.append(steerer_loc)

control_loc = '/home/luk02485/development/ControlNet/ControlNetHome'
if control_loc not in sys.path :
    sys.path.append(control_loc)
    
from ControlNetHome.cldm.cldm import ControlLDM
from ControlNetHome.ldm.util import default
from ControlNetHome.cldm.logger import ImageLogger
from ControlNetHome.cldm.model import create_model, load_state_dict
from ControlNetHome.cldm.ddim_hacked import DDIMSampler

from data_jhu import CrowdDataSetv2, MapConfig
from data_nwpu import CrowdDataSet

from inference import ckpt_search

class TrainedControlNet(ControlLDM) :

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):

        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.ddim_sampler = DDIMSampler(self)
        
    @torch.no_grad()
    def sample(self, prompt : Union[str, List[str]] , density : torch.Tensor, method = 'DDPM') :

        assert len(density.shape) > 2 and len(density.shape) <= 4, 'usage : [B,1,H,W] or [1,H,W]'
        original_dim = tuple(density.shape[-2:])
        if density.shape[-2:] != (512,512) :
            prev_area = density.shape[-1]*density.shape[-2]
            new_area = 512**2
            scaler = prev_area / new_area
            density = T.Resize((512,512), interpolation=T.InterpolationMode.NEAREST_EXACT)(density) * scaler
        if len(density.shape) == 4 and density.shape[0] > 1 :
            densities = list(torch.tensor_split(density,density.shape[0]))
            for k,dens in enumerate(densities) :
                densities[k] = self.sample(dens, method )
        elif len(density.shape) == 4 :
            densities = self.sample(densities, method)
        else :
            densities = density.unsqueeze(0)
        if method == 'DDPM' :
            densities = self.DDPM_generate(dict(density=densities, txt=prompt))
        try :
            densities = T.Resize((original_dim), interpolation=T.InterpolationMode.NEAREST_EXACT)(densities)
            densities = self.process(densities)
        except :
            for dens in densities : 
                dens = T.Resize((original_dim), interpolation=T.InterpolationMode.NEAREST_EXACT)(dens)
                dens = self.process(dens)
        return densities

    def process(self,img) :
        img = torch.clamp(img.squeeze(0), -1., 1.).cpu()
        img = ( img + 1.0 ) /2.0
        img = img.transpose(0,1).transpose(1,2).squeeze(-1)
        img = img.numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        #img = img.rotate(-90, expand = True)
        return img
    

    @torch.no_grad()
    def DDPM_generate(self, input : dict ) -> torch.Tensor :
        # input : {'density' : map, 'txt' : prompt, 'hint' : map }

        density = input['density']
        txt = [input['txt']] if type(input['txt'])==str else input['txt']
        #assuming densityalready in correct dimension (latent)
        '''if density.shape[0] == 1 :
            density = torch.unsqueeze(ei.rearrange(density, 'b c h w -> b h w c'), 0)
        '''
        shadow_x = 2 * torch.randn(1,3,512,512) - 1
        density = ei.rearrange(density, 'b c h w -> b h w c')
        shadow_x = ei.rearrange(shadow_x, 'b c h w -> b h w c')
        
        #form batch and get input here
        batch = {
            'jpg' : shadow_x, 
            'txt' : txt, 
            'hint' : density, 
            'gaussian' : density, 
            'count' : torch.tensor([density.sum().item()])

        }
        print(batch.keys())
        x, cond = self.get_input(batch, k=self.first_stage_key, dropout = False) #passes through encoder
        # encode into latent dimension here
        x_start = x#2 * torch.rand((1,4,64,64), device = self.device) -1
        
        # use get_input for this
        progress = tqdm(total=1000)
        step = 10
        for t in reversed(range(0,1000,step)) :
            time = torch.tensor([t], device = self.device)
            #noise = default(noise, lambda: torch.randn_like(x_start)) if t > 1 else torch.zeros(x_start.shape)
            noise = torch.randn_like(x_start) if t > 1 else torch.zeros(x_start.shape, device = self.device)
            #x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) This is forward process
            model_output = self.apply_model(x_start, time, cond)

            x_start = self.predict_reconstructed_from_noise(x_t=model_output, t=time, noise = noise) + torch.sqrt(self.betas[t]) * torch.randn_like(noise)
            progress.update(1*step)

        reconstructed = self.decode_first_stage(x_start)
        progress.close()

        return reconstructed

    @torch.no_grad()
    def DDIM_generate(self, input : dict, ddim_steps = 50, ddim_eta=0.0, plot_denoise_rows=False,  unconditional_guidance_scale=9.0) -> dict :
        log = dict()
        use_ddim =  ddim_steps is not None 
        z, c = self.get_input(input, self.first_stage_key, dropout = False)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = z.shape[0]
        samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
        x_samples = self.decode_first_stage(samples)
        log["samples"] = process_2(x_samples, return_tensor=True)
        if plot_denoise_rows:
            denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            log["denoise_row"] = denoise_grid
        
        
        '''if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            '''
        return log 
       
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
            
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

@torch.no_grad()
def large_DDIM_generate(model,fetched_data,nb, allow_access_other_devices = False) :
    '''
    Does not process or enhance the images, just splits the generation process in smaller batches.
    By setting allow_access_other_devices to True or an integer, this will send the samples to the specified(int) device
    or to the device with most available memory that is not model.device.
    If left to False, the tensors will be send to cpu.
    '''
    if allow_access_other_devices :
        device_list = [i for i in range(torch.cuda.device_count())]
        try : 
            device_list.pop(model.device.index)
        except TypeError:
            #model is on cpu
            pass
        try:
            device = int(allow_access_other_devices)
        except ValueError:
            available_memory = [torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i) for i in device_list]
            device = torch.device(available_memory.index(max(available_memory)))
    else :
        device = torch.device('cpu')

    #utility func
    def check(d) :
        out = 0
        for key in d.keys() :
            try :
                out += int(key) * d[key]
            except Exception as e :
                out += int(key)* check(d[key])
        return out 
    #utility func
    def section_length(x, restrict_divider_size = False ):
        ppgc,rest = x//10, x-(x//10)*10
        nb10 = ppgc
        nb5 = 0
        if rest >= 5 :
            x = x-x//10*10
            ppgc,rest = x//5, x-(x//5)*5
            nb5 = ppgc
        if restrict_divider_size and nb10>10:
            nb10 = section_length(nb10)
        return {'10' : nb10, '5' : nb5, '1' : rest}
    #utility func
    def section_indexing(x) :
        d = section_length(x)
        i = 0
        d_ = {}
        for key in d.keys() :
            d_[key] = []    
            for _ in range(d[key]) :
                if key != '1':     
                    d_[key].append([n for n in range(i,i+int(key))])
                    i += int(key)
                else :
                    d_[key] = [[n for n in range(i,i+d[key])]]
        indexing = []
        for key in d_ :
            print(key,d_[key])
            indexing += d_[key]
        return indexing
    #utility func
    def section_batch(data, indexes):   
        sectionned_data = {
                        'jpg' : data['jpg'][indexes],
                        'hint' : data['hint'][indexes],
                        'count' : data['count'][indexes],
                        'gaussian' : data['gaussian'][indexes],
                        'txt' : [data['txt'][k] for k in indexes] 
                        }
        return sectionned_data
    
    N = fetched_data['hint'].shape[0]
    print(f'N : {N} \n'
                f'datakeys : {fetched_data.keys()} \n'
        )
    for key in fetched_data.keys() :
        if isinstance(fetched_data[key], torch.Tensor) :
            print(f'{key} : {fetched_data[key].shape}')
        else :
            print(fetched_data[key])
    if N <= 10 :
        fake_samples = model.DDIM_generate(input=fetched_data)['samples']
    else :
        sections = section_indexing(N)
        print(f'N:{N}, sections : {sections}')
        sectionned_data = section_batch(fetched_data, sections.pop(0))
        fake_samples = model.DDIM_generate(input=sectionned_data)['samples'].to(device)
        for index in sections :
            print(f'index :  {index}')
            sectionned_data = section_batch(fetched_data, index)
            print(f'sectionned_data.shape (density) : {sectionned_data["hint"].shape}')
            fake_samples = torch.cat((fake_samples,
                                        model.DDIM_generate(
                                            input=sectionned_data)['samples'].to(device)
                                            ))                
    return fake_samples.to(model.device)

def sequential_generate(model, ratio : float ,quantity : list ) ->  Tuple[dict,dict] :
    
    config = MapConfig()
    TestSet = CrowdDataSetv2(config=config,set='test')
    assert int(np.ceil((1-ratio)*max(quantity))) <= TestSet.__len__(),  f'Not enough test data for {int(np.ceil((1-ratio)*quantity))} real images. Quantity of real images available : {TestSet.__len__()}. '
    
    try : 
        quantity = sorted(quantity)
    except TypeError :
        print('Passed a single quantity in sequential_generate !')
        quantity = [quantity]

    assert ratio>=0 and ratio <= 1
    def approx(n,r):
        if r <= 0.5 :
            return int(np.ceil(n*r))
        elif r > 0.5 :
            return int(np.floor(n*r))
        
    norm = ['MAE_fro', 'MSE_fro']
    dinstances = dict()
    cinstances = dict()
    nb_fake, nb_real = 0, 0
    for q in quantity :
        nb_fake = approx(q, ratio) - nb_fake
        nb_real = approx(q, 1-ratio) - nb_real
        

#%%

#%%

@torch.no_grad()
def evaluation(model, ratio = [0.05, 0.1, 0.25, 0.3, 0.5, 0.6, 0.8, 1], quantity = 100) -> Tuple[dict,dict] :
    
    #(dead) utility func
    def cat(t1 : Optional[torch.Tensor], t2 : Optional[torch.Tensor]) -> Optional[torch.Tensor] :
        if t1 is None :
            return t2 
        if t2 is None : 
            return t1 
        elif isinstance(t1,torch.Tensor) and isinstance(t2,torch.Tensor) :
            return torch.cat((t1,t2))
    #utility func
    def move_to_cpu(d : dict ) -> dict :
        if isinstance(d, dict):
            return {k: move_to_cpu(d[k]) for k in d}
        elif isinstance(d, list):
            return [move_to_cpu(v) for v in d]
        elif isinstance(d, tuple):
            return tuple(move_to_cpu(v) for v in d)
        elif torch.is_tensor(d):
            return d.cpu().tolist()
        else:
            return d   
        
    config = MapConfig()
    TestSet = CrowdDataSetv2(config=config,set='test')
    assert int(np.ceil((1-max(ratio))*quantity)) <= TestSet.__len__(),  f'Not enough test data for {int(np.ceil((1-prct)*quantity))} real images. Quantity of real images available : {TestSet.__len__()}. '

    try : 
        ratio = sorted(ratio)
    except TypeError :
        ratio = sorted([ratio])
        
    norm = ['MAE_fro', 'MSE_fro']
    dinstances = dict()
    cinstances = dict()

    progress = tqdm(total = len(ratio))
    for prct in ratio :
        print(f'[Creating instance {int(np.ceil(prct*100))}/{int(np.ceil((1-prct)*100))} - images]\n')
        
        #only real data
        if prct == 0 :
            real_data = rd_fetch_data(nb=int(np.ceil((1-prct)*quantity)), test_set=True)#make_single_batch(items)
            real_samples = process_2(ei.rearrange(real_data['jpg'], 'N H W C -> N C H W'))

            real_samples = real_samples.to(model.counter.device)
            result_real = model.counter.get_count(real_samples)[1]
            err = error(
                result_real.to(DEVICE), 
                real_data['gaussian'].to(DEVICE),
                norm =  norm
            )
            dinstances[f'{prct}'] = err[0]
            cinstances[f'{prct}'] = err[1]
            progress.update(1)
            continue

        #only artificial data
        elif prct == 1 :
            fetched_data = rd_fetch_data(nb=int(np.ceil(prct*quantity)))
            fake_samples = large_DDIM_generate(model, fetched_data, nb=int(np.ceil(prct*quantity)), allow_access_other_devices = False)                
            fake_samples = process_2(fake_samples)

            fake_samples = fake_samples.to(model.counter.device)
            result_fake = model.counter.get_count(fake_samples)[1]
            err = error(
                result_fake.to(DEVICE), 
                fetched_data['gaussian'].to(DEVICE),
                norm = norm
            )
            dinstances[f'{prct}'] = err[0]
            cinstances[f'{prct}'] = err[1]
            progress.update(1)
            continue 

        real_data = rd_fetch_data(nb=int(np.ceil((1-prct)*quantity)), test_set=True)#make_single_batch(items)
        real_samples = process_2(ei.rearrange(real_data['jpg'], 'N H W C -> N C H W'), return_tensor = True)
        
        print(' [Generating images from density] ...')
        fetched_data = rd_fetch_data(nb=int(np.ceil(prct*quantity)))
        fake_samples = large_DDIM_generate(model, fetched_data, nb=int(np.ceil(prct*quantity)), allow_access_other_devices = False)              
        fake_samples = process_2(fake_samples, return_tensor = True)
        
        print(' [Computing loss] ... ')
        real_samples = real_samples.to(model.counter.device)
        fake_samples = fake_samples.to(model.counter.device)
        result_real, result_fake = model.counter.get_count(real_samples)[1], model.counter.get_count(fake_samples)[1]

        err = error(
            torch.cat((result_real,result_fake)).to(DEVICE), 
            torch.cat((real_data['gaussian'],fetched_data['gaussian'])).to(DEVICE),
            norm = norm
            )
        print('done !')
        
        dinstances[f'{prct}'] = err[0]
        cinstances[f'{prct}'] = err[1]
        progress.update(1)
    progress.close()

    dinstances = move_to_cpu(dinstances)
    cinstances = move_to_cpu(cinstances)
    with open(f'test-density-{quantity}.json','w') as json_file:
        json.dump(dinstances, json_file)
    with open(f'test-count-{quantity}.json','w') as json_file:
        json.dump(cinstances, json_file)
        
    return dinstances,cinstances 

    
def log_evaluation(model,ratio = [0.05, 0.1, 0.25, 0.3, 0.5, 0.6, 0.8, 1], use_save = True, quantity = 100) :

    assert len(ratio) == 8, 'the plots only handle 8 ratios so far and will result in faield plots if not 8.'
    if use_save and os.path.exists(f'.//test-{quantity}.json'):
        with open(f'test-{quantity}.json', 'r') as json_file :
            data = json.load(json_file)
    else :
        data = evaluation(model, ratio=ratio, quantity=quantity)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('MSE/MAE-loss \n'
                    f'Diff. between STEERER approx. densities over a batch({quantity}) of images and their true densities,\n '
                    ' with different ratios of artificial data.\n'
                    f'(mixed crowd density ~5-10,000)')
    dataMSE, dataMAE = [], []
    for k, prct in enumerate(data.keys()):
            
        dataMSE.append({
            'mean': data[prct]['loss-MAE_fro'],
            'med': data[prct]['loss-MSE_fro'],
            'q1': data[prct]['q-0.25-MSE_fro'],
            'q3': data[prct]['q-0.75-MSE_fro'],
            'whislo': data[prct]['min-MSE_fro'],
            'whishi': data[prct]['max-MSE_fro'],
            'fliers': []
        })
        dataMAE.append({
            'mean': data[prct]['loss-MAE_fro'],
            'med': data[prct]['loss-MAE_fro'],
            'q1': data[prct]['q-0.25-MAE_fro'],
            'q3': data[prct]['q-0.75-MAE_fro'],
            'whislo': data[prct]['min-MAE_fro'],
            'whishi': data[prct]['max-MAE_fro'],
            'fliers': []
        })

    colors = ['#1f77b4', '#ff7f0e']
    b1 = ax1.bxp(dataMSE, positions=range(1,9), showmeans=False, showcaps=True, showbox=True, showfliers=False,patch_artist=True,boxprops=dict( facecolor=colors[0],alpha=0.4))
    ax1.set(xlim=(0, 9), xticks=range(1,9), xticklabels=[f'{int(r*100)}% ' for r in ratio])
    ax1.set_title('MSE')
    ax1.set_yscale('log')

    b2 = ax2.bxp(dataMAE, positions=range(1,9), showmeans=False, showcaps=True, showbox=True, showfliers=False, patch_artist=True,boxprops=dict( facecolor=colors[1],alpha=0.4))
    ax2.set(xlim=(0, 9), xticks=range(1,9), xticklabels=[f'{int(r*100)}% ' for r in ratio])
    ax2.set_title('MAE')
    ax2.set_yscale('log')

    ax1.grid(True, which="major", linestyle='--', linewidth=0.5)
    ax2.grid(True, which="major", linestyle='--', linewidth=0.5)

    interquartile_range_b = mpatches.Patch(color=colors[0], alpha=0.4, label='IQR') #interquartile range
    interquartile_range_o = mpatches.Patch(color=colors[1], alpha=0.4, label='IQR') #interquartile range
    mean = mlines.Line2D([], [], color='orange',
                            markersize=15, label='mean')
    min_max = mlines.Line2D([], [], color='black',
                            markersize=15, label='min/max')

    fig.legend(handles=[interquartile_range_b,interquartile_range_o,mean,min_max], labels = ['IQR-MSE', 'IQR-MAE', 'Mean', 'Min/Max'], loc='upper right')
    #ax2.legend(['MAE'], loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'boxplot-{quantity}-MSE_for-MAE_fro')

    plt.show()
    plt.close('fig')
    
    if use_save and os.path.exists(f'test-{10}.json'):
        with open(f'test-{10}.json', 'r') as json_file :
            q1 = json.load(json_file)
    else :
        q1 = evaluation(model, ratio=ratio, quantity=10)
    if use_save and os.path.exists(f'test-{20}.json'):
        with open(f'test-{20}.json', 'r') as json_file :
            q2 = json.load(json_file)
    else :
        q2 = evaluation(model, ratio=ratio, quantity=20)
    if use_save and os.path.exists(f'test-{40}.json'):
        with open(f'test-{40}.json', 'r') as json_file :
            q3 = json.load(json_file)
    else :
        q3 = evaluation(model, ratio=ratio, quantity=40)
    if use_save and os.path.exists(f'test-{50}.json'):
        with open(f'test-{50}.json', 'r') as json_file :
            q4 = json.load(json_file)
    else :
        q4 = evaluation(model, ratio=ratio, quantity=50)
    if use_save and os.path.exists(f'test-{70}.json'):
        with open(f'test-{70}.json', 'r') as json_file :
            q5 = json.load(json_file)
    else :
        q5 = evaluation(model, ratio=ratio, quantity=70)
    q6 = data

    data = [q1,q2,q3,q4,q5,q6]
    histo_dict = dict()
    for r in ratio :
        histo_dict[f'{r}'] = tuple([round(q[f'{r}']['loss-MAE_fro'], 2) for q in data])

    quants = [10,20,40,50,70,100] 
    x = np.arange(len(quants))
    width = 0.115
    multiplier = -4
    fig, ax = plt.subplots(layout='constrained', figsize=(24, 12))

    alphas = np.clip([0 + k*(1/7) for k in range(len(ratio))], 0, 1)
    alphas_ = alphas[::-1]
    for i, (r,loss) in enumerate(histo_dict.items()) :
        
        offset = width * multiplier
        try :
            label = f'{int(np.ceil(float(r)*100))}%'
        except ValueError :
            label = f'{np.ceil(float(r)*100)}%'
            
        rects = ax.bar(x+offset, 
                       loss, 
                       width, 
                       label = label,
                       #color = 'blue',
                       )
        for rect in rects.patches :
            if i >= 6 :
                rect.set_facecolor((0, 0.545, 0.545,alphas[i]))
                rect.set_edgecolor((0,0,0,alphas_[i]))
            else :    
                rect.set_facecolor((0,0,1,alphas[i]))
                rect.set_edgecolor((0,0,0,alphas_[i]))
        ax.bar_label(rects,padding=3)
        multiplier += 1   
    
    ax.set_ylabel('MAE',fontsize=12)
    ax.set_xlabel('batch size')
    ax.set_title('MSE-loss of diff. between recreated densities and true densities for increasing batch size of images')
    ax.set_xticks(x, quants,fontsize=18)
    ax.legend(loc='upper left', ncols=8, fontsize = 16)

    plt.show()

def get_dloss(norm, pred, truth) -> Tuple[torch.Tensor, torch.Tensor] :
    '''
    [Warning] MAE_fro and MSE_fro will both return losses of higher magnitude than MAE and MSE.
    Any frobenius mean will return a mean over the given qunatity of matrices.
    MAE and MSE is (true) vector approach so it will average over the WxH dimensions of your images
    '''
    with torch.no_grad():
        
        diff = pred-truth
        
        if norm == 'MAE' :
            to_mean = abs(diff).flatten()
            return torch.mean(to_mean), to_mean #F.l1_loss(pred, truth)
        if norm == 'MSE' :
            to_mean = (diff**2).flatten()
            return torch.mean(to_mean), to_mean #F.mse_loss(pred,truth)
        if norm == 'MAE_fro':
            dist = torch.linalg.norm(pred - truth, 
                                ord = 'fro',
                                dim =(2,3)
                                )
            return  torch.mean(dist), dist
        if norm == 'MSE_fro' :
            dist = torch.linalg.norm(pred - truth, 
                                ord = 'fro',
                                dim =(2,3)
                                ) ** 2
            return torch.mean(dist), dist
        
    raise ValueError(f"Unexpected norm value: {norm}. Expected 'MAE', 'MSE', 'MAE_fro', or 'MSE_fro'.")

def get_closs(norm, pred, truth) -> Tuple[torch.Tensor, torch.Tensor] :
    '''
    Computes count loss
    '''
    assert len(pred.shape) == 4 and len(truth.shape) == 4, f'requires smallest batch to be of size 1,1,512,512 in order to compute difference. Current shape pred:{pred.shape}/truth:{truth.shape}' 
    
    with torch.no_grad():
        pred = torch.tensor( [ pred[k].sum().item() for k in range(pred.shape[0]) ] )
        truth = torch.tensor( [ truth[k].sum().item() for k in range(truth.shape[0]) ] )
        diff = pred-truth
        
        if norm == 'MAE' :
            return F.l1_loss(pred,truth), diff #F.l1_loss(pred, truth)
        if norm == 'MSE' :
            return F.mse_loss(pred,truth), diff #F.mse_loss(pred,truth)
        
    raise ValueError(f"Unexpected norm value: {norm}. Expected 'MAE' or 'MSE'.")
'''
def error( pred, truth, norm : str, quantiles = True, min_max = True) -> Tuple[dict,dict] :
    'type : MAE, MSE, MAE_fro, MSE_fro'

    def mask(arr,eps = 0) -> torch.tensor:
        mask = arr > eps
        return arr[mask]
    
    output_dens = dict()
    output_count = dict()
    if isinstance(norm,list) :
        for nor in norm :    
            output_dens[f'loss-{nor}'], dvalues = get_dloss(nor,pred,truth)
            output_count[f'loss-{nor}'], cvalues = get_closs(nor,pred,truth)
            if quantiles :
                dvalues = dvalues.cpu()
                output_dens[f'q-0.75-{nor}'] = np.quantile(dvalues, q = 0.75)
                output_dens[f'q-0.25-{nor}'] = np.quantile(dvalues, q = 0.25)
                cvalues = cvalues.cpu()
                output_count[f'q-0.75-{nor}'] = np.quantile(cvalues, q = 0.75)
                output_count[f'q-0.25-{nor}'] = np.quantile(cvalues, q = 0.25)
            if min_max :
                cheat = np.percentile(dvalues,[0,100])
                output_dens[f'min-{nor}'] = cheat[0]
                output_dens[f'max-{nor}'] = cheat[-1]
                cheat = np.percentile(cvalues,[0,100])
                output_count[f'min-{nor}'] = cheat[0]
                output_count[f'max-{nor}'] = cheat[-1]

    elif isinstance(norm,str) :
        output_dens['loss'], dvalues = get_dloss(norm,pred,truth)
        output_count['loss'], cvalues = get_closs(nor,pred,truth)
        if quantiles :
                dvalues = dvalues.cpu()
                output_dens[f'q-0.75-{norm}'] = np.quantile(dvalues, q = 0.75)
                output_dens[f'q-0.25-{norm}'] = np.quantile(dvalues, q = 0.25)
                cvalues = cvalues.cpu()
                output_count[f'q-0.75-{norm}'] = np.quantile(cvalues, q = 0.75)
                output_count[f'q-0.25-{norm}'] = np.quantile(cvalues, q = 0.25)
        if min_max :
            cheat = np.percentile(dvalues,[0,100])
            output_dens[f'min-{norm}'] = cheat[0]
            output_dens[f'max-{norm}'] = cheat[-1]
            cheat = np.percentile(cvalues,[0,100])
            output_count[f'min-{norm}'] = cheat[0]
            output_count[f'max-{norm}'] = cheat[-1]
    return output_dens

'''
def error(pred, truth, norm: str, quantiles=True, min_max=True) -> Tuple[dict, dict]:
    'type : MAE, MSE, MAE_fro, MSE_fro'

    def mask(arr, eps=0) -> torch.tensor:
        mask = arr > eps
        return arr[mask]

    def compute_metrics(nor, pred, truth):
        output_dens = {}
        output_count = {}

        output_dens[f'loss-{nor}'], dvalues = get_dloss(nor, pred, truth)
        output_count[f'loss-{nor}'], cvalues = get_closs(nor, pred, truth)

        if quantiles:
            dvalues = dvalues.cpu()
            output_dens[f'q-0.75-{nor}'] = np.quantile(dvalues, q=0.75)
            output_dens[f'q-0.25-{nor}'] = np.quantile(dvalues, q=0.25)

            cvalues = cvalues.cpu()
            output_count[f'q-0.75-{nor}'] = np.quantile(cvalues, q=0.75)
            output_count[f'q-0.25-{nor}'] = np.quantile(cvalues, q=0.25)

        if min_max:
            cheat = np.percentile(dvalues, [0, 100])
            output_dens[f'min-{nor}'] = cheat[0]
            output_dens[f'max-{nor}'] = cheat[-1]

            cheat = np.percentile(cvalues, [0, 100])
            output_count[f'min-{nor}'] = cheat[0]
            output_count[f'max-{nor}'] = cheat[-1]

        return output_dens, output_count

    if isinstance(norm, list):
        output_dens = {}
        output_count = {}
        for nor in norm:
            dens, count = compute_metrics(nor, pred, truth)
            output_dens.update(dens)
            output_count.update(count)

    elif isinstance(norm, str):
        output_dens, output_count = compute_metrics(norm, pred, truth)

    return output_dens, output_count

def make_single_batch(input : list[dict]) -> dict :

    assert all(input[0].keys() == d.keys() for d in input), 'Not all dictionaries contain the same keys !'

    batch = {key : [d[key] for d in input ] for key in input[0].keys()}
    batch['jpg'] = torch.stack(batch['jpg'])    #shape N, H, W, C
    batch['hint'] =  torch.stack(batch['hint'])
    batch['gaussian'] =  torch.stack(batch['gaussian'])
    print(f'make_single_batch : {batch["count"]}')
    batch['count'] = torch.tensor(batch['count'])

    return batch

def rd_fetch_data(nb : int, return_as_batch = True, test_set = False) -> Optional[torch.Tensor] :
    '''
    the tensors in the out-dict are of dimension H,W,C !!!
    '''
    print(f'    fetching [{nb}] data[img/dens/gaussian] ...')
    
    config = MapConfig
    if test_set:
        train_set = CrowdDataSetv2(config)
        config.load_dir = 'val'
        val_set = CrowdDataSetv2(config,set='val')
        config = MapConfig()
        config.save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/'
        nwpuset = CrowdDataSet(config)
        search_densities = ConcatDataset([train_set, val_set, nwpuset])
    else :
        config = MapConfig()
        TestSet = CrowdDataSetv2(config=config,set='test')
        search_densities = TestSet

    indexes = np.random.choice([k for k in range(search_densities.__len__())], size = nb, replace= False)
    if return_as_batch :
        return make_single_batch([search_densities.__getitem__(n) for n in indexes])
    return [search_densities.__getitem__(n) for n in indexes]

def process_2(img : torch.Tensor, return_tensor = False) -> torch.Tensor :
    '''
    will return a tensor if a batch tensor is passed. If a single tensor is passed, a PIL.Image may be returned instead.
    '''
    device = img.device

    if len(img.shape)==4 and img.shape[0] > 1 :
        imgs = list(torch.tensor_split(img, img.shape[0]))

        processed_images = [process_2(tens, return_tensor= True) for tens in imgs]
        for tens in processed_images :
            assert tens.shape[0] == 1, f'tens.shape = {tens.shape}' # for .cat below

        return torch.cat(processed_images,0)
    
    if len(img.shape) == 4 :
        img = img.squeeze(0)
    img = torch.clamp(img, -1., 1.).cpu()
    img = (img + 1.0) / 2.0
    img = img.transpose(0, 1).transpose(1, 2)  # Change shape to H, W, C
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    #img = img.rotate(-90, expand=True)
    
    if return_tensor :
        img = T.ToTensor()(img).to(device=device).unsqueeze(0)

    return img

#%%
try:
    resume_path = ckpt_search()
except : 
    resume_path = './ControlNetHome/models/control_sd15_ini_v2.ckpt'#'./saves/checkpoints/crowdnet_dict-epoch-60.ckpt'#crowdnet_dict12epochs.ckpt'#'./models/control_sd15_ini.ckpt'

STEERER_path = '/home/luk02485/development/ControlNet/STEERER/JHU_mae_54.5_mse_240.6.pth'
resume_path = './lightning_logs/version_25/checkpoints/epoch=19-step=880.ckpt'
batch_size = 16
logger_freq = 300
learning_rate = 2e-5#1e-5   SET TO PAPER VALUES
sd_locked = True
only_mid_control = False
AVAILABLE_GPU = [1,2]

model = create_model('./ControlNetHome/models/trained_cldm.yaml',location=None).cpu()

interm = load_state_dict(resume_path, STEERER_path=STEERER_path,location=None)

model.load_state_dict(interm,strict = False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# set all parts to same device
DEVICE = 0
model = model.to(torch.device(DEVICE))
model.counter = model.counter.to(torch.device(DEVICE))
model.first_stage_model.device = model.first_stage_model.to(torch.device(DEVICE))
model.cond_stage_model.device = torch.device(DEVICE)

#%%
model = None
log_evaluation(model)
#%%

def testing(nb = 0, gs = 1) :
    pass
    '''
    map = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/map/0017.pt',map_location= torch.device(1))
    image = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/img/0017.pt',map_location=torch.device(1))
    gaussian = torch.load('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/train/gaussians/0017-2048.pt',map_location=torch.device(1))
    count = torch.tensor([gaussian.sum().item()], device = torch.device(1))
    prompt = 'a group of people standing on a sidewalk'

    map_ = torch.unsqueeze(ei.rearrange(map, 'c h w -> h w c' ), 0)
    image_ = torch.unsqueeze(ei.rearrange(image, 'c h w -> h w c' ), 0)


    batch = {'jpg' : image_, 'txt' : [prompt], 'hint' : map_, 'gaussian' : gaussian, 'count' : count}

    #model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
    samples = model.DDIM_generate(input=batch, unconditional_guidance_scale=gs)['samples']

    import matplotlib.pyplot as plt 

    fig, axes = plt.subplots(1,3,figsize=(15,3))
    axes[0].imshow(image.cpu().squeeze(0).permute(1,2,0))
    axes[0].axis('off')
    axes[0].set_title(f'txt : {prompt}')

    axes[1].imshow(gaussian.squeeze(0).cpu())
    axes[1].axis('off')
    axes[1].set_title(f'count : {count}')

    axes[2].imshow(samples)
    axes[2].axis('off')
    axes[2].set_title('samples')

    plt.savefig(f'./saves/20-8-24-runv1-9460steps-magn-regularizer/testing/nb={nb}-gs={gs}')
    plt.show()
    plt.close('fig')
    '''

'''
testing(1,1)
testing(1,0)
testing(1,0)
testing(1,5)
testing(1,5)
testing(1,10)
testing(1,10)
testing(1,15)
testing(1,20)
testing(1,50)
testing(1,50)
testing(1,100)
'''
# %%
