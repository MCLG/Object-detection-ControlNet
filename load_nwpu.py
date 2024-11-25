import torch
import os 
import numpy as np
from scipy.io import loadmat
from PIL import Image
import json
from einops import rearrange 
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from typing import Tuple, List, Optional
import sys 
import random as rd
from tqdm import tqdm 

project_root = os.path.abspath(os.path.dirname(__file__))

gmm_path = os.path.join(project_root, 'ControlNetHome/tools/')
if gmm_path not in sys.path:
    sys.path.append(gmm_path)

from ControlNetHome.tools.divergence_loss import DivergenceLoss
from data import crop_around_center, rotate_image, get_dimensions, create_dir, clear_dir

def density_map(points : list,
                boxes  : list,
                img_dim : tuple,
                precision = torch.float32 ) -> torch.Tensor:

    dmap =  torch.zeros(1,img_dim[1],img_dim[0], device = DEVICE, dtype=precision)

    x = torch.arange(0, img_dim[1], dtype=precision, device = DEVICE )
    y = torch.arange(0, img_dim[0], dtype=precision, device = DEVICE)
    Y,X = torch.meshgrid(x, y, indexing='ij')

    for k,coord in enumerate(boxes) :

        var_x, var_y = 4, 4
        scaler = 2*np.pi * 4 
        gaussian = torch.exp(-((1/var_x)*(X - points[k][0])**2 + (1/var_y)*(Y - points[k][1])**2) / 2) / scaler
        for c in range(dmap.shape[0]):
            dmap[c,:,:] += gaussian
    
    return dmap

def load_map_and_tensor(id : str) -> Tuple[dict,int] :

    img_path = os.path.join(RAW_DATA_LOC, f'images/{id}.jpg')
    try :
        img = Image.open(img_path)
        if img.mode != 'RGB' : 
            return
        elif img.mode == 'RGBA' :
            background = Image.new('RGB', img.size, (255,255,255))
            background.paste(img, mask=img.split()[3])
    except :
        print(f'Image with id : {id} does not exist at location : {img_path}')
        return

    try :
        gt_path = os.path.join(RAW_DATA_LOC, f'jsons/{id}.json')
        with open(gt_path, 'r') as file:
            data = json.load(file)
            points, boxes = data.pop('points'), data.pop('boxes')
    except :
        try :
            gt_path = os.path.join(RAW_DATA_LOC, f'mats/{id}.mat')
            mat = loadmat(gt_path)
            points, boxes = mat.pop('annPoints'), mat.pop('annBoxes')
        except FileNotFoundError :
            print(f'No annotation  file (.json or .mat) exist for id :{id}')
            return

    dens = density_map( points=points,
                        boxes = boxes,
                        img_dim=img.size)
    img = ToTensor()(img)
    
    _, h, w = dens.shape 
    dens_surface = h*w
    got_resized = -1
    if h < 512 and w < 512:
        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bicubic', align_corners=False).squeeze(0)
        img *= (dens_surface/(512**2))
        dens = F.interpolate(dens.unsqueeze(0), size=(512, 512), mode='bicubic', align_corners=False).squeeze(0)
        dens *= (dens_surface/(512**2))
        got_resized = 0
    elif h < 512:
        img = F.interpolate(img.unsqueeze(0), size=(512, w), mode='bicubic', align_corners=False).squeeze(0)
        img *= (dens_surface/(w*512))
        dens = F.interpolate(dens.unsqueeze(0), size=(512, w), mode='bicubic', align_corners=False).squeeze(0)
        dens *= (dens_surface/(w*512))
        got_resized = 1
    elif w < 512:
        img = F.interpolate(img.unsqueeze(0), size=(h, 512), mode='bicubic', align_corners=False).squeeze(0)
        img *= (dens_surface/(h*512))
        dens = F.interpolate(dens.unsqueeze(0), size=(h, 512), mode='bicubic', align_corners=False).squeeze(0)
        dens *= (dens_surface/(h*512))
        got_resized = 2

    return dict(image = img, density = dens), got_resized

def margin_slice(dimension : int, slice_size : int = 512) -> list :
    #Input : a line [0;dimension] is sliced in segments of lenght slice_size, with adaptive margin mL.
    assert dimension > slice_size

    if dimension < 1.75*slice_size:
        return [(0,dimension)]
    margin = int(np.floor((slice_size**2)/dimension))
    slices = [(k*slice_size- margin*k, (k+1)*slice_size - margin*k -1) for k in range(0,dimension//slice_size + 1)]
    return slices

def extract_rotated_tensors(img : torch.Tensor, dmap : torch.Tensor, angles : tuple = (10,45)) -> list :
    _, h, w = img.shape
    corners512 = get_dimensions(h,w)
    corners725 = get_dimensions(h,w, 725)
    cropped_images = list()

    for coords in corners512:
        x1, y1, x2, y2 = coords  
        cropped_img =  image = img[:, x1:x2, y1:y2]
        cropped_dmap = dmap[:, x1:x2, y1:y2]
        
        count = round(cropped_dmap.sum().item())
        if count > MAXIMAL_DENSITY or count < MINIMAL_DENSITY :
            continue
        cropped_images.append(dict(image = cropped_img, density = cropped_dmap))

    for coords in corners725:
        x1, y1, x2, y2 = coords
        cropped_img =  image = img[:, x1:x2, y1:y2]
        cropped_dmap = dmap[:, x1:x2, y1:y2]
        angle = rd.randint(angles[0], angles[1])
        rot_img = crop_around_center(rotate_image(cropped_img,angle), 512, 512)
        rot_dmap = crop_around_center(rotate_image(cropped_dmap,angle), 512, 512)
        
        count = round(rot_dmap.sum().item())
        if count > MAXIMAL_DENSITY or count < MINIMAL_DENSITY :
            continue
        cropped_images.append(dict(image = rot_img, density = rot_dmap))
    
    return cropped_images

def augment(xy : dict, got_resized : int) -> list :

    def get_sliced_tensors(t : torch.Tensor, slices : list, dim):
        sliced_tensors = list()
        for (i, j) in slices:
            if dim == 1:
                sliced_tensor = t[:, i:j, :]
            elif dim == 2 :
                sliced_tensor = t[:, :, i:j]
            sliced_tensors.append(sliced_tensor)
        return sliced_tensors

    def list_2_dictlist(l1,l2, id_l1 : str, id_l2 : str) -> List[dict]:
        assert len(l1) == len(l2) 
        list_d = list()
        for k in range(len(l1)) :
            d = {f'{id_l1}' : l1[k], f'{id_l2}' : l2[k]}
            list_d.append(d)
        return list_d
    
    img, dmap = xy.values()

    '''if got_resized == -1 :
        #for large image we extract horizontal and slightly rotated images
        d = extract_rotated_tensors(img, dmap)
    elif got_resized == 1 :
        
        'w = img.shape[2]
        slices = margin_slice(w, 512)
        imgs = get_sliced_tensors(img, slices, 2)
        densities = get_sliced_tensors(dmap, slices, 2)
        d = extract_rotated_tensors(img, densities)
        #print(f'{len(imgs)=}, {len(densities)=}')
        #d = list_2_dictlist(imgs,densities, 'image', 'density')
        #print(f'{d[0].keys()=}')
    elif got_resized == 2 :
        d = extract_rotated_tensors(img, dmap)
        h = img.shape[1]
        slices = margin_slice(h, 512)
        imgs = get_sliced_tensors(img, slices, 1)
        densities = get_sliced_tensors(dmap, slices, 1)
        
        d = extract_rotated_tensors(img, densities)
        #d = list_2_dictlist(imgs,densities, 'image', 'density')'''
    if got_resized == 0 :
        if dmap.sum() == 0 :
            return
        d = [xy]
    else :
        d = extract_rotated_tensors(img, dmap)

    # filter out the zero maps and filter out the over 200 objects maps
    # Blip here ??? --> if yes then return a COMPLETE dict
    if not d or len(d) == 0 :
        return
    
    return d

def load_set(K : int = -1) :
    i = 0
    id_digit_range = 6
    num_id = f'{1:0{id_digit_range}d}'  #create name id for new files

    labels = f'{DATA_LOC}/train/label.csv'
    with open(labels, 'w') as file:
        file.write('id,prompt\n')

    mean_extractor = DivergenceLoss(device=DEVICE,downscaling_factor = 8)

    N = len([1 for k in os.listdir(os.path.join(RAW_DATA_LOC,'images'))])
    progress_bar = tqdm(total=N, desc='Processing...', unit='images')
    # Load --> augment --> retrieve means -- > save all
    for file in os.listdir(os.path.join(RAW_DATA_LOC,'images/')) :
        id_, ext = os.path.splitext(file) 
        if ext != '.jpg' :
            continue 
               
        #id_ = '3348' #'2880' #'1681' #'2106' #'3811'#3390'#'2964'   
        try :
            d, got_resized = load_map_and_tensor(id_)   # returned a (dict,int)
        except TypeError :
            continue                                    # returned None 
        im,d = d.values()

        # prompt of large image
        inputs = PROCESSOR(ToPILImage()(im),text = 'a group of people ', return_tensors="pt").to(DEVICE)
        out = MODEL.generate(**inputs, max_new_tokens = 10)
        prompt_large_pic = PROCESSOR.decode(out[0], skip_special_tokens=True)

        # slice and rotate images
        dic = dict(image=im, density=d)
        dic = augment(dic, got_resized) #returns list of tensors or None
    
        if dic :
            for k,d in enumerate(dic) :

                image,dens = d.values()
                #Get new prompt 
                inputs = PROCESSOR(ToPILImage()(image),text = 'a group of people ', return_tensors="pt").to(DEVICE)
                out = MODEL.generate(**inputs, max_new_tokens = 10)
                prompt = PROCESSOR.decode(out[0], skip_special_tokens=True)
                if not any(word in prompt for word in OBJECT_DICTIONARY) or any(word in prompt for word in NEG_OBJECT_DICTIONARY) :
                    prompt = prompt_large_pic

                #get means positions
                d_mean, _ = mean_extractor.extract_means(dens.unsqueeze(0))
                d_mean = d_mean[0]
                
                torch.save(image.cpu(),f'{DATA_LOC}/train/img/{num_id}.pt')
                torch.save(dens.cpu(),f'{DATA_LOC}/train/map/{num_id}.pt')
                torch.save(d_mean.cpu(), f'{DATA_LOC}/train/mean/{num_id}.pt')
                with open(labels, 'a') as file:
                    file.write(f'{num_id}.pt, {prompt}\n')

                new_id = int(num_id) + 1
                num_id = f'{new_id:0{id_digit_range}d}'

                if i == K :
                    print(f'Finished processing {K} images. ')
                    return
                i += 1
            progress_bar.update(1)
    progress_bar.close()

def main() :

    import argparse

    OBJECT_DICTIONARY = ['group', 'people', 'crowd', 'man', 'men', 'woman', 'women', 'child','children', 'graduate', 'graduation', 'ceremony']
    '''['man', 'men', 'woman', 'women', 'group', 'mass', 'mob', 'people', 'crowd', 
                'human', 'person', 'highschooler', 'male', 'student', 'boy', 'girl', 
                'son', 'daughter', 'father', 'mother', 'gathering', 'protest', 'run',
                'race', 'parade', 'skier', 'team', 'game', 'stage', 'passenger', 'stadium', 
                'fan', 'band','market','street', 'couple','audience', 'conference', 'marathon', 'lecture']'''
    NEG_OBJECT_DICTIONARY = ['skateboard']

    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("raw_data_loc", help="Path to raw files is required input. ")
    parser.add_argument("data_loc", help="Path to saved processed imgs/maps/means is required input. ")
    parser.add_argument("device", help="Optional input device", nargs = "?", default = 0, type = int)
    parser.add_argument("minimal_density", help="Optional input minimal_density", nargs = "?", default = 3, type = int)
    parser.add_argument("maximal_density", help="Optional input maximal_density", nargs = "?", default = 150, type = int)
    args = parser.parse_args()
    
    # Split the line by commas and strip any leading/trailing whitespace from each word
    word_list = [word.strip() for word in line.split(',')]

    RAW_DATA_LOC = args.raw_data_loc #'/net/vid-raxus/storage/deeplearning/datasets/nwpu'
    DATA_LOC = args.data_loc#'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density'
    DEVICE = torch.device(args.device)
    MINIMAL_DENSITY = args.minimal_density
    MAXIMAL_DENSITY = args.maximal_density

    from transformers import BlipProcessor, BlipForConditionalGeneration
    PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    #clear_dir(DATA_LOC)
    create_dir(DATA_LOC)
    load_set(K = -1)
    
    '''N = sum([1 for k in os.listdir(os.path.join(DATA_LOC,'train/img'))])
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2,N, figsize=(N*4,4))
    for k,file in enumerate(os.listdir(os.path.join(DATA_LOC,'train/img')) ):
        print(f'{file=}')
        img = torch.load(os.path.join(DATA_LOC,f'train/img/{file}'))
        dens = torch.load(os.path.join(DATA_LOC,f'train/map/{file}'))
        mean = torch.load(os.path.join(DATA_LOC,f'train/mean/{file}'))
        print(f'{mean=}')
        ax[0,k].imshow(img.permute(1,2,0).cpu())
        #ax[0,k].axis('off')

        ax[1,k].imshow(dens.permute(1,2,0).cpu())
        #ax[1,k].axis('off')
    
    plt.savefig('./load_nwpu.png')'''
 
if __name__ == '__main__':
    main()

        

