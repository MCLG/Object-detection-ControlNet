import os
import sys 
import torch
import json
import numpy as np 
from PIL import Image 
from typing import Union, Optional, List
from torchvision.transforms.functional import rotate as tensor_rotate, hflip
from torchvision.transforms import ToTensor, ToPILImage, Resize,InterpolationMode
from einops import rearrange 

from dataclasses import dataclass
from scipy.io import loadmat
from tqdm import tqdm 
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

DEVICE = torch.device(0)    #device for map generation and imag splitting
BLIP_DEVICE = torch.device(0)   #device for caption
ROOT_DIR_NWPU = '/net/vid-raxus/storage/deeplearning/datasets/nwpu/'
ROOT_DIR_JHU = '/net/vid-raxus/storage/deeplearning/datasets/jhu/jhu_crowd_v2.0/'

@dataclass
class MapConfig :
    include_box_size : bool = False  #if false, all objects have fixed variance (4,4)
    scale_gaussian : bool = True    # true gaussian or not. DO NOT CHANGE
    save_as : str = 'Tensor'        #or 'PIL'. DO NOT CHANGE
    type : str = 'HeatMap'          #'RGB' resulting control dimension will be 1,512,512 or 'HeatMap' for control dimension 3,512,512. DO NOT CHANGE 
    
    #change here for different save directory
    save_dir : str = "/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/"  #location to save processed data
    
    save_for : str = 'train'        # or 'val' or 'test'. In use but not needed. creates a sub_directory 'save_for' in which processed data is stored.
    load_dir : str = "train/"       # 'val/' or 'test/'. Only relevant for JHU set. Set to '' for other data sets but respect the above mentioned folder structure.

config = MapConfig()
if __name__ == '__main__':
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

class CCNetSet(Dataset):
    
    def __init__(self, img_dir : str, map_dir : str, csv_loc : str, mean_dir : Optional[str] = None):
        self.img_dir = img_dir
        self.map_dir = map_dir
        self.mean_dir = mean_dir
        self.data = self.csv2dict(csv_loc)       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_path = os.path.join(self.map_dir, item["id"])
        target_path = os.path.join(self.img_dir, item["id"])

        source = torch.permute(torch.load(source_path, map_location=torch.device('cpu')).to(dtype=torch.float32),(2,1,0))
        target = torch.permute(torch.load(target_path, map_location=torch.device('cpu')).to(dtype=torch.float32),(2,1,0))
        target = 2*target - 1 #input to range [-1,1]
        
        prompt = item['prompt']
        
        if self.mean_dir :
            mean_path = os.path.join(self.mean_dir, item["id"])
            means = torch.load(mean_path , map_location=torch.device('cpu')).to(dtype=torch.float32)
            return dict(jpg=target, txt=prompt, hint=source, mean=means)

        return dict(jpg=target, txt=prompt, hint=source)
    
    def csv2dict(self, csv_loc) -> List[dict]:
        from csv import DictReader 
        data = []
        assert csv_loc[-4:] =='.csv', f'{csv_loc=} not leading to a .csv file'
        
        with open(csv_loc, mode='r', newline='') as csvfile:
            csv_reader = DictReader(csvfile)
            for row in csv_reader:
                data.append(row)
        
        return data
    
    def display(self, idx) :
        import matplotlib.pyplot as plt 
        jpg, prompt, source = self.__getitem__(idx).values()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        #scale to range [0,1]
        jpg = (jpg+1)/2

        axes[0].imshow(jpg.cpu())
        axes[1].imshow(source.cpu())
        fig.text(0.25, 0.05, prompt , ha='center', fontsize=12)

        plt.show()

def custom_collate(batch):
    # Custom collate_fn that handles batches with tensors with different dimension in the first position. 
    default_collate_bool = (len(batch[0].keys()) < 4 )
    if default_collate_bool :
        return default_collate(batch)
    jpg, txt, hint, mean = [], [], [], []
    for item in batch :
        jpg.append(item['jpg'])
        txt.append(item['txt'])
        hint.append(item['hint'])
        mean.append(item['mean'])
    jpg = torch.stack(jpg)
    hint = torch.stack(hint)
    new_batch = {
        'jpg' : jpg,
        'txt' : txt,
        'hint' : hint,
        'mean' : mean
    }
    return new_batch
    
def rotate_image(image : Union[Image.Image,torch.Tensor] , angle : float ) -> Union[Image.Image, torch.Tensor] :
    try :
        image = image.rotate(angle, expand = 1)
    except AttributeError :
        image = tensor_rotate(image, angle, expand = True)
    return image 

def create_dir(base_path: str):
    structure = [
        'train/img',
        'train/map',
        'train/mean'
    ]
    
    for dir_path in structure:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")

def clear_dir(base_path: str):

    target_subdirectories = [
        'train/img',
        'train/map',
        'train/mean'
    ]
    try :
        os.remove(os.path.join(base_path,'labels.csv'))
    except FileNotFoundError : 
        pass
    for subdirectory in target_subdirectories:
        full_subdirectory_path = os.path.join(base_path, subdirectory)
        
        if os.path.exists(full_subdirectory_path) and os.path.isdir(full_subdirectory_path):

            for item in os.listdir(full_subdirectory_path):
                item_path = os.path.join(full_subdirectory_path, item)
                
                if os.path.isfile(item_path) and item.endswith(('.pt', '.csv')):
                    os.remove(item_path)  
                
        else:
            print(f"Subdirectory does not exist: {full_subdirectory_path}")

def largest_rotated_rect(w : float , h : float , angle : float ) -> tuple:
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(np.floor(angle / (np.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = w * np.cos(alpha) + h * np.sin(alpha)
    bb_h = w * np.sin(alpha) + h * np.cos(alpha)

    gamma = np.arctan(bb_w/ bb_w) if (w < h) else np.arctan(bb_w/ bb_w)

    delta = np.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image : Union[Image.Image, torch.Tensor],
    width : int , height : int ) -> Union[Image.Image, torch.Tensor]:
    """
    Given a PIL.Image or tensor with len.shape = 3, crops it to the given width and height,
    around it's centre point
    
    Careful : if tensor, handles dimensions C, H, W whereas PIL.Image is W, H
    
    """
    assert isinstance(image,Image.Image) or (len(image.shape) == 3 and image.shape[0] <= 3)
    try :
        image_size = (image.shape[1], image.shape[2])
    except AttributeError :
        image_size = (image.size[0], image.size[1])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    try : 
        image = image.crop((x1,y1,x2,y2))
    except AttributeError :
        image = image[:, x1:x2, y1:y2]

    return image

def get_dimensions(width: int, height: int, box_size: int = 512) -> list:
    n_width, n_height = width // box_size, height // box_size
    corners = []

    for i in range(1, n_height + 1):
        y_upper = (i - 1) * box_size 
        y_lower = i * box_size     

        for j in range(0, n_width):
            x_left = j * box_size  
            x_right = (j + 1) * box_size 
            corners.append((x_left, y_upper, x_right, y_lower))

    return corners

def get_cropped_images(image : Union[Image.Image, torch.Tensor], box_size : int = 512) -> list :
    '''
    [WARNING] if image is a PIL.Image then will handle image as shape C, W, H.
    If tensor and it has been converted from a PIL.Image, width and height will be exchanged
    and so you have to make sure that image tensor also has shape C, W, H
    '''
    assert isinstance(image, Image.Image) or (len(image.shape) == 3 and image.shape[0] <= 3) 
    try : 
        size = image.shape[1:]
    except AttributeError :
        size = image.size 
    corners = get_dimensions(size[0], size[1], box_size)
    try :
        cropped_images = [image.crop(coords) for coords in corners]
    except AttributeError : 
        cropped_images = list()
        for coords in corners:
            x1, y1, x2, y2 = coords    
            cropped_images.append(image[:, x1:x2, y1:y2])
    
    if isinstance(image,torch.Tensor) :
        cropped_images = [rearrange(im, 'C H W -> C W H') for im in cropped_images]
    return cropped_images

def density_map(config : MapConfig, 
                points : list,
                boxes  : list,
                img_dim : tuple,
                precision = torch.float32 ) -> torch.Tensor:

    if config.type == 'RGB':
        map = torch.zeros(3,img_dim[1],img_dim[0], device = DEVICE, dtype=precision)
    elif config.type == 'HeatMap':
        map =  torch.zeros(1,img_dim[1],img_dim[0], device = DEVICE, dtype=precision)
    else :
        print("Error: Invalid argument 'MapConfig.type' was provided. Choices are 'RGB' or 'HeatMap'. ")
        sys.exit(1)

    x = torch.arange(0, img_dim[1], dtype=precision, device = DEVICE )
    y = torch.arange(0, img_dim[0], dtype=precision, device = DEVICE)
    Y,X = torch.meshgrid(x, y, indexing='ij')

    if config.scale_gaussian:          
        
        for k,coord in enumerate(boxes) :
            if config.include_box_size :
                    
                if len(boxes[k]) == 2 :
                    var_x= max(4, coord[0])
                    var_y= max(4, coord[1])
                else :
                    var_x = max(4, coord[2] - coord[0])
                    var_y = max(4, coord[3] - coord[1])
            else :
                var_x, var_y = 4, 4

            assert var_x > 0 and var_y > 0, f'found negative box size : ({var_x},{var_y}) - {boxes}'

            scaler = 2*np.pi * np.sqrt(var_x*var_y) 
            gaussian = torch.exp(-((1/var_x)*(X - points[k][0])**2 + (1/var_y)*(Y - points[k][1])**2) / 2) / scaler
            for c in range(map.shape[0]):
                map[c,:,:] += gaussian
    else :
        print(f'Else statement for config.scale_gaussian = {not config.scale_gaussian} not written as case not relevant anymore')
        sys.exit(1)

    if config.save_as == 'Tensor':
        return map
    elif config.save_as == 'PIL':
        return ToPILImage()(map)  # NOT RECOMMENDED --> you will lose some points
    else : 
        print("Error: Invalid argument 'MapConfig.save_as' was provided. Choices are 'Tensor' or 'PIL'. ")
        sys.exit(1)

def spatial_content_nwpu(config : MapConfig
                        , id : str) -> dict :

    img_path = ROOT_DIR_NWPU+f'images/{id}.jpg'
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
        gt_path = ROOT_DIR_NWPU+f'jsons/{id}.json'
        with open(gt_path, 'r') as file:
                    data = json.load(file)
                    points, boxes = data.pop('points'), data.pop('boxes')
    except :
        try :
            gt_path = ROOT_DIR_NWPU+f'mats/{id}.mat'
            mat = loadmat(gt_path)
            points, boxes = mat.pop('annPoints'), mat.pop('annBoxes')
        except FileNotFoundError :
            print(f'No annotation  file (.json or .mat) exist for id :{id}')
            return

    dens = density_map(config = config,
                        points=points,
                        boxes = boxes,
                        img_dim=img.size)

    return dict(image = img, density = dens)   

def extract_smaller_pairs(img : torch.Tensor, dens : torch.Tensor,
 angles : Optional[list] = [45, -45, 90, -90, 135, -135, 180],
 random_flip = True ,
 minimal_density = None) -> list :

    def is_empty(density : torch.Tensor) -> bool :
        val = density.sum().item()
        if val < minimal_density :
            return True 
        return False 
    
    def crop(img : torch.Tensor , dens : torch.Tensor ) -> tuple :
        img = rearrange(img, 'C H W -> C W H')
        icropped_512 = get_cropped_images(img,512) 
        icropped_725 = get_cropped_images(img,725)  #725 is smallest dim of square s.t. for any rotation of angle in 0-90, a 512,512 square with x-axis aligned is contained in it.
        dens = rearrange(dens, 'C H W -> C W H')
        dcropped_512 = get_cropped_images(dens,512) 
        dcropped_725 = get_cropped_images(dens,725)
        return icropped_512, icropped_725, dcropped_512, dcropped_725
    
    #try :    
    icropped_512, icropped_725, dcropped_512, dcropped_725 = crop(img, dens)
    if not minimal_density :
        minimal_density = dens.sum().item() / len(dcropped_725)

    mask_512, mask_725 = torch.tensor([not is_empty(crop) for crop in dcropped_512]), torch.tensor([not is_empty(crop) for crop in dcropped_725])
   
    icropped_512, dcropped_512 = torch.stack(icropped_512)[mask_512], torch.stack(dcropped_512)[mask_512]
    icropped_725, dcropped_725 = torch.stack(icropped_725)[mask_725], torch.stack(dcropped_725)[mask_725]
     
    if angles is None :
        angles = np.random.uniform(low=5, high=180, size=5) + np.random.uniform(low=-180, high=5, size=5) 
    rotation_list = list()

    for k in range(len(icropped_725)):
        img, dens = icropped_725[k], dcropped_725[k]
        rotation_list.append( [
            (crop_around_center(rotate_image(img,alph), 512, 512),
            crop_around_center(rotate_image(dens,alph), 512, 512)
            ) for alph in angles
        ] )
    rotation_list = [pair for rotated_pair_list in rotation_list for pair in rotated_pair_list] 

    #apply mask a second time as rotating can remove all objects
    rotation_list = [pair for pair in rotation_list if not is_empty(pair[1])]

    for k in range(icropped_512.shape[0]) :
        pair = icropped_512[k], dcropped_512[k]
        rotation_list.append(pair)
    
    if random_flip :
        prob_flip = np.random.uniform(low = 0, high = 1, size = len(rotation_list))
        for k, p in enumerate(prob_flip) : 
            if p > 0.5 :
                rotation_list[k] = (hflip(rotation_list[k][0]), hflip(rotation_list[k][1]))
    return rotation_list if len(rotation_list) != 0 else None

def wrapper_extract_smaller_pairs(img : torch.Tensor, dens : torch.Tensor,
 angles : Optional[list] = [45, -45, 90, -90, 135, -135, 180],
 random_flip = True ,
 minimal_density = None) -> list :
 
    def adjust_dimensions(img, dens) :
        size = max(img.shape[1],726), max(img.shape[2],726)
        og_surface = dens.shape[1] * dens.shape[2]
        new_surface = size[0] * size[1]
        reshape = Resize(size=size, interpolation=InterpolationMode.NEAREST_EXACT)

        img = reshape(img)
        gaussian_scaler = og_surface / new_surface  
        dens = reshape(dens) * gaussian_scaler

        return img,dens 
    copy_img = img.clone()
    copy_dens = dens.clone()
    if img.shape[1] <= 726 or img.shape[2] <= 726 :        
        img, dens = adjust_dimensions(img, dens)
        
        split_image = extract_smaller_pairs(img, dens, angles, random_flip, minimal_density) 
        if split_image :
            return split_image
    else :
        split_image = extract_smaller_pairs(img, dens, angles, random_flip, minimal_density) 

        if split_image :
            return split_image

        _, width, height = img.shape
        top_crop = height // 6
        bottom_crop = height - height // 6
        img = img[:, top_crop:bottom_crop, :]
        dens = dens[:, top_crop:bottom_crop, :]
        if img.shape[1] <= 726 or img.shape[2] <= 726 :
            img, dens = adjust_dimensions(img, dens)
        split_image = extract_smaller_pairs(img, dens, angles, random_flip, minimal_density)
        if split_image :
            return split_image 
    
    #if the img is large dimension with few spreaded people :
    try :
        trial = extract_smaller_pairs(copy_img, copy_dens, angles, random_flip, minimal_density=10 )
    except RuntimeError :
        trial = extract_smaller_pairs(img, dens, angles, random_flip, minimal_density=10 )
    if not trial :
        try :
            trial = extract_smaller_pairs(copy_img, copy_dens, angles, random_flip, minimal_density=1 )
        except :
            trial = extract_smaller_pairs(img, dens, angles, random_flip, minimal_density=1 )
    return trial
# %%    

def spatial_content_jhu(config : MapConfig
                        , id : str) -> dict :
    img_path = f'{ROOT_DIR_JHU}{config.load_dir}images/{id}.jpg'
    gt_path = f'{ROOT_DIR_JHU}{config.load_dir}gt/{id}.txt'

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

    points = []
    boxes = []
    with open(gt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y, w, h, o, b = map(int, line.split())
            if o != 3: # not include occlusions
                points.append([x, y])
                boxes.append([w, h])

    dens = density_map(config = config,
                        points = points,
                        boxes = boxes,
                        img_dim=img.size)

    return dict(image = img, density = dens)   

def create_data(config : MapConfig, blip_device = BLIP_DEVICE, id_digit_range = 7) :
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(blip_device)
    tensorize = ToTensor()
    
    id = f'{1:0{id_digit_range}d}'
    largest_possible_id = ''.join(['9' for _ in range(id_digit_range)])

    save_path = f'{config.save_dir}{config.save_for}'
    print(f'{save_path=}')
    try :
        os.mkdir(save_path)
        print(f'Created folder {save_path=}')
    except FileExistsError :
        pass
    
    img_path = f'{save_path}/img'
    map_path = f'{save_path}/map'
    try :
        os.mkdir(img_path)
    except FileExistsError :
        pass 
    try :
        os.mkdir(map_path)
    except FileExistsError :
        pass 

    labels = f'{save_path}/label.csv'
    with open(labels, 'w') as file:
        file.write('id,prompt\n')

    if config.save_as == 'Tensor' :
        extension = 'pt'
    elif config.save_as == 'PIL' :
        extension = 'jpg'
    else :
        raise ValueError(f'Invalid {config.save_as=} passed. Choices are "Tensor" or "PIL". ')
    
    #TODO: handle case were sub directory test and val of jhu set does not exist
    #search in jhu
    jhu_images = f'{ROOT_DIR_JHU}{config.load_dir}images'
    N = sum([1 for item in os.listdir(jhu_images) if os.path.isfile(os.path.join(jhu_images, item))])
    bar = tqdm(total=N, desc='Processing...', unit='images')
    print(f'Starting loading from JHU-set (total_images={N})')
    for name in os.listdir(jhu_images) :
        try :
            id_,extension_ = name.split('.')
        except ValueError :
            continue
        if extension_ != 'jpg' :
            continue 
        try :    
            img, dens = spatial_content_jhu(config, id_).values()
        except (ValueError, AttributeError)  :
            continue    # spatial_content_jhu returned None
        img = tensorize(img)
        batch_pictures = wrapper_extract_smaller_pairs(img,dens, minimal_density = None)

        if batch_pictures is None or len(batch_pictures) == 0:
            print(f'(Found tricky image) {batch_pictures is None=}\n'
                  f'{name=} from Jhu-set')
            continue
        for image,density in batch_pictures :
            
            #rotating tensors may have rotated objects out of frame --> assures we have no false positives
            if density.sum().item() < 1 :
                continue

            inputs = processor(ToPILImage()(image),text = 'a photograph of ', return_tensors="pt").to(blip_device)
            out = model.generate(**inputs, max_new_tokens = 10)
            prompt = processor.decode(out[0], skip_special_tokens=True)
            
            torch.save(image.cpu(),f'{save_path}/img/{id}.{extension}')
            torch.save(density.cpu(),f'{save_path}/map/{id}.{extension}')
            with open(labels, 'a') as file:
                file.write(f'{id}.{extension}, {prompt}\n')
            bar.set_postfix({'id': id})

            new_id = int(id) + 1
            id = f'{new_id:0{id_digit_range}d}'
            if id == largest_possible_id :
                print(f'{largest_possible_id=} reached ! Cannot create new images from data - consider augmenting {id_digit_range=}. ')
                bar.close()
                return
        bar.update(1)
    bar.close()

    #search in nwpu
    nwpu_images = f'{ROOT_DIR_NWPU}images'
    N = sum([1 for item in os.listdir(nwpu_images) if os.path.isfile(os.path.join(nwpu_images, item))])
    bar = tqdm(total=N, desc='Processing...', unit='images')
    print(f'Starting loading from NWPU-set (total_images={N})')
    for name in os.listdir(nwpu_images) :
        try :
            id_,extension_ = name.split('.')
        except ValueError :
            continue
        if extension_ != 'jpg' :
            continue 
        try :    
            img, dens = spatial_content_nwpu(config, id_).values()
        except (ValueError, AttributeError) :
            continue    # spatial_content_nwpu returned None or extract no square out of image (latter is unlikely)

        img = tensorize(img)
        batch_pictures = wrapper_extract_smaller_pairs(img,dens, minimal_density = None)

        if batch_pictures is None or len(batch_pictures) == 0:
            print(f'(Found tricky image) {batch_pictures is None=}\n'
                  f'{name=} from NWPU-set')
            continue
        for image,density in batch_pictures :

            if density.sum().item() < 10 :
                continue
            
            inputs = processor(ToPILImage()(image),text = 'a photograph of a crowd of people', return_tensors="pt").to(blip_device)
            out = model.generate(**inputs, max_new_tokens = 20)
            prompt = processor.decode(out[0], skip_special_tokens=True)
            
            torch.save(image.cpu(),f'{save_path}/img/{id}.{extension}')
            torch.save(density.cpu(),f'{save_path}/map/{id}.{extension}')
            with open(labels, 'a') as file:
                file.write(f'{id}.{extension}, {prompt}\n')
            bar.set_postfix({'id': id})

            new_id = int(id) + 1
            id = f'{new_id:0{id_digit_range}d}'
            if id == largest_possible_id :
                print(f'{largest_possible_id=} reached ! Cannot create new images from data - consider augmenting {id_digit_range=}. ')
                bar.close()
                return
        bar.update(1)
    bar.close()

#%%
if __name__=="__main__" :
    #TODO: verify if about to create files already image exist and skip otherwise 
    #print(f'{config=}')  
    #create_data(config)
    print('This is an old file, use "data_tools/load_dataset.py" instead. ')
