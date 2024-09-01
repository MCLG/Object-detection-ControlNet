#%%
from data_jhu import MapConfig
import torch
import json
from typing import Tuple
import numpy as np
import os, sys
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from copy import deepcopy
from tqdm import tqdm
import csv 
import argparse as arg
from data_jhu import CrowdDataSetv2

if __name__ == "__main__" :    
    DEVICE = torch.device('cpu')        # specify device to create tensor maps here

    ROOT_DIR = '/net/vid-raxus/storage/deeplearning/datasets/nwpu/'
    config = MapConfig()

    # Change here for different save location of processed imgs and  densities
    if os.path.exists('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/'):
        config.save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/'
    else :
        os.mkdir('/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/')
        config.save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/'
        
'''
    This files loads the NWPU-dataset for trainin. It requires the 'ROOT_DIR' to be ordered as such :
    ./nwpu/
        images/
            idXXXX.jpg
            ...
        mats/
            idXXXX.mat
            ...
        jsons/
            idXXXX.json
''' 

class CrowdDataSet(CrowdDataSetv2) :
    def __init__(self, config : MapConfig, load_as_PIL= False, setfor = 'train') :
        super().__init__(config,load_as_PIL,set = setfor)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        source_path = f'{self.config.save_dir}{self.set}/map/'+item['id']
        target_path = f'{self.config.save_dir}{self.set}/img/'+item['id']
        id, ext = item['id'].split('.') 
        gaussian_path = f'{self.config.save_dir}{self.set}/gaussians/'+id+'-2048'+f'.{ext}'
    
        source = torch.permute(torch.load(source_path, map_location=torch.device('cpu')).to(dtype=torch.float32),(2,1,0))
        target = torch.permute(torch.load(target_path, map_location=torch.device('cpu')).to(dtype=torch.float32),(2,1,0))
        source = torch.rot90(source, -1, [0,1])
        target = torch.rot90(target, -1,[0,1])
        
        gaussian = torch.load(gaussian_path, map_location=torch.device('cpu')).to(dtype=torch.float32)
        gaussian = torch.flip(gaussian, dims = [2])
    
        if self.load_as_PIL :
            source = T.ToPILImage()(source)
            target = T.ToPILImage()(target)

        prompt = item['Prompt'].split(',')
        if self.include_count :#and len(prompt) != 2 :
            #if config.save_dir != '/net/vid-raxus/storage/deeplearning/users/luk02485/control_nwpu/':
            #This case handles a jhu fetch0


            count = int(prompt[-1])
            prompt = ''.join(prompt[:-1])


            #count = int(prompt[2])
            #prompt = f'{prompt[0]}{prompt[1]}'
            return dict(jpg=target, txt=prompt, hint=source, count = count, gaussian = gaussian)
            #raise KeyError(f'MapConfig not fitting given data, CSV file at {self.config.save_dir}{self.set}/label.csv does not include crowd count. To Fix this, run write_csv_file() with MapConfig.include_count = True !')
        #elif self.include_count :
        #    count = int(prompt[1])
        #    prompt = f'{prompt[0]}'
        #    return dict(jpg=target, txt=prompt, hint=source, count = count, gaussian = gaussian)
        else :
            prompt = ''.join(prompt[:-1])
            return dict(jpg=target, txt=prompt, hint=source, gaussian = gaussian)
#%%
def density_map(config : MapConfig, 
                points : list,
                boxes  : list,
                img_dim : Tuple[int,int],
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
    Y,X = torch.meshgrid(x, y)

    if config.include_box_size:
        if config.scale_gaussian:          
           
            for k,coord in enumerate(boxes) :
                var_x = max(4, coord[2] - coord[0])
                var_y = max(4, coord[3] - coord[1])
                
                assert var_x > 0 and var_y > 0, f'found negative box size : ({var_x},{var_y}) - {boxes}'

                scaler = 2*np.pi * np.sqrt(var_x*var_y)
                gaussian = torch.exp(-((1/var_x)*(X - points[k][0])**2 + (1/var_y)*(Y - points[k][1])**2) / 2) / scaler
                for c in range(map.shape[0]):
                    map[c,:,:] += gaussian
        else :
            print(f'Else statement for config.scale_gaussian = {not config.scale_gaussian} not written as case not relevant anymore')
            sys.exit(1)
    else :
        print(f'Else statement for config.include_box_size = {not config.include_box_size} not written as case not relevant anymore')
        sys.exit(1)

    if config.save_as == 'Tensor':
        return map
    elif config.save_as == 'PIL':
        return T.ToPILImage()(map)  # NOT RECOMMENDED --> you will lose some points
    else : 
        print("Error: Invalid argument 'MapConfig.save_as' was provided. Choices are 'Tensor' or 'PIL'. ")
        sys.exit(1)
        
# %%
'''
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photo of a crowd of people"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
'''



def load(config,size = -1, dim = (512,512), include_fakes = False) :

    # Load BLIP model for captioning the pictures
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    def retrieve_img_content(config : MapConfig
                         , id : str) -> dict :

        img_path = ROOT_DIR+f'images/{id}.jpg'
        try :
            img = Image.open(img_path)
            inputs = processor(img, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs)
            prompt = processor.decode(out[0], skip_special_tokens=True)
        except :
            print(f'Image with id : {id} does not exist at location : {img_path}')
            return

        try :
            gt_path = ROOT_DIR+f'jsons/{id}.json'
            with open(gt_path, 'r') as file:
                        data = json.load(file)
                        points, boxes = data.pop('points'), data.pop('boxes')
        except :
            try :
                gt_path = ROOT_DIR+f'mats/{id}.mat'
                mat = loadmat(gt_path)
                points, boxes = mat.pop('annPoints'), mat.pop('annBoxes')
            except FileNotFoundError :
                print(f'No annotation  file (.json or .mat) exist for id :{id}')
                return

        dens = density_map(config = config,
                            points=points,
                            boxes = boxes,
                            img_dim=img.size)
        
        return dict(image = img, density = dens, prompt = prompt, count = round(dens.sum().item()))

    if not os.path.exists(f'{config.save_dir}/{config.save_for}') :
        os.mkdir(f'{config.save_dir}/{config.save_for}')

    folder = ['gaussians', 'img', 'map']
    dir = os.path.join(f'{config.save_dir}/{config.save_for}')
    for subfolder in folder :
        try :
            os.makedirs(os.path.join(dir, subfolder))
        except :
            print(f'{subfolder} already exists in {dir}')

    if config.save_as == 'Tensor':
        file_extension = ".pt"
    elif config.save_as == 'PIL':
        file_extension = ".jpg"
    else:
        print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'PIL'. ")
        sys.exit(1)

    print(f'Writing CSV file for {config.save_for} from {ROOT_DIR}\n'
          f'include_count = {config.CSV_include_count}\n')

    labels = {}
    total = size
    total = len([f for f in os.listdir(f'{ROOT_DIR}images') if os.path.isfile(os.path.join(f'{ROOT_DIR}images', f)) and size<0])
    progress = tqdm(total=total)

    for filename in os.listdir(f'{ROOT_DIR}images'):
        id = os.path.splitext(filename)[0]
        content = retrieve_img_content(id=id, config=config)

        if content is None :
            continue 
        if not include_fakes and content['count'] == 0 :  # DO not include images with no crowds
            continue

        if config.CSV_include_count :
            labels[id+file_extension] = content['prompt'], content['count']
        else :
            labels[id+file_extension] = content['prompt']

        image,  density, gaussian = process(content['image'], content['density'], dim = dim)

        if config.save_as == 'Tensor':
                torch.save(image,config.save_dir+config.save_for+"/img/"+id+file_extension)
                torch.save(density,config.save_dir+config.save_for+"/map/"+id+file_extension)
                
                torch.save(gaussian,config.save_dir+config.save_for+"/gaussians/"+id+'-'+'2048'+file_extension)

        elif config.save_as == 'PIL':
            image = T.ToPILImage()(image)
            label = T.ToPILImage()(label)
            torch.save(image,config.save_dir+config.save_for+"/img/"+id+file_extension)
            torch.save(density,config.save_dir+config.save_for+"/map/"+id+file_extension)

            gaussian = T.ToPILImage()(gaussian)
            torch.save(gaussian,config.save_dir+config.save_for+"/gaussians/"+id+'-'+'2048'+file_extension)
        else: 
            print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'PIL'. ")
            sys.exit(1)
        
        size -= 1
        progress.update(1)
        if size == 0  :
            break
    progress.close()

    file_path = config.save_dir+config.save_for+'/label.csv'

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'Prompt'])
        
        for key, value in labels.items():
            if config.CSV_include_count :
                string_count =  f'{value[0]}, {value[1]}'
                writer.writerow([key,string_count])
            else:
                writer.writerow([key,value])

    print(f'Done ! Added {len(labels)} prompts to file {config.save_dir} !')

        
def process(img : torch.Tensor, dens : torch.Tensor, dim) -> Tuple[torch.Tensor]:
    """
    Processes an image and its corresponding label. Assumes they have same dimension and label map is corrcetly oriented
    For future work: change transform into a list of torchvision.transforms  while respecting the current order.
    Ideally define a process dataclass and define the process steps globally on top of this file.
    """
    original_field = dens.shape[1]*dens.shape[2]

    transform = T.Resize(dim, interpolation = T.InterpolationMode.NEAREST_EXACT)
    
    gaussian = deepcopy(dens)
    img = transform(img)
    dens = transform(dens)

    new_field = dim[0] * dim[1]

    gaussian_preserving_ratio = original_field/new_field
    dens = gaussian_preserving_ratio * dens

    gaussian_preserving_ratio = original_field / (1536 * 2048)
    gaussian = T.Resize((1536,2048),interpolation=T.InterpolationMode.NEAREST_EXACT)(gaussian) * gaussian_preserving_ratio

    p = np.random.uniform()
    if p > 0.5:
        img = T.RandomHorizontalFlip(1)(img)
        dens = T.RandomHorizontalFlip(1)(dens)
        gaussian = T.RandomHorizontalFlip(1)(gaussian)

    img = T.ToTensor()(img)
    #dens = T.ToTensor()(dens)
    #gaussian = T.ToTensor()(gaussian)
    img = T.Normalize([0.5],[0.5])(img)
    #dens = T.Normalize([0.5],[0.5])(dens)
    
    return img,dens,gaussian


if __name__ == "__main__" :

    print('Usage : two arguments required : "datatype_usage" (str:train/val/test) and "numb_files_to_process" (int or -1 for all available files) ')
    
    parser = arg.ArgumentParser(description='Process three arguments.')

    parser.add_argument('arg1', type=str, help='train/val/test')
    parser.add_argument('arg2', type=int, help='amount of imgs to process')

    args = parser.parse_args()
    s = str(args.arg1)
    if s != 'train' and s != 'val' and s != 'test':
        print(f'Invalid argument {args.arg1} passed for data type. Choices are "train", "val", "test". ')
        sys.exit(1)
    print(f'Data to load: {args.arg1}')

    try:
        filesize = int(args.arg2)
    except:
        print('Select a number of files or choose "-1" to process the entire NWPU set (a decent amount of images do not have annotations). ') 
        sys.exit(1)   
    print(f'Processing {args.arg2} images\n')

    conf_save_for = s
    print(f'save_dir = {config.save_dir}\n')
    load(config=config, size=filesize)

