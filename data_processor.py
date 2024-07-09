#%%
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from torchvision import transforms as T
from PIL import Image
from typing import List, Tuple
from copy import deepcopy
import torch
import sys
import os
import csv
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import argparse as arg
from tqdm import tqdm

@dataclass
class MapConfig :
    include_box_size : bool = False
    scale_gaussian : bool = False   #for type = 'RGB' this will return a map that seems black but still has the gt info
    save_as : str = 'Tensor'  #or Tensor
    type : str = 'RGB'    #'RGB' or 'HeatMap' -> if Tensor then has dim 1,512,512 instead of 3,512,512
    save_dir : str = "/net/vid-raxus/storage/deeplearning/users/luk02485/control_net/"  #location to save processed maps
    save_for : str = 'train' # or 'val' or 'test'
    load_dir : str = "train" # 'val' or 'test'
    CSV_include_count : bool = True

ROOT_DIR = "/net/vid-raxus/storage/deeplearning/datasets/jhu/jhu_crowd_v2.0/"

class CrowdDataSetv2(Dataset):
    """
        Version of CrowDataSet direcetly reading and saving the csv file as List[dictionnary]. Has key names adjusted to the lllyasviel Control Net.
    """
    def __init__(self, config : MapConfig, load_as_PIL = False, set = 'train'):
        # choose set as 'val' or 'test' 
        self.load_as_PIL = load_as_PIL
        self.set = set
        self.config = config
        self.include_count = self.config.CSV_include_count
        self.data = self.csv2dict()       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_path = f'{self.config.save_dir}{self.set}/map/'+item['id']
        target_path = f'{self.config.save_dir}{self.set}/img/'+item['id']
        id, ext = item['id'].split('.')
        gaussian_paths = [f'{self.config.save_dir}{self.set}/gaussians/'+id+'-2048'+f'.{ext}',
                          f'{self.config.save_dir}{self.set}/gaussians/'+id+'-1024'+f'.{ext}',
                          f'{self.config.save_dir}{self.set}/gaussians/'+id+'-512'+f'.{ext}',
                          f'{self.config.save_dir}{self.set}/gaussians/'+id+'-256'+f'.{ext}']


        # Weird behaviour here : if using cv2 or Image.open() then the error AttributeError or UnidentifiedImageError come out 
        # at some point in this function. It is as if i cannot properly open these images. Openning them as Tensors and then converting them works fine
        #source = torch.open(source_path)
        #target = cv2.imread(target_path)
        #print(type(source))

        source = torch.permute(torch.load(source_path).to(dtype=torch.float32),(2,1,0))
        target = torch.permute(torch.load(target_path).to(dtype=torch.float32),(2,1,0))
        gaussians = list()
        for path in gaussian_paths :
            gaussians.append(torch.load(path).to(dtype=torch.float32))
        
        if self.load_as_PIL :
            source = T.ToPILImage()(source)
            target = T.ToPILImage()(target)

        prompt = item['Prompt'].split(',')
        if self.include_count and len(prompt) != 3 :
            raise KeyError(f'MapConfig not fitting given data, CSV file at {self.config.save_dir}{self.set}/label.csv does not include crowd count. To Fix this, run write_csv_file() with MapConfig.include_count = True !')
        elif self.include_count :
            count = int(prompt[2])
            prompt = f'{prompt[0]}{prompt[1]}'
            return dict(jpg=target, txt=prompt, hint=source, gaussians = gaussians, count = count)
        else :
            prompt = f'{prompt[0]}{prompt[1]}'
            return dict(jpg=target, txt=prompt, hint=source, gaussians = gaussians)
    
    def csv2dict(self) -> List[dict]:
        data = []
        csv_path = f'{self.config.save_dir}{self.set}/label.csv'
        assert csv_path[-4:] =='.csv', 'path not leading to a .csv file'
        
        with open(csv_path, mode='r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                data.append(row)
        
        return data

def density_map(config : MapConfig, 
                gt_path : str, 
                img_dim : Tuple[int,int] ) -> torch.Tensor :
    """
    INPUT : 
        config : MapConfig - configuration of how the label map for each img is to be generated.
        gt_path : location of a csv file holding the head annotations.
        img_din : dimension of map to create (width,height)

    RETURNS :
        Gaussian density map of shape : img_dim.

    if include_box_size = True, then the gaussian clouds will have modified variance
        | Width 0  |
        | 0 Height |
    if scale_gaussian = True, then each gaussian cloud is a probability distribution.
    if type = ''RGBTrue'', then returns a 3,shape Tensor or RGB image

    Having both scale_gaussian to True and type = 'RGB' returns a almost black map. Yet it still holds the necessary informations
    """

    if config.type == 'RGB':
        map = torch.zeros(3,img_dim[1],img_dim[0])
    elif config.type == 'HeatMap':
        map =  torch.zeros(1,img_dim[1],img_dim[0])
    else :
        print("Error: Invalid argument 'MapConfig.type' was provided. Choices are 'RGB' or 'HeatMap'. ")
        sys.exit(1)
    
    x = torch.arange(0, img_dim[1], dtype=torch.float32)
    y = torch.arange(0, img_dim[0], dtype=torch.float32)
    Y,X = torch.meshgrid(x, y)
    if config.include_box_size:
        if config.scale_gaussian:

            with open(gt_path, 'r') as file:
                for line in file:
                    xywhob = line.split()
                    x,y,width,height = int(xywhob[0]), int(xywhob[1]), max(4,int(xywhob[2])), max(4,int(xywhob[3]))

                    scaler = 2*np.pi * np.sqrt(width*height)
                    gaussian = torch.exp(-((1/width)*(X - x)**2 + (1/height)*(Y - y)**2) / 2) / scaler
                    for c in range(map.shape[0]):
                        map[c,:,:] += gaussian
        else:
            with open(gt_path, 'r') as file:
                for line in file:
                    xywhob = line.split()
                    x,y,width,height = int(xywhob[0]), int(xywhob[1]), max(4,int(xywhob[2])), max(4,int(xywhob[3]))

                    gaussian =  torch.exp(-((1/width)*(X - x)**2 + (1/height)*(Y - y)**2) / 2)
                    for c in range(map.shape[0]):
                        map[c,:,:] += gaussian
    else:
        var = 4
        if config.scale_gaussian :
            scaler = 2 * np.pi * var

            with open(gt_path, 'r') as file:
                for line in file:
                    xywhob = line.split()
                    x = int(xywhob[0])
                    y = int(xywhob[1])

                    gaussian = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * var)) / scaler
                    for c in range(map.shape[0]):
                        map[c,:,:] += gaussian
        else :
            with open(gt_path, 'r') as file:
                for line in file:
                    xywhob = line.split()
                    x = int(xywhob[0])
                    y = int(xywhob[1])

                    gaussian = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * var))
                    for c in range(map.shape[0]):
                        map[c,:,:] += gaussian

    if config.save_as == 'Tensor':
        return map
    elif config.save_as == 'PIL':
        return T.ToPILImage()(map)
    else : 
        print("Error: Invalid argument 'MapConfig.save_as' was provided. Choices are 'Tensor' or 'PIL'. ")
        sys.exit(1)

def process(img : torch.Tensor, 
            label : torch.Tensor,
            gaussian : torch.Tensor,
            transform : T.transforms ) -> Tuple[torch.Tensor] :
    """
    Processes an image and its corresponding label. Assumes they have same dimension and label map is corrcetly oriented
    For future work: change transform into a list of torchvision.transforms  while respecting the current order.
    Ideally define a process dataclass and define the process steps globally on top of this file.
    """
    img = transform(img)
    label = transform(label)
    p = np.random.uniform()
    if p > 0.5:
        img = T.RandomHorizontalFlip(1)(img)
        label = T.RandomHorizontalFlip(1)(label)
        gaussian = T.RandomHorizontalFlip(1)(gaussian)
    img = T.Normalize([0.5],[0.5])(img)
    label = T.Normalize([0.5],[0.5])(label)
    #gaussian = T.Normalize([0.5],[0.5])(gaussian) <-------------------------------------- IDK IF INCLUDE HERE ???????
    return img,label,gaussian

def load_to(config : MapConfig,
            img_dim : int = 512,
            root_dir : str = ROOT_DIR,
            #save_to : str = "./jhu_crowd_v2.0/processed/train/img/",
            save_as : str = 'Tensor', #or JPEG
            batch : List[str] = []
            ) -> None :
    """
    INPUT: 
        config : MapConfig - configuration of how the label map for each img is to be generated, includes config.save_to location to which to save the labels.
        root_dir : location of the imgs.
        save_to : location to save the processed imgs.
        save_as : choice between 'Tensor' or 'JPEG', in future could include other format if necessary.
        batch : (for testing purposes) pass a list of img id (str) e.g. 0001.pt/.jpg to process and save.

    The function processes all the data i.e. creationg of label maps, pre-processing and saves it in new directory
    Necessary step before loading a torch.DataSet. Checks for duplicates before running.

    Addded gaussian_config variable. this will always create a [1,H,W] gaussian maps and save in folder config.save_dir/config.save_for/gaussians/
    in order to be used by STEERER.
    """
    print(f'Starting processing {len(batch)} images !\n'
    'Config: \n'
    f'  dim: {img_dim}\n'
    f'  include_box_size: {config.include_box_size}\n'
    f'  True_gaussian: {config.scale_gaussian}\n'
    f'  save_directory: {config.save_dir}/{config.save_for}\n')

    gaussian_config = deepcopy(config)
    gaussian_config.include_box_size = True 
    gaussian_config.scale_gaussian = True
    gaussian_config.type = 'HeatMap'
    
    if len(batch) > 0:
        progress = tqdm(total = len(batch))
        for id in batch:
            id,extension = id.split('.')[0], '.' + id.split('.')[1]

            try :
                path = root_dir+config.save_for+'/'+'images/'+id+".jpg"
                image = Image.open(path)
            except FileNotFoundError :
                print("SKIP: no existing .jpg to path: ", path)
                continue
            
            save_path_img = config.save_dir+config.save_for+"/img/"+id+extension
            save_path_label = label,config.save_dir+config.save_for+"/map/"+id+extension
            if os.path.exists(save_path_img) and os.path.exists(save_path_label):
                progress.update(1)
                continue
            
            gt_file = f"{root_dir}{config.save_for}/gt/{id}.txt".format(id=id)
            config.save_as = 'Tensor'   #enforce Tensor output regardless because we are normalizing afterwards    
            label = density_map(config,gt_path= gt_file, img_dim=image.size)
            gaussian = density_map(gaussian_config,gt_path= gt_file, img_dim=image.size)

            image = T.ToTensor()(image)
            image,label,gaussian = process(image,label,gaussian, transform = T.Resize((img_dim,img_dim)))
            
            dimensions = ['2048','1024','512','256']
            gaussians = list()
            gaussians.append(T.Resize((1536,2048),interpolation=T.InterpolationMode.BICUBIC)(gaussian))
            gaussians.append(T.Resize((768,1024),interpolation=T.InterpolationMode.BICUBIC)(gaussian))
            gaussians.append(T.Resize((384,512),interpolation=T.InterpolationMode.BICUBIC)(gaussian))
            gaussians.append(T.Resize((192,256),interpolation=T.InterpolationMode.BICUBIC)(gaussian))

            if save_as == 'Tensor':
                torch.save(image,config.save_dir+config.save_for+"/img/"+id+extension)
                torch.save(label,config.save_dir+config.save_for+"/map/"+id+extension)
                for gaussian in range(4) :
                    torch.save(gaussians.pop(0),config.save_dir+config.save_for+"/gaussians/"+id+'-'+dimensions.pop(0)+extension)

            elif save_as == 'PIL':
                image = T.ToPILImage()(image)
                label = T.ToPILImage()(label)
                torch.save(image,config.save_dir+config.save_for+"/img/"+id+extension)
                torch.save(label,config.save_dir+config.save_for+"/map/"+id+extension)
                for gaussian in range(4) :
                    torch.save(gaussians.pop(0),config.save_dir+config.save_for+"/gaussians/"+id+'-'+dimensions.pop(0)+extension)
            else: 
                print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'PIL'. ")
                sys.exit(1)
            progress.update(1)
        progress.close()
        print(f'Saved {len(batch)} images - location : {config.save_dir}{config.save_for}\n')
    '''
    else:
        root_dir = root_dir+config.save_for+'/'+'images/'
        image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        for path in image_paths:

            image = Image.open(path)

            if image.mode != 'RGB':
                continue 

            id = os.path.basename(path).split('.')[0]
            if os.path.exists(config.save_dir+config.save_for+id+'.pt') or os.path.exists(config.save_dir+config.save_for+id+'.jpg'):
                continue

            if save_as == 'Tensor':
                extension = '.pt'
            elif save_as == 'JPEG':
                extension = '.jpg'
            else:
                print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'JPEG'. ")
                sys.exit(1)

            image = T.ToTensor()(image)
            image,label = process(image,label, transform = T.Resize((img_dim,img_dim)))

            if save_as == 'Tensor':
                torch.save(image,config.save_dir+config.save_for+"/img/"+id+extension)
                torch.save(label,config.save_dir+config.save_for+"/map/"+id+extension)
            elif save_as == 'PIL':
                image = T.ToPILImage()(image)
                label = T.ToPILImage()(label)
                torch.save(image,config.save_dir+config.save_for+"/img/"+id+extension)
                torch.save(label,config.save_dir+config.save_for+"/map/"+id+extension)
            else: 
                print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'PIL'. ")
                sys.exit(1)
    '''

def write_csv_file(config : MapConfig,
                   filesize : int = 10,
                   return_id_list : bool = False
                   ) -> None :
    """
    INPUT : 
        path : path to an existing csv file holding all the infos on the imgs excluding annotations.
    Assumes format 'id,count,location,weather type'.
        save_to : location to save new csv file.
        include_count : boolean whether or not to include count of heads in img.
        save_as : choice between 'Tensor' or 'JPEG', in future could include other format if necessary.
        file-size : (for testing purposes) only write a file_size lines in file.
        return_id_list : (for testing purposes) serves as the batch for load_to().

    Assumes the location of the images are known. choose file_size = -1 to write entire
    
    """
    print(f'Writing CSV file for {config.save_for} from {ROOT_DIR}{config.load_dir}\n'
          f'include_count = {config.CSV_include_count}')
    save_as = config.save_as
    include_count = config.CSV_include_count

    labels = {}
    weather = {0: "no weather degradation", 1 : "fog or haze", 2 : "rain", 3: "snow"}
    path = f"{ROOT_DIR}{config.load_dir}/image_labels.txt"
    save_to = config.save_dir + config.save_for + '/'

    if save_as == 'Tensor':
        file_extension = ".pt"
    elif save_as == 'PIL':
        file_extension = ".jpg"
    else:
        print("Error: Invalid argument 'save_as'. Choices are 'Tensor' or 'JPEG'. ")
        sys.exit(1)
    
    k = 1
    with open(path, 'r') as file:
        for line in file:
            content = line.strip().split(',')
            if content[4] == '1':   #not including distractor images (no faces included)
                continue
            
            id = content[0]
            corresponding_img = Image.open(f"{ROOT_DIR}{config.load_dir}/images/"+id+".jpg")
            if corresponding_img.mode != 'RGB':
                continue

            count = int(content[1])
            prompt = content[2]
            weather_id = int(content[3])
            weather_prompt = weather[weather_id]
            prompt = 'a photo of a crowd of people in a '+prompt+", "+weather_prompt
            
            if include_count :
                labels[id+file_extension] = prompt,count
            else :
                labels[id+file_extension] = prompt

            if k == filesize :
                break 
            k += 1

    file_path = save_to+'label.csv'

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'Prompt'])
        
        for key, value in labels.items():
            if include_count :
                string_count =  f'{value[0]}, {value[1]}'
                writer.writerow([key,string_count])
            else:
                writer.writerow([key,value])
    print(f'Done ! Added {len(labels)} prompts to file {config.save_dir} !')
    if return_id_list :
        return list(labels.keys())

#%%
'''

@dataclass
class MapConfig :
    include_box_size : bool = False
    scale_gaussian : bool = False   #for type = 'RGB' this will return a map that seems black but still has the gt info
    save_as : str = 'Tensor'  #or Tensor
    type : str = 'RGB'    #'RGB' or 'HeatMap' -> if Tensor then has dim 1,512,512 instead of 3,512,512
    save_dir : str = "/net/vid-raxus/deeplearning/users/luk02485/control_net/"  #location to save processed maps
    save_for : str = 'train' # or 'val' or 'test'
    load_dir : str = "train"
    CSV_include_count : bool = True

ROOT_DIR = "/net/vid-raxus/deeplearning/datasets/jhu/jhu_crowd_v2.0/"
'''

if __name__ == "__main__":

    print('Usage : two arguments required : "datatype_usage" (str:train/val/test/all) and "numb_files_to_process" (int or -1 for all available files) ')

    parser = arg.ArgumentParser(description='Process three arguments.')

    parser.add_argument('arg1', type=str, help='train/val/test/all')
    parser.add_argument('arg2', type=int, help='amount of imgs to process')

    args = parser.parse_args()
    s = str(args.arg1)
    if s != 'train' and s != 'val' and s != 'test' and s != 'all':
        print(f'Invalid argument {args.arg1} passed for data type. Choices are "train", "val", "test" and "all". ')
        sys.exit(1)
    print(f'Data to load: {args.arg1}')

    try:
        filesize = int(args.arg2)
    except:
        print('Select a number of files or choose "-1" to process the entire Jhu set. ') 
        sys.exit(1)   
    print(f'Processing {args.arg2} images\n')

    if s != 'all':
        print(f'[{s} IMAGES]\n')
        config =MapConfig()
        conf_save_for = s
        conf_load_dir = s

        keys_list = write_csv_file(config,return_id_list=True,filesize=filesize)
        load_to(config,batch=keys_list)

    elif s == 'all':

        print('[TRAIN IMAGES]\n')
        trainconfig = MapConfig()
        keys_list = write_csv_file(trainconfig,return_id_list=True,filesize=filesize)
        load_to(trainconfig,batch=keys_list)

        print('[VALIDATION IMAGES]\n')
        valconfig = MapConfig()
        valconfig.save_for = 'val'
        valconfig.load_dir = 'val'

        keys_list = write_csv_file(valconfig,return_id_list=True,filesize=filesize)
        load_to(valconfig,batch=keys_list)

        print('[TESTING IMAGES]\n')
        testconfig = MapConfig()
        testconfig.save_for = 'test'
        testconfig.load_dir = 'test'

        keys_list = write_csv_file(testconfig,return_id_list=True,filesize=filesize)
        load_to(testconfig,batch=keys_list)



#%%
# TESTING NEW CROWDDATASET WITH 'gaussians' key added --------- USAGE OF STEERER IS HERE
'''
config = MapConfig() 
data = CrowdDataSetv2(config)

x = data.__getitem__(0)
print(x['txt'],x['hint'].shape,x['jpg'].shape,x['gaussians'][0].shape)
#plt.imshow(x['jpg'])
#plt.show()
#plt.imshow(x['hint'])
#plt.show()
#for g in x['gaussians']:
#    plt.imshow(g.permute(2,1,0))
#    plt.show()

#%%

# CALLING STEERER GIVEN DATA POINT X FROM CROWDDATASETV2.
    # key 'gaussians' is useless right now until we can influcence the inference using the training label of STEERER.
    # Currently outputs counts of heads without positions. The head count is far from accurate.
import STEERER.inference as inference
counter = inference.CounterWrapper(device='cuda:1')
counter = counter.to(counter.device)
temp_x = x['jpg'].permute((2,1,0)).to(counter.device)

out = counter.get_count(temp_x, None)
print(out)

# calliung a non processed image from JHU set :
img = Image.open('/net/vid-raxus/deeplearning/datasets/jhu/jhu_crowd_v2.0/train/images/0001.jpg')
img = T.ToTensor()(img)
out = counter.get_count(img)
print(out)
'''
#TODO :
    # use STEERER regardless, write te training function
    # load entire training, validation and testing set 
# %%
