# {"img_id": "0001.jpg", "human_num": 234, "points": [[65, 1507],
import os
import json
import torch 
from PIL import Image 
from torchvision.transforms import ToPILImage

path_images = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/img'
path_maps = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/map'
path_means = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/mean'

json_save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/jsons'
if not os.path.exists(json_save_dir) :
    os.makedirs(json_save_dir)
images_asjpeg_save_dir = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/images'
if not os.path.exists(images_asjpeg_save_dir) :
    os.makedirs(images_asjpeg_save_dir)

train_file = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/train.txt'
train_lines = []

for ifile in os.listdir(path_images) :
    file_name = ifile.split('.')[0]
    file_mean = os.path.join(path_means, ifile)
    file_img = os.path.join(path_images, ifile)

    #convert means into list and get count
    mean = torch.load(file_mean)
    human_num = mean.shape[0]
    mean_list = mean.tolist()

    #convvert tensor to jpg and save it 
    img = ToPILImage()(torch.load(file_img).cpu())
    img.save(f'{images_asjpeg_save_dir}/{file_name}.jpg', 'JPEG')

    d = {
        "img_id" : f'{file_name}.jpg',
        "human_num" : human_num,
        "points" : mean_list
        }
    # Save the dictionary as a minified JSON file
    with open(f'{json_save_dir}/{file_name}.json', 'w') as f:
        json.dump(d, f)

    # add id to train.txt
    train_lines.append(f'{file_name}\n')

with open(train_file, "a") as file:
    file.writelines(train_lines)
