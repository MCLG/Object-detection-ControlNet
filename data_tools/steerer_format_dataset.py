import os
import json
import torch 
from tqdm import tqdm
from PIL import Image 
from torchvision.transforms import ToPILImage

# This files formats the processed training data for our model into STEERER training data. Use this if you have different data and wish to train 
# a version of STEERER on your specific data. If you use this version of STEERER, it will influence the behaviour of the loss functions in our model
# by more accurately predicting object positions from the gt tensors and thus reducing the loss magnitude from small location difference between gt location
# and sampled location.
#
# Running this file will only create the folders "images/" and "jsons/" and the file "train.txt" needed to train STEERER. For the remaining .txt files we suggest manually
# creating them. For the directory organisation and training of STEERER go to : https://github.com/taohan10200/STEERER 
# By default, we reproduce the file structure from SHHB/ (see https://github.com/taohan10200/STEERER)
#
# Assuming your data is organized as :
#
# dataset/
#   train/
#       img/
#       mean/
#       map/
#
# simply run "python steerer_format_dataset.py arg1 arg2" where arg1 is the location of "dataset/" and arg2 the save location you want the data to be.

def main() :
    import argparse

    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("path_data", help="Path to data img/mean/map (.pt files). ")
    parser.add_argument("path_save", help="Your save file path ")
    args = parser.parse_args()

    if not os.path.exists(args.path_data) :
        raise FileNotFoundError(f'Path to Data : "{args.path_data}" not found. ')
    if not os.path.exists(args.path_save) :
        os.mkdir(args.path_save)
    
    path_images = os.path.join(args.path_data,'train/img')      #'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/img'
    path_maps = os.path.join(args.path_data,'train/map')        #'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/map'
    path_means = os.path.join(args.path_data,'train/mean')      #'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/mean'
    if not all(os.path.exists(p) for p in [path_images, path_maps, path_means]) :
        raise FileNotFoundError(f'Empty dataset : "{args.path_data}". ')

    json_save_dir = os.path.join(args.path_save,'SHHB/jsons')           #'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/jsons'
    if not os.path.exists(json_save_dir) :
        os.makedirs(json_save_dir)
    images_asjpeg_save_dir = os.path.join(args.path_save, 'SHHB/images') #'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/images'
    if not os.path.exists(images_asjpeg_save_dir) :
        os.makedirs(images_asjpeg_save_dir)

    train_file = os.path.join(args.path_save, 'SHHB/train.txt')#'/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/SHHB/train.txt'
    train_lines = []
    progress = tqdm(total=len(os.listdir(path_images)))
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
        progress.update(1)
    progress.close()

    with open(train_file, "a") as file:
        file.writelines(train_lines)

if __name__ == '__main__' :
    main()