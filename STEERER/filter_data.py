import csv
import os
import sys
import tqdm 

p = os.path.join(os.getcwd(), 'ControlNetHome')
if p not in sys.path :
    sys.path.append(p)

p = os.path.join(os.getcwd(), 'STEERER')
if p not in sys.path :
    sys.path.append(p)

# execute from project_root_dir with "python -m STEERER.filter_data arg1 arg2 arg3"

from STEERER.steerer_inference import CounterWrapper
import torch
import shutil

def main() :
    import argparse
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("loc_data", help="Path to your processed data (ends with .../train/) (map/img/mean) ")
    parser.add_argument("loc_new_data", help="Path to save your filtered data (map/img/mean) will create a new ./train file inside it. ")
    parser.add_argument("conf", help="a number between 10 and 100. We recommend at least 10. This sets the accuracy to filter out files : if relative_abs_error > conf => remove file. ")
    args = parser.parse_args()

    assert args.loc_data != args.loc_new_data, f'[WARNING] you were about to overwrite your data, with {args.loc_data=} and {args.loc_new_data=}. If thats what you wanted, delete this assert. '
    
    loc_new_data = os.path.join(args.loc_new_data,'train')
    if not os.path.exists(loc_new_data) :
        os.makedirs(loc_new_data, exist_ok=True)

    assert 0 < int(args.conf) <= 100
    confidence = int(args.conf)/100

    to_delete,keep = filter(
        loc_data = args.loc_data,
        eps = confidence
        )

    print(f"Number of files to remove: {len(to_delete)}, Number of files to keep: {len(keep)}, ")
    user_input = input(f"Do you want to proceed sending the remaining files to {loc_new_data} ? (y/no): ").strip().lower()


    while True:
        user_input = input(f"Do you want to proceed sending the remaining files to {loc_new_data} ? (y/no): ").strip().lower()
        if user_input == 'y':
            break
        elif user_input == 'no':
            print("Operation aborted. No files were sent.")
            return
        else:
            print("Invalid input. Please type 'y' to proceed or 'no' to abort.")
    
    print("Proceeding ...")
    csv_loc = os.path.join(args.loc_data, 'label.csv')
    send(   
        loc_new_data = loc_new_data,
        old_data_loc = args.loc_data,
        csv_loc = csv_loc,
        keep = keep,
        delete = to_delete
        )
    print(f'finished copying that to {loc_new_data} ! ')

def load_csv_to_dict(file_path):
    data_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            data_dict[row[0]] = row[1]
    return header, data_dict

def delete_rows_by_ids(data_dict, ids_to_delete):
    for id_to_delete in ids_to_delete:
        if id_to_delete in data_dict:
            del data_dict[id_to_delete]
    return data_dict

def save_dict_to_csv(header, data_dict, output_file_path):
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for id_, prompt in data_dict.items():
            writer.writerow([id_, prompt]) 

def send(loc_new_data : str, old_data_loc : str, csv_loc :str, keep : list, delete : list) :

    img_dir = os.path.join(old_data_loc,'img')
    dens_dir = os.path.join(old_data_loc, 'map')
    mean_dir = os.path.join(old_data_loc, 'mean')

    new_img_dir = os.path.join(loc_new_data,'img')
    new_dens_dir = os.path.join(loc_new_data, 'map')
    new_mean_dir = os.path.join(loc_new_data, 'mean')

    if not os.path.exists(new_img_dir) :
        os.makedirs(new_img_dir)
    if not os.path.exists(new_dens_dir) :
        os.makedirs(new_dens_dir)
    if not os.path.exists(new_mean_dir) :
        os.makedirs(new_mean_dir)

    header, dic = load_csv_to_dict(csv_loc)
    dic = delete_rows_by_ids(data_dict = dic, ids_to_delete = delete)
    save_dict_to_csv(header = header, data_dict = dic, output_file_path = os.path.join(loc_new_data,'label.csv'))
    bar = tqdm.tqdm(total=len(keep))
    for f in keep : 
        file = os.path.join(img_dir,f)
        shutil.copy2(file,new_img_dir)

        file = os.path.join(dens_dir,f)
        shutil.copy2(file,new_dens_dir)

        file = os.path.join(mean_dir,f)
        shutil.copy2(file,new_mean_dir)

        bar.update(1)
    bar.close()


def filter(loc_data : str, eps : int, DEVICE = torch.device(0))  :

    Steerer = CounterWrapper('STEERER/nwpu_pre_trained.pth').to(DEVICE)

    img_dir = os.path.join(loc_data,'img')
    dens_dir = os.path.join(loc_data, 'map')
    mean_dir = os.path.join(loc_data, 'mean')
    
    deleted_files = list()
    keep = list()
    bar = tqdm.tqdm(total = len(os.listdir(img_dir)))
    for f in os.listdir(img_dir):

        image = torch.load(os.path.join(img_dir,f), map_location = DEVICE)
        density = torch.load(os.path.join(dens_dir,f), map_location = DEVICE)

        approx_count = Steerer.get_count(image).sum().item()
        true_count = density.sum().item()
            
        score = abs(true_count-approx_count)/true_count

        if approx_count < 1 :
            deleted_files.append(f)
        elif score > eps :
            deleted_files.append(f)
            bar.update(1)
            continue
        keep.append(f)
        bar.update(1)
    bar.close()
    return deleted_files,keep

if __name__ == '__main__' :
    main()