import os 
import torch
import sys
import random

from STEERER.steerer_inference import CounterWrapper
from torchvision.transforms import Resize
import matplotlib.pyplot as plt 

from ControlNetHome.tools.divergence_loss import DivergenceLoss
from STEERER.lib.models.build_counter import  freeze_model

'''
    This file evaluates the performance of STEERER wrt. MSE/MAE/w2. Computes a forward pass of STEERER given the gt images, and the maps 
    are compared with the gt-maps. To run in console :

    'python -m STEERER.test_counter arg1 arg2 arg3' 
    
    from the root_dir. Results are printed in newly created folder './test_counter'.
    arg1 : the folder where your data lies, must contain /map, /img, /mean directories
    arg2 : temp file location to save intermediate tensors. Choose '0' if you want default loc in current dir.
    arg3 : device to use
'''

def main() :
    import argparse

    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("loc_data", help="Path to your processed data (map/img/mean) ")
    parser.add_argument("loc_temp_file", help="temporary location to store files")
    parser.add_argument("device", help="id of gpu to use")
    args = parser.parse_args()

    DATAFOLDER = args.loc_data                      # location of your data
    img_folder = os.path.join(DATAFOLDER,'img/')
    map_folder = os.path.join(DATAFOLDER, 'map/')
    mean_folder = os.path.join(DATAFOLDER, 'mean/')
    saved_maps_dir = args.loc_temp_file             # temp folder to save approximated density maps
    try : 
        default_val = int(saved_maps_dir)
        saved_maps_dir = None
    except :
        continue
    DEVICE = torch.device(int(args.device))
    densities = [10, 50, 100]                       # choose densities. They should be ordered.
    N = 25                                          # how many samples per density

    print(
        f'Testing counter performance \n',
        f'  {DATAFOLDER=}\n',
        f'  loc_temp_file={saved_maps_dir}\n',
        f'  {DEVICE=}\n',
        f'  {densities=}\n',
        f'  average_over_N={N}'
    )
    plot_res(
        DATAFOLDER, 
        img_folder, 
        map_folder, 
        mean_folder, 
        saved_maps_dir, 
        DEVICE,
        densities,
        N
        )
    
def plot_res(
    DATAFOLDER, 
    img_folder, 
    map_folder,     
    mean_folder, 
    saved_maps_dir, 
    DEVICE,
    densities,
    N
    ) :

    local_save_folder = './test_counter'
    if not os.path.exists(local_save_folder) :
        os.mkdir(local_save_folder)

    test_dic = dict(very_low=[], low=[], mid=[], high=[])
    file_dir = os.listdir(img_folder)
    random.shuffle(file_dir)
    print(f'Fetching test data ... ')
    for id_ in file_dir :

        dmap = torch.load(os.path.join(map_folder,id_))
        if dmap.sum().item() < densities[0] :
            if len(test_dic['very_low']) < N :
                img = torch.load(os.path.join(img_folder, id_))
                mean = torch.load(os.path.join(mean_folder, id_))
                test_dic['very_low'].append((img,dmap,mean))


        elif dmap.sum().item() > densities[0] and dmap.sum().item() < densities[1] :
            if len(test_dic['low']) < N :
                img = torch.load(os.path.join(img_folder, id_))
                mean = torch.load(os.path.join(mean_folder, id_))
                test_dic['low'].append((img,dmap,mean))

        elif dmap.sum().item() > densities[1] and dmap.sum().item() < densities[2] :
            if len(test_dic['mid']) < N :
                img = torch.load(os.path.join(img_folder, id_))
                mean = torch.load(os.path.join(mean_folder, id_))
                test_dic['mid'].append((img,dmap,mean))

        elif dmap.sum().item() > densities[2] :
            if len(test_dic['high']) < N :
                img = torch.load(os.path.join(img_folder, id_))
                mean = torch.load(os.path.join(mean_folder, id_))
                test_dic['high'].append((img,dmap,mean))
        stop = all([len(test_dic[key]) == N for key in test_dic.keys()])
        if stop :
            break
    print(f'Done !')
    path = 'STEERER/nwpu_pre_trained.pth'
    
    if saved_maps_dir :
        dmap_save_path = saved_maps_dir
    else :
        dmap_save_path = './temp_test_counter'
    steerer = CounterWrapper(path = path).to(DEVICE)
    freeze_model(steerer)

    resize = Resize(size=(512,512))
    l1_loss = []
    l2_loss = []
    print(f'Computing average TV/MSE loss ...')
    for key in test_dic.keys() :

        img_batch = torch.stack([test_dic[key][k][0] for k in range(N)]).to(DEVICE)
        dmap_batch = torch.stack([test_dic[key][k][1] for k in range(N)]).to(DEVICE)

        approx_maps = steerer.get_count(img_batch)
        approx_maps = resize(approx_maps)

        #save the maps 
        torch.save(approx_maps,os.path.join(dmap_save_path,f'{key}.pt'))

        l2_loss.append(sum(torch.sum(abs(dmap_batch - approx_maps)**2, dim = (2,3))).item()/N)
        
        dmap_batch = dmap_batch/torch.sum(dmap_batch, dim=(2,3), keepdim = True)
        approx_maps = approx_maps/torch.sum(approx_maps, dim=(2,3), keepdim = True)
        diff = 0.5*abs(dmap_batch - approx_maps)#B C H W 
        
        TV_losses = torch.sum(diff, dim=(2,3))
        
        l1_loss.append(sum(TV_losses).item()/N)
    print(f'Done !')
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    densities = ['<10', '<50', '<100', '>100']
    bar_labels_l1 = ['blue', 'blue', 'blue', 'blue']
    bar_labels_l2 = ['orange', 'orange', 'orange', 'orange']
    #bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax[0].bar(densities, l1_loss, color=bar_labels_l1)
    ax[1].bar(densities, l2_loss, color=bar_labels_l2)
    
    ax[0].set_ylabel('TV', fontsize=18)
    ax[1].set_ylabel('MSE',fontsize=18)
    
    #ax.set_title('Fruit supply by kind and color')
    #ax.legend(title='Fruit color')

    plt.savefig(f'{local_save_folder}/tv_mse.png')
    print(f'Figures saved in {local_save_folder}/tv_mse.png')
    w2 = DivergenceLoss(downscaling_factor = 8, extract_type = 'ggm-em', device = DEVICE, eval_on_low_dim = False)
    w2_loss = []
    diag_scale = 1.9073486328125e-06
    print(r'Computing average Wasserstein loss ... ')
    for key in test_dic.keys() :
        mean_batch = [test_dic[key][k][2].to(DEVICE) for k in range(N)]
        dens_batch = torch.load(os.path.join(dmap_save_path,f'{key}.pt')).to(DEVICE)
        loss = w2.wasserstein2(dens_batch, mean_batch,include_spread_loss = False).mean().item()   

        w2_loss.append(loss*diag_scale)
    print('Done !')

    fig, ax = plt.subplots(figsize=(10,6))
    densities = ['<10', '<50', '<100', '>100']
    bar_labels_l1 = ['blue', 'blue', 'blue', 'blue']
    ax.bar(densities,w2_loss, color = bar_labels_l1)
    ax.set_ylabel(r'$\mathcal{W}_2$',fontsize=18)

    plt.savefig(f'{local_save_folder}/w2.png')
    print(f'Figures saved in {local_save_folder}/w2.png')

    f, ax = plt.subplots(3,4, figsize=(16,12))
    for k,key in enumerate(test_dic.keys()) :

        dmap_batch = torch.stack([test_dic[key][k][1].to(DEVICE) for k in range(N)])
        dens_ = torch.load(os.path.join(dmap_save_path,f'{key}.pt')).to(DEVICE)

        min_k = torch.argmin(abs(torch.sum(dmap_batch, dim=(1,2,3)) - torch.sum(dens_, dim = (1,2,3))))

        dmap_batch = test_dic[key][min_k][1]
        img_batch = test_dic[key][min_k][0]
        dens_ = dens_[min_k]

        #first row is image
        ax[0,k].imshow(img_batch.permute(1,2,0).cpu())
        ax[0,k].set_yticklabels([])
        ax[0,k].set_xticklabels([])

        #second row is true dmap
        ax[1,k].imshow(dmap_batch.squeeze(0).cpu())
        ax[1,k].set_yticklabels([])
        ax[1,k].set_xticklabels([])

        #thrid row is approx dmap
        ax[2,k].imshow(dens_.squeeze(0).cpu())
        ax[2,k].set_yticklabels([])
        ax[2,k].set_xticklabels([])

        ax[2,k].set_xlabel(f'{key}',fontsize=18)

    ax[0,0].set_ylabel(f'target images',fontsize=18)
    ax[1,0].set_ylabel(f'control maps',fontsize=18)
    ax[2,0].set_ylabel(f'approximated maps',fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{local_save_folder}/samples.png') 
    print(f'Figures saved in {local_save_folder}/samples.png')

if __name__ == '__main__' :
    main()
