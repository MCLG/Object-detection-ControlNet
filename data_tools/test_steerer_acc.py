
import os, random 
import matplotlib.pyplot as plt 
from torchvision.transforms import Resize
from ControlNetHome.tools.divergence_loss import to_dmap, DivergenceLoss
from STEERER.steerer_inference import CounterWrapper

def find_normalizing_constant(repetition : int = 10, nb_samples : int = 100, concat_to_size : int = 10 ) -> torch.float32 :
    pass


def test_data_set() :
    path = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_var_4_limited_density/train/'
    n_choices = 10
    f, ax = plt.subplots(n_choices,4, figsize = (15,5*n_choices))

    loss = DivergenceLoss(downscaling_factor = 8, device = 0)
    counter = CounterWrapper().to(0)
    for k in range(n_choices) :
        sample = random.choice(os.listdir(os.path.join(path,'img/')))
        id_sample, extension = sample.split('.')
        print(f'{k=} - {id_sample=}')
        dmap = torch.load(os.path.join(path,f'map/{sample}'))
        mean = torch.load(os.path.join(path,f'mean/{sample}'))
        sample = torch.load(os.path.join(path,f'img/{sample}'))

        ax[k,0].imshow(sample.cpu().permute(1,2,0))
        ax[k,0].axis('off')
        ax[k,0].set_title('target')

        ax[k,1].imshow(dmap.cpu().permute(1,2,0))
        ax[k,1].axis('off')
        ax[k,1].set_title('gt')

        mean_map = to_dmap(mean)
        ax[k,2].imshow(mean_map.cpu())
        ax[k,2].axis('off')
        ax[k,2].set_title('gt GMM means')

        steerer_map = counter.get_count(sample.to(0), mode = 'val')
        w2_diff = loss.wasserstein2(b_dmap = steerer_map, b_gt = [mean.to(0)], include_spread_loss = False)
        print(f'{w2_diff=}')
        ax[k,3].imshow(Resize(size=(512,512))(steerer_map).squeeze(0).permute(1,2,0).cpu())
        ax[k,3].axis('off')
        ax[k,3].set_title(f'steerer map, w_2={w2_diff}')
        
    plt.savefig('./debbug.png') 