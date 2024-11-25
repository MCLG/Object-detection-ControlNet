#%%
import os
import glob 
import sys 
import torch 
import numpy as np
from typing import Optional, Iterable, Tuple, List
import matplotlib.pyplot as plt
import time 
#from torchvision.transforms import Resize, InterpolationMode
from sklearn.cluster import DBSCAN, KMeans
#from sklearn.mixture import GaussianMixture
from torch.nn.functional import interpolate
from gmm_torch.gmm import GaussianMixture
from scipy.optimize import linear_sum_assignment
from torch.autograd import Function

def ckpt_search() -> str :
    '''
    searches through lightning_logs for the models dict of the latest code version. 
    '''
    path = os.curdir + '/lightning_logs/version_*'
    versions = glob.glob(path) 
    checkpoints = []
    while len(checkpoints) == 0:
        try:
            ver = versions.pop()
        except Exception as e:
            print(f'Error : {e} occured. No existing ckpt of previous model version exists !')
            sys.exit(1)

        checkpoints = glob.glob(ver + '/checkpoints/epoch=*')
    return checkpoints[0]


@torch.no_grad()
def fast_mean_clustering(means : int, 
                         data : torch.Tensor,
                         gpu : torch.device = torch.device('cpu'), 
                         iter = 100) -> torch.Tensor :
    '''
    space_size : height x width 
    '''
    #torch.cuda.reset_peak_memory_stats()
    #start_time = time.time()

    assert len(data.shape) == 2 #N x 2

    data = data.float().to(gpu)
    initial_means = data[np.random.randint(0,data.shape[0],means)]
    eye_mat = torch.eye(means,device=gpu)
    means = torch.mm(eye_mat,initial_means)

    for i in range(iter) :
        diff_from_means = means.unsqueeze(0) - data.unsqueeze(1)
        dist_to_means = diff_from_means.pow(2).mean(2)
        idx_mean = dist_to_means.argmin(1)
        a = eye_mat[idx_mean].t()
        means = torch.mm(a,data)/a.sum(1,keepdim=True)

    return torch.round(means)

def to_dmap(means : torch.Tensor, cov : Optional[torch.Tensor] = None, display : bool = False,
    size = (512,512), device='cuda:1'):

    canvas = torch.zeros(size)
    if cov is None :
        for pos in means:
            
            circle_size = 4
            x, y = int(pos[0].item()), int(pos[1].item())
            x_start = max(0, x - circle_size // 2)
            x_end = min(size[0], x + circle_size // 2 + 1)
            y_start = max(0, y - circle_size // 2)
            y_end = min(size[1], y + circle_size // 2 + 1)
            
            canvas[x_start:x_end, y_start:y_end] = 1
    else :
        for pos, c in zip(means,cov) :
            x, y = int(pos[0].item()), int(pos[1].item())
            x_start = max(0, x - c[0] // 2)
            x_end = min(size[0], x + c[0] // 2 + 1)
            y_start = max(0, y - c[1] // 2)
            y_end = min(size[1], y + c[1] // 2 + 1)
            
            canvas[int(x_start):int(x_end), int(y_start):int(y_end)] = 1
    if not display :
        return canvas
    canvas_cpu = canvas.cpu()
    plt.imshow(canvas_cpu, cmap='gray')
    plt.show()

    return canvas

class AutogradPixelSelection(Function):
    @staticmethod
    def forward(ctx, tensor):
        # tensor shape : H,W

        ctx.save_for_backward(tensor)
        pixels = (tensor > 0).nonzero(as_tuple = False).float()
        
        return pixels

    @staticmethod
    def backward(ctx, grad_output):
        # choice of pixel value assignment from position : average variation in both x,y coordinates from previous (forward) computation
        tensor, = ctx.saved_tensors        
        active_pixels = (tensor > 0).nonzero(as_tuple = False)
        grad_input = torch.zeros_like(tensor)
        
        if len(active_pixels) > 0:
            for i, active_pixel in enumerate(active_pixels):
                row, col = active_pixel
                grad_input[row][col] = torch.mean(grad_output[i])

        return grad_input, None

class DivergenceLoss() :
    # choose min downsclaing_factor s.t. self.cluster_dim**2 is greater than the max coung in an image of your dataset
    def __init__(self,
    downscaling_factor : int, 
    extract_type = 'ggm-em', 
    device = torch.device('cpu'),
    eval_on_low_dim : bool = False):

        # options :
        #       downscaling_factor : for less memory usage and speed increase we downscale the input tensors to HxW // downscaling_factor
        #                            needs to be chosen s.t. max_count in an image in your dataset < (H.W // downscaling_factor)**2
        #       extract_type : fixed as this is the most efficient one and is differentiable
        #       device : device on which the model is initialized. Input tensors need to be sent to same device before.
        #       eval_on_low_dim : if True then the means of the gaussian clouds will not be rescaled to fit the original dim HxW of the map. 
        #                         This will reduce the magnitude of the w2 loss.
        #
        # We do not recommend using this for density maps with a count that exceeds (???). For our largest count in data set (3646) :
        #   peak_memory=17019.1630859375Mb (without backprop.)
        #   time=292.62189269065857s        (without backpop.)
        #   w2=tensor([0.7508, 0.9593], device='cuda:0', grad_fn=<AddBackward0>)
        #
        # We recommend a max count of 200 objects per density map for an effective forward and backward pass (max 2s per batch),
        # otherwise increase the downscaling_factor.
        
        self.extract_type = extract_type
        self.downscaling_factor = downscaling_factor
        self.cluster_dim = 512//self.downscaling_factor
        self.device = device
        self.cov_loss = torch.nn.MSELoss()
        self.eval_on_low_dim = eval_on_low_dim  

    # Not used
    def closest_to_means(self, X : torch.Tensor, means : torch.Tensor)  -> torch.Tensor:

        cost_matrix = torch.stack([torch.stack([torch.linalg.norm(x - mu) for x in X]) for mu in means])
        _, Xmin_to_mean_idx = linear_sum_assignment(cost_matrix.detach().cpu())
        
        return X[Xmin_to_mean_idx]

    def fixed_state_moments_init(self, data : torch.Tensor, K : int, seed = 42) -> Tuple[torch.Tensor] :
        # We need to fix the state in the init (and in gmm.py init) otherwise an exact reconstruction dmap
        # of the gt density map will not result in W2(gt_dmap, dmap) = 0, provided include_spread_loss = False.
        np_random_state = np.random.get_state()
        np.random.seed(seed)

        rd_pick_init = np.random.randint(0, data.shape[0], size = K)
        mu_init = data[rd_pick_init].clone().unsqueeze(0)

        rd_pick_init = np.random.randint(0, data.shape[0], size = K)
        d_init = data[rd_pick_init].clone().unsqueeze(0)
        var_init_ = torch.mean((d_init-mu_init)**2, dim = 1, keepdim = True )
        var_init = var_init_.expand(-1, K, -1)

        np.random.set_state(np_random_state)
        return mu_init, var_init

    def extract_means(self, dmap: torch.Tensor ) -> Tuple[list,list] :
        #if dmap.requires_grad is True :
        #    dmap.retain_grad()

        #dmap = dmap.to(self.device)
        if len(dmap.shape) != 4 :
            dmap_ = dmap.unsqueeze(0)
        else : 
            dmap_ = dmap

        og_surface = dmap_.shape[2]*dmap.shape[3]
        object_c = dmap_.sum(dim = (1,2,3))
        dmap_low_res = interpolate(size=(self.cluster_dim,self.cluster_dim),mode= 'nearest-exact', input=dmap_)
        dmap_low_res_ = dmap_low_res * (og_surface/(self.cluster_dim**2))

        non_zero_points = []
        for k in range(dmap_low_res_.shape[0]) :
            pixels = AutogradPixelSelection.apply(dmap_low_res_.squeeze(1)[k])
            #non_zero_points = torch.stack((non_zero_points, pixels), dim=0) if non_zero_points is not None else pixels
            non_zero_points.append(pixels)
        if self.extract_type == 'ggm-em': 
            means, covariances = [], []
            for k,non_z_elements in enumerate(non_zero_points) :
                n_components = object_c[k].item()
                
                # it is necessary to make the initialization of gmm to be data dependent for gradient tracking
                #TODO: memory x speed check with this 

                mu_init, var_init = self.fixed_state_moments_init(non_z_elements, round(n_components))
                
                gm = GaussianMixture(n_components=round(n_components),
                    covariance_type = 'diag', 
                    n_features = 2,
                    mu_init = mu_init,     #torch.Tensor (1, k, d)
                    var_init = var_init).to(self.device)      #torch.Tensor (1, k, d)
                
                gm.fit(non_z_elements)

                #gm_mu = self.closest_to_means(non_z_elements, gm.mu.clone().squeeze(0)) # Too slow !!!

                gm_mu = gm.mu.clone().squeeze(0)
                gm_var = gm.var.clone().squeeze(0)

                #means = torch.stack((means,gm_mu), dim=0) if means is not None else gm_mu
                #covariances = torch.stack((covariances,gm_var), dim=0) if covariances is not None else gm_var
                if not self.eval_on_low_dim :
                    means.append(gm_mu * self.downscaling_factor)
                else : 
                    means.append(gm_mu)
                covariances.append(gm_var)
        else :
            raise AttributeError(f'd_loss.extract_type={self.extract_type} not implemented. ')

        return means, covariances

    def wasserstein2(self, b_dmap : torch.Tensor, b_gt : List[torch.Tensor], include_spread_loss = True) -> torch.Tensor:
        # Input : b_dmap.shape B,1,N,2 
        #        b_gt : list of gt_tensors with length B.
        #               Each element is the mean of a gaussian cloud on the x,y grid. We assume gt covariances are fixed to diag(4,4)
        #        include_spread_loss : If False, will ignore the variance ||cov(b_dmap)^0.5 - cov(b_gt)^0.5||_frob^2 in the loss computation.
        #
        # Output : returns a tensor of shape B,1 which holds the average Wasserstein2 distance between two gaussian clouds from b_dmap and b_gt.
                
        assert b_dmap.shape[0] == len(b_gt), f'Batch mismatch between entries and gt : {b_dmap.shape=} and {len(b_gt)=}. '
        
        means,cov = self.extract_means(b_dmap)

        #TODO :Check if sort before or not is faster :
        #means,_ = torch.sort(means,dim=-2)
        #gt,_ = torch.sort(b_gt,dim=-2)

        mean_averages_batch = torch.stack([
            self.w2_average_over_means(gaus=means[i], gaus_gt=b_gt[i])[0]
            for i in range(b_dmap.shape[0])
        ])
        if include_spread_loss :
            cov_averages_batch = torch.stack([
                self.w2_average_over_var(variances = cov[i])
                for i in range(b_dmap.shape[0])
            ])
        else :
            cov_averages_batch = torch.zeros(mean_averages_batch.shape, device = self.device)
        wasserstein_loss = mean_averages_batch + cov_averages_batch
        
        return wasserstein_loss 

    def w2_average_over_means(self, gaus : torch.Tensor, gaus_gt : torch.Tensor, 
    eval_only_matching_means : bool = False,
    unmatched_penalty_scale : float = .01) -> Tuple[torch.Tensor, list] :

        # TODO : what to do with ranges 0,512 ??? --> distance becomes huge
        # gaus, gaus_gt : two arrays containing the 2D means of a gaussian cloud. gaus_gt is the ground truth. 
        # This distinction makes a difference in the unmatched_mean_penalty mechanism

        # We need to fit the closest means from gaus and gaus_gt to one another and calucalte their squared euclidean norm.
        # This is equivalent to solving a minimum weight matching problem in bipartite graphs where the edges are induced by the matrix 
        #   C = [ || gaus[i]-gaus_gt[j] ||_2^2 ]_i,j
        # and the optimization problem :
        #   min_X SUM_i,j C_i,j * X_i,j     X_ij={0,1}
        # Optional : 
        #   eval_only_matching_means : bool = False (default). If True then if N gaus match exactly with N gaus_gt and gaus_gt.len > N, the remaining 
        #                                                      means in gaus_gt will not be evaluated in the loss. If False, then the remaining means.                                        
        
        cost_matrix = torch.stack([torch.stack([torch.linalg.norm(a-b)**2 for b in gaus_gt]) for a in gaus])
        indexes = linear_sum_assignment(cost_matrix.detach().cpu())

        id_gaus, id_gaus_gt = indexes
        min_costs = torch.stack([cost_matrix[i1][i2] for i1, i2 in zip(id_gaus,id_gaus_gt)])
        w2_mean = torch.mean(min_costs)

        if not eval_only_matching_means :
            unmatched_nb = gaus.shape[0] -gaus_gt.shape[0]

            if unmatched_nb > 0 :
                # too many means generated by model
                penalty_scale = unmatched_penalty_scale * (1 + unmatched_nb/gaus.shape[0])

                remove = torch.ones(gaus.shape[0], dtype = bool)
                remove[id_gaus] = False         #mask selecting all indices but the ones used in w2_mean
                penalty_means = gaus[remove]

                penalty = penalty_scale * torch.sum(torch.linalg.norm(penalty_means, dim=1)**2)/penalty_means.shape[0]
                w2_mean += penalty

            elif unmatched_nb < 0 :
                # not enough means generated by model
                penalty_scale = unmatched_penalty_scale * (1 + abs(unmatched_nb)/gaus_gt.shape[0])

                remove = torch.ones(gaus_gt.shape[0], dtype = bool)
                remove[id_gaus_gt] = False         #mask selecting all indices but the ones used in w2_mean
                penalty_means = gaus_gt[remove]

                penalty = penalty_scale * torch.sum(torch.linalg.norm(penalty_means, dim=1)**2)/penalty_means.shape[0]
                w2_mean += penalty

            else : 
                #perfect matching !
                pass

        return w2_mean, indexes


    def w2_average_over_var(self, variances : torch.Tensor) -> torch.Tensor :
        # We assumed the gt data has fixed variance 2,2. All gaussians have diagonal covariances.

        tensor_of_twos = torch.tensor([2.,2.], device = self.device).repeat(variances.shape[0],1,1).squeeze(1)
        variances = variances.sqrt()
        average_frob_dist = self.cov_loss(variances, tensor_of_twos)
        
        return average_frob_dist

    def grad_total_variation_norm_w_2norm_sq(self, dmap : torch.Tensor, gt_dmap : torch.Tensor) -> torch.Tensor :
        # computes the gradient of the TV-norm between 2 density maps (as true density function maps) and computes the squared 2-norm of it.

        assert len(dmap.shape) == 4 and dmap.shape == gt_dmap
        
        sum_gt_dmap = torch.sum(gt_dmap,dim=(2,3), keepdim= True)
        true_dmap = dmap/torch.sum(dmap,dim=(2,3), keepdim= True)
        gt_true_dmap = dmap/sum_gt_dmap
        v = gt_true_dmap - true_dmap
        sgn_v = torch.sgn(v)

        grad_TV = -0.5*(sgn_v/sum_gt_dmap - (sgn_v*gt_true_dmap)/(sum_gt_dmap**2))
        normed_grad_TV = torch.linalg.norm(grad_TV, dim = (2,3))**2

        return normed_grad_TV
    
    def TV_norm(self, dmap : torch.Tensor, gt_dmap : torch.Tensor) -> torch.Tensor :
        # computes the TV-norm between 2 density maps (as true density function maps)

        assert len(dmap.shape) == 4 and dmap.shape == gt_dmap
        
        true_dmap = dmap/torch.sum(dmap,dim=(2,3), keepdim = True)
        gt_true_dmap = dmap/torch.sum(gt_dmap,dim=(2,3), keepdim = True)

        dist = true_dmap - gt_true_dmap
        TV_norm = 0.5*(torch.sum(dist, dim = (2,3)))

        return TV_norm  



def main() :
    from tqdm import tqdm 
    gpu = torch.device(0)
    data_map_folder = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/map/'
    max_count = 200
    max_memory_usage = 1000 #in Mb
    max_time = 9 #seconds
    
    W2 = DivergenceLoss(downscaling_factor = 8, extract_type = 'ggm-em', device = gpu, eval_on_low_dim = False)
    results = dict()
    bar = tqdm(total = len([n for n in os.listdir(data_map_folder)]))
    for map_file in os.listdir(data_map_folder) :
        path_to_map = os.path.join(data_map_folder,map_file)
        dmap = torch.load(path_to_map).requires_grad_(True)

        if dmap.sum().item() > max_count :
            continue
        batch = dmap.unsqueeze(0).to(gpu)

        #Fetch approx gt as list of len=B containing tensors of shape K,2 
        gt_means, _ = W2.extract_means(batch.detach().requires_grad_(False))

        #performance tracking
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()  # Record memory usage before

        #Compute W2
        out = W2.wasserstein2(b_dmap=batch, b_gt=gt_means, include_spread_loss = True)    
        end = time.time()

        time_needed = end-start
        peak_memory = torch.cuda.max_memory_allocated()/(1024**2)

        if peak_memory > max_memory_usage :
            mem = {f'{map_file}' : {'peak_memory (Mb)' : peak_memory, 'count' : dmap.sum().item(), 'time (s)' : time_needed}}
            results.update(mem)
        elif time_needed > max_time :
            mem = {f'{map_file}' : {'peak_memory (Mb)' : peak_memory, 'count' : dmap.sum().item(), 'time (s)' : time_needed}}
            results.update(mem)
        bar.update(1)

    print(f'Saving results ...')
    import json
    file_path = os.path.join(os.getcwd(), 'benchmark_dataset_on_wasserstein2.json')
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)
    print(f"Dictionary saved to {file_path}")

    bar.close()
if __name__ == '__main__' :

    '''
    Biggest counts :
    0000001.pt 122c
    0000036.pt
    0000202.pt 608c     <-- 
    0000231.pt 629c
    0000238.pt 1118c
    0000574.pt 1360c
    0000576.pt 1516c
    0000577.pt 1520c
    0007843.pt 1697c
    0015581.pt 3646c
    '''
    #To be included in thesis :::
    '''
    i have two arrays of dimension N,2 and M,2 which represents gaussian means. 
    I wish to compute the wasserstein2 distance (ignoring the variance for now) of those gaussians using the closed form 
    W2(g1,g2) = ||mu1-mu2||_2^2. this has to be done for each element in those arrays and then we average the distances.
     Small issue : the arras are not sorted so we have to determine which mu1 corresponds to which mu2 in both arrays. 
     I have solved this by solving a  minimum weight matching problem in bipartite graphs where the edges are induced by the matrix 
          C = [ || gaus[i]-gaus_gt[j] ||_2^2 ]_i,j
    This works. and it computes the W2 distances for each gaussian. 
    However one issue remains. What hsould i do if N>M or M<N ? I cannot just ignore it because i will use this as a loss function to my model
     and if i ignore then the model can just generalize by not generating any mean in gaus_gt (output of model, gaus ground truth). 
     I cannot also set a point, say the origin 0,0 as a reference and the remainin M-N or N-M gaussians of g2 or g1 to average the distances of them to the origin. 
     This could also lead to a generalization where my model only produces gaus_gt to be 0,0 centered. 
     Finally maybe we penalize with the ground truth ? If say K are unmatched in gaus (ground truth) then yes 
     i can average with the origin and this would make sense. However what to do if instead K are unmatched in gaus_gt, 
     i.e. more generated means that actual ground truth means ?
    '''

    '''w2losser = d_loss(l_type = 4, downscaling_factor = 8, extract_type = 'ggm-em')

    gaus = torch.randint(0,513,size=(51,2)).float()
    gt = torch.randint(0,513,size=(27,2)).float()
    cost,indexes = w2losser.w2_average_over_means(gaus,gt)
    print(f'{gaus=}')
    print(f'{gt=}')
    print(f'{cost=}')
    print(f'{type(cost)=}')
    print(f'{indexes=}')
    covariances = torch.randint(0,6,size=(51,2)).float().to(0)
    cov_wasserstein = w2losser.w2_average_over_var(covariances)
    print(f'{cov_wasserstein=}')'''

    #tensor = torch.zeros(size=(1,1,5,5))
    #tensor[0,0,1,1] = 5.
    #tensor[0,0,4,4] = 2.
    #tensor[0,0,2,4] = 1.
    #tensor[0,0,3,4] = 7.
    #tensor[0,0,1,4] = 99.
    #tensor[0,0,4,2] = 2000.
    #tensor.requires_grad = True
    #print(f'{tensor.requires_grad=}')
    #modification = AutogradPixelSelection.apply(tensor.squeeze(1)[0])
    #print(f'{modification.requires_grad=}')
    #out = (modification**2-78.).sum()
    #print(f'{modification=}')
    #out.backward()
    #print(f'{tensor.grad=}')


    main()
    '''gpu = torch.device(0)
    path_to_map = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/map/0000202.pt'
    dmap = torch.load(path_to_map).requires_grad_(True)

    path_to_map2 = '/net/vid-raxus/storage/deeplearning/users/luk02485/ccnet_fixed_var_4/train/map/0000036.pt'
    dmap2 = torch.load(path_to_map2).requires_grad_(True)

    losser = DivergenceLoss(l_type = 4, downscaling_factor = 8, extract_type = 'ggm-em', device = gpu, eval_on_low_dim = False)

    # create Batch B,1,512,512
    batch = torch.stack((dmap,dmap2),dim=0).to(gpu)
    batch.retain_grad() #for test
    print()

    #Fetch approx gt as list of len=B containing tensors of shape K,2 
    means, _ = losser.extract_means(batch.detach().requires_grad_(False))

    #performance tracking
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()  # Record memory usage before

    #batch = torch.rand(size=(1,1,512,512), requires_grad = True ).to(gpu)
    #batch.retain_grad()
    #means, _ = losser.extract_means(batch.detach().requires_grad_(False))

    #Compute W2
    out = losser.wasserstein2(b_dmap=batch, b_gt=means, include_spread_loss = True)    

    end = time.time()
    peak_memory = torch.cuda.max_memory_allocated()/(1024**2)

    print(f'{peak_memory=}Mb')
    print(f'time={end-start}s')
    print(f'w2={out}')

    loss = torch.mean(out)
    loss.backward()
    
    print(f' : {batch.grad=}, {(torch.all(batch.eq(0)))=}')
    
    f, ax = plt.subplots(1,4,figsize=(12,6))
    ax[0].imshow(dmap.detach().cpu().squeeze(0))

    dmap_r = to_dmap(means[0].squeeze(0),size=(512,512))
    ax[1].imshow(dmap_r.detach().cpu())

    dmap_r = to_dmap(means[1].squeeze(0),size=(512,512))
    ax[2].imshow(dmap_r.detach().cpu())

    ax[3].imshow(dmap2.detach().cpu().squeeze(0))
    plt.savefig('./k_cluster.png')
    '''