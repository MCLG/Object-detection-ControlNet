import os
import glob 
import sys 
import torch 
import numpy as np
import torch.nn.functional as F
from typing import Optional, Iterable, Tuple, List
import matplotlib.pyplot as plt
import time 
#from torchvision.transforms import Resize, InterpolationMode
from sklearn.cluster import DBSCAN, KMeans
#from sklearn.mixture import GaussianMixture
from torch.nn.functional import interpolate
from tools.gmm_torch.gmm import GaussianMixture
from scipy.optimize import linear_sum_assignment
from torch.autograd import Function

# execute as  'python -m tools.divergence_loss path_to_train_dir device scale_bool' from /Project_dir/ControlNetHome/
# example :
'''
     python -m tools.divergence_loss /net/vid-raxus/storage/deeplearning/users/luk02485/ProcessedData/train 1 1
'''

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

        # OPTIONS :
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

    '''def diffbar_sampling(data : torch.Tensor, K : int, replace = False) :
        
        assert (len(data) >= K and not replace) or replace
        if K == len(data) and not replace :
            return data
        
        dist = Normal(0.,1.)
        sampled_data = list()
        while len(sampled_data) != K :
            event_shape = torch.Size([len(data)])
            samples = dist.rsample(event_shape)
            for p, x in zip(samples, data) :
                if p > .5 :
                    sampled_data.append(x)
                    if not replace :
                        data.remove(x)
                if len(sampled_data) == K :
                    break
        return torch.stack(sampled_data)'''

    def fixed_state_moments_init(self, data : torch.Tensor, K : int, seed = 42) -> Tuple[torch.Tensor] :
        # We need to fix the state in the init (and in gmm.py init) otherwise an exact reconstruction dmap
        # of the gt density map will not result in W2(gt_dmap, dmap) = 0, provided include_spread_loss = False.
        #torch_random_state = torch.random.get_rng_state()
        #torch.manual_seed(seed)
        
        d_min, d_max = data.min(), data.max()
        data = (data.squeeze(1) - d_min) / (d_max - d_min)

        #rd_pick_init = np.random.randint(0, data.shape[0], size = K)
        #mu_init = data[rd_pick_init].clone().unsqueeze(0)
        #mu_init = self.diffbar_sampling(data=data, K=K).unsqueeze(0)
        indices1 = torch.randperm(data.shape[0])[:K]
        mu_init = data[indices1].unsqueeze(0)
        
        #rd_data = self.diffbar_sampling(data=data, K=data.shape[0])
        indices2 = torch.randperm(data.shape[0])[:K]
        rd_data = data[indices2].unsqueeze(0)
        
        #rd_pick_init = np.random.randint(0, data.shape[0], size = K)
        #d_init = data[rd_pick_init].clone().unsqueeze(0)
        var_init_ = torch.mean((rd_data-mu_init)**2, dim = 1, keepdim = True )
        var_init = var_init_.expand(-1, K, -1)

        #torch.random.manual_seed(torch_random_state)

        return mu_init*(d_max - d_min) + d_min, var_init*(d_max - d_min) + d_min

    def extract_means(self, dmap: torch.Tensor ) -> Tuple[list,list] :
        
        # Given a tensor with values ranging between 0 and 1, will return the list of 2D-means of a GMM by performing EM-algo.
        # The number of means is determined by the sum of the tensors pixel value (assumes you are passing a gaussian density map)
        # Handles Batch inputs.
        #
        # Input : size B,1,512,512 or 1,512,512
        # Output : (Tuple) : ([(N,2)] with len = B ) x 2 
        #
        # If the sum of the density map is less than < 1, it will handle it as a 0 map and return empty tensors.

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

                # delete try - except in the future.
                try :
                    n_components = round(object_c[k].item())#non_z_elements.shape[0]
                except : 
                    print(f'{object_c=}')
                    print(f'{dmap=}')                    
                    n_components = round(object_c[k].item())
                
                if n_components != 0 and non_z_elements.numel() > 0:

                    if non_z_elements.shape[0] < n_components :
                        print(f'Dimension reduction in DivergenceLoss.extracting_means resulted in more centroids to data points to map ! Setting centroids to available data')
                        n_components = non_z_elements.shape[0]

                    # it is necessary to make the initialization of gmm to be data dependent for gradient tracking                
                    mu_init, var_init = self.fixed_state_moments_init(non_z_elements, n_components)
                    
                    #print(f'{mu_init.shape=}, {n_components=}, {non_z_elements.shape=}')

                    #save states
                    #np_state = np.random.get_state()
                    #torch_state = torch.random.get_rng_state()
                    #np.random.seed(42)
                    #torch.manual_seed(42)
                
                    gm = GaussianMixture(n_components=n_components,
                        covariance_type = 'diag', 
                        n_features = 2,
                        mu_init = mu_init,     #torch.Tensor (1, k, d)
                        var_init = var_init,
                        cluster_dim = self.cluster_dim).to(self.device)      #torch.Tensor (1, k, d)
                    
                    gm.fit(non_z_elements)

                    # set the states back to normal
                    #torch.random.set_rng_state(torch_state)
                    #np.random.set_state(np_state)

                    #gm_mu = self.closest_to_means(non_z_elements, gm.mu.clone().squeeze(0)) # Too slow !!!
                    
                    gm_mu = gm.mu.clone().squeeze(0) if len(gm.mu.shape) == 3 else gm.mu.clone()
                    gm_var = gm.var.clone().squeeze(0)if len(gm.var.shape) == 3 else gm.var.clone()

                else :
                    gm_mu = torch.empty(size=(0,2), device = self.device)
                    gm_var = torch.empty(size=(0,2), device = self.device)
                
                if not self.eval_on_low_dim :
                    means.append(gm_mu * self.downscaling_factor)
                else : 
                    means.append(gm_mu)
                covariances.append(gm_var)
        else :
            raise AttributeError(f'd_loss.extract_type={self.extract_type} not implemented. ')

        return means, covariances

    def wasserstein2(
        self, 
        b_dmap : torch.Tensor, 
        b_gt : List[torch.Tensor], 
        include_spread_loss = True,
        space_scaler : Optional[float] = None
        ) -> torch.Tensor:

        # Input : b_dmap.shape B,1,N,2 
        #        b_gt : list of gt_tensors with length B.
        #               Each element is the mean of a gaussian cloud on the x,y grid. We assume gt covariances are fixed to diag(4,4)
        #        include_spread_loss : If False, will ignore the variance ||cov(b_dmap)^0.5 - cov(b_gt)^0.5||_frob^2 in the loss computation.
        #
        # Output : returns a tensor of shape B,1 which holds the average Wasserstein2 distance between two gaussian clouds from b_dmap and b_gt.
        #
        # Optional : space_scaler : is a magnitude scaler to scale the dimension down. This would typically be used outside of this function in the
        # training loop to scale down the importance of a loss term. We propose passing this directly inside this function.
        # In that way we can prevent the values to explode for very large map sizes that are very far from the ground truth.
        # This holds by linearity but increases numerical error. Worst-case W2 for 512,512 maps is around 104857600 so no overflow error and this can be ignored.

        assert b_dmap.shape[0] == len(b_gt), f'Batch mismatch between entries and gt : {b_dmap.shape=} and {len(b_gt)=}. '
        
        means,cov = self.extract_means(b_dmap)
        
        mean_averages_batch = torch.stack([
            self.w2_average_over_means(
                gaus=means[i], 
                gaus_gt=b_gt[i],
                space_scaler=space_scaler
                )[0]
            for i in range(b_dmap.shape[0])
        ])
        if include_spread_loss :
            cov_averages_batch = torch.stack([
                self.w2_average_over_var(
                    variances = cov[i],
                    space_scaler=space_scaler)
                for i in range(b_dmap.shape[0])
            ])
        else :
            cov_averages_batch = torch.zeros(mean_averages_batch.shape, device = self.device)
        wasserstein_loss = mean_averages_batch + cov_averages_batch
        
        return wasserstein_loss, means

    def w2_average_over_means(
        self, gaus : torch.Tensor, 
        gaus_gt : torch.Tensor, 
        eval_only_matching_means : bool = False,
        unmatched_penalty_scale : float = .01,
        space_scaler : Optional[float] = None ) -> Tuple[torch.Tensor, list] :

        # gaus, gaus_gt : two arrays containing the 2D means of a gaussian cloud. gaus_gt is the ground truth. 
        #                 
        # This distinction makes a difference in the unmatched_mean_penalty mechanism

        # We need to fit the closest means from gaus and gaus_gt to one another and calucalte their squared euclidean norm.
        # This is equivalent to solving a minimum weight matching problem in bipartite graphs where the edges are induced by the matrix 
        #   C = [ || gaus[i]-gaus_gt[j] ||_2^2 ]_i,j
        # and the optimization problem :
        #   min_X SUM_i,j C_i,j * X_i,j     X_ij={0,1}
        # Optional : 
        #   eval_only_matching_means : bool = False (default). If True then if N gaus match exactly with N gaus_gt and gaus_gt.len > N, the remaining 
        #                                                      means in gaus_gt will not be evaluated in the loss. If False, then the remaining means.                                        
        #   space_scaler : see wasserstein2() function for description.

        assert gaus_gt.shape[0] != 0, 'Empty true tensor passed for gt annotations !'

        ls = space_scaler if space_scaler else 1.

        if gaus.shape[0] == 0 :
            id_gaus_gt = indexes = []
            w2_mean = torch.tensor(0., device = self.device)
        else :    
            cost_matrix = torch.stack([torch.stack([torch.linalg.norm(a-b)**2 for b in gaus_gt]) for a in gaus])
            indexes = linear_sum_assignment(cost_matrix.detach().cpu())

            id_gaus, id_gaus_gt = indexes
            min_costs = torch.stack([cost_matrix[i1][i2] for i1, i2 in zip(id_gaus,id_gaus_gt)])
            w2_mean = torch.mean(min_costs) * ls

        if not eval_only_matching_means :
            unmatched_nb = gaus.shape[0] -gaus_gt.shape[0]

            if unmatched_nb > 0 :
                # too many means generated by model
                penalty_scale = unmatched_penalty_scale * (1 + unmatched_nb/gaus.shape[0])

                remove = torch.ones(gaus.shape[0], dtype = bool)
                remove[id_gaus] = False         #mask selecting all indices but the ones used in w2_mean
                penalty_means = gaus[remove] 
                
                # debbug ############
                #if penalty_means.shape[0] == 0 :
                #    print(f'DIVISION BY 0 at {__file__} line 327')
                #    sys.exit(0)
                ########################
                penalty = penalty_scale * torch.sum(
                    torch.linalg.norm(penalty_means, dim=1)**2
                    )/penalty_means.shape[0] * ls

            elif unmatched_nb < 0 :
                # not enough means generated by model
                penalty_scale = unmatched_penalty_scale * (1 + abs(unmatched_nb)/gaus_gt.shape[0])

                remove = torch.ones(gaus_gt.shape[0], dtype = bool)
                remove[id_gaus_gt] = False         #mask selecting all indices but the ones used in w2_mean

                penalty_means = gaus_gt[remove] 
                
                # debbug ############
                #if penalty_means.shape[0] == 0 :
                #    print(f'DIVISION BY 0 HERE')
                #    sys.exit(0)
                ########################

                penalty = penalty_scale * torch.sum(
                    torch.linalg.norm(penalty_means, dim=1)**2
                    )/penalty_means.shape[0] * ls

            else : 
                #perfect matching !
                penalty = 0.

        return w2_mean + penalty, indexes


    def w2_average_over_var(self, variances : torch.Tensor) -> torch.Tensor :
        # We assumed the gt data has fixed variance 2,2. All gaussians have diagonal covariances.

        ls = space_scaler if space_scaler else 1.

        tensor_of_twos = torch.tensor([2.,2.], device = self.device).repeat(variances.shape[0],1,1).squeeze(1)
        variances = variances.sqrt()
        average_frob_dist = self.cov_loss(variances, tensor_of_twos) * ls
        
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
        
        assert len(dmap.shape) == 4 and dmap.shape == gt_dmap.shape
        dmap_scalors = torch.sum(dmap,dim=(2,3), keepdim = True)
        dmap_scalors_safe = torch.where(dmap_scalors != 0., dmap_scalors, other = torch.tensor(1., dtype=dmap.dtype, device=dmap.device))
        true_dmap = dmap/dmap_scalors_safe 
        gt_dmap_scalor = torch.sum(gt_dmap,dim=(2,3), keepdim = True)
        gt_true_dmap = gt_dmap/gt_dmap_scalor
     
        dist = gt_dmap_scalor * abs(true_dmap - gt_true_dmap)
        TV_norm = 0.5*(torch.sum(dist, dim = (2,3)))

        return TV_norm  

def main() :

    import argparse

    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("path_map_data", help="Path to the maps (.pt files). ")
    parser.add_argument("device", help="device to use")
    parser.add_argument("scale_loss", help="if '0' the w2 loss will not be scaled and will be huge for large density map space. If '1', will scale down for density_maps of size 512,512. ")
    args = parser.parse_args()

    test_data_set_memory_consumption(
        device = torch.device(int(args.device)),
        data_path = args.path_map_data,
        scale = bool(args.scale_loss)
        )

def test_data_set_memory_consumption(
    device : torch.device, 
    data_path,
    scale = True) :

    torch.autograd.set_detect_anomaly(True)

    space_scaler = 1.9073486328125e-05 if scale else 1.

    mean_folder = os.path.join(data_path,'mean')
    density_folder = os.path.join(data_path,'map')
    
    print(f'searching for largest count ... ')
    device = torch.device(device)
    W2 = DivergenceLoss(downscaling_factor = 8, extract_type = 'ggm-em', device = device, eval_on_low_dim = False)

    max_error = 0
    max_time = 0
    peak_memory = 0
    mem_list, time_list = [], []

    from tqdm import tqdm
    bar = tqdm(total = len(os.listdir(density_folder)))

    files = os.listdir(density_folder)
    for i in range(0, len(files), 2):
        
        # problematic files : file1='000395.pt', file2='000396.pt'
        file1 = files[i]
        file2 = files[i + 1] if i + 1 < len(files) else None
        if file2 :
        
            density1 = torch.load(os.path.join(density_folder, file1)).to(device)
            density2 = torch.load(os.path.join(density_folder, file2)).to(device)

            mean1 = torch.load(os.path.join(mean_folder,file1)).to(device)
            mean2 = torch.load(os.path.join(mean_folder,file2)).to(device)

            batch1 = torch.stack([density1,density2])
            batch2 = torch.stack([torch.zeros_like(density1),density2])

            mean_batch = [mean1, mean2]

            torch.cuda.reset_peak_memory_stats()
            start_memory1 = torch.cuda.memory_allocated()
            start1 = time.time()

            batch1.requires_grad = True
            batch1.retain_grad()

            out1 = W2.wasserstein2(
                    batch1, 
                    mean_batch,
                    include_spread_loss= False,
                    space_scaler = space_scaler
                )
            out1.mean().backward()
            out1 = torch.max(out1)

            end1 = time.time()
            time1 = end1 - start1
            peak_memory1 = torch.cuda.max_memory_allocated()/(1024**2)
            peak_memory = max(peak_memory, peak_memory1)
            mem_list.append(peak_memory1)

            torch.cuda.reset_peak_memory_stats()
            start_memory2 = torch.cuda.memory_allocated()
            start2 = time.time()

            batch2.requires_grad = True
            batch2.retain_grad()

            out2 = W2.wasserstein2(
                    batch2, 
                    mean_batch,
                    include_spread_loss= False,
                    space_scaler = space_scaler
                )
            out2.mean().backward()
            out2 = torch.max(out2)

            end2 = time.time()
            time2 = end2 - start2
            peak_memory2 = torch.cuda.max_memory_allocated()/(1024**2)  
            peak_memory = max(peak_memory, peak_memory2)
            mem_list.append(peak_memory2)

            max_error = max(max_error, out1.item(), out2.item())
            max_time = max(max_time, time1, time2)

            time_list.append(time1,time2)

        else :
            break

        bar.update()
    bar.close()
    
    print(f'{max_error=}, {peak_memory=}, {max_time=}')

if __name__ == '__main__' :
    main()

    