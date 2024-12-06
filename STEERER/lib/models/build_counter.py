import torch
import torch.nn as nn
import torch.nn.functional as F
from STEERER.lib.models.backbones.backbone_selector import BackboneSelector
from STEERER.lib.models.heads.head_selector import  HeadSelector
from STEERER.lib.models.heads.moe import upsample_module
from STEERER.lib.utils.Gaussianlayer import Gaussianlayer

from ControlNetHome.ldm.modules.diffusionmodules.util import checkpoint
from functools import partial 

class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.tensor([1,0.5, 0.25, 0.125])
        sigma = torch.tensor(-torch.log(2*sigma))
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num
        self.count = 0

    def forward(self, input):
        loss = 0
        # import pdb
        # pdb.set_trace()
        for i in range(self.v_num):
            loss += input[i]*0.5*torch.exp(-self.sigma[i]) #/(2 * self.sigma[i] ** 2)
        loss +=0.01* torch.exp(0.5*self.sigma).sum()  #(torch.exp(-self.sigma).sum()-2)**2 #torch.log(self.sigma.pow(2).prod())
        self.count+=1
        if self.count %100 == 0:
            print(self.sigma.data)
        return loss

def freeze_model(model):
    for (name, param) in model.named_parameters():
            param.requires_grad = False

'''
if hasattr(self.counter, "enable_input_require_grads"):
            self.counter.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.counter.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
'''

class Baseline_Counter(nn.Module):
    def __init__(self, config=None,weight=200, route_size=(64,64),device=torch.device('cpu')):
        super(Baseline_Counter, self).__init__()
        
        self.config = config
        self.resolution_num = config.resolution_num

        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)

        self.gaussian_maximum=self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()
        
        if self.config.counter_type == 'withMOE':
            self.multi_counters = HeadSelector(self.config.head).get_head()
            self.counter_copy=HeadSelector(self.config.head).get_head()
            freeze_model(self.counter_copy)
            self.upsample_module = upsample_module(self.config.head)

        elif self.config.counter_type == 'single_resolution':
            self.count_head = HeadSelector(self.config.head).get_head()
        else:
            raise ValueError('COUNTER must be basleline or withMOE')
        self.weight = weight
        self.route_size = (route_size[0] //(2**self.resolution_num[0]),
                            route_size[1] //(2**self.resolution_num[0]))
        self.label_start = self.resolution_num[0]
        self.label_end = self.resolution_num[-1]+1
    '''
    # NEW wrapper for checkpointing
    def upsample_module_wrapper(self, input) :
        in_list,self.multi_counters,self.counter_copy = input
        return self.upsample_module(in_list,self.multi_counters,self.counter_copy)
    '''
    def backbone_call(self, inputs) :
        # https://github.com/huggingface/transformers/issues/23170
        '''def make_outputs_require_grad(module, input, output, val : bool):
            for tens in output :
                tens.requires_grad_(val)

        if inputs.requires_grad :
            hook = partial(make_outputs_require_grad, val = True)
            checkpoint_ = True
        else :
            hook = partial(make_outputs_require_grad, val = False)
            checkpoint_ = False
        self.backbone.register_forward_hook(hook)
        '''
        checkpoint_ = inputs.requires_grad
        if checkpoint_ :
            return torch.utils.checkpoint.checkpoint(self.backbone, inputs, use_reentrant = False)
        return self.backbone(inputs)

        '''return checkpoint(
            self.backbone,(inputs,),
            self.backbone.parameters(),
            checkpoint_
            ) # try with torch.utils checkpoint
        '''
    def upsample_module_call(self, in_list) :
        # https://github.com/huggingface/transformers/issues/23170
        '''def make_outputs_require_grad(module, input, output, val : bool):
            for tens in output :
                tens.requires_grad_(val)

        if all(t.requires_grad for t in in_list) :
            hook = partial(make_outputs_require_grad, val = True)
            checkpoint_ = True
        else :
            hook = partial(make_outputs_require_grad, val = False)
            checkpoint_ = False
        self.upsample_module.register_forward_hook(hook)'''
        
        checkpoint_ = all(t.requires_grad for t in in_list)
        if checkpoint_ :
            return torch.utils.checkpoint.checkpoint(self._upsample_module_call,
            (in_list, self.multi_counters, self.counter_copy), 
            use_reentrant = False)
        return self.upsample_module(in_list, self.multi_counters, self.counter_copy)
        '''return checkpoint(
            self.upsample_module,
            (in_list,self.multi_counters,self.counter_copy),
            self.upsample_module.parameters(),
            checkpoint_ 
            )'''
    def _upsample_module_call(self, inputs) :
        in_list, self.multi_counters, self.counter_copy = inputs
        return self.upsample_module(in_list, self.multi_counters, self.counter_copy)
            
    def forward(self,inputs, labels=None, mode='train', context = 'compute_grad'):
        #added the compute_grad as context manager

        #this is for an assert to track good behaviour in grad tracking during training :
        grad_track_is_on = False
        if inputs.requires_grad :
            grad_track_is_on = True 

        if self.config.counter_type == 'single_resolution':
            x_list = self.backbone(inputs)
            x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
            y = [x_list[0]]
            for i in range(1, len(x_list)):
                y.append(F.upsample(x_list[i], size=(x0_h, x0_w), mode='bilinear'))
            y = torch.cat(y, 1)

            outputs = self.count_head(y)

            # used for flops calculating and model testing
            if labels is None:
                return  outputs

            labels = labels[0].unsqueeze(1)
            labels =  self.gaussian(labels)

            if mode =='train' or mode =='val':
                loss = self.mse_loss(outputs, labels*self.weight)
                gt_cnt = labels.sum().item()
                pre_cnt = outputs.sum().item()/self.weight

                result = {
                    'x4': {'gt': gt_cnt, 'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))},
                    'x8': {'gt': 0, 'error': 0},
                    'x16': {'gt': 0, 'error': 0},
                    'x32': {'gt': 0, 'error': 0},
                    'acc1': {'gt': 0, 'error': 0},
                    'losses':loss,
                    'pre_den':
                        {
                            '1':outputs/self.weight,
                         },

                    'gt_den':{'1':labels}

                }
                return  result

            elif mode == 'test':
                return outputs / self.weight

        elif self.config.counter_type == 'withMOE':
            result = {'pre_den':{},'gt_den':{}}
            in_list = self.backbone_call(inputs)

            #this is for an assert to track good behaviour in grad tracking during training :
            if grad_track_is_on :
                assert all([t.requires_grad for t in in_list]), f'{grad_track_is_on=}, {inputs.requires_grad=} ... are we training ?'
                #print(f'{[t.grad_fn for t in in_list]=}')

            self.counter_copy.load_state_dict(self.multi_counters.state_dict())
            freeze_model(self.counter_copy)

            in_list = in_list[self.resolution_num[0]:self.resolution_num[-1]+1]
            out_list = self.upsample_module_call(in_list)
            
            #this is for an assert to track good behaviour in grad tracking during training :
            if grad_track_is_on :
                assert all([t.requires_grad for t in out_list]), f'{grad_track_is_on=}, {inputs.requires_grad=} ... are we training ?'
                #print(f'{[t.grad_fn for t in out_list]=}')
            # import pdb
            # pdb.set_trace()
            
            out_list = out_list[0]
            
            if labels is None:
                out = out_list/self.weight  #only output we are interested in 
                return  out
            
            ############################### RELEVANT ONLY UP UNTIL HERE ####################################################################
            label_list = []

            labels = labels[self.label_start:self.label_end]

            for i, label in enumerate(labels):
                label_list.append(self.gaussian(label.unsqueeze(1))*self.weight)

            # moe_label,score_gt = self.get_moe_label(out_list, label_list, (64,64))

            # import numpy as np
            # import cv2
            # import pdb
            # pred_color_map= moe_label.cpu().numpy()
            # np.save('./exp/moe/{}.npy'.format(moe_label.size(2)),pred_color_map)
            # pred_color_map = cv2.applyColorMap(
            #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            # cv2.imwrite('./exp/moe/moe_label_{}.png'.format(moe_label.size(2)), pred_color_map)
            # pdb.set_trace()

            if mode =='val':
                print(f'{out_list[0].requires_grad=}, {type(self.weight)=}')
                result.update({'losses':self.mse_loss(out_list[0],label_list[0])})
                result['pre_den'].update({'1': out_list[0]/self.weight})
                result['pre_den'].update({'2': out_list[-3]/self.weight})
                result['pre_den'].update({'4': out_list[-2]/self.weight})
                result['pre_den'].update({'8': out_list[-1]/self.weight})

                result['gt_den'].update({'1': label_list[0]/self.weight})
                result['gt_den'].update({'2': label_list[-3]/self.weight})
                result['gt_den'].update({'4': label_list[-2]/self.weight})
                result['gt_den'].update({'8': label_list[-1]/self.weight})
                
                return result

            moe_label,score_gt = self.get_moe_label(out_list, label_list, self.route_size)
            mask_gt = torch.zeros_like(score_gt)
            if mode =='train' or mode =='val':
                mask_gt = mask_gt.scatter_(1,moe_label, 1)

            loss_list = []

            outputs = torch.zeros_like(out_list[0])
            label_patch = torch.zeros_like(label_list[0])

            result.update({'acc1': {'gt':0, 'error':0}})

            # import pdb
            # pdb.set_trace()
            mask_add = torch.ones_like(mask_gt[:,0].unsqueeze(1))
            for i in range(mask_gt.size(1)):

                kernel = (int(self.route_size[0] / (2 ** i)), int(self.route_size[1] / (2 ** i)))
                loss_mask = F.upsample_nearest(mask_add,   size=(out_list[i].size()[2:]))
                hard_loss=self.mse_loss(out_list[i]*loss_mask,label_list[i]*loss_mask)
                loss_list.append(hard_loss)

                # import pdb
                # pdb.set_trace()

                if i == 0:
                    label_patch += (label_list[0] * F.upsample_nearest(mask_gt[:,i].unsqueeze(1),
                                                           size=(out_list[i].size()[2:])))
                    label_patch = F.unfold(label_patch, kernel,  stride=kernel)
                    B_, _, L_ = label_patch.size()
                    label_patch = label_patch.transpose(2, 1).view(B_, L_, kernel[0], kernel[1])
                else:
                    gt_slice  = F.unfold(label_list[i], kernel,stride=kernel)
                    B, KK, L = gt_slice.size()

                    pick_gt_idx = (moe_label.flatten(start_dim=1) == i).unsqueeze(2).unsqueeze(3)
                    gt_slice = gt_slice.transpose(2,1).view(B, L,kernel[0], kernel[1])
                    pad_w, pad_h =(self.route_size[1] - kernel[1])//2, (self.route_size[0] - kernel[0])//2
                    gt_slice = F.pad(gt_slice, [pad_w,pad_w,pad_h,pad_h], "constant", 0.2)
                    gt_slice = (gt_slice * pick_gt_idx)
                    label_patch += gt_slice

                gt_cnt = (label_list[i]*loss_mask).sum().item()/self.weight
                pre_cnt = (out_list[i]*loss_mask).sum().item()/self.weight
                result.update({f"x{2**(self.resolution_num[i]+2)}": {'gt': gt_cnt,
                                            'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))}})
                mask_add -=mask_gt[:,i].unsqueeze(1)

            B_num, C_num, H_num, W_num =  out_list[0].size()
            patch_h, patch_w = H_num // self.route_size[0], W_num // self.route_size[1]
            label_patch =label_patch.view(B_num, patch_h*patch_w, -1).transpose(1,2)
            label_patch = F.fold(label_patch, output_size=(H_num, W_num), kernel_size=self.route_size, stride=self.route_size)

            if mode =='train' or mode =='val':
                loss = 0
                if self.config.baseline_loss:
                    loss = loss_list[0]
                else:
                    for i in range(len(self.resolution_num)):
                        # if self.config.loss_weight:
                        loss +=loss_list[i]*self.config.loss_weight[i]
                        # else:
                        #     loss += loss_list[i] /(2**(i))

                for i in ['x4','x8','x16', 'x32']:
                    if i not in result.keys():
                        result.update({i: {'gt': 0, 'error': 0}})
                result.update({'moe_label':moe_label})
                result.update({'losses':torch.unsqueeze(loss,0)})
                result['pre_den'].update({'1':out_list[0]/self.weight})
                result['pre_den'].update({'8': out_list[-1]/self.weight})
                result['gt_den'].update({'1': label_patch/self.weight})
                result['gt_den'].update({'8': label_list[-1]/self.weight})

                return result

            elif mode == 'test':

                return outputs / self.weight


    def get_moe_label(self, out_list, label_list, route_size):
        """
        :param out_list: (N,resolution_num,H, W) tensor
        :param in_list:  (N,resolution_num,H, W) tensor
        :param route_size: 256
        :return:
        """
        B_num, C_num, H_num, W_num =  out_list[0].size()
        patch_h, patch_w = H_num // route_size[0], W_num // route_size[1]
        errorInslice_list = []

        for i, (pre, gt) in enumerate(zip(out_list, label_list)):
            #pre, gt= pre.detach(), gt.detach()                             ###############MODIFIED
            kernel = (int(route_size[0]/(2**i)), int(route_size[1]/(2**i)))

            weight = torch.full(kernel,1/(kernel[0]*kernel[1])).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            error= (pre - gt)**2
            patch_mse=F.conv2d(error, weight,stride=kernel)

            weight = torch.full(kernel,1.).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            # mask = (gt>0).float()
            # mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            patch_error=F.conv2d(error, weight,stride=kernel)  #(pre-gt)*(gt>0)
            fractions = F.conv2d(gt, weight, stride=kernel)

            instance_mse = patch_error/(fractions+1e-10)

            errorInslice_list.append(patch_mse + instance_mse)


        score = torch.cat(errorInslice_list, dim=1)
        moe_label = score.argmin(dim=1, keepdim=True)
        print(f'{score.requires_grad=}\n'
            f'{moe_label.shape=}\n'
            f'{score.shape=}')


        # mask = mask.view(mask.size(0),mask.size(1),patch_h, patch_w).float()
        # import pdb
        # pdb.set_trace()

        return  moe_label, score



class Baseline_Classifier(nn.Module):
    def __init__(self, config=None):
        super(Baseline_Classifier, self).__init__()
        self.config = config
        self.backbone = BackboneSelector(self.config).get_backbone()

        # self.cls_head0 = HeadSelector(self.config.head0).get_head()
        # self.cls_head1 = HeadSelector(self.config.head1).get_head()
        # self.cls_head2 = HeadSelector(self.config.head2).get_head()
        # self.cls_head3 = HeadSelector(self.config.head3).get_head()
        # self.wrap_clshead = nn.ModuleList([HeadSelector(self.config.head0).get_head(),
        #                                  HeadSelector(self.config.head1).get_head(),
        #                                  HeadSelector(self.config.head2).get_head(),
        #                                  HeadSelector(self.config.head3).get_head()])

        self.wrap_clshead = HeadSelector(self.config.head0).get_head()
        self.counter = 0

    def forward(self,x, batch_idx=None):
        # if batch_idx is not None:
        #     seed = batch_idx % 4
        #     x_list= self.backbone(x,seed)
        #     x = self.wrap_clshead[seed](x_list[seed:])
        # import pdb
        # pdb.set_trace()
        #     return x
        # else:
        x_list= self.backbone(x)
        return self.wrap_clshead(x_list)

        # y = self.wrap_clshead[0](x_list[0:])
        # for i in range(1,4,1):
        #     x_list= self.backbone(x,i)
        #     y = y+self.wrap_clshead[i](x_list[i:])
        # return  y/4

if __name__ == "__main__":
    from mmcv import  Config
    cfg_data = Config.fromfile(
        '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/configs/NWPU/hrformer_b.py')

    print(cfg_data)
    # import pdb
    # pdb.set_trace()
    model = Baseline_Counter(cfg_data.network)
    print(model)

##

