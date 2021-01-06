import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb

# def soft_threshold(w, th):
# 	'''
# 	pytorch soft-sign function
# 	'''
# 	with torch.no_grad():
# 		temp = torch.abs(w) - th
# 		# print('th:', th)
# 		# print('temp:', temp.size())
# 		return torch.sign(w) * nn.functional.relu(temp)

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class AddTrainableMask(ABC):

    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):

        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model, init_mask_dict=None):

    if init_mask_dict is None:
        
        mask1_train = nn.Parameter(torch.ones_like(model.net_layer[0].weight))
        mask1_fixed = nn.Parameter(torch.ones_like(model.net_layer[0].weight), requires_grad=False)
        mask2_train = nn.Parameter(torch.ones_like(model.net_layer[1].weight))
        mask2_fixed = nn.Parameter(torch.ones_like(model.net_layer[1].weight), requires_grad=False)
        
    else:
        mask1_train = nn.Parameter(init_mask_dict['mask1_train'])
        mask1_fixed = nn.Parameter(init_mask_dict['mask1_fixed'], requires_grad=False)
        mask2_train = nn.Parameter(init_mask_dict['mask2_train'])
        mask2_fixed = nn.Parameter(init_mask_dict['mask2_fixed'], requires_grad=False)

    AddTrainableMask.apply(model.net_layer[0], 'weight', mask1_train, mask1_fixed)
    AddTrainableMask.apply(model.net_layer[1], 'weight', mask2_train, mask2_fixed)
 
        
def generate_mask(model):

    mask_dict = {}
    mask_dict['mask1'] = torch.zeros_like(model.net_layer[0].weight)
    mask_dict['mask2'] = torch.zeros_like(model.net_layer[1].weight)

    return mask_dict


def subgradient_update_mask(model, args):

    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    model.net_layer[0].weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.net_layer[0].weight_mask_train.data))
    model.net_layer[1].weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.net_layer[1].weight_mask_train.data))


def get_mask_distribution(model, if_numpy=True):

    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero] # 13264

    weight_mask_tensor0 = model.net_layer[0].weight_mask_train.flatten()    # 22928
    nonzero = torch.abs(weight_mask_tensor0) > 0
    weight_mask_tensor0 = weight_mask_tensor0[nonzero]

    weight_mask_tensor1 = model.net_layer[1].weight_mask_train.flatten()    # 22928
    nonzero = torch.abs(weight_mask_tensor1) > 0
    weight_mask_tensor1 = weight_mask_tensor1[nonzero]

    weight_mask_tensor = torch.cat([weight_mask_tensor0, weight_mask_tensor1]) # 112
    # np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())
    if if_numpy:
        return adj_mask_tensor.detach().cpu().numpy(), weight_mask_tensor.detach().cpu().numpy()
    else:
        return adj_mask_tensor.detach().cpu(), weight_mask_tensor.detach().cpu()
    

def plot_mask_distribution(model, epoch, acc_test, path):

    print("Plot Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    if not os.path.exists(path): os.makedirs(path)
    adj_mask, weight_mask = get_mask_distribution(model)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    plt.hist(adj_mask)
    plt.title("adj mask")
    plt.xlabel('mask value')
    plt.ylabel('times')

    plt.subplot(1,2,2)
    plt.hist(weight_mask)
    plt.title("weight mask")
    plt.xlabel('mask value')
    plt.ylabel('times')
    plt.suptitle("Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    plt.savefig(path + '/mask_epoch{}.png'.format(epoch))


def get_final_mask(model, percent):

    print("-" * 100)
    print("Begin pruning percent: [{:.2f} %]".format(percent * 100))

    adj_mask, wei_mask = get_mask_distribution(model, if_numpy=False)
    adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)

    adj_total = adj_mask.shape[0]
    wei_total = wei_mask.shape[0]
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * percent)
    adj_thre = adj_y[adj_thre_index]
    print("Adj    pruning threshold: [{:.6f}]".format(adj_thre))
    wei_thre_index = int(wei_total * percent)
    wei_thre = wei_y[wei_thre_index]
    print("Weight pruning threshold: [{:.6f}]".format(wei_thre))
    
    mask_dict = {}
    ori_adj_mask = model.adj_mask1_train.detach().cpu()
    ori_adj_mask.add_((2 * torch.rand(ori_adj_mask.shape) - 1) * 1e-5)

    mask_dict['adj_mask'] = get_each_mask(ori_adj_mask, adj_thre)
    mask_dict['weight1_mask'] = get_each_mask(model.net_layer[0].state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(model.net_layer[1].state_dict()['weight_mask_train'], wei_thre)

    print("Finish pruning, Sparsity: Adj:[{:.4f} %], Weight:[{:.4f} %]"
        .format((mask_dict['adj_mask'].sum() / adj_total) * 100, 
         ((mask_dict['weight1_mask'].sum() + mask_dict['weight2_mask'].sum()) / wei_total) * 100))
    print("-" * 100)
    return mask_dict


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, adj_percent, wei_percent):

    adj_mask, wei_mask = get_mask_distribution(model, if_numpy=False)
    #adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)
    adj_total = adj_mask.shape[0]
    wei_total = wei_mask.shape[0]
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]
    
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    mask_dict = {}
    ori_adj_mask = model.adj_mask1_train.detach().cpu()
    # ori_adj_mask.add_((2 * torch.rand(ori_adj_mask.shape) - 1) * 1e-5)
    mask_dict['adj_mask'] = get_each_mask(ori_adj_mask, adj_thre)
    mask_dict['weight1_mask'] = get_each_mask(model.net_layer[0].state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(model.net_layer[1].state_dict()['weight_mask_train'], wei_thre)

    return mask_dict


def print_sparsity(model):

    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight1_total = model.net_layer[0].weight_mask_train.numel()
    weight2_total = model.net_layer[1].weight_mask_train.numel()
    weight_total = weight1_total + weight2_total

    weight1_nonzero = model.net_layer[0].weight_mask_train.sum().item()
    weight2_nonzero = model.net_layer[1].weight_mask_train.sum().item()
    weight_nonzero = weight1_nonzero + weight2_nonzero

    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
    .format(adj_spar, wei_spar))
    print("-" * 100)

    return adj_spar, wei_spar


def add_trainable_mask_noise(model):
    
    model.adj_mask1_train.requires_grad = False
    model.net_layer[0].weight_mask_train.requires_grad = False
    model.net_layer[1].weight_mask_train.requires_grad = False
    c = 1e-5
    rand1 = (2 * torch.rand(model.adj_mask1_train.shape) - 1) * c
    rand1 = rand1.to(model.adj_mask1_train.device) 
    rand1 = rand1 * model.adj_mask1_train
    model.adj_mask1_train.add_(rand1)

    rand2 = (2 * torch.rand(model.net_layer[0].weight_mask_train.shape) - 1) * c
    rand2 = rand2.to(model.net_layer[0].weight_mask_train.device)
    rand2 = rand2 * model.net_layer[0].weight_mask_train
    model.net_layer[0].weight_mask_train.add_(rand2)

    rand3 = (2 * torch.rand(model.net_layer[1].weight_mask_train.shape) - 1) * c
    rand3 = rand3.to(model.net_layer[1].weight_mask_train.device)
    rand3 = rand3 * model.net_layer[1].weight_mask_train
    model.net_layer[1].weight_mask_train.add_(rand3)

    model.adj_mask1_train.requires_grad = True
    model.net_layer[0].weight_mask_train.requires_grad = True
    model.net_layer[1].weight_mask_train.requires_grad = True

    

