import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb
import torch.nn.init as init
import math
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


def add_mask(model):

    for i in range(28):
        mask_train = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight))
        mask_fixed = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight), requires_grad=False)
        AddTrainableMask.apply(model.gcns[i].mlp[0], 'weight', mask_train, mask_fixed)


def subgradient_update_mask(model, args):

    model.edge_mask1_train.grad.data.add_(args.s1 * torch.sign(model.edge_mask1_train.data))
    for i in range(28):
        model.gcns[i].mlp[0].weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.gcns[i].mlp[0].weight_mask_train.data))


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


##### random pruning #######
def random_pruning(model, adj_percent, wei_percent):

    model.adj_mask1_train.requires_grad = False
    model.net_layer[0].weight_mask_train.requires_grad = False
    model.net_layer[1].weight_mask_train.requires_grad = False

    adj_nonzero = model.adj_mask1_train.nonzero()
    wei1_nonzero = model.net_layer[0].weight_mask_train.nonzero()
    wei2_nonzero = model.net_layer[1].weight_mask_train.nonzero()

    adj_total = adj_nonzero.shape[0]
    wei1_total = wei1_nonzero.shape[0]
    wei2_total = wei2_nonzero.shape[0]

    adj_pruned_num = int(adj_total * adj_percent)
    wei1_pruned_num = int(wei1_total * wei_percent)
    wei2_pruned_num = int(wei2_total * wei_percent)

    adj_index = random.sample([i for i in range(adj_total)], adj_pruned_num)
    wei1_index = random.sample([i for i in range(wei1_total)], wei1_pruned_num)
    wei2_index = random.sample([i for i in range(wei2_total)], wei2_pruned_num)

    adj_pruned = adj_nonzero[adj_index].tolist()
    wei1_pruned = wei1_nonzero[wei1_index].tolist()
    wei2_pruned = wei2_nonzero[wei2_index].tolist()

    for i, j in adj_pruned:
        model.adj_mask1_train[i][j] = 0
    
    for i, j in wei1_pruned:
        model.net_layer[0].weight_mask_train[i][j] = 0
    
    for i, j in wei2_pruned:
        model.net_layer[1].weight_mask_train[i][j] = 0
    
    model.adj_mask2_fixed = model.adj_mask1_train
    model.net_layer[0].weight_mask_fixed = model.net_layer[0].weight_mask_train
    model.net_layer[1].weight_mask_fixed = model.net_layer[1].weight_mask_train

    model.adj_mask1_train.requires_grad = True
    model.net_layer[0].weight_mask_train.requires_grad = True
    model.net_layer[1].weight_mask_train.requires_grad = True

    
def print_sparsity(model):

    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight1_total = model.net_layer[0].weight_mask_fixed.numel()
    weight2_total = model.net_layer[1].weight_mask_fixed.numel()
    weight_total = weight1_total + weight2_total

    weight1_nonzero = model.net_layer[0].weight_mask_fixed.sum().item()
    weight2_nonzero = model.net_layer[1].weight_mask_fixed.sum().item()
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

    
def soft_mask_init(model, init_type, seed):

    setup_seed(seed)
    if init_type == 'all_one':
        add_trainable_mask_noise(model)
    elif init_type == 'kaiming':
        
        init.kaiming_uniform_(model.adj_mask1_train, a=math.sqrt(5))
        # init.constant_(model.adj_mask1_train, 1.0)
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.kaiming_uniform_(model.net_layer[0].weight_mask_train, a=math.sqrt(5))

        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True

        init.kaiming_uniform_(model.net_layer[1].weight_mask_train, a=math.sqrt(5))

        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True


    elif init_type == 'normal':
        mean = 1.0
        std = 0.1
        init.normal_(model.adj_mask1_train, mean=mean, std=std)
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.normal_(model.net_layer[0].weight_mask_train, mean=mean, std=std)

        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True

        init.normal_(model.net_layer[1].weight_mask_train, mean=mean, std=std)

        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True

    elif init_type == 'uniform':
        a = 0.8
        b = 1.2
        init.uniform_(model.adj_mask1_train, a=a, b=b)
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.uniform_(model.net_layer[0].weight_mask_train, a=a, b=b)

        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True

        init.uniform_(model.net_layer[1].weight_mask_train, a=a, b=b)

        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True

    else:
        assert False

    

