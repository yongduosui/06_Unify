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


def save_all(model, rewind_weight, optimizer, imp_num, epoch, save_path, save_name='default'):
    
    state = {
            'imp_num': imp_num,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'rewind_weight_mask': rewind_weight,
            'optimizer_state_dict': optimizer.state_dict()
        }

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}.pth'.format(save_path, save_name)
    torch.save(state, filename)


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


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

    mask_train_ebd = nn.Parameter(torch.ones_like(model.embedding_h.weight), requires_grad=True)
    mask_fixed_ebd = nn.Parameter(torch.ones_like(model.embedding_h.weight), requires_grad=False)
    AddTrainableMask.apply(model.embedding_h, 'weight', mask_train_ebd, mask_fixed_ebd)

    for i in range(3):

        mask_train_gcn = nn.Parameter(torch.ones_like(model.layers[i].apply_mod.linear.weight), requires_grad=True)
        mask_fixed_gcn = nn.Parameter(torch.ones_like(model.layers[i].apply_mod.linear.weight), requires_grad=False)
        AddTrainableMask.apply(model.layers[i].apply_mod.linear, 'weight', mask_train_gcn, mask_fixed_gcn)

        mask_train_mlp = nn.Parameter(torch.ones_like(model.MLP_layer.FC_layers[i].weight), requires_grad=True)
        mask_fixed_mlp = nn.Parameter(torch.ones_like(model.MLP_layer.FC_layers[i].weight), requires_grad=False)
        AddTrainableMask.apply(model.MLP_layer.FC_layers[i], 'weight', mask_train_mlp, mask_fixed_mlp)


def add_trainable_mask_noise(model, c=1e-5):

    model.edge_mask1_train.requires_grad = False
    rand = (2 * torch.rand(model.edge_mask1_train.shape) - 1) * c
    rand = rand.to(model.edge_mask1_train.device) 
    rand = rand * model.edge_mask1_train
    model.edge_mask1_train.add_(rand)
    model.edge_mask1_train.requires_grad = True

    model.embedding_h.weight_mask_train.requires_grad = False
    rand = (2 * torch.rand(model.embedding_h.weight_mask_train.shape) - 1) * c
    rand = rand.to(model.embedding_h.weight_mask_train.device) 
    rand = rand * model.embedding_h.weight_mask_train
    model.embedding_h.weight_mask_train.add_(rand)
    model.embedding_h.weight_mask_train.requires_grad = True

    for i in range(3):

        model.layers[i].apply_mod.linear.weight_mask_train.requires_grad = False
        rand = (2 * torch.rand(model.layers[i].apply_mod.linear.weight_mask_train.shape) - 1) * c
        rand = rand.to(model.layers[i].apply_mod.linear.weight_mask_train.device) 
        rand = rand * model.layers[i].apply_mod.linear.weight_mask_train
        model.layers[i].apply_mod.linear.weight_mask_train.add_(rand)
        model.layers[i].apply_mod.linear.weight_mask_train.requires_grad = True

        model.MLP_layer.FC_layers[i].weight_mask_train.requires_grad = False
        rand = (2 * torch.rand(model.MLP_layer.FC_layers[i].weight_mask_train.shape) - 1) * c
        rand = rand.to(model.MLP_layer.FC_layers[i].weight_mask_train.device) 
        rand = rand * model.MLP_layer.FC_layers[i].weight_mask_train
        model.MLP_layer.FC_layers[i].weight_mask_train.add_(rand)
        model.MLP_layer.FC_layers[i].weight_mask_train.requires_grad = True


def subgradient_update_mask(model, args):

    model.edge_mask1_train.grad.data.add_(args.s1 * torch.sign(model.edge_mask1_train.data))
    for i in range(3):
        model.layers[i].apply_mod.linear.weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.layers[i].apply_mod.linear.weight_mask_train.data))
        model.MLP_layer.FC_layers[i].weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.MLP_layer.FC_layers[i].weight_mask_train.data))


def get_soft_mask_distribution(model):

    adj_mask_vector = model.edge_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_vector) > 0
    adj_mask_vector = adj_mask_vector[nonzero]

    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))

    weight_mask = model.embedding_h.weight_mask_train.flatten()
    nonzero = torch.abs(weight_mask) > 0
    weight_mask = weight_mask[nonzero]
    weight_mask_vector = torch.cat((weight_mask_vector, weight_mask))

    for i in range(3):
        weight_mask = model.layers[i].apply_mod.linear.weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask) > 0
        weight_mask = weight_mask[nonzero]
        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask))

        weight_mask = model.MLP_layer.FC_layers[i].weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask) > 0
        weight_mask = weight_mask[nonzero]
        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask))

    return adj_mask_vector.detach().cpu(), weight_mask_vector.detach().cpu()


def plot_mask_distribution(model, epoch, acc_test, path):
    # print("Plot Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    if not os.path.exists(path): os.makedirs(path)
    adj_mask, weight_mask = get_soft_mask_distribution(model)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    plt.hist(adj_mask.numpy())
    plt.title("adj mask")
    plt.xlabel('mask value')
    plt.ylabel('times')

    plt.subplot(1,2,2)
    plt.hist(weight_mask.numpy())
    plt.title("weight mask")
    plt.xlabel('mask value')
    plt.ylabel('times')
    plt.suptitle("Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    plt.savefig(path + '/mask_epoch{}.png'.format(epoch))


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, rewind_weight, adj_percent, wei_percent):

    adj_mask, wei_mask = get_soft_mask_distribution(model)
    #adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)
    adj_total = adj_mask.shape[0] # 2358104
    wei_total = wei_mask.shape[0] # 39627
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]

    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]
    ### create mask dict 
    
    ori_edge_mask = model.edge_mask1_train.detach().cpu()
    rewind_weight['edge_mask1_train'] = get_each_mask(ori_edge_mask, adj_thre)
    rewind_weight['edge_mask2_fixed'] = rewind_weight['edge_mask1_train']

    ori_ebd_mask = model.embedding_h.weight_mask_train.detach().cpu()
    rewind_weight['embedding_h.weight_mask_train'] = get_each_mask(ori_ebd_mask, wei_thre)
    rewind_weight['embedding_h.weight_mask_fixed'] = rewind_weight['embedding_h.weight_mask_train']

    
    for i in range(3):

        key_train = 'layers.{}.apply_mod.linear.weight_mask_train'.format(i)
        key_fixed = 'layers.{}.apply_mod.linear.weight_mask_fixed'.format(i)
        rewind_weight[key_train] = get_each_mask(model.layers[i].apply_mod.linear.state_dict()['weight_mask_train'], wei_thre)
        rewind_weight[key_fixed] = rewind_weight[key_train]

        key_train_mlp = 'MLP_layer.FC_layers.{}.weight_mask_train'.format(i)
        key_fixed_mlp = 'MLP_layer.FC_layers.{}.weight_mask_fixed'.format(i)
        rewind_weight[key_train_mlp] = get_each_mask(model.MLP_layer.FC_layers[i].state_dict()['weight_mask_train'], wei_thre)
        rewind_weight[key_fixed_mlp] = rewind_weight[key_train_mlp]

    return rewind_weight


def random_pruning(model, adj_percent, wei_percent):

    model.edge_mask1_train.requires_grad = False
    adj_total = model.edge_mask1_train.numel()
    adj_pruned_num = int(adj_total * adj_percent)
    adj_nonzero = model.edge_mask1_train.nonzero()
    adj_pruned_index = random.sample([i for i in range(adj_total)], adj_pruned_num)
    adj_pruned_list = adj_nonzero[adj_pruned_index].tolist()
    for i, j in adj_pruned_list:
        model.edge_mask1_train[i][j] = 0
        model.edge_mask2_fixed[i][j] = 0
    model.edge_mask1_train.requires_grad = True

    model.embedding_h.weight_mask_train.requires_grad = False
    wei_total = model.embedding_h.weight_mask_train.numel()
    wei_pruned_num = int(wei_total * wei_percent)
    wei_nonzero = model.embedding_h.weight_mask_train.nonzero()
    wei_pruned_index = random.sample([i for i in range(wei_total)], wei_pruned_num)
    wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()
    for i, j in wei_pruned_list:
        model.embedding_h.weight_mask_train[i][j] = 0
        model.embedding_h.weight_mask_fixed[i][j] = 0
    model.embedding_h.weight_mask_train.requires_grad = True

    for i in range(3):
        
        model.layers[i].apply_mod.linear.weight_mask_train.requires_grad = False
        wei_total = model.layers[i].apply_mod.linear.weight_mask_train.numel()
        wei_pruned_num = int(wei_total * wei_percent)
        wei_nonzero = model.layers[i].apply_mod.linear.weight_mask_train.nonzero()
        wei_pruned_index = random.sample([j for j in range(wei_total)], wei_pruned_num)
        wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()
        for ii, (ai, wj) in enumerate(wei_pruned_list):
            model.layers[i].apply_mod.linear.weight_mask_train[ai][wj] = 0
            model.layers[i].apply_mod.linear.weight_mask_fixed[ai][wj] = 0
        model.layers[i].apply_mod.linear.weight_mask_train.requires_grad = True

        model.MLP_layer.FC_layers[i].weight_mask_train.requires_grad = False
        wei_total = model.MLP_layer.FC_layers[i].weight_mask_train.numel()
        wei_pruned_num = int(wei_total * wei_percent)
        wei_nonzero = model.MLP_layer.FC_layers[i].weight_mask_train.nonzero()
        wei_pruned_index = random.sample([j for j in range(wei_total)], wei_pruned_num)
        wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()
        for ii, (ai, wj) in enumerate(wei_pruned_list):
            model.MLP_layer.FC_layers[i].weight_mask_train[ai][wj] = 0
            model.MLP_layer.FC_layers[i].weight_mask_fixed[ai][wj] = 0
        model.MLP_layer.FC_layers[i].weight_mask_train.requires_grad = True




def print_sparsity(model):

    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.edge_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight_total = 0
    weight_nonzero = 0

    weight_total += model.embedding_h.weight_mask_fixed.numel()
    weight_nonzero += model.embedding_h.weight_mask_fixed.sum().item()

    for i in range(3):

        weight_total += model.layers[i].apply_mod.linear.weight_mask_fixed.numel()
        weight_nonzero += model.layers[i].apply_mod.linear.weight_mask_fixed.sum().item()

        weight_total += model.MLP_layer.FC_layers[i].weight_mask_fixed.numel()
        weight_nonzero += model.MLP_layer.FC_layers[i].weight_mask_fixed.sum().item()
    
    wei_spar = weight_nonzero * 100 / weight_total

    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]".format(adj_spar, wei_spar))
    print("-" * 100)

    return adj_spar, wei_spar


