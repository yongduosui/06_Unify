import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import matplotlib.pyplot as plt
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

        mask_weight = getattr(module, self._tensor_name + "_mask_weight")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_weight * orig_weight
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_weight", mask.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model, init_mask_dict=None):

    if init_mask_dict is None:
        
        mask1 = nn.Parameter(torch.ones_like(model.net_layer[0].weight))
        AddTrainableMask.apply(model.net_layer[0], 'weight', mask1)

        mask2 = nn.Parameter(torch.ones_like(model.net_layer[1].weight))
        AddTrainableMask.apply(model.net_layer[1], 'weight', mask2)

    else:

        mask1 = nn.Parameter(init_mask_dict['mask1'])
        AddTrainableMask.apply(model.net_layer[0], 'weight', mask1)

        mask2 = nn.Parameter(init_mask_dict['mask2'])
        AddTrainableMask.apply(model.net_layer[1], 'weight', mask2)
        


def generate_mask(model):

    mask_dict = {}
    mask_dict['mask1'] = torch.zeros_like(model.net_layer[0].weight)
    mask_dict['mask2'] = torch.zeros_like(model.net_layer[1].weight)

    return mask_dict


def subgradient_update_mask(model, args):

    model.adj_mask.grad.data.add_(args['s1'] * torch.sign(model.adj_mask.data))
    model.net_layer[0].weight_mask_weight.grad.data.add_(args['s2'] * torch.sign(model.net_layer[0].weight_mask_weight.data))
    model.net_layer[1].weight_mask_weight.grad.data.add_(args['s2'] * torch.sign(model.net_layer[1].weight_mask_weight.data))



def get_mask_distribution(model):

    adj_mask_tensor = model.adj_mask.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero] # 13264

    weight_mask_tensor = model.net_layer[0].weight_mask_weight.flatten()    # 22928
    weight_mask_tensor = torch.cat((weight_mask_tensor, model.net_layer[1].weight_mask_weight.flatten())) # 112

    np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())

    return adj_mask_tensor, weight_mask_tensor
    

def pruning(model, percent):

    
    adj_total = model.adj_mask.numel()

    weight_total = 0
    weight_total += model.net_layer[0].weight_mask_weight.numel()
    weight_total += model.net_layer[1].weight_mask_weight.numel()

    adj_mask_weight = model.adj_mask.data.flatten().abs().clone()

    weight_mask_weight = model.net_layer[0].weight_mask_weight.flatten().abs().clone()    # 22928
    weight_mask_weight = torch.cat((weight_mask_weight, model.net_layer[1].weight_mask_weight.flatten().abs().clone())) # 112

    adj_y, adj_i = torch.sort(adj_mask_weight)
    wei_y, wei_i = torch.sort(weight_mask_weight)

    adj_thre_index = int(adj_total * percent)
    adj_thre = adj_y[adj_thre_index]
    print("adj pruning the:{}".format(adj_thre))
    wei_thre_index = int(weight_total * percent)
    wei_thre = wei_y[wei_thre_index]
    print("weight pruning the:{}".format(wei_thre))
    
    adj_mask = torch.zeros(adj_total)






    bn = torch.zeros(total)
    index = 0


    

