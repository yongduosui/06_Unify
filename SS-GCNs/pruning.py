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

    mask_tensor = model.adj_mask.flatten()
    nonzero = torch.abs(mask_tensor) > 0
    mask_tensor = mask_tensor[nonzero]
    
    mask_tensor = torch.cat((mask_tensor, model.net_layer[0].weight_mask_weight), 1)
    mask_tensor = torch.cat((mask_tensor, model.net_layer[1].weight_mask_weight), 1)
    
    plt.hist(mask_tensor, bins=1000)
    plt.xlabel('mask')
    plt.ylabel('value')
    plt.savefig('./mask_distribution.png')
    return mask_tensor
    

# def pruning(model, percent):

#     total = 0
#     total += model.adj_mask.numel()
#     total += model.net_layer[0].weight_mask_weight.numel()
#     total += model.net_layer[1].weight_mask_weight.numel()

#     bn = torch.zeros(total)

#     index = 0

    

