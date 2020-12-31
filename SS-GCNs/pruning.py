import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random

def soft_threshold(w, th):
	'''
	pytorch soft-sign function
	'''
	with torch.no_grad():
		temp = torch.abs(w) - th
		# print('th:', th)
		# print('temp:', temp.size())
		return torch.sign(w) * nn.functional.relu(temp)

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

        setattr(module, self._tensor_name, self.apply_weight(module))

    def apply_mask(self, module):

        mask_weight = getattr(module, self._tensor_name + "_mask_weight")
        orig_weight = getattr(module, self._tensor_name)
        pruned_weight = mask_weight * orig_weight
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_weight", mask.to(dtype=orig.dtype))

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model, init_mask_dict=None):

    if init_mask_dict is None:
        import pdb; pdb.set_trace()
        mask1 = nn.Parameter(torch.ones_like(model.net_layer[0].weight))
        AddTrainableMask.apply(model.net_layer[0], 'weight', mask1)

        mask2 = nn.Parameter(torch.ones_like(model.net_layer[1].weight))
        AddTrainableMask.apply(model.net_layer[1], 'weight', mask2)

    else:
        pass
