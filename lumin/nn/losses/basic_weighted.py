from typing import Optional

from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch
from torch.tensor import Tensor


@weak_module
class WeightedMSE(nn.MSELoss):
    r'''
    Class for computing Mean Squared-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMSE()
        >>> loss = WeightedMSE(weights)
    '''

    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input:Tensor, target:Tensor):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


@weak_module
class WeightedMAE(nn.L1Loss):
    r'''
    Class for computing Mean Absolute-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMAE()
        >>> loss = WeightedMAE(weights)
    '''
    
    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input:Tensor, target:Tensor):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


@weak_module
class WeightedCCE(nn.NLLLoss):
    r'''
    Class for computing Categorical Cross-Entropy loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedCCE()
        >>> loss = WeightedCCE(weights)
    '''

    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input:Tensor, target:Tensor):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)
