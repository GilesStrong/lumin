from typing import Optional

import torch.nn as nn
import torch
from torch.tensor import Tensor

__all__ = ['WeightedMSE', 'WeightedMAE', 'WeightedCCE']


class WeightedMSE(nn.MSELoss):
    r'''
    Class for computing Mean Squared-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMSE()
        >>>
        >>> loss = WeightedMSE(weights)
    '''

    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''

        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


class WeightedMAE(nn.L1Loss):
    r'''
    Class for computing Mean Absolute-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMAE()
        >>>
        >>> loss = WeightedMAE(weights)
    '''
    
    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''

        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


class WeightedCCE(nn.NLLLoss):
    r'''
    Class for computing Categorical Cross-Entropy loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedCCE()
        >>>
        >>> loss = WeightedCCE(weights)
    '''

    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='mean')
        self.weights = weight
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''

        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)
