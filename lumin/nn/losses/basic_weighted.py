from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch


@weak_module
class WeightedMSE(nn.MSELoss):
    '''Loss class for weighted mean squared error'''
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


@weak_module
class WeightedMAE(nn.L1Loss):
    '''Loss class for weighted mean absolute error'''
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)


@weak_module
class WeightedCCE(nn.NLLLoss):
    '''Loss class for negative log likelihood loss with more flexible weightings'''
    def __init__(self, weight=None):
        super().__init__(reduction='mean')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target))
        else:                        return super().forward(input, target)
