from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch


@weak_module
class WeightedMSE(nn.MSELoss):
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weight = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weight is not None:
            return torch.mean(self.weight*super().forward(input, target))
        else:
            return super().forward(input, target)


@weak_module
class WeightedMAE(nn.L1Loss):
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weight = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weight is not None:
            return torch.mean(self.weight*super().forward(input, target))
        else:
            return super().forward(input, target)


