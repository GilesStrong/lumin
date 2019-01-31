from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch


@weak_module
class WeightedMSE(nn.MSELoss):
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weightss = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None:
            return torch.mean(self.weights*super().forward(input, target))
        else:
            return super().forward(input, target)


@weak_module
class WeightedMAE(nn.L1Loss):
    def __init__(self, weight=None):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None:
            return torch.mean(self.weights*super().forward(input, target))
        else:
            return super().forward(input, target)


@weak_module
class WeightedCCE(nn.NLLLoss):
    def __init__(self, weight=None):
        super().__init__(reduction='mean')
        self.weights = weight
        
    @weak_script_method
    def forward(self, input, target):
        if self.weights is not None:
            return torch.mean(self.weights*super().forward(input, target))
        else:
            return super().forward(input, target)
