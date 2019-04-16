from torch._jit_internal import weak_module, weak_script_method
import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable


@weak_module
class SignificanceLoss(nn.Module):
    '''Significance-based loss function a la 1806.00322'''
    def __init__(self, weight:Tensor, sig_wgt=float, bkg_wgt=float, func=Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        super().__init__()
        self.weight,self.sig_wgt,self.bkg_wgt,self.func = weight.squeeze(),sig_wgt,bkg_wgt,func
    
    @weak_script_method
    def forward(self, input, target):
        input, target = input.squeeze(), target.squeeze()
        sig_wgt = (target*self.weight)*self.sig_wgt/torch.dot(target.squeeze(), self.weight)
        bkg_wgt = ((1-target)*self.weight)*self.bkg_wgt/torch.dot(1-target.squeeze(), self.weight)
        s = torch.dot(sig_wgt*input, target)
        b = torch.dot(bkg_wgt*input, (1-target))
        return 1/self.func(s, b)
