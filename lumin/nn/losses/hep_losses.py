import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable

__all__ = ['SignificanceLoss']


class SignificanceLoss(nn.Module):
    r'''
    General class for implementing significance-based loss functions, e.g. Asimov Loss (https://arxiv.org/abs/1806.00322).
    For compatability with using basic PyTorch losses, event weights are passed during initialisation rather than when computing the loss.

    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss
        sig_wgt: total weight of signal events
        bkg_wgt: total weight of background events
        func: callable which returns a float based on signal and background weights

    Examples::
        >>> loss = SignificanceLoss(weight, sig_weight=sig_weight,
        ...                         bkg_weight=bkg_weight, func=calc_ams_torch)
        >>>
        >>> loss = SignificanceLoss(weight, sig_weight=sig_weight,
        ...                         bkg_weight=bkg_weight,
        ...                         func=partial(calc_ams_torch, br=10))        
    '''

    def __init__(self, weight:Tensor, sig_wgt=float, bkg_wgt=float, func=Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        super().__init__()
        self.weight,self.sig_wgt,self.bkg_wgt,self.func = weight.squeeze(),sig_wgt,bkg_wgt,func
    
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''

        input, target = input.squeeze(), target.squeeze()
        # Reweight accordign to batch size
        sig_wgt = (target*self.weight)*self.sig_wgt/torch.dot(target, self.weight)
        bkg_wgt = ((1-target)*self.weight)*self.bkg_wgt/torch.dot(1-target, self.weight)
        # Compute Signal and background weights without a hard cut
        s = torch.dot(sig_wgt*input, target)
        b = torch.dot(bkg_wgt*input, (1-target))
        return 1/self.func(s, b)  # Return inverse of significance (would negative work better?)
