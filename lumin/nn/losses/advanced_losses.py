from typing import Optional

import torch
from torch import nn, Tensor

from ...utils.misc import to_device

__all__ = ['WeightedFractionalMSE', 'WeightedBinnedHuber', 'WeightedFractionalBinnedHuber']


class WeightedFractionalMSE(nn.MSELoss):
    r'''
    Class for computing the Mean fractional Squared-Error loss (<Delta^2/true>) with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedFractionalMSE()
        >>>
        >>> loss = WeightedFractionalMSE(weights)
    '''

    def __init__(self, weight:Optional[Tensor]=None):
        super().__init__(reduction='none')
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

        if self.weights is not None: return torch.mean(self.weights*super().forward(input, target)/target)
        else:                        return torch.mean(super().forward(input, target)/target)


class WeightedBinnedHuber(nn.MSELoss):
    r'''
    Class for computing the Huberised Mean Squared-Error loss (<Delta^2>) with optional weights per prediction.
    Losses soft-clamped with Huber like term above adaptive percentile in bins of the target.
    The thresholds used to transition from MSE to MAE per bin are initialised using the first batch of data as the value of the specified percentile in each bin,
    subsequently, the thresholds evolve according to: T <- (1-mom)*T + mom*T_batch, where T_batch are the percentiles comuted on the current batch, and mom(emtum)
    lies between [0,1]

    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        perc: quantile of data in each bin above which to use MAE rather than MSE
        bins: tensor of edges for the binning of the target data
        mom: momentum for the running average of the thresholds
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedBinnedHuber(perc=0.68)
        >>>
        >>> loss = WeightedBinnedHuber(perc=0.68, weights=weights)
    '''

    def __init__(self, perc:float, bins:Tensor, mom=0.1, weight:Optional[Tensor]=None):
        super().__init__(reduction='none')
        self.perc,self.bins,self.weights,self.mom = perc,bins,weight,mom
        self.kths = to_device(torch.zeros_like(self.bins[:-1])-1)

    def _compute_losses(self, input:Tensor, target:Tensor) -> Tensor:
        loss = super().forward(input, target)  # MSE
        # MAE
        for i in range(len(self.bins)-1):
            m = (target >= self.bins[i])*(target < self.bins[i+1])
            if m.sum() == 0: continue
            kth = loss[m].view(-1).kthvalue(1+round(self.perc*(loss[m].numel()-1))).values.detach()
            if self.kths[i] < 0: self.kths[i] = kth
            else:                self.kths[i].lerp_(kth, self.mom)
            m = m*(loss > self.kths[i])
            d = torch.sqrt(self.kths[i])
            loss[m] = self.kths[i]+(2*d*((torch.abs(input[m]-target[m]))-d))
        if self.weights is not None: loss = loss*self.weights
        return loss
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''
        
        loss = self._compute_losses(input, target)
        return torch.mean(loss)


class WeightedFractionalBinnedHuber(WeightedBinnedHuber):
    r'''
    Class for computing the Huberised Mean fractional Squared-Error loss (<Delta^2/true>) with optional weights per prediction.
    Losses soft-clamped with Huber like term above adaptive percentile in bins of the target.
    The thresholds used to transition from MSE to MAE per bin are initialised using the first batch of data as the value of the specified percentile in each bin,
    subsequently, the thresholds evolve according to: T <- (1-mom)*T + mom*T_batch, where T_batch are the percentiles comuted on the current batch, and mom(emtum)
    lies between [0,1]

    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.
    
    Arguments:
        perc: quantile of data in each bin above which to use MAE rather than MSE
        bins: tensor of edges for the binning of the target data
        mom: momentum for the running average of the thresholds
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss
    '''
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        r'''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor
        
        Returns:
            (weighted) loss
        '''
        
        loss = self._compute_losses(input, target)
        loss = loss/target
        return torch.mean(loss)
