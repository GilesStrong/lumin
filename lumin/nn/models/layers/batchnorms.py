from fastcore.all import store_attr
import math
import torch
from torch import nn, Tensor, tensor

__all__ = ['LCBatchNorm1d', 'RunningBatchNorm1d', 'RunningBatchNorm2d', 'RunningBatchNorm3d']


class LCBatchNorm1d(nn.Module):
    r'''
    Wrapper class for 1D batchnorm to make it run over (Batch x length x channel) data for use in NNs designed to be broadcast across matrix data.

    Arguments:
        bn: base 1D batchnorm module to call
    '''

    def __init__(self, bn:nn.BatchNorm1d):
        super().__init__()
        self.bn = bn
    
    def forward(self, x:Tensor) -> Tensor: return self.bn(x.transpose(-1,-2)).transpose(-1,-2)


class RunningBatchNorm1d(nn.Module):
    r'''
    1D Running batchnorm implementation from fastai (https://github.com/fastai/course-v3) distributed under apache2 licence.
    Modifcations: Adaptation to 1D & 3D, add eps in mom1 calculation, type hinting, docs

    Arguments:
        nf: number of features/channels
        mom: momentum (fraction to add to running averages)
        n_warmup: number of warmup iterations (during which variance is clamped)
        eps: epsilon to prevent division by zero
    '''

    def __init__(self, nf:int, mom:float=0.1, n_warmup:int=20, eps:float=1e-5):
        super().__init__()
        store_attr()
        self._set_params()

    def _set_params(self) -> None:
        self.weight = nn.Parameter(torch.ones(self.nf,1))
        self.bias = nn.Parameter(torch.zeros(self.nf,1))
        self.register_buffer('sums', torch.zeros(1,self.nf,1))
        self.register_buffer('sqrs', torch.zeros(1,self.nf,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.dims = (0,2)

    def update_stats(self, x:Tensor) -> None:
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        s = x.sum(self.dims, keepdim=True)
        ss = (x*x).sum(self.dims, keepdim=True)
        c = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1+self.eps))
        self.sums.lerp_(s, mom1)
        self.sqrs.lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs

    def forward(self, x:Tensor) -> Tensor:
        squeeze = False
        if len(x.shape) == 2:
            squeeze = True
            x = x.unsqueeze(-1)
        if self.training: self.update_stats(x)
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < self.n_warmup): varns.clamp_min_(0.01)
        factor = self.weight/(varns+self.eps).sqrt()
        offset = self.bias-means*factor
        x = x*factor+offset
        if squeeze: x = x.squeeze(-1)
        return x


class RunningBatchNorm2d(RunningBatchNorm1d):
    r'''
    2D Running batchnorm implementation from fastai (https://github.com/fastai/course-v3) distributed under apache2 licence.
    Modifcations: add eps in mom1 calculation, type hinting, docs

    Arguments:
        nf: number of features/channels
        mom: momentum (fraction to add to running averages)
        eps: epsilon to prevent division by zero
    '''

    def _set_params(self) -> None:
        self.weight = nn.Parameter(torch.ones(self.nf,1,1))
        self.bias = nn.Parameter(torch.zeros(self.nf,1,1))
        self.register_buffer('sums', torch.zeros(1,self.nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,self.nf,1,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.dims = (0,2,3)

    def forward(self, x:Tensor) -> Tensor:
        if self.training: self.update_stats(x)
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < self.n_warmup): varns.clamp_min_(0.01)
        factor = self.weight/(varns+self.eps).sqrt()
        offset = self.bias-means*factor
        return x*factor+offset


class RunningBatchNorm3d(RunningBatchNorm2d):
    r'''
    3D Running batchnorm implementation from fastai (https://github.com/fastai/course-v3) distributed under apache2 licence.
    Modifcations: Adaptation to 3D, add eps in mom1 calculation, type hinting, docs

    Arguments:
        nf: number of features/channels
        mom: momentum (fraction to add to running averages)
        eps: epsilon to prevent division by zero
    '''

    def _set_params(self) -> None:
        self.weight = nn.Parameter(torch.ones(self.nf,1,1,1))
        self.bias = nn.Parameter(torch.zeros(self.nf,1,1,1))
        self.register_buffer('sums', torch.zeros(1,self.nf,1,1,1))
        self.register_buffer('sqrs', torch.zeros(1,self.nf,1,1,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.dims = (0,2,3,4)
