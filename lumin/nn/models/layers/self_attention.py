from typing import Callable, Optional, Any
import math
from fastcore.all import store_attr

import torch
from torch import nn, Tensor

from .activations import lookup_act
from .batchnorms import LCBatchNorm1d
from ..initialisations import lookup_normal_init

__all__ = ['SelfAttention']


class SelfAttention(nn.Module):
    r'''
    Class for applying self attention (Vaswani et al. 2017 (https://arxiv.org/abs/1706.03762)) to features per vertex.
    
    Arguments:
        n_fpv: number of features per vertex to expect
        n_a: width of self attention representation (paper recommends n_fpv//4)
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d` 
    '''
    
    def __init__(self, n_fpv:int, n_a:int, do:float=0, bn:bool=False, act:str='relu',
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d):
        super().__init__()
        store_attr()
        self.q = self._get_layer(self.n_fpv, self.n_a)
        self.k = self._get_layer(self.n_fpv, self.n_a)
        self.v = self._get_layer(self.n_fpv, self.n_fpv)
        self.out = self._get_out()
        
    def _get_out(self) -> nn.Sequential:
        layers = [self._get_layer(self.n_fpv, self.n_fpv)]
        if self.act != 'linear': layers.append(self.lookup_act(self.act))
        if self.bn: layers.append(LCBatchNorm1d(self.bn_class(self.n_fpv)))
        if self.do: 
            if self.act == 'selu': layers.append(nn.AlphaDropout(self.do))
            else:                  layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
        
    def _get_layer(self, fan_in:int, fan_out:int) -> nn.Module:   
        l = nn.Linear(fan_in, fan_out)
        self.lookup_init('linear', fan_in, fan_out)(l.weight)
        nn.init.zeros_(l.bias)
        return l
    
    def forward(self, x:Tensor) -> Tensor:  # B N C
        r'''
        Augments features per vertex
        
        Arguemnts:
            x: incoming data (batch x vertices x features)
            
        Returns:
            augmented features (batch x vertices x new features)
        '''
            
        a = (self.q(x)@self.k(x).transpose(-1,-2))/math.sqrt(self.n_a)  # B N N
        a = torch.softmax(a, dim=-1)  # Softmax norm columns
        sa = a@self.v(x)  # B N C
        return x+self.out(sa)  # B N C
    
    def get_out_size(self) -> int: return self.n_fpv


class OffsetSelfAttention(SelfAttention):
    r'''
    Class for applying offset-self attention (Guo et al. 2020 (https://arxiv.org/abs/2012.09688)) to features per vertex.
    
    Arguments:
        n_fpv: number of features per vertex to expect
        n_a: width of self attention representation (paper recommends n_fpv//4)
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d` 
    '''
    
    def forward(self, x:Tensor) -> Tensor:  # B N C
        r'''
        Augments features per vertex
        
        Arguemnts:
            x: incoming data (batch x vertices x features)
            
        Returns:
            augmented features (batch x vertices x new features)
        '''
        
        a = self.q(x)@self.k(x).transpose(-1,-2)  # B N N
        a = torch.softmax(a, dim=-2)  # Softmax norm rows
        a = a/(a.sum(-1, keepdim=True)+1e-17)  # L1 norm columns
        sa = a@self.v(x)  # B N C
        return x+self.out(x-sa)  # B N C
