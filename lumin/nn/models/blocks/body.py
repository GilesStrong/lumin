from typing import Optional, Callable, Any
import numpy as np

import torch.nn as nn
import torch
from torch import Tensor

from ..layers.activations import lookup_act
from ..initialisations import lookup_normal_init
    

class FullyConnected(nn.Module):
    r'''
    Fully connected set of hidden layers. Designed to be passed as a 'body' to :class:ModelBuilder.
    Supports batch normalisation and dropout.
    Order is dense->activation->BN->DO, except when res is true in which case the BN is applied after the addition.
    Can optionaly have skip connections between each layer (res=true).
    Alternatively can concatinate layers (dense=true)
    growth_rate parameter can be used to adjust the width of layers according to width+(width*(depth-1)*growth_rate)

    Arguments:
        depth: number of hidden layers. If res==True and depth is even, depth will be increased by one.
        width: base width of each hidden layer
        do: if not None will add dropout layers with dropout rates do
        bn: whether to use batch normalisation
        act: string representation of argument to pass to lookup_act
        res: whether to add an additative skip connection every two dense layers. Mutually exclusive with dense.
        dense: whether to perform layer-wise concatinations after every layer. Mutually exclusion with res.
        growth_rate: rate at which width of dense layers should increase with depth beyond the initial layer. Ignored if res=True. Can be negative.
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable

    Examples::
        >>> body = FullyConnected(n_in=32, depth=4, width=100, act='relu')
        >>> body = FullyConnected(n_in=32, depth=4, width=200, act='relu', growth_rate=-0.3)
        >>> body = FullyConnected(n_in=32, depth=4, width=100, act='swish', do=0.1, res=True)
        >>> body = FullyConnected(n_in=32, depth=6, width=32, act='selu', dense=True, growth_rate=0.5)
        >>> body = FullyConnected(n_in=32, depth=6, width=50, act='prelu', bn=True, lookup_init=lookup_uniform_init)
    '''

    def __init__(self, n_in:int, depth:int, width:int, do:float, bn:bool, act:str, res:bool, dense:bool, growth_rate:int,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False):
        super().__init__()
        self.n_in,self.depth,self.width,self.do,self.bn,self.act,self.res,self.dense,self.growth_rate = n_in,depth,width,do,bn,act,res,dense,growth_rate
        self.lookup_init,self.lookup_act,self.freeze = lookup_init,lookup_act,freeze

        if self.res:
            self.depth = 1+int(np.floor(self.depth/2))  # One upscale layer + each subsequent block will contain 2 layers
            self.res_bns = nn.ModuleList([nn.BatchNorm1d(self.width) for d in range(self.depth-1)])
            self.layers = nn.ModuleList([self._get_layer(idx=d, fan_in=self.width, fan_out=self.width)
                                         if d > 0 else self._get_layer(idx=d, fan_in=self.n_in, fan_out=self.width)
                                         for d in range(self.depth)])
        elif self.dense:
            self.layers = nn.ModuleList([self._get_layer(idx=d,fan_in=self.n_in+(self.width*d)+np.sum([int(self.width*growth_rate*i) for i in range(d)]),
                                                         fan_out=self.width+int(self.width*d*self.growth_rate))
                                         if d > 0 else self._get_layer(d, self.n_in, self.width)
                                         for d in range(self.depth)]) 
        else:
            self.layers = nn.ModuleList([self._get_layer(idx=d, fan_in=self.width+int(self.width*(d-1)*self.growth_rate),
                                                         fan_out=self.width+int(self.width*d*self.growth_rate))
                                         if d > 0 else self._get_layer(idx=d, fan_in=self.n_in, fan_out=self.width)
                                         for d in range(self.depth)])
            
        if self.freeze: self.freeze_layers

    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self):
        r'''
        Make parameters untrainable
        '''

        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self):
        r'''
        Make parameters trainable
        '''

        for p in self.parameters(): p.requires_grad = True
    
    def _get_layer(self, idx:int, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> None:
        fan_in  = self.width if fan_in  is None else fan_in
        fan_out = self.width if fan_out is None else fan_out
        if fan_in  < 1: fan_in  = 1
        if fan_out < 1: fan_out = 1        
        
        layers = []
        for i in range(2 if self.res and idx > 0 else 1):
            layers.append(nn.Linear(fan_in, fan_out))
            self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
            if self.act != 'linear': layers.append(self.lookup_act(self.act))
            if self.bn and i == 0:  layers.append(nn.BatchNorm1d(fan_out))  # In case of residual, BN will be added after addition
            if self.do: 
                if self.act == 'selu': layers.append(nn.AlphaDropout(self.do))
                else:                  layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
    
    def forward(self, x:Tensor) -> Tensor:
        if self.dense:
            for l in self.layers[:-1]: x = torch.cat((l(x), x), -1)
            x = self.layers[-1](x)
        else:
            for i, l in enumerate(self.layers):
                if self.res and i > 0:
                    x = l(x)+x
                    x = self.res_bns[i-1](x)  # Renormalise after addition
                else:
                    x = l(x)
        return x
    
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''
        
        sz = self.width+int(self.width*(self.depth-1)*self.growth_rate)
        return sz if sz > 0 else 1
