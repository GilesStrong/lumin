from typing import Optional, Callable

import torch.nn as nn
import torch
from torch import Tensor

from ..layers.activations import lookup_act
from ..initialisations import lookup_normal_init


class FullyConnected(nn.Module):
    '''Fully connected set of hidden layers.
    Can optionaly have skip connections between each layer (res=true).
    Alternatively can concatinate layers (dense=true)'''
    def __init__(self, depth:int, width:int, do:float, bn:bool, act:str, res:bool, dense:bool,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],nn.Module]=lookup_act, freeze:bool=False):
        super().__init__()
        self.depth,self.width,self.do,self.bn,self.act,self.res,self.dense = depth,width,do,bn,act,res,dense,
        self.lookup_init,self.lookup_act,self.freeze = lookup_init,lookup_act,freeze

        self.layers = nn.ModuleList([self.get_layer(d) for d in range(depth)])
        if dense: self.layers += [self.get_layer(depth, self.width*(2**(self.depth)), self.width)]
        if self.freeze: self.freeze_layers

    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self):
        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self):
        for p in self.parameters(): p.requires_grad = True
    
    def get_layer(self, idx:int, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> None:
        width = self.width if not self.dense else self.width*(2**idx)
        fan_in  = width if fan_in  is None else fan_in
        fan_out = width if fan_out is None else fan_out
        
        layers = [nn.Linear(fan_in, fan_out)]
        self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
        if self.act != 'linear': layers.append(self.lookup_act(self.act))
        if self.bn or self.res: layers.append(nn.BatchNorm1d(fan_out))
        if self.do: 
            if self.act == 'selu':
                layers.append(nn.AlphaDropout(self.do))
            else:
                layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
    
    def forward(self, x:Tensor) -> Tensor:
        if self.dense:
            for l in self.layers[:-1]: x = torch.cat((l(x), x), 1)
            x = self.layers[-1](x)
        else:
            for l in self.layers: x = l(x)+x if self.res else l(x)
        return x
    
    def get_out_size(self) -> int: return self.width
