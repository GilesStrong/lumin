from typing import Optional, Callable, Any, List, Dict
import numpy as np
from functools import partial

import torch.nn as nn
import torch
from torch import Tensor

from ..layers.activations import lookup_act
from ..initialisations import lookup_normal_init
from .abs_block import AbsBlock


class AbsBody(AbsBlock):
    def __init__(self, n_in:int, feat_map:Dict[str,List[int]],
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False):
        super().__init__(lookup_init=lookup_init, freeze=freeze)
        self.n_in,self.feat_map,self.lookup_act = n_in,feat_map,lookup_act
    

class FullyConnected(AbsBody):
    r'''
    Fully connected set of hidden layers. Designed to be passed as a 'body' to :class:ModelBuilder.
    Supports batch normalisation and dropout.
    Order is dense->activation->BN->DO, except when res is true in which case the BN is applied after the addition.
    Can optionaly have skip connections between each layer (res=true).
    Alternatively can concatinate layers (dense=true)
    growth_rate parameter can be used to adjust the width of layers according to width+(width*(depth-1)*growth_rate)

    Arguments:
        n_in: number of inputs to the block
        feat_map: dictionary mapping input features to the model to outputs of head block
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
        >>> body = FullyConnected(n_in=32, feat_map=head.feat_map, depth=4, width=100, act='relu')
        >>> body = FullyConnected(n_in=32, feat_map=head.feat_map, depth=4, width=200, act='relu', growth_rate=-0.3)
        >>> body = FullyConnected(n_in=32, feat_map=head.feat_map, depth=4, width=100, act='swish', do=0.1, res=True)
        >>> body = FullyConnected(n_in=32, feat_map=head.feat_map, depth=6, width=32, act='selu', dense=True, growth_rate=0.5)
        >>> body = FullyConnected(n_in=32, feat_map=head.feat_map, depth=6, width=50, act='prelu', bn=True, lookup_init=lookup_uniform_init)
    '''

    def __init__(self, n_in:int, feat_map:Dict[str,List[int]], depth:int, width:int, do:float=0, bn:bool=False, act:str='relu', res:bool=False,
                 dense:bool=False, growth_rate:int=0, lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False):
        super().__init__(n_in=n_in, feat_map=feat_map, lookup_init=lookup_init, lookup_act=lookup_act, freeze=freeze)
        self.depth,self.width,self.do,self.bn,self.act,self.res,self.dense,self.growth_rate = depth,width,do,bn,act,res,dense,growth_rate

        if self.res:
            self.depth = 1+int(np.floor(self.depth/2))  # One upscale layer + each subsequent block will contain 2 layers
            self.res_bns = nn.ModuleList([nn.BatchNorm1d(self.width) for d in range(self.depth-1)])
            self.layers = nn.ModuleList([self._get_layer(idx=d, fan_in=self.width, fan_out=self.width)
                                         if d > 0 else self._get_layer(idx=d, fan_in=self.n_in, fan_out=self.width)
                                         for d in range(self.depth)])
        elif self.dense:
            self.layers = []
            for d in range(self.depth):
                self.layers.append(self._get_layer(idx=d, fan_in=self.n_in if d == 0 else self.n_in+np.sum([l[0].out_features for l in self.layers]),
                                   fan_out=max(1,self.width+int(self.width*d*self.growth_rate))))
            self.layers = nn.ModuleList(self.layers)
        else:
            self.layers = nn.ModuleList([self._get_layer(idx=d, fan_in=self.width+int(self.width*(d-1)*self.growth_rate),
                                                         fan_out=self.width+int(self.width*d*self.growth_rate))
                                         if d > 0 else self._get_layer(idx=d, fan_in=self.n_in, fan_out=self.width)
                                         for d in range(self.depth)])
            
        if self.freeze: self.freeze_layers

    def _get_layer(self, idx:int, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> nn.Module:
        fan_in  = self.width if fan_in  is None else fan_in
        fan_out = self.width if fan_out is None else fan_out
        if fan_in  < 1: fan_in  = 1
        if fan_out < 1: fan_out = 1        
        
        layers = []
        for i in range(2 if self.res and idx > 0 else 1):
            layers.append(nn.Linear(fan_in, fan_out))
            self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
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
        
        return self.layers[-1][0].out_features


class MultiBlock(AbsBody):
    r'''
    Body block allowing outputs of head block to be split amongst a series of body blocks.
    Output is the concatination of all sub-body blocks.
    Optionally, single-neuron 'bottleneck' layers can be used to pass an input to each sub-block based on a learned function of the input features that block
    would otherwise not receive, i.e. a  highly compressed representation of the rest of teh feature space.

    Arguments:
        n_in: number of inputs to the block
        feat_map: dictionary mapping input features to the model to outputs of head block
        blocks: list of uninstantciated :class:AbsBody blocks to which to pass a subsection of the total inputs. Note that partials should be used to set any
            relevant parameters at initialisation time
        feats_per_block: list of lists of names of features to pass to each :class:AbsBody, not that the feat_map provided by :class:AbsHead will map features
            to their relavant head outputs
        bottleneck: if true, each block will receive the output of a single neuron which takes as input all the features which each given block does not directly
            take as inputs
        bottleneck_act: if set to a string representation of an activation function, the output of each bottleneck neuron will be passed throguh the defined
            activation function before being passed to their associated blocks
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable

    Examples::
        >>> body = MultiBlock(blocks=[partial(FullyConnected, depth=1, width=50, act='swish'),
                                      partial(FullyConnected, depth=6, width=55, act='swish', dense=True, growth_rate=-0.1)],
                              feats_per_block=[[f for f in train_feats if 'DER_' in f], [f for f in train_feats if 'PRI_' in f]])
        >>> body = MultiBlock(blocks=[partial(FullyConnected, depth=1, width=50, act='swish'),
                                      partial(FullyConnected, depth=6, width=55, act='swish', dense=True, growth_rate=-0.1)],
                              feats_per_block=[[f for f in train_feats if 'DER_' in f], [f for f in train_feats if 'PRI_' in f]]
                              bottleneck=True)
        >>> body = MultiBlock(blocks=[partial(FullyConnected, depth=1, width=50, act='swish'),
                                      partial(FullyConnected, depth=6, width=55, act='swish', dense=True, growth_rate=-0.1)],
                              feats_per_block=[[f for f in train_feats if 'DER_' in f], [f for f in train_feats if 'PRI_' in f]]
                              bottleneck=True, bottleneck_act='swish')
    '''

    def __init__(self, n_in:int, feat_map:Dict[str,List[int]], blocks:List[partial], feats_per_block:List[List[str]],
                 bottleneck_sz:int=0, bottleneck_act:Optional[str]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False):
        super().__init__(n_in=n_in, feat_map=feat_map, lookup_init=lookup_init, lookup_act=lookup_act, freeze=freeze)
        self.feats_per_block,self.bottleneck_sz,self.bottleneck_act = feats_per_block,bottleneck_sz,bottleneck_act
        self.blocks,self.n_out,self.masks,self.bottleneck_blocks = [],0,[],None
        
        if self.bottleneck_sz > 0:
            self.bottleneck_blocks,self.bottleneck_sz_masks = [],[]
            for i, fs in enumerate(self.feats_per_block):
                tmp_map = {f: feat_map[f] for f in feat_map if f not in feats_per_block[i]}
                self.bottleneck_sz_masks.append([i for f in tmp_map for i in tmp_map[f]])
                self.bottleneck_blocks.append(self._get_bottleneck(self.bottleneck_sz_masks[-1]))
            self.bottleneck_blocks = nn.ModuleList(self.bottleneck_blocks)

        for i, b in enumerate(blocks):
            tmp_map = {f: feat_map[f] for f in feat_map if f in feats_per_block[i]}
            self.masks.append([i for f in tmp_map for i in tmp_map[f]])
            self.blocks.append(b(n_in=len(self.masks[-1])+self.bottleneck_sz, feat_map=tmp_map, lookup_init=self.lookup_init,
                                 lookup_act=self.lookup_act, freeze=self.freeze))
            self.n_out += self.blocks[-1].get_out_size()
        self.blocks = nn.ModuleList(self.blocks)

    def _get_bottleneck(self, mask:List[int]) -> nn.Module:
        layers = [nn.Linear(len(mask), self.bottleneck_sz)]
        if self.bottleneck_act is None:
            init = self.lookup_init('linear', len(mask), self.bottleneck_sz) 
        else:
            init = self.lookup_init(self.bottleneck_act, len(mask), self.bottleneck_sz)
            layers.append(self.lookup_act(self.bottleneck_act))
        init(layers[0].weight)
        return nn.Sequential(*layers)

    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Total number of outputs accross all blocks
        '''
        
        return self.n_out
    
    def forward(self, x:Tensor) -> Tensor:
        y = None
        for i, b in enumerate(self.blocks):
            if self.bottleneck_sz:
                a = self.bottleneck_blocks[i](x[:,self.bottleneck_sz_masks[i]])
                tmp_x = torch.cat((x[:,self.masks[i]], a), -1)
            else:
                tmp_x = x[:,self.masks[i]]
            out = b(tmp_x)
            if y is None: y = out
            else:         y = torch.cat((y, out), -1)
        return y
