from typing import List, Callable, Any, Optional, Tuple
from fastcore.all import store_attr
import numpy as np

import torch
from torch import nn, Tensor

from ..initialisations  import lookup_normal_init
from ..layers.batchnorms import LCBatchNorm1d
from ..layers.activations import lookup_act

__all__ = ['GravNetLayer']


class GravNetLayer(nn.Module):
    r'''
    Single GravNet GNN layer (Qasim, Kieseler, Iiyama, & Pierini, 2019 https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9).
    Designed to be used as a sub-layer of a head block, e.g. :class:`~lumin.nn.models.blocks.head.GravNetHead`
    Passes features per vertex through NN to compute new features & coordinates of vertex in latent space.
    Vertex then receives additional features based on aggregations of distance-weighted features for k-nearest vertices in latent space
    Second NN transforms features per vertex.
    Input (batch x vertices x features) --> Output (batch x vertices x new features)
    
    Arguments:
        n_fpv: number of features per vertex to expect
        n_s: number of latent-spatial dimensions to compute
        n_lr: number of features to compute per vertex for latent representation
        k: number of neighbours (including self) each vertex should consider when aggregating latent-representation features
        agg_methods: list of functions to use to aggregate distance-weighted latent-representation features
        n_out: number of output features to compute per vertex
        cat_means: if True, will extend the incoming features per vertex by including the means of all features across all vertices
        f_slr_depth: number of layers to use for the latent rep. NN
        f_out_depth: number of layers to use for the output NN
        potential: function to control distance weighting (default is the exp(-d^2) potential used in the paper)
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d` 
    '''
    
    def __init__(self, n_fpv:int, n_s:int, n_lr:int, k:int, agg_methods:List[Callable[[Tensor],Tensor]], n_out:int,
                 cat_means:bool=True, f_slr_depth:int=1, f_out_depth:int=1, potential:Callable[[Tensor],Tensor]=lambda x: torch.exp(-10*(x**2)),
                 do:float=0, bn:bool=False, act:str='relu',
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d):
        super().__init__()
        store_attr()
        if self.cat_means: self.n_fpv *= 2
        self.f_slr = self._get_nn(fan_in=self.n_fpv, width=2*self.n_s+self.n_lr, fan_out=self.n_s+self.n_lr, depth=self.f_slr_depth)
        self.f_out = self._get_nn(fan_in=self.n_fpv+(len(self.agg_methods)*self.n_lr), width=2*self.n_out, fan_out=self.n_out, depth=self.f_out_depth)
        
    def _get_nn(self, fan_in:int, width:int, fan_out:int, depth:int) -> nn.Module:
        return nn.Sequential(*[self._get_layer(fan_in if i == 0 else width, width if i+1 < depth else fan_out) for i in range(depth)])
    
    def _get_layer(self, fan_in:int, fan_out:int) -> nn.Module:   
        layers = []
        layers.append(nn.Linear(fan_in, fan_out))
        self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if self.act != 'linear': layers.append(self.lookup_act(self.act))
        if self.bn:              layers.append(self.bn_class(LCBatchNorm1d(fan_out)))
        if self.do: 
            if self.act == 'selu': layers.append(nn.AlphaDropout(self.do))
            else:                  layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
    
    def _knn(self, dists:Tensor) -> Tuple[Tensor,Tensor,Tensor]:
        idxs = dists.argsort()
        i = np.arange(dists.shape[0])[:,None,None]
        j = np.arange(dists.shape[1])[None,:,None]
        return i,j,idxs[:,:,:self.k]
    
    def forward(self, x:Tensor) -> Tensor:
        r'''
        Pass batch of vertices through GravNet layer and return new features per vertex
        
        Arguments:
            x: Incoming data (batch x vertices x features)
            
        Returns:
            Data with new features per vertex (batch x vertices x new features)
        '''
        
        # Concat means
        if self.cat_means: x = torch.cat([x,x.mean(1).unsqueeze(-1).repeat_interleave(repeats=x.shape[1],dim=-1).transpose(1,2)],dim=2)
        
        # Compute spatial and vertex features
        slr = self.f_slr(x)
        s,lr = slr[:,:,:self.n_s],slr[:,:,self.n_s:]
        
        # kNN
        d_jk = torch.norm(s[:,:,None]-s[:,None], dim=-1)
        f_ijk = torch.repeat_interleave(lr[:,None], repeats=lr.shape[1], dim=1)
        idxs = self._knn(d_jk)
        d_jk,f_ijk = d_jk[idxs],f_ijk[idxs]
        
        # Aggregate feats
        v_jk = self.potential(d_jk)
        ft_ijk = f_ijk*v_jk.unsqueeze(-1)
        fp = [x]
        for agg in self.agg_methods: fp.append(agg(ft_ijk))
        fp = torch.cat(fp,dim=-1)
        
        # Output
        return self.f_out(fp)
