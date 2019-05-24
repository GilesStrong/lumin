import numpy as np
from typing import Dict, Optional, List, Callable
from glob import glob
from collections import OrderedDict
from pathlib import Path
import os

import torch.nn as nn
from torch.tensor import Tensor
import torch

from ..helpers import Embedder
from ..initialisations import lookup_normal_init
from ..layers.activations import lookup_act
from ....plotting.plot_settings import PlotSettings
from ....plotting.interpretation import plot_embedding


class CatEmbHead(nn.Module):
    '''Standard model head for columnar data. Provides inputs for continuous features, and embeddign matrices for categorical inputs'''
    def __init__(self, n_cont_in:int, n_out:int, act:str, do:float, do_cont:float, do_cat:float, bn:bool, cat_embedder:Optional[Embedder]=None, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],nn.Module]=lookup_act, freeze:bool=False):
        super().__init__()
        self.n_cont_in,self.n_out,self.do,self.do_cont,self.do_cat,self.bn,self.cat_embedder = n_cont_in,n_out,do,do_cont,do_cat,bn,cat_embedder
        self.act,self.lookup_init,self.lookup_act,self.freeze = act,lookup_init,lookup_act,freeze
        if self.cat_embedder is None: self.cat_embedder = Embedder([], [])
        if self.cat_embedder.n_cat_in > 0: 
            self.embeds = nn.ModuleList([nn.Embedding(ni, no) for _, ni, no in self.cat_embedder])
            if self.cat_embedder.emb_load_path is not None: self.load_embeds()
            if self.do_cat   > 0: self.emb_do     = nn.Dropout(self.do_cat)
        input_sz = self.n_cont_in if self.cat_embedder.n_cat_in == 0 else self.n_cont_in+np.sum(self.cat_embedder.emb_szs)
        if self.do_cont  > 0: self.cont_in_do = nn.Dropout(self.do_cont)
        self.dense = self.get_dense(input_sz, self.n_out, self.act)
        if self.freeze: self.freeze_layers()
    
    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self):
        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self):
        for p in self.parameters(): p.requires_grad = True

    def get_dense(self, fan_in:int, fan_out:int, act:str) -> nn.Module:
        layers = []
        layers.append(nn.Linear(fan_in, fan_out))
        self.lookup_init(act, fan_in, fan_out)(layers[-1].weight)
        if act != 'linear': layers.append(self.lookup_act(act))
            
        if self.bn: layers.append(nn.BatchNorm1d(fan_out))
        if self.do: 
            if act == 'selu': layers.append(nn.AlphaDropout(self.do))
            else:             layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
        
    def forward(self, x_in:Tensor) -> Tensor:
        if self.cat_embedder.n_cat_in > 0:
            x_cat = x_in[:,self.n_cont_in:].long()
            x = torch.cat([emb(x_cat[:,i]) for i, emb in enumerate(self.embeds)], dim=1)
            if self.do_cat > 0: x = self.emb_do(x)
        if self.n_cont_in > 0:
            x_cont = x_in[:,:self.n_cont_in]
            if self.do_cont > 0: x_cont = self.cont_in_do(x_cont) 
            x = torch.cat((x, x_cont), dim=1) if self.cat_embedder.n_cat_in > 0 else x_cont
        return self.dense(x)
    
    def load_embeds(self, path:Optional[Path]=None) -> None:
        path = self.cat_embedder.emb_load_path if path is None else path
        avail = {x.index(x[:-3]): x for x in glob(f'{path}/*.h5') if x[x.rfind('/')+1:-3] in self.cat_embedder.cat_names}
        print(f'Loading embedings: {avail}')
        for i in avail: self.embeds[i].load_state_dict(torch.load(avail[i], map_location='cpu'))
            
    def save_embeds(self, path:Path) -> None:
        os.makedirs(path, exist_ok=True)
        for i, name in enumerate(self.cat_embedder.cat_names): torch.save(self.embeds[i].state_dict(), path/f'{name}.h5')
            
    def get_embeds(self) -> Dict[str,OrderedDict]: return {n: self.embeds[i].state_dict() for i, n in enumerate(self.cat_embedder.cat_names)}
    
    def get_out_size(self) -> int: return self.n_out

    def plot_embeds(self, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
        for i, n in enumerate(self.cat_embedder.cat_names): plot_embedding(self.embeds[i].state_dict(), n, savename=savename, settings=settings)
