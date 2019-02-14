import numpy as np
from typing import Dict, Optional, List
from glob import glob
from collections import OrderedDict
from pathlib import Path
import os

import torch.nn as nn
from torch.tensor import Tensor
import torch

from ....plotting.plot_settings import PlotSettings
from ....plotting.interpretation import plot_embedding


class CatEmbHead(nn.Module):
    def __init__(self, n_cont_in:int, n_cat_in:int, emb_szs:Optional[List[int]], do_cont:float, do_cat:float,
                 cat_names:Optional[List[str]], emb_load_path:Optional[Path], freeze:bool=False):
        super().__init__()
        self.layers = []
        self.n_cont_in,self.n_cat_in,self.emb_szs,self.do_cont,self.do_cat,self.cat_names,self.emb_load_path,self.freeze = n_cont_in,n_cat_in,emb_szs,do_cont,do_cat,cat_names,emb_load_path,freeze
        if self.n_cat_in > 0:
            self.embeds = nn.ModuleList([nn.Embedding(ni, no) for ni, no in self.emb_szs])
            self.layers.append(self.embeds)
        if self.emb_load_path is not None: self.load_embeds()
        self.out_size = self.n_cont_in if self.n_cat_in == 0 else self.n_cont_in+np.sum(self.emb_szs[:,1])
        if self.do_cat   > 0:
            self.emd_do  = nn.Dropout(self.do_cat)
            self.layers.append(self.emd_do)
        if self.do_cont  > 0:
            self.do = nn.Dropout(self.do_cont)
            self.layers.append(self.do_cont)
        if self.n_cat_in > 0:
            self.bn = nn.BatchNorm1d(self.out_size)
            self.layers.append(self.bn)
        if self.freeze: self.freeze_layers()
    
    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self):
        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self):
        for p in self.parameters(): p.requires_grad = True
        
    def forward(self, x_in:Tensor) -> Tensor:
        if self.n_cat_in > 0:
            x_cat = x_in[:,self.n_cont_in:].long()
            x = torch.cat([emb(x_cat[:,i]) for i, emb in enumerate(self.embeds)], dim=1)
            if self.do_cat > 0: x = self.emd_do(x)
        if self.n_cont_in > 0:
            x_cont = x_in[:,:self.n_cont_in]
            if self.do_cont > 0: x_cont = self.do(x_cont) 
            x = torch.cat((x, x_cont), dim=1) if self.n_cat_in > 0 else x_cont
        if self.n_cat_in > 0: x = self.bn(x)
        return x
    
    def load_embeds(self, path:Optional[Path]=None) -> None:
        path = self.emb_load_path if path is None else path
        avail = {x.index(x[:-3]): x for x in glob(f'{path}/*.h5') if x[x.rfind('/')+1:-3] in self.cat_names}
        print(f'Loading embedings: {avail}')
        for i in avail: self.embeds[i].load_state_dict(torch.load(avail[i]))
            
    def save_embeds(self, path:Path) -> None:
        os.makedirs(path, exist_ok=True)
        for i, name in enumerate(self.cat_names): torch.save(self.embeds[i].state_dict(), path/f'{name}.h5')
            
    def get_embeds(self) -> Dict[str,OrderedDict]: return {n: self.embeds[i].state_dict() for i, n in enumerate(self.cat_names)}
    
    def get_out_size(self) -> int: return self.out_size

    def plot_embeds(self, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
        for i, n in enumerate(self.cat_names): plot_embedding(self.embeds[i].state_dict(), n, savename=savename, settings=settings)
