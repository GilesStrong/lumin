import numpy as np
from typing import Dict, Optional, Callable
from glob import glob
from collections import OrderedDict
from pathlib import Path
import os

import torch.nn as nn
from torch.tensor import Tensor
import torch

from ..helpers import Embedder
from ..initialisations import lookup_normal_init
from ....plotting.plot_settings import PlotSettings
from ....plotting.interpretation import plot_embedding


class CatEmbHead(nn.Module):
    r'''
    Standard model head for columnar data.
    Provides inputs for continuous features and embedding matrices for categorical inputs, and uses a dense layer to upscale to width of network body.
    Designed to be passed as a 'head' to :class:ModelBuilder.
    Supports batch normalisation and dropout (at separate rates for pre-dense continuous, pre-dense categorical embeddings, and post dense).
    Continuous features are expected to be the first n_cont_in columns of input tensors and categorical features the remaining columns.
    Embedding arguments for categorical features are set using a :class:Embedder.

    Arguments:
        n_cont_in: number of continuous inputs to expect
        do_cont: if not None will add a dropout layer with dropout rate do acting on the continuous inputs prior to concatination wih the categorical embeddings
        do_cat: if not None will add a dropout layer with dropout rate do acting on the categorical embeddings prior to concatination wih the continuous inputs
        cat_embedder: :class:Embedder providing details of how to embed categorical inputs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable

    Examples::
        >>> head = CatEmbHead(n_cont_in=30)
        >>> head = CatEmbHead(n_cont_in=25, cat_embedder=Embedder.from_fy(train_fy))
        >>> head = CatEmbHead(n_cont_in=25, cat_embedder=Embedder.from_fy(train_fy), do_cont=0.1, do_cat=0.05)
        >>> head = CatEmbHead(n_cont_in=25, cat_embedder=Embedder.from_fy(train_fy), lookup_init=lookup_uniform_init)
    '''

    def __init__(self, n_cont_in:int, do_cont:float=0, do_cat:float=0, cat_embedder:Optional[Embedder]=None, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__()
        self.n_cont_in,self.do_cont,self.do_cat,self.cat_embedder,self.lookup_init,self.freeze = n_cont_in,do_cont,do_cat,cat_embedder,lookup_init,freeze
        if self.cat_embedder is None: self.cat_embedder = Embedder([], [])
        if self.cat_embedder.n_cat_in > 0: 
            self.embeds = nn.ModuleList([nn.Embedding(ni, no) for _, ni, no in self.cat_embedder])
            if self.cat_embedder.emb_load_path is not None: self._load_embeds()
            if self.do_cat   > 0: self.emb_do     = nn.Dropout(self.do_cat)
        self.n_out = self.n_cont_in if self.cat_embedder.n_cat_in == 0 else self.n_cont_in+np.sum(self.cat_embedder.emb_szs)
        if self.do_cont  > 0: self.cont_in_do = nn.Dropout(self.do_cont)
        if self.freeze: self.freeze_layers()
    
    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self) -> None:
        r'''
        Make parameters untrainable
        '''

        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self) -> None:
        r'''
        Make parameters trainable
        '''

        for p in self.parameters(): p.requires_grad = True
        
    def forward(self, x_in:Tensor) -> Tensor:
        if self.cat_embedder.n_cat_in > 0:
            x_cat = x_in[:,self.n_cont_in:].long()
            x = torch.cat([emb(x_cat[:,i]) for i, emb in enumerate(self.embeds)], dim=1)
            if self.do_cat > 0: x = self.emb_do(x)
        if self.n_cont_in > 0:
            x_cont = x_in[:,:self.n_cont_in]
            if self.do_cont > 0: x_cont = self.cont_in_do(x_cont) 
            x = torch.cat((x, x_cont), dim=1) if self.cat_embedder.n_cat_in > 0 else x_cont
        return x
    
    def _load_embeds(self, path:Optional[Path]=None) -> None:
        path = self.cat_embedder.emb_load_path if path is None else path
        avail = {x.index(x[:-3]): x for x in glob(f'{path}/*.h5') if x[x.rfind('/')+1:-3] in self.cat_embedder.cat_names}
        print(f'Loading embedings: {avail}')
        for i in avail: self.embeds[i].load_state_dict(torch.load(avail[i], map_location='cpu'))
            
    def save_embeds(self, path:Path) -> None:
        r'''
        Save learned embeddings to path.
        Each categorical embedding matic will be saved as a separate state_dict with name equal to the feature name as set in cat_embedder

        Arguments:
            path: path to which to save embedding weights
        '''
        
        os.makedirs(path, exist_ok=True)
        for i, name in enumerate(self.cat_embedder.cat_names): torch.save(self.embeds[i].state_dict(), path/f'{name}.h5')
            
    def get_embeds(self) -> Dict[str,OrderedDict]:
        r'''
        Get state_dict for every embedding matrix.

        Returns:
            Dictionary mapping categorical features to learned embedding matrix
        '''

        return {n: self.embeds[i].state_dict() for i, n in enumerate(self.cat_embedder.cat_names)}
    
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''
        
        return self.n_out

    def plot_embeds(self, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
        r'''
        Plot representations of embedding matrices for each categorical feature.

        Arguments:
            savename: if not None, will save copy of plot to give path
            settings: :class:PlotSettings class to control figure appearance
        '''
        
        for i, n in enumerate(self.cat_embedder.cat_names): plot_embedding(self.embeds[i].state_dict(), n, savename=savename, settings=settings)
