import numpy as np
from typing import Dict, Optional, Callable, List, Any, Union, Tuple
from glob import glob
from collections import OrderedDict
from pathlib import Path
import os
from abc import abstractmethod

import torch.nn as nn
from torch.tensor import Tensor
import torch

from ..helpers import CatEmbedder
from ..initialisations import lookup_normal_init
from ..layers.activations import lookup_act
from ....plotting.plot_settings import PlotSettings
from ....plotting.interpretation import plot_embedding
from .abs_block import AbsBlock
from ....utils.misc import to_device

__all__ = ['CatEmbHead']


class AbsHead(AbsBlock):
    def __init__(self, cont_feats:List[str], cat_embedder:Optional[CatEmbedder]=None, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__(lookup_init=lookup_init, freeze=freeze)
        self.cont_feats,self.cat_embedder = cont_feats,cat_embedder
        self.n_cont_in = len(cont_feats)
        self.n_cat_in,self.cat_feats = (self.cat_embedder.n_cat_in,self.cat_embedder.cat_names) if self.cat_embedder is not None else (0,[])
        self.n_matrix_in,self.matrix_feats = 0,[]

    # TODO Make abtsract wrt data format

    @abstractmethod
    def _map_outputs(self) -> Dict[str,List[int]]: pass


class AbsMatrixHead(AbsHead):
    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str], row_wise:bool=True,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, cat_embedder=None, lookup_init=lookup_init, freeze=freeze)
        self.vecs,self.fpv,self.row_wise,self.lookup_act = vecs,feats_per_vec,row_wise,lookup_act
        self.n_v,self.n_fpv = len(self.vecs),len(self.fpv)
        self._build_lookup()
            
    def _build_lookup(self) -> None:
        shp = (self.n_v,self.n_fpv) if self.row_wise else (self.n_fpv,self.n_v)
        lookup = torch.zeros(shp, dtype=torch.long)
        missing = torch.zeros(shp, dtype=torch.uint8)
        
        if self.row_wise:
            for i, v in enumerate(self.vecs):
                for j, c in enumerate(self.fpv):
                    f = f'{v}_{c}'
                    if f in self.cont_feats: lookup[i,j]  = self.cont_feats.index(f)
                    else:                    missing[i,j] = 1
        else:
            for j, v in enumerate(self.vecs):
                for i, c in enumerate(self.fpv):
                    f = f'{v}_{c}'
                    if f in self.cont_feats: lookup[i,j]  = self.cont_feats.index(f)
                    else:                    missing[i,j] = 1
        self.missing,self.lookup = to_device(missing.flatten()),to_device(lookup.flatten())
    
    def _get_matrix(self, x:Tensor) -> Tensor:
        mat = x[:,self.lookup]
        mat[:,self.missing] = 0
        mat = mat.reshape((x.size(0),len(self.vecs),len(self.fpv)) if self.row_wise else (x.size(0),len(self.fpv),len(self.vecs))) 
        return to_device(mat)

    @abstractmethod
    def forward(self, x:Tensor, to_matrix:bool=False) -> Tensor:
        r'''
        Pass tensor through head

        Arguments:
            x: input tensor
        
        Returns
            Resulting tensor
        '''

        pass


class CatEmbHead(AbsHead):
    r'''
    Standard model head for columnar data.
    Provides inputs for continuous features and embedding matrices for categorical inputs, and uses a dense layer to upscale to width of network body.
    Designed to be passed as a 'head' to :class:`~lumin.nn.models.model_builder.ModelBuilder`.
    Supports batch normalisation and dropout (at separate rates for continuous features and categorical embeddings).
    Continuous features are expected to be the first len(cont_feats) columns of input tensors and categorical features the remaining columns.
    Embedding arguments for categorical features are set using a :class:`~lumin.nn.models.helpers.CatEmbedder`.

    Arguments:
        cont_feats: list of names of continuous input features
        do_cont: if not None will add a dropout layer with dropout rate do acting on the continuous inputs prior to concatination wih the categorical embeddings
        do_cat: if not None will add a dropout layer with dropout rate do acting on the categorical embeddings prior to concatination wih the continuous inputs
        cat_embedder: :class:`~lumin.nn.models.helpers.CatEmbedder` providing details of how to embed categorical inputs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable

    Examples::
        >>> head = CatEmbHead(cont_feats=cont_feats)
        >>>
        >>> head = CatEmbHead(cont_feats=cont_feats,
        ...                   cat_embedder=CatEmbedder.from_fy(train_fy))
        >>>
        >>> head = CatEmbHead(cont_feats=cont_feats,
        ...                   cat_embedder=CatEmbedder.from_fy(train_fy),
        ...                   do_cont=0.1, do_cat=0.05)
        >>>
        >>> head = CatEmbHead(cont_feats=cont_feats,
        ...                   cat_embedder=CatEmbedder.from_fy(train_fy),
        ...                   lookup_init=lookup_uniform_init)
    '''

    def __init__(self, cont_feats:List[str], do_cont:float=0, do_cat:float=0, cat_embedder:Optional[CatEmbedder]=None, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__(cont_feats=cont_feats, cat_embedder=cat_embedder, lookup_init=lookup_init, freeze=freeze)
        self.do_cont,self.do_cat, = do_cont,do_cat
        if self.cat_embedder is None: self.cat_embedder = CatEmbedder([], [])
        if self.cat_embedder.n_cat_in > 0:
            self.embeds = nn.ModuleList([nn.Embedding(ni, no) for _, ni, no in self.cat_embedder])
            if self.cat_embedder.emb_load_path is not None: self._load_embeds()
            if self.do_cat   > 0: self.emb_do     = nn.Dropout(self.do_cat)
        self.n_out = self.n_cont_in if self.cat_embedder.n_cat_in == 0 else self.n_cont_in+np.sum(self.cat_embedder.emb_szs)
        if self.do_cont  > 0: self.cont_in_do = nn.Dropout(self.do_cont)
        if self.freeze: self.freeze_layers()
        self._map_outputs()

    def _map_outputs(self):
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = [i]
        offset = self.n_cont_in
        for f, _, sz in self.cat_embedder:
            self.feat_map[f] = list(range(offset, offset+sz))
            offset += sz
        
    def forward(self, x:Tensor) -> Tensor:
        x_out = x
        if self.cat_embedder.n_cat_in > 0:
            x_cat = x[:,self.n_cont_in:].long()
            x_out = torch.cat([emb(x_cat[:,i]) for i, emb in enumerate(self.embeds)], dim=1)
            if self.do_cat > 0: x = self.emb_do(x)
        if self.n_cont_in > 0:
            x_cont = x[:,:self.n_cont_in]
            if self.do_cont > 0: x_cont = self.cont_in_do(x_cont) 
            x_out = torch.cat((x_cont, x_out), dim=1) if self.cat_embedder.n_cat_in > 0 else x_cont
        return x_out
    
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
            settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
        '''
        
        for i, n in enumerate(self.cat_embedder.cat_names): plot_embedding(self.embeds[i].state_dict(), n, savename=savename, settings=settings)


class MultiHead(AbsHead):
    r'''

    '''

    def __init__(self, cont_feats:List[str], matrix_head:Callable[[Any],AbsMatrixHead], flat_head:Callable[[Any],AbsHead]=CatEmbHead,
                 cat_embedder:Optional[CatEmbedder]=None, lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, cat_embedder=cat_embedder, lookup_init=lookup_init, freeze=freeze)
        self._set_feats(matrix_head)
        self.flat_head = flat_head(cont_feats=[f for f in self.flat_feats if f in cont_feats], cat_embedder=self.cat_embedder,
                                   lookup_init=self.lookup_init, freeze=self.freeze, **kargs)
        self.matrix_head = matrix_head(cont_feats=self.matrix_feats, cat_embedder=None, lookup_init=self.lookup_init, freeze=self.freeze, **kargs)
        self.cont_feats,self.n_cont_in = self.flat_head.cont_feats,self.flat_head.n_cont_in  # Only flat cont feats
        self._map_outputs()
        self._build_lookups()
        
    def _set_feats(self, matrix_head:Callable[[Any],AbsMatrixHead]) -> None:
        self.feats = self.cont_feats+self.cat_feats
        self.matrix_feats,tmp_fs = [],[]
        for v in matrix_head.keywords['vecs']:
            for c in matrix_head.keywords['feats_per_vec']:
                tmp_fs.append(f'{v}_{c}')  # All features for matrix elements
        for f in self.cont_feats:
            if f in tmp_fs: self.matrix_feats.append(f)  # Only subset of features present in flattened data, same ordering
        self.n_matrix_in = len(self.matrix_feats)
        self.flat_feats = [f for f in self.feats if f not in self.matrix_feats]
        
    def _map_outputs(self) -> None:
        self.feat_map = {**self.flat_head.feat_map}
        for f in self.matrix_head.feat_map:
            self.feat_map[f] = [self.matrix_head.feat_map[f][i]+self.flat_head.get_out_size() for i in self.matrix_head.feat_map[f]]
        
    def _build_lookups(self) -> None:
        self.flat_lookup = torch.zeros(len(self.flat_feats), dtype=torch.long)
        for i,f in enumerate(self.flat_feats): self.flat_lookup[i] = self.feats.index(f)
        self.matrix_lookup = torch.zeros(len(self.matrix_feats), dtype=torch.long)
        for i,f in enumerate(self.matrix_feats): self.matrix_lookup[i] = self.matrix_feats.index(f)
    
    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        if isinstance(x, tuple):
            return torch.cat((self.flat_head(x[0]),self.matrix_head(x[1])), 1)
        else:
            return torch.cat((self.flat_head(x[:,self.flat_lookup]),self.matrix_head(x[:,self.matrix_lookup])), 1)
    
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''
        
        return self.flat_head.get_out_size()+self.matrix_head.get_out_size()


class InteractionNet(AbsMatrixHead):
    r'''
    
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 intfunc_depth:int, intfunc_width:int, intfunc_out_sz:int,
                 outfunc_depth:int, outfunc_width:int, outfunc_out_sz:int, agg_method:str,
                 do:float=0, bn:bool=False, act:str='relu',
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, row_wise=False,
                         lookup_act=lookup_act, lookup_init=lookup_init, freeze=freeze)
        self.intfunc_depth,self.intfunc_width,self.intfunc_out_sz = intfunc_depth,intfunc_width,intfunc_out_sz
        self.outfunc_depth,self.outfunc_width,self.outfunc_out_sz = outfunc_depth,outfunc_width,outfunc_out_sz
        self.do,self.bn,self.act = do,bn,act
        self._check_agg_method(agg_method)
        
        self.n_e = self.n_v*(self.n_v-1)
        self.mat_rr,self.mat_rs = self._get_mat_r()
        self.mat_rr_t = self.mat_rr.t()
        self.fr = self._get_nn(fan_in=2*self.n_fpv, width=self.intfunc_width, fan_out=self.intfunc_out_sz, depth=self.intfunc_depth)
        self.fo = self._get_nn(fan_in=self.n_fpv+self.intfunc_out_sz, width=self.outfunc_width, fan_out=self.outfunc_out_sz, depth=self.outfunc_depth)
        self._map_outputs()
    
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))
            
    def _check_agg_method(self, agg_method:str) -> None:
        agg_method = agg_method.lower()
        ms = ['sum', 'flatten']
        if agg_method not in ms: raise ValueError(f'{agg_method} not in {ms}')
        self.agg_method = agg_method
    
    def _get_nn(self, fan_in:int, width:int, fan_out:int, depth:int) -> nn.Module:
        return nn.Sequential(*[self._get_layer(fan_in if i == 0 else width, width if i+1 < depth else fan_out) for i in range(depth)])
    
    def _get_layer(self, fan_in:int, fan_out:int) -> nn.Module:   
        layers = []
        layers.append(nn.Linear(fan_in, fan_out))
        self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if self.act != 'linear': layers.append(self.lookup_act(self.act))
        if self.bn:  layers.append(nn.BatchNorm1d(fan_out))
        if self.do: 
            if self.act == 'selu': layers.append(nn.AlphaDropout(self.do))
            else:                  layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)
    
    def _get_mat_r(self) -> Tuple[Tensor,Tensor]:
        mat_rr,mat_rs = torch.zeros((self.n_v,self.n_e)),torch.zeros((self.n_v,self.n_e))
        for i in range(self.n_e):
            j = i % self.n_v
            mat_rr[j,(i+1) % self.n_e] = 1
            mat_rs[j,i] = 1
        return to_device(mat_rr),to_device(mat_rs)

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        if isinstance(x, tuple): x = x[1]
        mat_i = self._get_matrix(x) if len(x.shape) <= 2 else x
        mat_o = torch.cat((mat_i@self.mat_rr, mat_i@self.mat_rs), 1)
        
        # Transpose+reshape trick from https://github.com/eric-moreno/IN/blob/master/gnn.py
        mat_o = torch.transpose(mat_o, 1, 2)
        mat_o = self.fr(mat_o.reshape(-1, 2*self.n_fpv)).reshape(-1, self.n_e, self.intfunc_out_sz)
        mat_o = torch.transpose(mat_o, 1, 2)
        
        mat_o = mat_o@self.mat_rr_t
        mat_o = torch.cat((mat_i,mat_o), 1)
        
        mat_o = torch.transpose(mat_o, 1, 2)
        mat_o = self.fo(mat_o.reshape(-1, self.n_fpv+self.intfunc_out_sz)).reshape(-1, self.n_v, self.outfunc_out_sz)

        if self.agg_method == 'sum':       return mat_o.sum(1)
        elif self.agg_method == 'flatten': return mat_o.reshape(x.size(0), -1)
    
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''
        
        if self.agg_method == 'sum':       return self.outfunc_out_sz
        elif self.agg_method == 'flatten': return self.outfunc_out_sz*self.n_v
