import numpy as np
from typing import Dict, Optional, Callable, List, Any, Union, Tuple
from glob import glob
from collections import OrderedDict
from pathlib import Path
import os
from abc import abstractmethod, ABCMeta
from functools import partial
from distutils.version import LooseVersion

import torch.nn as nn
from torch import Tensor
import torch

from ..helpers import CatEmbedder
from ..initialisations import lookup_normal_init
from ..layers.activations import lookup_act
from ..layers.batchnorms import LCBatchNorm1d
from .gnn_blocks import AbsGraphFeatExtractor, GraphCollapser
from ....plotting.plot_settings import PlotSettings
from ....plotting.interpretation import plot_embedding
from .abs_block import AbsBlock
from ....utils.misc import to_device, is_partially
from .conv_blocks import Conv1DBlock, Res1DBlock, ResNeXt1DBlock

__all__ = ['CatEmbHead', 'MultiHead', 'GNNHead', 'RecurrentHead', 'AbsConv1dHead', 'LorentzBoostNet', 'AutoExtractLorentzBoostNet']


class AbsHead(AbsBlock, metaclass=ABCMeta):
    def __init__(self, cont_feats:List[str], cat_embedder:Optional[CatEmbedder]=None, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__(lookup_init=lookup_init, freeze=freeze)
        self.cont_feats,self.cat_embedder = cont_feats,cat_embedder
        self.n_cont_in = len(cont_feats)
        self.n_cat_in,self.cat_feats = (self.cat_embedder.n_cat_in,self.cat_embedder.cat_names) if self.cat_embedder is not None else (0,[])
        self.n_matrix_in,self.matrix_feats = 0,[]

    # TODO Make abtsract wrt data format

    @abstractmethod
    def _map_outputs(self) -> Dict[str,List[int]]:
        r'''
        Returns a one-to-one/many mapping of features to output nodes of block

        Returns:
            Dictionary mapping feature names to associated output nodes
        '''

        pass


class AbsMatrixHead(AbsHead):
    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str], row_wise:bool=True,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        super().__init__(cont_feats=cont_feats, cat_embedder=None, lookup_init=lookup_init, freeze=freeze)
        if row_wise is None: raise ValueError("row_wise is None, possibly a custom inheriting class did not change the class-attribute value. \
                                               Please set to True or False.")
        self.vecs,self.fpv,self.row_wise,self.lookup_act,self.bn_class = vecs,feats_per_vec,row_wise,lookup_act,bn_class
        self.n_v,self.n_fpv = len(self.vecs),len(self.fpv)
        self._build_lookup()
            
    def _build_lookup(self) -> None:
        r'''
        Builds lookup-tables necessary to map flattened data to correct locations for reshaping into a matrix.
        Also handles missing data, i.e. elements in the matrix which do not exist in the flattened data
        '''

        shp = (self.n_v,self.n_fpv) if self.row_wise else (self.n_fpv,self.n_v)
        lookup  = torch.zeros(shp, dtype=torch.long)
        missing = torch.zeros(shp, dtype=torch.bool if LooseVersion(torch.__version__) >= LooseVersion("1.2") else torch.uint8)
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
        r'''
        Converts flat data to matrix via lookup-and-reshaping, elements not present in flat data are set to zero

        Arguments:
            x: flat data

        Returns:
            2D matrix on device
        '''

        mat = x[:,self.lookup]
        mat[:,self.missing] = 0
        mat = mat.reshape((x.size(0),len(self.vecs),len(self.fpv)) if self.row_wise else (x.size(0),len(self.fpv),len(self.vecs))) 
        return to_device(mat)

    @abstractmethod
    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Pass tensor through head

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will conver the data to a matrix
        
        Returns:
            Resulting tensor
        '''

        pass

    def _process_input(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Processes input data, converting to matrix if necessary.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will conver the data to a matrix
        
        Returns:
            Relevant data in matrix form
        '''

        if isinstance(x, tuple): x = x[1]
        return self._get_matrix(x) if len(x.shape) <= 2 else x


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
    Wrapper head to handel data containing flat continuous and categorical features, and matrix data.
    Flat inputs are passed through `flat_head`, and matrix inputs are passed through `matrix_head`. The outputs of both blocks are then concatenated together.
    Incoming data can either be: Completely flat, in which case the `matrix_head` should construct its own matrix from the data;
    or a tuple of flat data and the matrix, in which case the `matrix_head` will receive the data already in matrix format.

    Arguments:
        cont_feats: list of names of continuous and matrix input features
        matrix_head: Uninitialised (partial) head to handle matrix data e.g. :class:`~lumin.nn.models.blocks.head.InteractionNet`
        flat_head: Uninitialised (partial) head to handle flat data e.g. :class:`~lumin.nn.models.blocks.head.CatEmbHead`
        cat_embedder: :class:`~lumin.nn.models.helpers.CatEmbedder` providing details of how to embed categorical inputs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable

    Examples::
    >>> inet = partial(InteractionNet, intfunc_depth=2,intfunc_width=4,intfunc_out_sz=3,
    ...        outfunc_depth=2,outfunc_width=5,outfunc_out_sz=4,agg_method='flatten',
    ...        feats_per_vec=feats_per_vec,vecs=vecs, act='swish')
    ... multihead = MultiHead(cont_feats=cont_feats+matrix_feats, matrix_head=inet, cat_embedder=CatEmbedder.from_fy(train_fy))
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
        r'''
        Sorts out which features will be sent to the flat and matrix heads.
        Feature usage is (currently) exclusive, i.e. the same feature cannot be used as a matrix element and a flat-continuous input.

        Arguments:
            matrix_head: The unititialised `matrix_head`, which should have `vecs` and `feats_per_vec` keyword arguments
        '''

        self.feats = self.cont_feats+self.cat_feats
        self.matrix_feats,tmp_fs = [],[]
        if 'vecs' in matrix_head.keywords and 'feats_per_vec' in matrix_head.keywords:
            for v in matrix_head.keywords['vecs']:
                for c in matrix_head.keywords['feats_per_vec']:
                    tmp_fs.append(f'{v}_{c}')  # All features for matrix elements
        for f in self.cont_feats:
            if f in tmp_fs: self.matrix_feats.append(f)  # Only subset of features present in flattened data, same ordering
        self.n_matrix_in = len(self.matrix_feats)
        self.flat_feats = [f for f in self.feats if f not in self.matrix_feats]
        
    def _map_outputs(self) -> None:
        r'''
        Combines `feat_maps` of the matrix and flat heads, offsetting to indeces to account for the concatenated outputs.
        '''

        self.feat_map = {**self.flat_head.feat_map}
        for f in self.matrix_head.feat_map:
            self.feat_map[f] = [self.matrix_head.feat_map[f][i]+self.flat_head.get_out_size() for i in self.matrix_head.feat_map[f]]
        
    def _build_lookups(self) -> None:
        r'''
        Build lookup tables to direct flat and matrix features to correct heads when input is supplied as a single, fully flat tensor.
        '''
        
        self.flat_lookup = torch.zeros(len(self.flat_feats), dtype=torch.long)
        for i,f in enumerate(self.flat_feats): self.flat_lookup[i] = self.feats.index(f)
        self.matrix_lookup = torch.zeros(len(self.matrix_feats), dtype=torch.long)
        for i,f in enumerate(self.matrix_feats): self.matrix_lookup[i] = self.feats.index(f)
    
    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Pass incoming data through flat and matrix heads.
        If `x` is a `Tuple` then the first element is passed to the flat head and the secons is sent to the matrix head.
        Else the elements corresponding to flat dta are sent to the flat head and the elements corresponding to matrix elements are sent to the matrix head.

        Arguments:
            x: input data as either a flat `Tensor` or a `Tuple` of the form `[flat Tensor, matrix Tensor]`

        Returns:
            Concetanted outout of flat and matrix heads
        '''

        if isinstance(x, tuple):
            return torch.cat((self.flat_head(x[0]),self.matrix_head(x[1])), 1)
        else:
            return torch.cat((self.flat_head(x[:,self.flat_lookup]),self.matrix_head(x[:,self.matrix_lookup])), 1)
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Output size of flat head + output size of matrix head
        '''
        
        return self.flat_head.get_out_size()+self.matrix_head.get_out_size()


class GNNHead(AbsMatrixHead):
    r'''
    Encasulating class for applying graph neural-networks to features per vertex.
    New features are extracted per vertex via a :class:`~lumin.nn.models.blocks.gnn_blocks.AbsGraphFeatExtractor`, and then data is flattened via :class:`~lumin.nn.models.blocks.gnn_blocks.GraphCollapser`
    
    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly into matrix form.
    Reshaping (row-wise or column-wise) depends on the `row_wise` class attribute of the feature extractor. Data will be automatically converted to row-wise for processing by the grpah collaser.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.
        
    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. feature prefixes
        feats_per_vec: list of features per vertex, i.e. feature suffixes
        use_int_bn: If true, will apply batch norm to incoming features
        cat_means: if True, will extend the incoming features per vertex by including the means of all features across all vertices
        extractor: The :class:`~lumin.nn.models.blocks.gnn_blocks.AbsGraphFeatExtractor` class to instantiate to create new features per vertex
        collasper: The :class:`~lumin.nn.models.blocks.gnn_blocks.GraphCollapser` class to instantiate to collapse graph to flat data (batch x features)
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 extractor:Callable[[Any],AbsGraphFeatExtractor], collapser:Callable[[Any],GraphCollapser],
                 use_in_bn:bool=False, cat_means:bool=False, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, freeze=freeze,
                         row_wise=extractor.func.row_wise if is_partially else extractor.row_wise)
        self.cat_means,self.use_in_bn = cat_means,use_in_bn
        if self.use_in_bn: self.bn = LCBatchNorm1d(bn_class(self.n_fpv)) if self.row_wise else bn_class(self.n_fpv)
        if self.cat_means: self.n_fpv *= 2
        self.extractor = extractor(n_fpv=self.n_fpv, n_v=self.n_v)
        sz = self.extractor.get_out_size()
        self.collapser = collapser(n_fpv=sz[1], n_v=sz[0])
        self._map_outputs()
        if self.freeze: self.freeze_layers()

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Passes input through the GravNet head and returns a flat tensor.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will convert the data to a matrix
        
        Returns:
            Resulting tensor
        '''
        
        x = self._process_input(x)  # features per vtx per datapoint
        if self.use_in_bn: x = self.bn(x)
        if self.cat_means:
            if self.row_wise: x = torch.cat([x,x.mean(1).unsqueeze(2).repeat_interleave(repeats=x.shape[1],dim=2).transpose(1,2)],dim=2)
            else:             x = torch.cat([x,x.mean(2).unsqueeze(1).repeat_interleave(repeats=x.shape[2],dim=1).transpose(1,2)],dim=1)
        x = self.extractor(x)  # new features per vtx per datapoint
        if not self.row_wise: x = x.transpose(1,2)
        x = self.collapser(x)  # features per datapoint
        return x
    
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Width of output representation
        '''
        
        return self.collapser.get_out_size()


class OldInteractionNet(AbsMatrixHead):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.models.gnn_blocks.InteractionNet`.
        It is a copy of the old `InteractionNet` class used in lumin<=0.7.2. It will be removed in V0.9

    Implementation of the Interaction Graph-Network (https://arxiv.org/abs/1612.00222).
    Shown to be applicable for embedding many 4-momenta in e.g. https://arxiv.org/abs/1908.05318

    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly in column-wise matrix form.
    Matrices should/will be column-wise: each column is a seperate object (e.g. particle and jet) and each row is a feature (e.g. energy and mometum component).
    Matrix elements are expected to be named according to `{object}_{feature}`, e.g. `photon_energy`.
    `vecs` (vectors) should then be a list of objects, i.e. column headers, feature prefixes.
    `feats_per_vec` should be a list of features, i.e. row headers, feature suffixes.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.

    The penultimate stage of processing in the interaction net is a matrix, this must be processed into a flat tensor. `agg_method` controls how this is done:
    'sum' will sum over the embedded representations of each object meaning that the objects can be placed in any order, however some information will be lost
    during the aggregation. 'flatten' will flatten out the matrix preserving all the information, however the objects must be placed in some order each time.
    Additionally, the 'flatten' mode can potentially become quite large if many objects are embedded. A future comprimise might be to feed the embeddings into
    a recurrent layer to provide a smaller output which preserves more information than the summing.

    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. column headers, feature prefixes
        feats_per_vec: list of features per object, i.e. row headers, feature suffixes
        intfunc_depth: number of layers in the interaction-representation network
        intfunc_width: width of hidden layers in the interaction-representation network
        intfunc_out_sz: width of output layer of the interaction-representation network, i.e. the size of each interaction representation
        outfunc_depth: number of layers in the post-interaction network
        outfunc_width: width of hidden layers in the post-interaction network
        outfunc_out_sz: width of output layer of the post-interaction network, i.e. the size of each output representation
        agg_method: how to transform the output matrix, currently either 'sum' to sum across objects, or 'flatten' to flatten out the matrix
        do: dropout rate to be applied to hidden layers in the interaction-representation and post-interaction networks
        bn: whether batch normalisation should be applied to hidden layers in the interaction-representation and post-interaction networks
        act: activation function to apply to hidden layers in the interaction-representation and post-interaction networks
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
    
    Examples::
        >>> inet = InteractionNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                       intfunc_depth=2,intfunc_width=4,intfunc_out_sz=3,
        ...                       outfunc_depth=2,outfunc_width=5,outfunc_out_sz=4,agg_method='flatten')
        >>>
        >>> inet = InteractionNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                       intfunc_depth=2,intfunc_width=4,intfunc_out_sz=6,
        ...                       outfunc_depth=2,outfunc_width=5,outfunc_out_sz=8,agg_method='sum')
        >>>
        >>> inet = InteractionNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                       intfunc_depth=3,intfunc_width=4,intfunc_out_sz=3,
        ...                       outfunc_depth=3,outfunc_width=5,outfunc_out_sz=4,agg_method='flatten',
        ...                       do=0.1, bn=True, act='swish', lookup_init=lookup_uniform_init)
    '''

    # XXX remove in V0.9

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 intfunc_depth:int, intfunc_width:int, intfunc_out_sz:int,
                 outfunc_depth:int, outfunc_width:int, outfunc_out_sz:int, agg_method:str,
                 do:float=0, bn:bool=False, act:str='relu',
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, row_wise=False,
                         lookup_act=lookup_act, lookup_init=lookup_init, freeze=freeze, bn_class=bn_class)
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
        if self.freeze: self.freeze_layers()
    
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
        if self.bn:  layers.append(LCBatchNorm1d(self.bn_class(fan_out)))
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
        r'''
        Passes input through the interaction network and aggregates out down to a flat tensor.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will conver the data to a matrix
        
        Returns:
            Resulting tensor
        '''

        mat_i = self._process_input(x)
        mat_o = torch.cat((mat_i@self.mat_rr, mat_i@self.mat_rs), 1)
        
        mat_o = torch.transpose(mat_o, 1, 2)
        mat_o = self.fr(mat_o)
        mat_o = torch.transpose(mat_o, 1, 2)
        
        mat_o = mat_o@self.mat_rr_t
        mat_o = torch.cat((mat_i,mat_o), 1)
        
        mat_o = torch.transpose(mat_o, 1, 2)
        mat_o = self.fo(mat_o)
        mat_o = torch.transpose(mat_o, 1, 2)
        if self.agg_method == 'sum':       return mat_o.sum(2)
        elif self.agg_method == 'flatten': return mat_o.reshape(mat_i.size(0), -1)
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Width of output representation
        '''
        
        if self.agg_method == 'sum':       return self.outfunc_out_sz
        elif self.agg_method == 'flatten': return self.outfunc_out_sz*self.n_v


class RecurrentHead(AbsMatrixHead):
    r'''
    Recurrent head for row-wise matrix data applying e.g. RNN, LSTM, GRU.
    
    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly into matrix form.
    Matrices should/will be row-wise: each column is a seperate object (e.g. particle and jet) and each row is a feature (e.g. energy and mometum component).
    Matrix elements are expected to be named according to `{object}_{feature}`, e.g. `photon_energy`.
    `vecs` (vectors) should then be a list of objects, i.e. row headers, feature prefixes.
    `feats_per_vec` should be a list of features, i.e. column headers, feature suffixes.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.

    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. row headers, feature prefixes
        feats_per_vec: list of features per object, i.e. columns headers, feature suffixes
        depth: number of hidden layers to use
        width: size of each hidden state
        bidirectional: whether to set recurrent layers to be bidirectional
        rnn: module class to use for the recurrent layer, e.g. `torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`
        do: dropout rate to be applied to hidden layers
        act: activation function to apply to hidden layers, only used if rnn expects a nonliearity
        stateful: whether to return all intermediate hidden states, or only the final hidden states
        freeze: whether to start with module parameters set to untrainable
    
    Examples::
        >>> rnn = RecurrentHead(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs, depth=1, width=20)
        >>>
        >>> rnn = RecurrentHead(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                     depth=2, width=10, act='relu', bidirectional=True)
        >>>
        >>> lstm = RecurrentHead(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                      depth=1, width=10, rnn=nn.LSTM)
        >>>
        >>> gru = RecurrentHead(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs,
        ...                     depth=3, width=10, rnn=nn.GRU, bidirectional=True)
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 depth:int, width:int, bidirectional:bool=False, rnn:nn.RNNBase=nn.RNN,
                 do:float=0., act:str='tanh', stateful:bool=False, freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, row_wise=True, freeze=freeze)
        self.stateful,self.width,self.bidirectional = stateful,width,bidirectional
        p = partial(rnn, input_size=self.n_fpv, hidden_size=width, num_layers=depth, bias=True, batch_first=True, dropout=do, bidirectional=bidirectional)
        try:              self.rnn = p(nonlinearity=act)
        except TypeError: self.rnn = p()
        self._init_rnn(width)
        self._map_outputs()
        if self.freeze: self.freeze_layers()

    def _init_rnn(self, width:int) -> None:
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:        nn.init.zeros_(param)
            elif 'weight_ih' in name: nn.init.orthogonal_(param)    
            if isinstance(self.rnn, nn.RNN):
                if 'weight_hh' in name: nn.init.orthogonal_(self.rnn.state_dict()[name])
            elif isinstance(self.rnn, nn.LSTM):
                if 'bias' in name: self.rnn.state_dict()[name][width:width*2].fill_(1)  # Forget bias -> 1
                elif 'weight_hh' in name:
                    for i in range(4): nn.init.orthogonal_(self.rnn.state_dict()[name][i*width:(i+1)*width])
            elif isinstance(self.rnn, nn.GRU):
                if 'bias' in name: self.rnn.state_dict()[name][:width].fill_(-1)  # Reset bias -> -1
                elif 'weight_hh' in name:
                    for i in range(3): nn.init.orthogonal_(self.rnn.state_dict()[name][i*width:(i+1)*width])
            
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Passes input through the recurrent network.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will conver the data to a matrix
        
        Returns:
            if stateful, returns all hidden states, otherwise only returns last hidden state
        '''

        x = self._process_input(x)
        x,_ = self.rnn(x)
        if self.stateful: return x
        else:             return x[:,-1]
    
    def get_out_size(self) -> Union[int,Tuple[int,int]]:
        r'''
        Get size of output

        Returns:
            Width of output representation, or shape of output if stateful
        '''
        
        if self.stateful:
            return (self.n_v,2*self.width) if self.bidirectional else (self.n_v,self.width)
        else:
            return 2*self.width if self.bidirectional else self.width


class AbsConv1dHead(AbsMatrixHead):
    r'''
    Abstract wrapper head for applying 1D convolutions to column-wise matrix data.
    Users should inherit from this class and overload :meth:`~lumin.nn.models.blocks.heads.AbsConv1dHead.get_layers` to define their model.
    Some common convolutional layers are already defined (e.g. :class:`~lumin.nn.models.blocks.conv_blocks.ConvBlock` and
    :class:`~lumin.nn.models.blocks.conv_blocks.ResNeXt`), which are accessable using methods such as
    :meth`~lumin.nn.models.blocks.heads.AbsConv1dHead..get_conv1d_block`.
    For more complicated models, :meth:`~lumin.nn.models.blocks.heads.AbsConv1dHead.foward` can also be overwritten
    The output size of the block is automatically computed during initialisation by passing through random pseudodata.
    
    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly into matrix form.
    Matrices should/will be row-wise: each column is a seperate object (e.g. particle and jet) and each row is a feature (e.g. energy and mometum component).
    Matrix elements are expected to be named according to `{object}_{feature}`, e.g. `photon_energy`.
    `vecs` (vectors) should then be a list of objects, i.e. row headers, feature prefixes.
    `feats_per_vec` should be a list of features, i.e. column headers, feature suffixes.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.

    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. row headers, feature prefixes
        feats_per_vec: list of features per object, i.e. columns headers, feature suffixes
        act: activation function passed to `get_layers`
        bn: batch normalisation argument passed to `get_layers`
        layer_kargs: dictionary of keyword arguments which are passed to `get_layers`
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
    
    Examples::
        >>> class MyCNN(AbsConv1dHead):
        ...     def get_layers(self, act:str='relu', bn:bool=False, **kargs) -> Tuple[nn.Module, int]:    
        ...         layers = []
        ...         layers.append(self.get_conv1d_block(3, 16, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_block(16, 16, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_block(16, 32, stride=2, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_block(32, 32, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(nn.AdaptiveAvgPool1d(1))
        ...         layers = nn.Sequential(*layers)
        ...         return layers
        ...
        ... cnn = MyCNN(cont_feats=matrix_feats, vecs=vectors, feats_per_vec=feats_per_vec)
        >>>
        >>> class MyResNet(AbsConv1dHead):
        ...     def get_layers(self, act:str='relu', bn:bool=False, **kargs) -> Tuple[nn.Module, int]:    
        ...         layers = []
        ...         layers.append(self.get_conv1d_block(3, 16, stride=1, kernel_sz=3, act='linear', bn=False))
        ...         layers.append(self.get_conv1d_res_block(16, 16, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_res_block(16, 32, stride=2, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_res_block(32, 32, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(nn.AdaptiveAvgPool1d(1))
        ...         layers = nn.Sequential(*layers)
        ...         return layers
        ...
        ... cnn = MyResNet(cont_feats=matrix_feats, vecs=vectors, feats_per_vec=feats_per_vec)
        >>>
        >>> class MyResNeXt(AbsConv1dHead):
        ...     def get_layers(self, act:str='relu', bn:bool=False, **kargs) -> Tuple[nn.Module, int]:    
        ...         layers = []
        ...         layers.append(self.get_conv1d_block(3, 32, stride=1, kernel_sz=3, act='linear', bn=False))
        ...         layers.append(self.get_conv1d_resNeXt_block(32, 4, 4, 32, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_resNeXt_block(32, 4, 4, 32, stride=2, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(self.get_conv1d_resNeXt_block(32, 4, 4, 32, stride=1, kernel_sz=3, act=act, bn=bn))
        ...         layers.append(nn.AdaptiveAvgPool1d(1))
        ...         layers = nn.Sequential(*layers)
        ...         return layers
        ...
        ... cnn = MyResNeXt(cont_feats=matrix_feats, vecs=vectors, feats_per_vec=feats_per_vec)
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 act:str='relu', bn:bool=False, layer_kargs:Optional[Dict[str,Any]]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, row_wise=False, lookup_init=lookup_init, lookup_act=lookup_act,
                         freeze=freeze, bn_class=bn_class)
        if layer_kargs is None: layer_kargs = {}
        self.layers:nn.Module = self.get_layers(in_c=self.n_fpv, act=act, bn=bn, **layer_kargs)
        self.out_sz = self.check_out_sz()
        if self.freeze: self.freeze_layers()
        self._map_outputs()
            
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))
            
    def check_out_sz(self) -> int:
        r'''
        Automatically computes the output size of the head by passing through random data of the expected shape

        Returns:
            x.size(-1) where x is the outgoing tensor from the head
        '''

        x = torch.rand((1, self.n_fpv,self.n_v))
        training = self.training
        self.eval()
        x = self.forward(x)
        if training: self.train()
        return x.size(-1)
            
    def get_conv1d_block(self, in_c:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1, act:str='relu', bn:bool=False) -> Conv1DBlock:
        r'''
        Wrapper method to build a :class:`~lumin.nn.models.blocks.conv_blocks.ConvBlock` object.

        Arguments:
            in_c: number of input channels (number of features per object / rows in input matrix)
            out_c: number of output channels (number of features / rows in output matrix)
            kernel_sz: width of kernel, i.e. the number of columns to overlay
            padding: amount of padding columns to add at start and end of convolution.
                If left as 'auto', padding will be automatically computed to conserve the number of columns.
            stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
                stride 2 = kernel centred on ever other column and input size halved, et cetera.
            act: string representation of argument to pass to lookup_act
            bn: whether to use batch normalisation (order is weights->activation->batchnorm)

        Returns:
            Instantiated :class:`~lumin.nn.models.blocks.conv_blocks.ConvBlock` object
        '''
        
        return Conv1DBlock(in_c=in_c, out_c=out_c, kernel_sz=kernel_sz, padding=padding, stride=stride, act=act, bn=bn,
                           lookup_act=self.lookup_act, lookup_init=self.lookup_init, bn_class=self.bn_class)
    
    def get_conv1d_res_block(self, in_c:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1,act:str='relu', bn:bool=False) -> Res1DBlock:
        r'''
        Wrapper method to build a :class:`~lumin.nn.models.blocks.conv_blocks.Res1DBlock` object.

        Arguments:
            in_c: number of input channels (number of features per object / rows in input matrix)
            out_c: number of output channels (number of features / rows in output matrix)
            kernel_sz: width of kernel, i.e. the number of columns to overlay
            padding: amount of padding columns to add at start and end of convolution.
                If left as 'auto', padding will be automatically computed to conserve the number of columns.
            stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
                stride 2 = kernel centred on ever other column and input size halved, et cetera.
            act: string representation of argument to pass to lookup_act
            bn: whether to use batch normalisation (order is pre-activation: batchnorm->activation->weights)

        Returns:
            Instantiated :class:`~lumin.nn.models.blocks.conv_blocks.Res1DBlock` object
        '''

        return Res1DBlock(in_c=in_c, out_c=out_c, kernel_sz=kernel_sz, padding=padding, stride=stride, act=act, bn=bn,
                          lookup_act=self.lookup_act, lookup_init=self.lookup_init, bn_class=self.bn_class)
    
    def get_conv1d_resNeXt_block(self, in_c:int, inter_c:int, cardinality:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1,
                                 act:str='relu', bn:bool=False) -> ResNeXt1DBlock:
        r'''
        Wrapper method to build a :class:`~lumin.nn.models.blocks.conv_blocks.ResNeXt1DBlock` object.

        Arguments:
            in_c: number of input channels (number of features per object / rows in input matrix)
            inter_c: number of intermediate channels in groups
            cardinality: number of groups
            out_c: number of output channels (number of features / rows in output matrix)
            kernel_sz: width of kernel, i.e. the number of columns to overlay
            padding: amount of padding columns to add at start and end of convolution.
                If left as 'auto', padding will be automatically computed to conserve the number of columns.
            stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
                stride 2 = kernel centred on ever other column and input size halved, et cetera.
            act: string representation of argument to pass to lookup_act
            bn: whether to use batch normalisation (order is pre-activation: batchnorm->activation->weights)

        Returns:
            Instantiated :class:`~lumin.nn.models.blocks.conv_blocks.ResNeXt1DBlock` object
        '''

        return ResNeXt1DBlock(in_c=in_c, inter_c=inter_c, cardinality=cardinality, out_c=out_c, kernel_sz=kernel_sz, padding=padding, stride=stride, act=act,
                              bn=bn, lookup_act=self.lookup_act, lookup_init=self.lookup_init, bn_class=self.bn_class)
    
    @abstractmethod
    def get_layers(self, in_c:int, act:str='relu', bn:bool=False, **kargs) -> nn.Module:
        r'''
        Abstract function to be overloaded by user. Should return a single torch.nn.Module which accepts the expected input matrix data.
        
        '''
        
        # layers = []
        # layers.append(self.get_conv1d_block(in_c, 16, kernel_sz=7, padding=3, stride=2))
        # ...
        # layers = nn.Sequential(*layers)
        # return layers
        
        pass

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Passes input through the convolutional network.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will conver the data to a matrix
        
        Returns:
            Resulting tensor
        '''

        x = self._process_input(x)
        return self.layers(x).view(x.size(0),-1)
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Width of output representation
        '''
        
        return self.out_sz


class LorentzBoostNet(AbsMatrixHead):
    r'''
    Implementation of the Lorentz Boost Network (https://arxiv.org/abs/1812.09722), which takes 4-momenta for particles and learns new particles and reference
    frames from linear combinations of the original particles, and then boosts the new particles into the learned reference frames. Preset kernel functions are
    the run over the 4-momenta of the boosted particles to compute a set of veriables per particle. These functions can be based on pairs etc. of particles,
    e.g. angles between particles. (`LorentzBoostNet.comb` provides an index iterator over all paris of particles).
    
    A default feature extractor is provided which returns the (px,py,pz,E) of the boosted particles and the cosine angle between every pair of boosted particle.
    This can be overwritten by passing a function to the `feat_extractor` argument during initialisation, or overidding `LorentzBoostNet.feat_extractor`.

    .. Important::
        4-momenta should be supplied without preprocessing, and 4-momenta must be physical (E>=|p|). It is up to the user to ensure this, and not doing so may
        result in errors. A BatchNorm argument (`bn`) is available to preprocess the features extracted from the boosted particles prior to returning them.

    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly in row-wise matrix form.
    Matrices should/will be row-wise: each row is a seperate 4-momenta in the form (px,py,pz,E).
    Matrix elements are expected to be named according to `{particle}_{feature}`, e.g. `photon_E`.
    `vecs` (vectors) should then be a list of particles, i.e. row headers, feature prefixes.
    `feats_per_vec` should be a list of features, i.e. column headers, feature suffixes.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.

    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. row headers, feature prefixes
        feats_per_vec: list of features per object, i.e. column headers, feature suffixes
        n_particles: the number of particles and reference frames to learn
        feat_extractor: if not None, will use the argument as the function to extract features from the 4-momenta of the boosted particles.
        bn: whether batch normalisation should be applied to the extracted features
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
            Purely for inheritance, unused by class as is.
        lookup_act: function taking choice of activation function and returning an activation function layer. Purely for inheritance, unused by class as is.
        freeze: whether to start with module parameters set to untrainable.
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
    
    Examples::
        >>> lbn = LorentzBoostNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs, n_particles=6)
        >>>
        >>> def feat_extractor(x:Tensor) -> Tensor:  # Return masses of boosted particles, x dimensions = [batch,particle,4-mom]
        ...     momenta,energies =  x[:,:,:3], x[:,:,3:]
        ...     mass = torch.sqrt((energies**2)-torch.sum(momenta**2, dim=-1)[:,:,None])
        ...     return mass
        >>> lbn = InteractionNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs, n_particle=6, feat_extractor=feat_extractor)
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 n_particles:int, feat_extractor:Optional[Callable[[Tensor],Tensor]]=None, bn:bool=True,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        super().__init__(cont_feats=cont_feats,vecs=vecs,feats_per_vec=feats_per_vec,row_wise=True,lookup_init=lookup_init,lookup_act=lookup_act,freeze=freeze,
                         bn_class=bn_class)
        self.n_particles = n_particles
        self.comb = torch.combinations(torch.arange(0,self.n_particles))
        if feat_extractor is not None: self.feat_extractor = feat_extractor
        self.part_wgts,self.rf_wgts = self._get_wgts(),self._get_wgts()
        self.out_sz = None
        self.out_sz = self.check_out_sz()
        self.bn = self.bn_class(self.out_sz) if bn else None
        self._map_outputs()
        if self.freeze: self.freeze_layers()
    
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))
            
    def _get_wgts(self) -> nn.Module:
        return nn.Parameter(torch.randn(self.n_particles,1,self.n_v,1)/self.n_particles)
    
    def _get_particles(self, x:Tensor) -> Tensor:
        def _upscale(mat:Tensor, wgts:nn.Parameter) -> Tensor: return torch.transpose(torch.abs(wgts)*mat, 0,1).sum(2)
        
        # Get particles and reference frames
        parts,rfs = _upscale(x, self.part_wgts),_upscale(x, self.rf_wgts)
        parts,rfs = parts.reshape(-1,4),rfs.reshape(-1,4)
        
        # Compute boost vectors to reference frames and gamma factors
        b_vec = -(rfs/(rfs[:,3:4]+1e-10))[:,:3]
        b2 = b_vec.norm(dim=1)**2
        g = 1/(torch.sqrt(1-b2)+1e-10)[:,None]
        g2 = (g-1)/(b2+1e-10)[:,None]
        
        # Boost particles to reference frames
        bp = torch.sum(parts[:,:3]*b_vec, 1)[:,None]
        parts = torch.cat((parts[:,:3]+(g2*bp*b_vec)+(g*b_vec*parts[:,3][:,None]),(parts[:,3:]+bp)*g),1)
        return parts.reshape(-1,self.n_particles, 4)
    
    def feat_extractor(self, x:Tensor) -> Tensor:
        r'''
        Computes features from boosted particle 4-momenta. Incoming tensor `x` contains all 4-momenta for all particles for all datapoints in minibatch.
        Default function returns 4-momenta and cosine angle between all particles.

        Arguments:
            x: 3D incoming tensor with dimensions: [batch, particle, 4-mom (px,py,pz,E)]
        
        Returns:
            2D tensor with dimensions [batch, features]
        '''

        def _cos_delta(x:Tensor) -> Tensor:
            v0,v1 = x[:,:,0,:3],x[:,:,1,:3]
            d = torch.sum(v0*v1, dim=2)
            mag = v0.norm(dim=2)*v1.norm(dim=2)
            return d/mag

        bs = x.size(0)
        out = [x.reshape((bs,-1))]
        out.append(_cos_delta(x[:,self.comb]))
        return torch.cat(out, -1)

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Passes input through the LB network and aggregates down to a flat tensor via the feature extractor, optionally passing through a batchnorm layer.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will convert the data to a matrix
        
        Returns:
            Resulting tensor
        '''
        x = self._process_input(x)
        x = self._get_particles(x)
        x = self.feat_extractor(x)
        if self.out_sz is not None and self.bn is not None: x = self.bn(x)
        return x
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Width of output representation
        '''
        
        return self.out_sz
    
    def check_out_sz(self) -> int:
        r'''
        Automatically computes the output size of the head by passing through random data of the expected shape

        Returns:
            x.size(-1) where x is the outgoing tensor from the head
        '''
        
        x = torch.rand((1,self.n_v, self.n_fpv))
        training = self.training
        self.eval()
        x = self.forward(x)
        if training: self.train()
        return x.size(-1)


class AutoExtractLorentzBoostNet(LorentzBoostNet):
    r'''
    Modified version of :class:`~lumin.nn.models.blocks.head.LorentzBoostNet (implementation of the Lorentz Boost Network (https://arxiv.org/abs/1812.09722)).
    Rather than relying on fixed kernel functions to extract features from the boosted paricles, the functions are learnt during training via neural networks.
    
    Two netrowks are used, one to extract `n_singles` features from each particle and another to extract `n_pairs` features from each pair of particles.

    .. Important::
        4-momenta should be supplied without preprocessing, and 4-momenta must be physical (E>=|p|). It is up to the user to ensure this, and not doing so may
        result in errors. A BatchNorm argument (`bn`) is available to preprocess the 4-momenta of the boosted particles prior to passing them through the neural
        networks

    Incoming data can either be flat, in which case it is reshaped into a matrix, or be supplied directly in row-wise matrix form.
    Matrices should/will be row-wise: each row is a seperate 4-momenta in the form (px,py,pz,E).
    Matrix elements are expected to be named according to `{particle}_{feature}`, e.g. `photon_E`.
    `vecs` (vectors) should then be a list of particles, i.e. row headers, feature prefixes.
    `feats_per_vec` should be a list of features, i.e. column headers, feature suffixes.

    .. Note::
        To allow for the fact that there may be nonexistant features (e.g. z-component of missing energy), `cont_feats` should be a list of all matrix features
        which really do exist (i.e. are present in input data), and be in the same order as the incoming data. Nonexistant features will be set zero.

    Arguments:
        cont_feats: list of all the matrix features which are present in the input data
        vecs: list of objects, i.e. column headers, feature prefixes
        feats_per_vec: list of features per object, i.e. row headers, feature suffixes
        n_particles: the number of particles and reference frames to learn
        depth: the number of hidden layers in each network
        width: the number of neurons per hidden layer
        n_singles: the number of features to extract per individual particle
        n_pairs: the number of features to extract per pair of particles
        act: string representation of argument to pass to lookup_act. Activation should ideally have non-zero outputs to help deal with poorly normalised inputs
        do: dropout rate for use in networks
        bn: whether to use batch normalisation within networks. Inputs are passed through BN regardless of setting.
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer.
        freeze: whether to start with module parameters set to untrainable.
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
    
    Examples::
        >>> aelbn = AutoExtractLorentzBoostNet(cont_feats=matrix_feats, feats_per_vec=feats_per_vec,vecs=vecs, n_particles=6,
                                               depth=3, width=10, n_singles=3, n_pairs=2)
    '''

    def __init__(self, cont_feats:List[str], vecs:List[str], feats_per_vec:List[str],
                 n_particles:int, depth:int, width:int, n_singles:int=0, n_pairs:int=0, act:str='swish', do:float=0, bn:bool=False, 
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d, **kargs):
        self.n_singles,self.n_pairs = n_singles,n_pairs
        
        # Mock NNs to allow out_sz computation
        self.comb = torch.combinations(torch.arange(0,n_particles))
        self.single_nn = lambda x: torch.zeros((x.size(0),self.n_singles*self.n_particles))
        self.pair_nn   = lambda x: torch.zeros((x.size(0),self.n_pairs*len(self.comb)))
        self.pre_bn    = lambda x: x
        
        super().__init__(cont_feats=cont_feats, vecs=vecs, feats_per_vec=feats_per_vec, n_particles=n_particles,
                         bn=False, lookup_init=lookup_init, lookup_act=lookup_act, freeze=freeze, bn_class=bn_class)
        
        if n_singles > 0: self.single_nn = self._get_nn(n_in=4, depth=depth, width=width,n_out=n_singles, act=act, do=do, bn=bn,
                                                        lookup_act=lookup_act, lookup_init=lookup_init)
        if n_pairs   > 0: self.pair_nn   = self._get_nn(n_in=8, depth=depth, width=width, n_out=n_pairs, act=act, do=do, bn=bn,
                                                        lookup_act=lookup_act, lookup_init=lookup_init)
        self.pre_bn = self.bn_class(4*self.n_particles)
    
    def _get_nn(self, n_in:int, depth:int, width:int, n_out:int, act:str, do:bool, bn:bool,
                lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]],
                lookup_act:Callable[[str],Any]) -> nn.Sequential:
        return nn.Sequential(*[self._get_layer(n_in=n_in if i == 0 else width, n_out=width if i+1 < depth else n_out,
                                               act=act, do=do, bn=bn, lookup_init=lookup_init, lookup_act=lookup_act)
                               for i in range(depth)])
    
    def _get_layer(self, n_in:int, n_out:int, act:str, do:bool, bn:bool,
                   lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]], lookup_act:Callable[[str],Any]) -> nn.Sequential:   
        layers = []
        layers.append(nn.Linear(n_in, n_out))
        lookup_init(act, n_in, n_out)(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if act != 'linear': layers.append(lookup_act(act))
        if bn:  layers.append(LCBatchNorm1d(nn.BatchNorm1d(n_out)))
        if do: 
            if act == 'selu': layers.append(nn.AlphaDropout(do))
            else:             layers.append(nn.Dropout(do))
        return nn.Sequential(*layers)
    
    def feat_extractor(self, x:Tensor) -> Tensor:
        r'''
        Computes features from boosted particle 4-momenta. Incoming tensor `x` contains all 4-momenta for all particles for all datapoints in minibatch.
        `single_nn` broadcast to all boosted particles, and `pair_nn` broadcast to all paris of particles. Returned features are concatenated together.

        Arguments:
            x: 3D incoming tensor with dimensions: [batch, particle, 4-mom (px,py,pz,E)]
        
        Returns:
            2D tensor with dimensions [batch, features]
        '''

        bs = x.size(0)
        x = self.pre_bn(x.reshape((bs,-1))).reshape((bs,-1, 4))
        out = []
        if self.n_singles > 0: out.append(self.single_nn(x).reshape((bs,-1)))
        if self.n_pairs   > 0: out.append(self.pair_nn(x[:,self.comb].reshape(bs, len(self.comb), 8)).reshape((bs, -1)))
        if len(out) == 1: out = out[0]
        else:             out = torch.cat(out, -1)
        return out
