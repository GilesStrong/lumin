from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import warnings

from ..data.fold_yielder import FoldYielder

__all__ = ['CatEmbedder']


class CatEmbedder():
    r'''
    Helper class for embedding categorical features. Designed to be passed to :class:`~lumin.nn.models.model_builder.ModelBuilder`.
    Note that the classmethod :meth:`~lumin.nn.models.helpers.CatEmbedder.from_fy` may be used to instantiate an :class:`~lumin.nn.models.helpers.CatEmbedder`
    from a :class:`~lumin.nn.data.fold_yielder.FoldYielder`.

    Arguments:
        cat_names: list of names of catgorical features in order in which they will be passed as inputs columns
        cat_szs: list of cardinalities (number of unique elements) for each feature
        emb_szs: Optional list of embedding sizes for each feature.  If None, will use min(max_emb_sz, (1+sz)//2)
        max_emb_sz: Maximum size of embedding if emb_szs is None
        emb_load_path: if not None, will cause :class:`~lumin.nn.models.model_builder.ModelBuilder` to attempt to load pretrained embeddings from path

    Examples::
        >>> cat_embedder = CatEmbedder(cat_names=['n_jets', 'channel'],
                                       cat_szs=[5, 3])
        >>>
        >>> cat_embedder = CatEmbedder(cat_names=['n_jets', 'channel'],
                                       cat_szs=[5, 3], emb_szs=[2, 2])
        >>>
        >>> cat_embedder = CatEmbedder(cat_names=['n_jets', 'channel'],
                                       cat_szs=[5, 3], emb_szs=[2, 2], 
                                       emb_load_path=Path('weights'))
    '''

    # TODO: load pretrained embeddings to check sizes

    def __init__(self, cat_names:List[str], cat_szs:List[int], emb_szs:Optional[List[int]]=None, max_emb_sz:int=50,
                 emb_load_path:Optional[Union[Path,str]]=None):
        assert len(cat_names) == len(cat_szs), "Different number of feature names and feature cardinalities received"
        if emb_szs is not None: assert len(cat_szs) == len(emb_szs), "Different number of features and embedding sizes received"
        self.cat_names,self.cat_szs,self.emb_szs,self.max_emb_sz,self.emb_load_path = cat_names,cat_szs,emb_szs,max_emb_sz,emb_load_path
        if self.emb_szs is None: self.calc_emb_szs()
        if self.emb_load_path is not None and not isinstance(self.emb_load_path, Path): self.emb_load_path = Path(self.emb_load_path)
        self.n_cat_in = len(self.cat_szs)
        
    def __repr__(self) -> str:
        rep = ""
        for i in range(self.n_cat_in): rep += f'{self.cat_names[i]}:\t{self.cat_szs[i]} --> {self.emb_szs[i]}\n'
        if self.emb_load_path is not None: rep += f'\nLoading pretrained embeddings from: {self.emb_load_path}'
        return rep
    
    def __getitem__(self, key:Union[int,str]) -> Tuple[int,int]:
        if isinstance(key, int): return (self.cat_szs[key], self.emb_szs[key])
        else:                    return self[self.cat_names.index(key)]
    
    def __iter__(self) -> Tuple[int,int]:
        for name, sz, emb_sz in zip(self.cat_names, self.cat_szs, self.emb_szs): yield name, sz, emb_sz
            
    @classmethod
    def from_fy(cls, fy:FoldYielder,  emb_szs:Optional[List[int]]=None, max_emb_sz:int=50, emb_load_path:Optional[Union[Path,str]]=None):
        r'''
        Instantiate an :class:`~lumin.nn.models.helpers.CatEmbedder` from a :class:`~lumin.nn.data.fold_yielder.FoldYielder`, i.e. avoid having to pass
        cat_names and cat_szs.
        
        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` with training data
            emb_szs: Optional list of embedding sizes for each feature.  If None, will use min(max_emb_sz, (1+sz)//2)
            max_emb_sz: Maximum size of embedding if emb_szs is None
            emb_load_path: if not None, will cause :class:`~lumin.nn.models.model_builder.ModelBuilder` to attempt to load pretrained embeddings from path

        Returns:
            :class:`~lumin.nn.models.helpers.CatEmbedder`

        Examples::
            >>> cat_embedder = CatEmbedder.from_fy(train_fy)
            >>>
            >>> cat_embedder = CatEmbedder.from_fy(train_fy, emb_szs=[2, 2])
            >>>
            >>> cat_embedder = CatEmbedder.from_fy(
                    train_fy, emb_szs=[2, 2],
                    emb_load_path=Path('weights'))
        '''

        cat_names = fy.get_use_cat_feats()
        cat_szs = None
        # Get cardinalities
        for fld_id in range(len(fy)):
            tmp_max = fy.get_df(pred_name=None, targ_name=None, wgt_name=None, n_folds=1, fold_idx=fld_id, inc_inputs=True,
                                verbose=False, suppress_warn=True)[cat_names].max().values.astype(int)
            if cat_szs is None: cat_szs = tmp_max
            else:               cat_szs = np.maximum(cat_szs, tmp_max)
        cat_szs = list(1+cat_szs)  # zero-ordered, therefore cardinality is 1+max
        return cls(cat_names=cat_names, cat_szs=cat_szs, emb_szs=emb_szs, max_emb_sz=max_emb_sz, emb_load_path=emb_load_path)
            
    def calc_emb_szs(self) -> None:
        r'''
        Method used to set sizes of embeddings for each categorical feature when no embedding sizes are explicitly passed
        Uses rule of thumb of min(50, (1+cardinality)/2)
        '''
        
        self.emb_szs = [min(self.max_emb_sz, (1+sz)//2) for sz in self.cat_szs]
