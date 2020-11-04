import numpy as np
import pandas as pd
from typing import List, Optional, Union, Generator, Callable
from abc import ABC, abstractmethod

from torch.tensor import Tensor
from torch import optim

from ..data.batch_yielder import BatchYielder
from ..data.fold_yielder import FoldYielder
from ..callbacks.abs_callback import AbsCallback

__all__ = []


class OldAbsModel(ABC):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.models.abs_model.AbsModel`.
        It is a copy of the old `AbsModel` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self): pass
    
    @abstractmethod
    def fit(self, batch_yielder:BatchYielder, callbacks:List[AbsCallback]) -> float: pass
    
    @abstractmethod
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]=None) -> float: pass

    @abstractmethod   
    def predict(self, inputs, as_np:bool=True) -> Union[np.ndarray, Tensor]: pass


class AbsModel(ABC):
    '''Abstract model class for typing'''
    def __init__(self): pass
    
    @abstractmethod
    def fit(self, n_epochs:int, fy:FoldYielder, val_idx:int, bs:int, bulk_move:bool=True, train_on_weights:bool=True,
            cbs:Optional[Union[AbsCallback,List[AbsCallback]]]=None, opt:Optional[Callable[[Generator],optim.Optimizer]]=None,
            loss:Optional[Callable[[],Callable[[Tensor,Tensor],Tensor]]]=None, mask_inputs:bool=True) -> List[AbsCallback]: pass

    @abstractmethod   
    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool=True, pred_name:str='pred',
                mask_inputs:bool=True, pred_cb:PredHandler=PredHandler(), cbs:Optional[List[AbsCallback]]=None, bs:Optional[int]=None) \
        -> Union[np.ndarray, Tensor, None]: pass
