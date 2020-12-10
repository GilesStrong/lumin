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


class FitParams():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.epoch,self.sub_epoch = 0,0


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
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]) -> float: pass

    @abstractmethod   
    def predict(self, inputs, as_np:bool=True) -> Union[np.ndarray, Tensor]: pass


class AbsModel(ABC):
    '''Abstract model class for typing'''
    def __init__(self): pass
    
    @abstractmethod
    def fit(self, n_epochs:int, fy:FoldYielder, val_idx:int, bs:int, bulk_move:bool, train_on_weights:bool,
            cbs:Optional[Union[AbsCallback,List[AbsCallback]]], opt:Optional[Callable[[Generator],optim.Optimizer]],
            loss:Optional[Callable[[],Callable[[Tensor,Tensor],Tensor]]], mask_inputs:bool) -> List[AbsCallback]: pass

    @abstractmethod   
    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool, pred_name:str,
                mask_inputs:bool, pred_cb:AbsCallback, cbs:Optional[List[AbsCallback]], bs:Optional[int]) \
        -> Union[np.ndarray, Tensor, None]: pass
