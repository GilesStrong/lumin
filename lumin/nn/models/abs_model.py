import numpy as np
import pandas as pd
from typing import List, Optional, Union, Generator, Callable
from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch import optim

from ..data.fold_yielder import FoldYielder
from ..callbacks.abs_callback import AbsCallback

__all__ = []


class FitParams():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.epoch,self.sub_epoch = 0,0


class AbsModel(metaclass=ABCMeta):
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
