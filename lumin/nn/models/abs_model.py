import numpy as np
from typing import List, Optional, Union
from abc import ABC, abstractmethod

from torch.tensor import Tensor

from ..data.batch_yielder import BatchYielder
from ..callbacks.abs_callback import AbsCallback

__all__ = []


class AbsModel(ABC):
    '''Abstract model class for typing'''
    def __init__(self): pass
    
    @abstractmethod
    def fit(self, batch_yielder:BatchYielder, callbacks:List[AbsCallback]) -> float: pass
    
    @abstractmethod
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]=None) -> float: pass

    @abstractmethod   
    def predict(self, inputs, as_np:bool=True) -> Union[np.ndarray, Tensor]: pass
