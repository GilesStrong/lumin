import numpy as np
import pandas as pd
from typing import Union
from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from ....utils.misc import to_np


class AbsEndcap(nn.Module):
    '''Abstract class for constructing post training layer which performs furtehr calculation on NN outputs.
    Used when NN was trained to some proxy objective'''
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
    
    @abstractmethod
    def func(self, x:Tensor) -> Tensor: pass
        
    def forward(self, x:Tensor) -> Tensor: return self.func(x)
    
    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor], as_np:bool=True) -> Union[np.ndarray, Tensor]:
        x = self.model.predict(inputs, as_np=False)
        x = self.func(x)
        return to_np(x) if as_np else x
