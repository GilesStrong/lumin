import numpy as np
from typing import Optional, Union, Tuple, Callable

from ..initialisations import lookup_normal_init
from ....utils.misc import to_device

from torch.tensor import Tensor
import torch.nn as nn


class ClassRegMulti(nn.Module):
    '''Output block for (multi(class/label)) classification or regression tasks'''
    def __init__(self, n_in:int, n_out:int, objective:str, y_range:Optional[Union[Tuple,np.ndarray]]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init):
        super().__init__()
        self.n_in,self.n_out,self.objective,self.y_range,self.lookup_init = n_in,n_out,objective,y_range,lookup_init
        if self.y_range is not None:
            if not isinstance(self.y_range, np.ndarray): self.y_range = np.array(self.y_range)
            self.y_min = np.array(np.min(self.y_range, axis=-1))
            self.y_diff = np.abs(self.y_range.take([1], axis=-1)-self.y_range.take([0], axis=-1)).ravel()
            self.y_min, self.y_diff = to_device(Tensor(self.y_min)), to_device(Tensor(self.y_diff))
        self.build_layers()

    def __getitem__(self, key:int) -> nn.Module:
        if key == 0: return self.dense
        if key == 1: return self.act
        raise IndexError(f'Index {key} out of range')
        
    def build_layers(self) -> None:
        self.dense = nn.Linear(self.n_in, self.n_out)
        if 'class' in self.objective:
            if 'multiclass' in self.objective: 
                self.act = nn.LogSoftmax(1)
                init = self.lookup_init('softmax', self.n_in, self.n_out)
            else:
                self.act = nn.Sigmoid()
                init = self.lookup_init('sigmoid', self.n_in, self.n_out)
        else:
            if self.y_range is None:
                self.act = lambda x: x
                init = self.lookup_init('linear', self.n_in, self.n_out)
            else:
                self.act = nn.Sigmoid()
                init = self.lookup_init('sigmoid', self.n_in, self.n_out)
        init(self.dense.weight)
        
    def forward(self, x:Tensor) -> Tensor:
        x = self.dense(x)
        x = self.act(x)
        if self.y_range is not None: x = (self.y_diff*x)+self.y_min
        return x
        
    def get_out_size(self) -> int: return self.n_out
