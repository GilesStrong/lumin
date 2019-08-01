import numpy as np
from typing import Optional, Union, Tuple, Callable

from ..initialisations import lookup_normal_init
from ....utils.misc import to_device
from .abs_block import AbsBlock

from torch.tensor import Tensor
import torch.nn as nn


class AbsTail(AbsBlock):
    def __init__(self, n_in:int, n_out:int, objective:str, bias_init:Optional[float]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__(lookup_init=lookup_init, freeze=freeze)
        self.n_in,self.n_out,self.objective,self.bias_init = n_in,n_out,objective,bias_init


class ClassRegMulti(AbsTail):
    r'''
    Output block for (multi(class/label)) classification or regression tasks.
    Designed to be passed as a 'tail' to :class:ModelBuilder.
    Takes output size of network body and scales it to required number of outputs.
    For regression tasks, y_range can be set with per-output minima and maxima. The outputs are then adjusted according to ((y_max-y_min)*x)+self.y_min, where x
    is the output of the network passed through a sigmoid function. Effectively allowing regression to be performed without normalising and standardising the
    target values. Note it is safest to allow some leaway in setting the min and max, e.g. max = 1.2*max, min = 0.8*min 
    Output activation function is automatically set according to objective and y_range.

    Arguments:
        n_in: number of inputs to expect
        n_out: number of outputs required
        objective: string representation of network objective, i.e. 'classification', 'regression', 'multiclass'
        y_range: if not None, will apply rescaling to network outputs.
        lookup_init: function taking string representation of activation function, number of inputs, and number of outputs an returning a function to initialise
            layer weights.

    Examples::
        >>> tail = ClassRegMulti(n_in=100, n_out=1, objective='classification')
        >>> tail = ClassRegMulti(n_in=100, n_out=5, objective='multiclass')
        >>> y_range = (0.8*targets.min(), 1.2*targets.max())
            tail = ClassRegMulti(n_in=100, n_out=1, objective='regression', y_range=y_range)
        >>> min_targs = np.min(targets, axis=0).reshape(targets.shape[1],1)
            max_targs = np.max(targets, axis=0).reshape(targets.shape[1],1)
            min_targs[min_targs > 0] *=0.8
            min_targs[min_targs < 0] *=1.2
            max_targs[max_targs > 0] *=1.2
            max_targs[max_targs < 0] *=0.8
            y_range = np.hstack((min_targs, max_targs))
            tail = ClassRegMulti(n_in=100, n_out=6, objective='regression', y_range=y_range, lookup_init=lookup_uniform_init)
    '''

    # TODO: Automate y_range calculation with adjustable leeway

    def __init__(self, n_in:int, n_out:int, objective:str, y_range:Optional[Union[Tuple,np.ndarray]]=None, bias_init:Optional[float]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        super().__init__(n_in=n_in, n_out=n_out, objective=objective, bias_init=bias_init, lookup_init=lookup_init, freeze=freeze)
        self.y_range = y_range
        if self.y_range is not None:
            if not isinstance(self.y_range, np.ndarray): self.y_range = np.array(self.y_range)
            self.y_min = np.array(np.min(self.y_range, axis=-1))
            self.y_diff = np.abs(self.y_range.take([1], axis=-1)-self.y_range.take([0], axis=-1)).ravel()
            self.y_min, self.y_diff = to_device(Tensor(self.y_min)), to_device(Tensor(self.y_diff))
        self._build_layers()
        if self.freeze: self.freeze_layers

    def __getitem__(self, key:int) -> nn.Module:
        if key == 0: return self.dense
        if key == 1: return self.act
        raise IndexError(f'Index {key} out of range')
        
    def _build_layers(self) -> None:
        self.dense = nn.Linear(self.n_in, self.n_out)
        if 'class' in self.objective:
            if 'multiclass' in self.objective: 
                self.act = nn.LogSoftmax(1)
                init = self.lookup_init('softmax', self.n_in, self.n_out)
                bias = 1/self.n_out if self.bias_init is None else self.bias_init
            else:
                self.act = nn.Sigmoid()
                init = self.lookup_init('sigmoid', self.n_in, self.n_out)
                bias = 0.5 if self.bias_init is None else self.bias_init
        else:
            if self.y_range is None:
                self.act = lambda x: x
                init = self.lookup_init('linear', self.n_in, self.n_out)
                bias = 0 if self.bias_init is None else self.bias_init
            else:
                self.act = nn.Sigmoid()
                init = self.lookup_init('sigmoid', self.n_in, self.n_out)
                bias = 0.5 if self.bias_init is None else self.bias_init
        init(self.dense.weight)
        nn.init.constant_(self.dense.bias, val=bias)

    def forward(self, x:Tensor) -> Tensor:
        x = self.dense(x)
        x = self.act(x)
        if self.y_range is not None: x = (self.y_diff*x)+self.y_min
        return x
        
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''
        
        return self.n_out
