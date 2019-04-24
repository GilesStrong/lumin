from typing import Union, Tuple

from torch import Tensor

from .callback import Callback
from ..data.batch_yielder import BatchYielder


class BinaryLabelSmooth(Callback):
    '''Apply label smoothing to binary classes, based on arXiv:1512.00567'''
    def __init__(self, coefs:Union[float,Tuple[float,float]]=0):
        self.coefs = coefs if isinstance(coefs, tuple) else (coefs, coefs)
    
    def on_epoch_begin(self, batch_yielder:BatchYielder, **kargs) -> None:
        '''Apply smoothing at train-time'''
        batch_yielder.targets = batch_yielder.targets.astype(float)
        batch_yielder.targets[batch_yielder.targets == 0] = self.coefs[0]
        batch_yielder.targets[batch_yielder.targets == 1] = 1-self.coefs[1]
         
    def on_eval_begin(self, targets:Tensor, **kargs) -> None:
        '''Apply smoothing at test-time'''
        targets[targets == 0] = self.coefs[0]
        targets[targets == 1] = 1-self.coefs[1]
