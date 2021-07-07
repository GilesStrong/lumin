import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
import math

from ...utils.misc import to_device

from torch import Tensor

__all__ = ['BatchYielder']


'''
Todo
- Improve this/change to dataloader
'''


class BatchYielder:
    r'''
    Yields minibatches to model during training. Iteration provides one minibatch as tuple of tensors of inputs, targets, and weights.
    
    Arguments:
        inputs: input array for (sub-)epoch
        targets: target array for (sub-)epoch
        bs: batchsize, number of data to include per minibatch
        objective: 'classification', 'multiclass classification', or 'regression'. Used for casting target dtype.
        weights: Optional weight array for (sub-)epoch
        shuffle: whether to shuffle the data at the beginning of an iteration
        use_weights: if passed weights, whether to actually pass them to the model
        bulk_move: whether to move all data to device at once. Default is true (saves time), but if device has low memory you can set to False.
        input_mask: optionally only use Boolean-masked inputs
        drop_last: whether to drop the last batch if it does not contain `bs` elements
    '''

    def __init__(self, inputs:Union[np.ndarray,Tuple[np.ndarray,np.ndarray]], bs:int, objective:str, targets:Optional[np.ndarray]=None,
                 weights:Optional[np.ndarray]=None, shuffle:bool=True, use_weights:bool=True, bulk_move:bool=True, input_mask:Optional[np.ndarray]=None,
                 drop_last:bool=True):
        self.inputs,self.targets,self.weights,self.bs,self.objective,self.shuffle,self.use_weights,self.bulk_move,self.input_mask,self.drop_last = \
            inputs,targets,weights,bs,objective,shuffle,use_weights,bulk_move,input_mask,drop_last
        if isinstance(self.inputs, tuple): self.inputs,self.matrix_inputs = self.inputs
        else:                                          self.matrix_inputs = None
        if isinstance(self.inputs, pd.DataFrame): self.inputs = self.inputs.values
        if self.input_mask is not None: self.inputs = self.inputs[:,self.input_mask]

    def __iter__(self) -> List[Tensor]:
        r'''
        Iterate through data in batches.

        Returns:
            tuple of batches of inputs, targets, and weights as tensors on device
        '''

        full_idxs = np.arange(len(self.inputs))
        upper = len(full_idxs)
        if self.drop_last: upper -= self.bs-1
        if self.shuffle: np.random.shuffle(full_idxs)

        if self.bulk_move:
            inputs = to_device(Tensor(self.inputs))
            if self.targets is not None:
                if 'multiclass' in self.objective: targets = to_device(Tensor(self.targets).long().squeeze())
                else:                              targets = to_device(Tensor(self.targets))
            if self.weights is not None and self.use_weights: weights = to_device(Tensor(self.weights))
            else:                                             weights = None
            if self.matrix_inputs is not None: matrix_inputs = to_device(Tensor(self.matrix_inputs))
            else:                                              matrix_inputs = None

            for i in range(0, upper, self.bs):
                idxs = full_idxs[i:i+self.bs]
                x = inputs[idxs] if matrix_inputs is None else (inputs[idxs],matrix_inputs[idxs])
                y = None if self.targets is None else targets[idxs]
                w = None if weights is None else weights[idxs]
                yield x, y, w              

        else:
            for i in range(0, upper, self.bs):
                idxs = full_idxs[i:i+self.bs]
                if self.targets is not None:
                    if 'multiclass' in self.objective: y = to_device(Tensor(self.targets[idxs]).long().squeeze())
                    else:                              y = to_device(Tensor(self.targets[idxs]))
                else:
                    y = None
                if self.matrix_inputs is None: x =  to_device(Tensor(self.inputs[idxs]))
                else:                          x = (to_device(Tensor(self.inputs[idxs])),to_device(Tensor(self.matrix_inputs[idxs])))
                w = to_device(Tensor(self.weights[idxs])) if self.weights is not None and self.use_weights else None
                yield x, y, w

    def __len__(self): return len(self.inputs)//self.bs if self.drop_last else math.ceil(len(self.inputs)/self.bs)

    def get_inputs(self, on_device:bool=False) -> Union[Tensor, Tuple[Tensor,Tensor]]:
        if on_device:
            if self.matrix_inputs is None: return to_device(Tensor(self.inputs))
            else:                          return (to_device(Tensor(self.inputs)), to_device(Tensor(self.matrix_inputs)))
        else:
            if self.matrix_inputs is None: return self.inputs
            else:                          return (self.inputs, self.matrix_inputs)
