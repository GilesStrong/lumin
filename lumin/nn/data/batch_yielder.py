import numpy as np
from typing import List, Optional, Union, Tuple

from ...utils.misc import to_device

from torch.tensor import Tensor

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
        targets: targte array for (sub-)epoch
        bs: batchsize, number of data to include per minibatch
        objective: 'classification', 'multiclass classification', or 'regression'. Used for casting target dtype.
        weights: Optional weight array for (sub-)epoch
        shuffle: whether to shuffle the data at the beginning of an iteration
        use_weights: if passed weights, whether to actually pass them to the model
        bulk_move: whether to move all data to device at once. Default is true (saves time), but if device has low memory you can set to False.
    '''

    def __init__(self, inputs:Union[np.ndarray,Tuple[np.ndarray,np.ndarray]], targets:np.ndarray, bs:int, objective:str,
                 weights:Optional[np.ndarray]=None, shuffle:bool=True, use_weights:bool=True, bulk_move:bool=True):
        self.inputs,self.targets,self.weights,self.bs,self.objective,self.shuffle,self.use_weights,self.bulk_move,self.matrix_inputs = \
            inputs,targets,weights,bs,objective,shuffle,use_weights,bulk_move,None
        if isinstance(self.inputs, tuple): self.inputs,self.matrix_inputs = self.inputs

    def __iter__(self) -> List[Tensor]:
        r'''
        Iterate through data in batches.

        Returns:
            tuple of batches of inputs, targets, and weights as tensors on device
        '''

        if self.shuffle:
            if self.weights is not None and self.use_weights:
                if self.matrix_inputs is not None:
                    data = list(zip(self.inputs, self.targets, self.weights, self.matrix_inputs))
                    np.random.shuffle(data)
                    inputs, targets, weights, matrix_inputs = zip(*data)
                else:
                    data = list(zip(self.inputs, self.targets, self.weights))
                    np.random.shuffle(data)
                    inputs, targets, weights = zip(*data)
            else:
                if self.matrix_inputs is not None:
                    data = list(zip(self.inputs, self.targets, self.matrix_inputs))
                    np.random.shuffle(data)
                    inputs, targets, matrix_inputs = zip(*data)
                else:
                    data = list(zip(self.inputs, self.targets))
                    np.random.shuffle(data)
                    inputs, targets = zip(*data)
        else:
            inputs, targets, weights, matrix_inputs = self.inputs, self.targets, self.weights, self.matrix_inputs

        if self.bulk_move:
            inputs = to_device(Tensor(inputs))
            if 'multiclass' in self.objective: targets = to_device(Tensor(targets).long().squeeze())
            else:                              targets = to_device(Tensor(targets))
            if self.weights is not None and self.use_weights: weights = to_device(Tensor(weights))
            else:                                             weights = None
            if self.matrix_inputs is not None: matrix_inputs = to_device(Tensor(matrix_inputs))
            else:                                              matrix_inputs = None

            for i in range(0, len(inputs)-self.bs+1, self.bs):
                x = inputs[i:i+self.bs] if matrix_inputs is None else (inputs[i:i+self.bs],matrix_inputs[i:i+self.bs])
                w = None if weights is None else weights[i:i+self.bs]
                yield x, targets[i:i+self.bs], w              

        else:
            for i in range(0, len(inputs)-self.bs+1, self.bs):
                if 'multiclass' in self.objective: y = to_device(Tensor(targets[i:i+self.bs]).long().squeeze())
                else:                              y = to_device(Tensor(targets[i:i+self.bs]))
                if self.matrix_inputs is None: x = to_device(Tensor(inputs[i:i+self.bs]))
                else:                     x = (to_device(Tensor(inputs[i:i+self.bs])),to_device(Tensor(matrix_inputs[i:i+self.bs])))
                w = to_device(Tensor(weights[i:i+self.bs])) if self.weights is not None and self.use_weights else None
                yield x, y, w

    def __len__(self): return len(self.inputs)//self.bs

    def get_inputs(self, on_device:bool=False) -> Union[Tensor, Tuple[Tensor,Tensor]]:
        if on_device:
            if self.matrix_inputs is None: return to_device(Tensor(self.inputs))
            else:                          return (to_device(Tensor(self.inputs)), to_device(Tensor(self.matrix_inputs)))
        else:
            if self.matrix_inputs is None: return self.inputs
            else:                          return (self.inputs, self.matrix_inputs)
