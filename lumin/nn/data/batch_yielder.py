import numpy as np
from typing import List, Optional

from ...utils.misc import to_device

from torch.tensor import Tensor


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

    def __init__(self, inputs:np.ndarray, targets:np.ndarray, bs:int, objective:str,
                 weights:Optional[np.ndarray]=None, shuffle=True, use_weights:bool=True, bulk_move=True):
        self.inputs,self.targets,self.weights,self.bs,self.objective,self.shuffle,self.use_weights,self.bulk_move = \
            inputs,targets,weights,bs,objective,shuffle,use_weights,bulk_move

    def __iter__(self) -> List[Tensor]:
        r'''
        Iterate through data in batches.

        Returns:
            tuple of batches of inputs, targets, and weights as tensors on device
        '''

        if self.shuffle:
            if self.weights is not None and self.use_weights:
                data = list(zip(self.inputs, self.targets, self.weights))
                np.random.shuffle(data)
                inputs, targets, weights = zip(*data)
            else:
                data = list(zip(self.inputs, self.targets))
                np.random.shuffle(data)
                inputs, targets = zip(*data)
        else:
            inputs, targets, weights = self.inputs, self.targets, self.weights

        if self.bulk_move:
            inputs = to_device(Tensor(inputs))
            if 'multiclass' in self.objective: targets = to_device(Tensor(targets).long().squeeze())
            else:                              targets = to_device(Tensor(targets))
            if self.weights is not None and self.use_weights: weights = to_device(Tensor(weights))
            else:                                             weights = None
            
            for i in range(0, len(inputs)-self.bs+1, self.bs):
                if weights is None: yield inputs[i:i+self.bs], targets[i:i+self.bs], None
                else:               yield inputs[i:i+self.bs], targets[i:i+self.bs], weights[i:i+self.bs]                    

        else:
            for i in range(0, len(inputs)-self.bs+1, self.bs):
                if 'multiclass' in self.objective: y = Tensor(targets[i:i+self.bs]).long().squeeze()
                else:                              y = Tensor(targets[i:i+self.bs])
                if self.weights is not None and self.use_weights:
                    yield to_device(Tensor(inputs[i:i+self.bs])), to_device(y), to_device(Tensor(weights[i:i+self.bs]))
                else:
                    yield to_device(Tensor(inputs[i:i+self.bs])), to_device(y), None

    def __len__(self): return len(self.inputs)//self.bs
