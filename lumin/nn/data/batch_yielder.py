import numpy as np
from typing import List, Optional

from ...utils.misc import to_device

from torch.tensor import Tensor


'''
Todo
- Improve this/change to dataloader
'''


class BatchYielder:
    '''Yields minibatches to model during training. Default mode is to dump all data on GPU.
    Switch off `bulk_move` if memory capacity is an issue or split data into more folds'''
    def __init__(self, inputs:np.ndarray, targets:np.ndarray, bs:int, objective:str,
                 weights:Optional[np.ndarray]=None, shuffle=True, use_weights:bool=True, bulk_move=True):
        self.inputs,self.targets,self.weights,self.bs,self.objective,self.shuffle,self.use_weights,self.bulk_move = \
            inputs,targets,weights,bs,objective,shuffle,use_weights,bulk_move

    def __iter__(self) -> List[Tensor]:
        if self.shuffle:
            if self.weights is not None and self.use_weights:
                data = list(zip(self.inputs, self.targets, self.weights))
                np.random.shuffle(data)
                inputs, targets, weights = zip(*data)
            else:
                data = list(zip(self.inputs, self.targets))
                np.random.shuffle(data)
                inputs, targets = zip(*data)

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
