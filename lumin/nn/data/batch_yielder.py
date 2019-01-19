import numpy as np
from typing import List, Optional

from torch.tensor import Tensor


'''
Todo
- Improve this
'''


class BatchYielder:
    def __init__(self, inputs:np.ndarray, targets:np.ndarray, bs:int, objective:str, weights:Optional[np.ndarray]=None, shuffle=True, use_weights:bool=True):
        self.inputs,self.targets,self.weights,self.bs,self.objective,self.shuffle,self.use_weights = inputs,targets,weights,bs,objective,shuffle,use_weights

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

        for i in range(0, len(inputs)-self.bs+1, self.bs):
            if 'multiclass' in self.objective:
                y = Tensor(targets[i:i+self.bs]).long().squeeze()
            else:
                y = Tensor(targets[i:i+self.bs])
            if self.weights is not None and self.use_weights:
                if 'multiclass' in self.objective:
                    w = Tensor(np.mean(weights[i:i+self.bs], axis=0))
                else:
                    w = Tensor(weights[i:i+self.bs])
                yield Tensor(inputs[i:i+self.bs]), y, w
            else:
                yield Tensor(inputs[i:i+self.bs]), y, None

    def __len__(self):
        return len(self.inputs)//self.bs
