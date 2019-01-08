import numpy as np
from typing import List, Optional

from torch.tensor import Tensor


'''
Todo
- Improve this
'''


class BatchYielder:
    def __init__(self, inputs:np.ndarray, targets:np.ndarray, bs:int, weights:Optional[np.ndarray]=None, shuffle=True, use_weights:bool=True):
        self.inputs,self.targets,self.weights,self.bs,self.shuffle,self.use_weights = inputs,targets,weights,bs,shuffle,use_weights

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
            if self.weights is not None and self.use_weights:
                yield Tensor(inputs[i:i+self.bs]), Tensor(targets[i:i+self.bs]), Tensor(weights[i:i+self.bs])
            else:
                yield Tensor(inputs[i:i+self.bs]), Tensor(targets[i:i+self.bs]), None

    def __len__(self):
        return len(self.inputs)//self.bs
