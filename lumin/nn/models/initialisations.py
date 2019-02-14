import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple

from torch import Tensor
import torch.nn as nn


def lookup_init(act:str, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> Tuple[Callable[[Tensor, str],None],Dict[str,Any]]:
    '''Lookup weight initialisation for activation functions'''
    if act == 'relu':       return nn.init.kaiming_normal_, {'nonlinearity': 'relu'}
    if act == 'selu':       return nn.init.normal_,         {'std': 1/np.sqrt(fan_in)}
    if act == 'sigmoid':    return nn.init.xavier_normal_,  {}
    if act == 'logsoftmax': return nn.init.xavier_normal_,  {}
    if act == 'softmax':    return nn.init.xavier_normal_,  {}
    if act == 'linear':     return nn.init.xavier_normal_,  {}
    if 'swish' in act:      return nn.init.kaiming_normal_, {'nonlinearity': 'relu'}
    raise ValueError("Activation not implemented")
        