import numpy as np
from typing import Optional, Callable
from functools import partial

from torch import Tensor
import torch.nn as nn


def lookup_normal_init(act:str, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> Callable[[Tensor],None]:
    r'''
    Lookup for weight initialisation using Normal distributions

    Arguments:
        act: string representation of activation function
        fan_in: number of inputs to neuron
        fan_out: number of outputs from neuron
    
    Returns:
        Callable to initialise weight tensor
    '''

    if act == 'relu':       return partial(nn.init.kaiming_normal_, nonlinearity='relu')
    if act == 'prelu':      return partial(nn.init.kaiming_normal_, nonlinearity='relu')
    if act == 'selu':       return partial(nn.init.normal_, std=1/np.sqrt(fan_in))
    if act == 'sigmoid':    return nn.init.xavier_normal_
    if act == 'logsoftmax': return nn.init.xavier_normal_
    if act == 'softmax':    return nn.init.xavier_normal_
    if act == 'linear':     return nn.init.xavier_normal_
    if 'swish' in act:      return partial(nn.init.kaiming_normal_, nonlinearity='relu')
    raise ValueError("Activation not implemented")


def lookup_uniform_init(act:str, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> Callable[[Tensor],None]:
    r'''
    Lookup weight initialisation using Uniform distributions

    Arguments:
        act: string representation of activation function
        fan_in: number of inputs to neuron
        fan_out: number of outputs from neuron
    
    Returns:
        Callable to initialise weight tensor
    '''

    if act == 'relu':       return partial(nn.init.kaiming_uniform_, nonlinearity='relu')
    if act == 'prelu':      return partial(nn.init.kaiming_uniform_, nonlinearity='relu')
    if act == 'selu':       return partial(nn.init.uniform_, a=-1/np.sqrt(fan_in), b=1/np.sqrt(fan_in))
    if act == 'sigmoid':    return nn.init.xavier_uniform_
    if act == 'logsoftmax': return nn.init.xavier_uniform_
    if act == 'softmax':    return nn.init.xavier_uniform_
    if act == 'linear':     return nn.init.xavier_uniform_
    if 'swish' in act:      return partial(nn.init.kaiming_uniform_, nonlinearity='relu')
    raise ValueError("Activation not implemented")
    