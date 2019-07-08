from typing import Any

import torch
import torch.nn as nn


def lookup_act(act:str) -> Any:
    r'''
    Map activation name to class
    
    Arguments:
        act: string representation of activation function

    Returns:
        Class implementing requested activation function
    '''

    if act == 'relu':       return nn.ReLU()
    if act == 'prelu':      return nn.PReLU()
    if act == 'selu':       return nn.SELU()
    if act == 'sigmoid':    return nn.Sigmoid()
    if act == 'logsoftmax': return nn.LogSoftmax(1)
    if act == 'softmax':    return nn.Softmax(1)
    if act == 'linear':     return lambda x: x
    if 'swish' in act:      return Swish()
    raise ValueError("Activation not implemented")

        
class Swish(nn.Module):
    r'''
    Non-trainable Swish activation function https://arxiv.org/abs/1710.05941

    Arguments:
        inplace: whether to apply activation inplace

    Examples::
        >>> swish = Swish()
    '''

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x  # Do we need to return?
        else:
            return x*torch.sigmoid(x)
            