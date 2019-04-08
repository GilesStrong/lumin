import torch
import torch.nn as nn


def lookup_act(act:str) -> nn.Module:
    '''Map activation name to class'''
    if act == 'relu':       return nn.ReLU()
    if act == 'prelu':      return nn.PReLU()
    if act == 'selu':       return nn.SELU()
    if act == 'sigmoid':    return nn.Sigmoid()
    if act == 'logsoftmax': return nn.LogSoftmax(1)
    if act == 'softmax':    return nn.Softmax(1)
    if act == 'linear':     return nn.Linear()
    if 'swish' in act:      return Swish()
    raise ValueError("Activation not implemented")

        
class Swish(nn.Module):
    '''Non-trainable Swish activation function https://arxiv.org/abs/1710.05941'''
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)
            