import torch
import torch.nn as nn


def lookup_act(act:str):
        if act == 'relu':       return nn.ReLU()
        if act == 'selu':       return nn.SELU()
        if act == 'sigmoid':    return nn.Sigmoid()
        if act == 'logsoftmax': return nn.LogSoftmax()
        if act == 'linear':     return nn.Linear()
        if 'swish' in act:      return Swish()
        raise ValueError("Activation not implemented")

        
class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)
            