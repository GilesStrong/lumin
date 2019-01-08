import numpy as np
from typing import Dict, Union, Any, Callable, Tuple, Optional

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor

from .layers.activations import Swish

'''
Todo
- Better typing for nn._WeightedLoss
'''


class ModelBuilder(object):
    def __init__(self, objective:str, n_in:int, n_out:int,
                 model_args:Dict[str,Any]={}, opt_args:Dict[str,Any]={},
                 loss:Union[Any,'auto']='auto'):
        self.objective,self.n_in,self.n_out = objective.lower(),n_in,n_out
        self.parse_loss(loss)
        self.parse_model_args(model_args)
        self.parse_opt_args(opt_args)

    def parse_loss(self, loss:Union[Any,'auto']='auto') -> None:
        if loss is 'auto':
            if 'class' in self.objective:
                self.loss = nn.NLLLoss if self.n_out > 1 and 'multi' in self.objective else nn.BCELoss
            else:
                self.loss = nn.MSELoss
        else:   
                self.loss = loss

    def parse_model_args(self, model_args:Dict[str,Any]) -> None:
        model_args = {k.lower(): model_args[k] for k in model_args}
        self.width = 100    if 'width' not in model_args else model_args['width']
        self.depth = 4      if 'depth' not in model_args else model_args['depth']
        self.do    = 0      if 'do'    not in model_args else model_args['do']
        self.bn    = False  if 'bn'    not in model_args else model_args['bn']
        self.l2    = False  if 'l2'    not in model_args else model_args['l2']
        self.act   = 'relu' if 'act'   not in model_args else model_args['act'].lower()
    
    def parse_opt_args(self, opt_args:Dict[str,Any]) -> None:
        opt_args = {k.lower(): opt_args[k] for k in opt_args}
        self.opt = 'adam' if 'opt' not in opt_args else opt_args['opt']
        if self.opt not in ['adam', 'sgd']: raise ValueError('Optimiser not currently available')
        self.opt_args = {k: opt_args[k] for k in opt_args if k != 'opt'}        

    def build_opt(self, model:nn.Module) -> optim.Optimizer:
        if   self.opt == 'adam': return optim.Adam(model.parameters(), **self.opt_args)
        elif self.opt == 'sgd':  return optim.SGD(model.parameters(), **self.opt_args)

    def set_lr(self, lr:float) -> None:
        self.opt_args['lr'] = lr

    @staticmethod
    def lookup_init(act:str, fan_in:Optional[int]=None, fan_out:Optional[int]=None) -> Tuple[Callable[[Tensor, str],None],Dict[str,Any]]:
        if act == 'relu':       return nn.init.kaiming_normal_, {'nonlinearity': 'relu'}
        if act == 'selu':       return nn.init.normal_,         {'std': 1/np.sqrt(fan_in)}
        if act == 'sigmoid':    return nn.init.xavier_normal_,  {}
        if act == 'logsoftmax': return nn.init.xavier_normal_,  {}
        if act == 'linear':     return nn.init.xavier_normal_,  {}
        if 'swish' in act:      return nn.init.kaiming_normal_, {'nonlinearity': 'relu'}
        raise ValueError("Activation not implemented")

    @staticmethod
    def lookup_act(act:str):
        if act == 'relu':       return nn.ReLU()
        if act == 'selu':       return nn.SELU()
        if act == 'sigmoid':    return nn.Sigmoid()
        if act == 'logsoftmax': return nn.LogSoftmax()
        if act == 'linear':     return nn.Linear()
        if 'swish' in act:      return Swish()
        raise ValueError("Activation not implemented")

    def get_dense(self, fan_in:Optional[int]=None, fan_out:Optional[int]=None, act:Optional[int]=None) -> nn.Module:
        fan_in  = self.width if fan_in  is None else fan_in
        fan_out = self.width if fan_out is None else fan_out
        act     = self.act   if act     is None else act

        layers = []
        layers.append(nn.Linear(fan_in, fan_out))
        init, args = self.lookup_init(act, fan_in, fan_out)
        init(layers[-1].weight, **args)
        if act != 'linear': layers.append(self.lookup_act(act))

        if self.bn: layers.append(nn.BatchNorm1d(fan_in))
        if self.do: 
            if act == 'selu':
                layers.append(nn.AlphaDropout(self.do))
            else:
                layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)

    def get_head(self) -> nn.Module:
        return self.get_dense(self.n_in)

    def get_body(self, depth:int) -> nn.Module:
        return nn.Sequential(*[self.get_dense() for d in range(depth)])

    def get_tail(self) -> nn.Module:
        if 'class' in self.objective:
            if 'multi' in self.objective: 
                return self.get_dense(self.width, self.n_out, 'logsoftmax')
            else:
                return self.get_dense(self.width, self.n_out, 'sigmoid')
        else:
                return self.get_dense(self.width, self.n_out, 'linear')

    def build_model(self) -> nn.Module:
        return nn.Sequential(self.get_head(), self.get_body(self.depth-1), self.get_tail())

    def get_model(self) -> Tuple[nn.Module, optim.Optimizer, Any]:
        model = self.build_model()
        opt = self.build_opt(model)
        return model, opt, self.loss
