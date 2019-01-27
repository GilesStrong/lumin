
import numpy as np
from typing import Dict, Union, Any, Callable, Tuple, Optional
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor

from .layers.activations import lookup_act
from .initialisations import lookup_init
from .blocks.body import FullyConnected
from .blocks.head import CatEmbHead
from .blocks.tail import ClassRegMulti

'''
Todo
- Better typing for nn._WeightedLoss
'''


class ModelBuilder(object):
    def __init__(self, objective:str, n_cont_in:int, n_out:int, y_range:Optional[Union[Tuple,np.ndarray]]=None,
                 model_args:Dict[str,Any]={}, opt_args:Dict[str,Any]={}, cat_args:Dict[str,Any]=None,
                 loss:Union[Any,'auto']='auto', body:Callable[[int,int,float,bool,str,bool,bool],nn.Module]=FullyConnected,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Tuple[Callable[[Tensor, str],None],Dict[str,Any]]]=lookup_init,
                 lookup_act:Callable[[str], nn.Module]=lookup_act):
        self.objective,self.n_cont_in,self.n_out,self.y_range,self.body,self.lookup_init,self.lookup_act = objective.lower(),n_cont_in,n_out,y_range,body,lookup_init,lookup_act
        self.parse_loss(loss)
        self.parse_model_args(model_args)
        self.parse_opt_args(opt_args)
        self.parse_cat_args(cat_args)
    
    def parse_cat_args(self, cat_args) -> None:
        cat_args = {k.lower(): cat_args[k] for k in cat_args}
        self.n_cat_in      = 0    if 'n_cat_in'      not in cat_args else cat_args['n_cat_in']
        self.cat_szs       = None if 'cat_szs'       not in cat_args else cat_args['cat_szs']
        self.emb_szs       = None if 'emb_szs'       not in cat_args else cat_args['emb_szs']
        self.cat_names     = None if 'cat_names'     not in cat_args else cat_args['cat_names']
        self.emb_load_path = None if 'emb_load_path' not in cat_args else cat_args['emb_load_path']
        
        if self.cat_szs is None:
            self.n_cat_in = 0  # Treat cats as conts
            return
        if self.emb_szs is None:
            self.emb_szs = np.array([(sz, min(50, 1+(sz//2))) for sz in self.cat_szs])
        if isinstance(self.emb_load_path, str):
            self.emb_load_path = Path(self.emb_load_path)
            
    def parse_loss(self, loss:Union[Any,'auto']='auto') -> None:
        if loss is 'auto':
            if 'class' in self.objective:
                if self.n_out > 1 and 'multiclass' in self.objective:
                    self.loss = nn.NLLLoss
                else:
                    self.loss = nn.BCELoss
            
            else:
                self.loss = nn.MSELoss
        else:   
            self.loss = loss

    def parse_model_args(self, model_args:Dict[str,Any]) -> None:
        model_args   = {k.lower(): model_args[k] for k in model_args}
        self.width   = 100    if 'width'   not in model_args else model_args['width']
        self.depth   = 4      if 'depth'   not in model_args else model_args['depth']
        self.do      = 0      if 'do'      not in model_args else model_args['do']
        self.do_cat  = 0      if 'do_cat'  not in model_args else model_args['do_cat']
        self.do_cont = 0      if 'do_cont' not in model_args else model_args['do_cont']
        self.bn      = False  if 'bn'      not in model_args else model_args['bn']
        self.act     = 'relu' if 'act'     not in model_args else model_args['act'].lower()
        self.res     = False  if 'res'     not in model_args else model_args['res']
        self.dense   = False  if 'dense'   not in model_args else model_args['dense']
    
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

    def get_dense(self, fan_in:Optional[int]=None, fan_out:Optional[int]=None, act:Optional[int]=None, last_layer:bool=False) -> nn.Module:
        fan_in  = self.width if fan_in  is None else fan_in
        fan_out = self.width if fan_out is None else fan_out
        act     = self.act   if act     is None else act

        layers = []
        layers.append(nn.Linear(fan_in, fan_out))
        init, args = self.lookup_init(act, fan_in, fan_out)
        init(layers[-1].weight, **args)
        if act != 'linear': layers.append(self.lookup_act(act))
            
        if self.bn and not last_layer: layers.append(nn.BatchNorm1d(fan_out))
        if self.do and not last_layer: 
            if act == 'selu':
                layers.append(nn.AlphaDropout(self.do))
            else:
                layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)

    def get_head(self) -> nn.Module:
        inputs = CatEmbHead(self.n_cont_in, self.n_cat_in, self.emb_szs, self.do_cont, self.do_cat, self.cat_names, self.emb_load_path)
        linear = self.get_dense(inputs.get_out_size())
        return nn.Sequential(inputs, linear)

    def get_body(self, depth:int) -> nn.Module:
        return self.body(depth, self.width, self.do, self.bn, self.act, self.res, self.dense)

    def get_tail(self, n_in) -> nn.Module:
        return ClassRegMulti(n_in, self.n_out, self.objective, self.y_range)

    def build_model(self) -> nn.Module:
        head = self.get_head()
        body = self.get_body(self.depth-1)
        if hasattr(body, 'get_out_size'):
            out_size = body.get_out_size()
        else:
            *_, last = body.parameters()
            out_size = len(last)
        tail = self.get_tail(out_size)
        return nn.Sequential(head, body, tail)

    def get_model(self) -> Tuple[nn.Module, optim.Optimizer, Any]:
        model = self.build_model()
        opt = self.build_opt(model)
        return model, opt, self.loss

    def get_out_size(self) -> int:
        return self.tail.get_out_size
