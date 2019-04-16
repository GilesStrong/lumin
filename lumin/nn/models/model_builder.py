
import numpy as np
from typing import Dict, Union, Any, Callable, Tuple, Optional
from pathlib import Path
import pickle

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
import torch

from .layers.activations import lookup_act
from .initialisations import lookup_normal_init
from .blocks.body import FullyConnected
from .blocks.head import CatEmbHead
from .blocks.tail import ClassRegMulti
from ..losses.basic_weighted import WeightedCCE, WeightedMSE

'''
Todo
- Better typing for nn._WeightedLoss
'''


class ModelBuilder(object):
    '''Class to build models to specified architecture on demand along with an optimiser'''
    def __init__(self, objective:str, n_cont_in:int, n_out:int, y_range:Optional[Union[Tuple,np.ndarray]]=None,
                 model_args:Dict[str,Any]={}, opt_args:Dict[str,Any]={}, cat_args:Dict[str,Any]=None,
                 loss:Union[Any,'auto']='auto', head:nn.Module=CatEmbHead, body:nn.Module=FullyConnected, tail:nn.Module=ClassRegMulti,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],nn.Module]=lookup_act, pretrain_file:Optional[str]=None, freeze_head:bool=False, freeze_body:bool=False):
        self.objective,self.n_cont_in,self.n_out,self.y_range,self.head,self.body,self.tail  = objective.lower(),n_cont_in,n_out,y_range,head,body,tail
        self.lookup_init,self.lookup_act,self.pretrain_file,self.freeze_head,self.freeze_body = lookup_init,lookup_act,pretrain_file,freeze_head,freeze_body
        self.parse_loss(loss)
        self.parse_model_args(model_args)
        self.parse_opt_args(opt_args)
        self.parse_cat_args(cat_args)

    @classmethod
    def from_model_builder(cls, model_builder, pretrain_file:Optional[str]=None, freeze_head:bool=False, freeze_body:bool=False,
                           loss:Optional[Any]=None, opt_args:Optional[Dict[str,Any]]=None):
        if isinstance(model_builder, str):
            with open(model_builder, 'rb') as fin: model_builder = pickle.load(fin)
        cat_args = {'n_cat_in': model_builder.n_cat_in, 'cat_szs': model_builder.cat_szs,
                    'emb_szs': model_builder.emb_szs, 'cat_names': model_builder.cat_names}
        model_args = {'width': model_builder.width, 'depth': model_builder.depth,
                      'do': model_builder.do, 'do_cat': model_builder.do_cat, 'do_cont': model_builder.do_cont,
                      'bn': model_builder.bn, 'act': model_builder.act, 'res': model_builder.res, 'dense': model_builder.dense}
        return cls(objective=model_builder.objective, n_cont_in=model_builder.n_cont_in, n_out=model_builder.n_out, y_range=model_builder.y_range,
                   cat_args=cat_args, model_args=model_args, opt_args=opt_args if opt_args is not None else {},
                   loss=model_builder.loss if loss is None else loss, head=model_builder.head, body=model_builder.body, tail=model_builder.tail,
                   pretrain_file=pretrain_file, freeze_head=freeze_head, freeze_body=freeze_body)

    def parse_cat_args(self, cat_args) -> None:
        cat_args = {k.lower(): cat_args[k] for k in cat_args}
        self.n_cat_in      = 0    if 'n_cat_in'      not in cat_args else cat_args['n_cat_in']
        self.cat_szs       = None if 'cat_szs'       not in cat_args else cat_args['cat_szs']
        self.emb_szs       = None if 'emb_szs'       not in cat_args else cat_args['emb_szs']
        self.cat_names     = None if 'cat_names'     not in cat_args else cat_args['cat_names']
        self.emb_load_path = None if 'emb_load_path' not in cat_args else cat_args['emb_load_path']
        
        if self.cat_szs is None: self.n_cat_in = 0; return  # Treat cats as conts
        if self.emb_szs is None: self.emb_szs = np.array([(sz, min(50, 1+(sz//2))) for sz in self.cat_szs])
        if isinstance(self.emb_load_path, str): self.emb_load_path = Path(self.emb_load_path)
            
    def parse_loss(self, loss:Union[Any,'auto']='auto') -> None:
        if loss == 'auto':
            if 'class' in self.objective:
                if self.n_out > 1 and 'multiclass' in self.objective: self.loss = WeightedCCE
                else:                                                 self.loss = nn.BCELoss
            else:
                self.loss = WeightedMSE
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
        elif self.opt == 'sgd':  return optim.SGD(model.parameters(),  **self.opt_args)

    def set_lr(self, lr:float) -> None: self.opt_args['lr'] = lr

    def get_head(self) -> nn.Module:
        return self.head(n_cont_in=self.n_cont_in, n_cat_in=self.n_cat_in, cat_names=self.cat_names, n_out=self.width, act=self.act,
                         emb_szs=self.emb_szs, emb_load_path=self.emb_load_path, do=self.do, do_cont=self.do_cont, do_cat=self.do_cat, bn=self.bn,
                         lookup_init=self.lookup_init, lookup_act=self.lookup_act, freeze=self.freeze_head)

    def get_body(self, depth:int) -> nn.Module:
        return self.body(depth=depth, width=self.width, do=self.do, bn=self.bn, act=self.act, res=self.res, dense=self.dense,
                         lookup_init=self.lookup_init, lookup_act=self.lookup_act, freeze=self.freeze_body)

    def get_tail(self, n_in) -> nn.Module:
        return self.tail(n_in=n_in, n_out=self.n_out, objective=self.objective, y_range=self.y_range,lookup_init=self.lookup_init)

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

    def load_pretrained(self, model:nn.Module):
        state = torch.load(self.pretrain_file, map_location='cpu')
        print('Loading pretrained model')
        return model.load_state_dict(state['model'])

    def get_model(self) -> Tuple[nn.Module, optim.Optimizer, Any]:
        model = self.build_model()
        if self.pretrain_file is not None: self.load_pretrained(model)
        opt = self.build_opt(model)
        return model, opt, self.loss

    def get_out_size(self) -> int: return self.tail.get_out_size()
