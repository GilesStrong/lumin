import numpy as np
import pandas as pd
from typing import List, Optional, Union
from collections import OrderedDict

import torch
from torch.tensor import Tensor
import torch.nn as nn

from .abs_model import AbsModel
from .model_builder import ModelBuilder
from ..data.batch_yielder import BatchYielder
from ..callbacks.abs_callback import AbsCallback
from ...utils.misc import to_np
from ..data.fold_yielder import FoldYielder
from ..interpretation.features import get_nn_feat_importance
from ..metrics.eval_metric import EvalMetric
from ...utils.misc import to_device


class Model(AbsModel):
    def __init__(self, model_builder:ModelBuilder=None):
        self.model_builder = model_builder
        if self.model_builder is not None:
            self.model, self.opt, self.loss = self.model_builder.get_model()
            self.model = to_device(self.model)
            self.head = self.model[0][0]
            self.body = self.model[1]
            self.tail = self.model[2]
            self.objective = self.model_builder.objective

    def __repr__(self) -> str:
        return f'Model:\n{self.model.parameters}\n\nOptimiser:\n{self.opt}\n\nLoss:\n{self.loss}'

    def __getitem__(self, key:Union[int,str]) -> nn.Module:
        if isinstance(key, int):
            if key == 0: return self.head
            if key == 1: return self.body
            if key == 2: return self.tail
            raise IndexError(f'Index {key} out of range')
        if isinstance(key, str):
            if key == 'head': return self.head
            if key == 'body': return self.body
            if key == 'tail': return self.tail
            raise KeyError(key)
        raise ValueError(f'Expected string or int, recieved {key} of type {type(key)}')
        
    def fit(self, batch_yielder:BatchYielder, callbacks:List[AbsCallback]) -> float:
        self.model.train()
        self.stop_train = False
        losses = []
        for c in callbacks: c.on_epoch_begin(losses)

        for x, y, w in batch_yielder:
            for c in callbacks: c.on_batch_begin()
            y_pred = self.model(x)
            loss = self.loss(weight=w)(y_pred, y) if w is not None else self.loss()(y_pred, y)
            losses.append(loss.data.item())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            for c in callbacks: c.on_batch_end(logs={'loss': losses[-1]})
            if self.stop_train: break
        
        for c in callbacks: c.on_epoch_end()
        return np.mean(losses)
              
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]=None) -> float:
        self.model.eval()
        if 'multiclass' in self.objective:
            targets = targets.long().squeeze()
            weights = weights[0]
        else:
            targets = targets.float()
        y_pred = self.model(to_device(inputs.float()))
        loss = self.loss(weight=to_device(weights))(y_pred, to_device(targets)) if weights is not None else self.loss()(y_pred, to_device(targets))
        return loss.data.item()
            
    def predict(self, inputs, as_np:bool=True) -> Union[np.ndarray, Tensor]:
        self.model.eval()
        if isinstance(inputs, pd.DataFrame): inputs = Tensor(inputs.values)
        if not isinstance(inputs, Tensor): inputs = Tensor(inputs)
        pred = self.model(to_device(inputs.float()))
        if as_np:
            if 'multiclass' in self.objective:
                return np.exp(to_np(pred))
            else:
                return to_np(pred)
        else:
            return pred

    def get_weights(self) -> OrderedDict:
        return self.model.state_dict()

    def set_weights(self, weights:OrderedDict) -> None:
        self.model.load_state_dict(weights)

    def get_lr(self) -> float:
        return self.opt.param_groups[0]['lr']

    def set_lr(self, lr:float) -> None:
        self.opt.param_groups[0]['lr'] = lr

    def get_mom(self) -> float:
        return self.opt.param_groups[0]['lr']

    def set_mom(self, mom:float) -> None:
        if   'betas'    in self.opt.param_groups: self.opt.param_groups[0]['betas'][0] = mom
        elif 'momentum' in self.opt.param_groups: self.opt.param_groups[0]['lr']       = mom
    
    def save(self, name:str) -> None:
        torch.save({'model':self.model.state_dict(), 'opt':self.opt.state_dict()}, name)
        
    def load(self, name:str, model_builder:ModelBuilder=None) -> None:
        if model_builder is not None: self.model, self.opt, self.loss = model_builder.get_model()
        state = torch.load(name)
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        self.objective = self.model_builder.objective if model_builder is None else model_builder.objective

    def export2onnx(self, name:str, bs:int=1) -> None:
        if '.onnx' not in name: name += '.onnx'
        dummy_input = torch.rand(bs, self.model_builder.n_cont_in+self.model_builder.n_cat_in)
        torch.onnx.export(self.model, dummy_input, name)     

    def get_feat_importance(self, fold_yielder:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        return get_nn_feat_importance(self, fold_yielder, eval_metric)

    def get_out_size(self) -> int:
        return self.tail.get_out_size()
