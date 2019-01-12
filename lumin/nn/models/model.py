import numpy as np
from typing import List, Optional, Union
from collections import OrderedDict

import torch
from torch.tensor import Tensor

from .model_builder import ModelBuilder
from ..data.batch_yielder import BatchYielder
from ..callbacks.abs_callback import AbsCallback
from ...utils.misc import to_np


class Model():
    def __init__(self, model_builder:ModelBuilder=None):
        self.model_builder = model_builder
        if self.model_builder is not None: self.model, self.opt, self.loss = self.model_builder.get_model()
        
    def fit(self, batch_yielder:BatchYielder, callbacks:List[AbsCallback]) -> float:
        self.model.train()
        self.stop_train = False
        losses = []
        for c in callbacks: c.on_epoch_begin(losses)

        for x, y, w in batch_yielder:
            for c in callbacks: c.on_batch_begin()
            y_pred = self.model(x)
            loss = self.loss(weight=w)(y_pred, y)
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
        y_pred = self.model(inputs.float())
        loss = self.loss(weight=weights)(y_pred, targets.float())
        return loss.data.item()
            
    def predict(self, inputs, as_np:bool=True) -> Union[np.ndarray, Tensor]:
        self.model.eval()
        if not isinstance(inputs, Tensor): inputs = Tensor(inputs)
        pred = self.model(inputs.float())
        if as_np: return to_np(pred)
        else:     return pred

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

    def export2onnx(self, name:str, bs:int=1) -> None:
        if '.onnx' not in name: name += '.onnx'
        dummy_input = torch.rand(bs, self.model_builder.n_cont_in+self.model_builder.n_cat_in)
        torch.onnx.export(self.model, dummy_input, name)     