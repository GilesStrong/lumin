import numpy as np
import pandas as pd
from typing import List, Optional, Union
from collections import OrderedDict
from fastprogress import master_bar, progress_bar
import timeit
import warnings

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
from ...utils.statistics import uncert_round


class Model(AbsModel):
    '''Class to handle training and inference of NNs created via a ModelBuilder'''
    def __init__(self, model_builder:ModelBuilder=None):
        self.model_builder = model_builder
        if self.model_builder is not None:
            self.model, self.opt, self.loss = self.model_builder.get_model()
            self.model = to_device(self.model)
            self.head, self.body, self.tail = self.model[0], self.model[1], self.model[2]
            self.objective = self.model_builder.objective
            self.n_out = self.tail.get_out_size()
            self.parameters = self.model.parameters

    def __repr__(self) -> str: return f'Model:\n{self.model.parameters}\n\nOptimiser:\n{self.opt}\n\nLoss:\n{self.loss}'

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

    @classmethod
    def from_save(cls, name:str, model_builder:ModelBuilder) -> AbsModel:
        m = cls(model_builder)
        m.load(name)
        return m
        
    def fit(self, batch_yielder:BatchYielder, callbacks:Optional[List[AbsCallback]]=None) -> float:
        self.model.train()
        self.stop_train = False
        losses = []
        if callbacks is None: callbacks = []
        for c in callbacks: c.on_epoch_begin(batch_yielder=batch_yielder)

        for x, y, w in batch_yielder:
            for c in callbacks: c.on_batch_begin()
            y_pred = self.model(x)
            loss = self.loss(weight=w)(y_pred, y) if w is not None else self.loss()(y_pred, y)
            losses.append(loss.data.item())
            self.opt.zero_grad()
            for c in callbacks: c.on_backwards_begin(loss=loss)
            loss.backward()
            for c in callbacks: c.on_backwards_end(loss=loss)
            self.opt.step()
            
            for c in callbacks: c.on_batch_end(loss=losses[-1])
            if self.stop_train: break
        
        for c in callbacks: c.on_epoch_end(losses=losses)
        return np.mean(losses)
              
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]=None, callbacks:Optional[List[AbsCallback]]=None) -> float:
        if callbacks is None: callbacks = []
        self.model.eval()
        if 'multiclass' in self.objective: targets = targets.long().squeeze()
        else:                              targets = targets.float()
        y_pred = self.model(to_device(inputs.float()))
        for c in callbacks: c.on_eval_begin(inputs=inputs, targets=targets, weights=weights)
        loss = self.loss(weight=to_device(weights))(y_pred, to_device(targets)) if weights is not None else self.loss()(y_pred, to_device(targets))
        for c in callbacks: c.on_eval_end(loss=loss)        
        return loss.data.item()

    def predict_array(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool=True) -> Union[np.ndarray, Tensor]:
        self.model.eval()
        if isinstance(inputs, pd.DataFrame): inputs = Tensor(inputs.values)
        if not isinstance(inputs, Tensor): inputs = Tensor(inputs)
        pred = self.model(to_device(inputs.float()))
        if as_np:
            if 'multiclass' in self.objective: return np.exp(to_np(pred))
            else:                              return to_np(pred)
        else:
            return pred

    def predict_folds(self, fy:FoldYielder, pred_name:str='pred') -> None:
        times = []
        mb = master_bar(range(len(fy.foldfile)))
        for fold_idx in mb:
            fold_tmr = timeit.default_timer()
            if not fy.test_time_aug:
                fold = fy.get_fold(fold_idx)['inputs']
                pred = self.predict_array(fold)
            else:
                tmpPred = []
                pb = progress_bar(range(fy.aug_mult), parent=mb)
                for aug in pb:
                    fold = fy.get_test_fold(fold_idx, aug)['inputs']
                    tmpPred.append(self.predict_array(fold))
                pred = np.mean(tmpPred, axis=0)

            times.append((timeit.default_timer()-fold_tmr)/len(fold))
            if self.n_out > 1: fy.save_fold_pred(pred, fold_idx, pred_name=pred_name)
            else: fy.save_fold_pred(pred[:, 0], fold_idx, pred_name=pred_name)
        times = uncert_round(np.mean(times), np.std(times, ddof=1)/np.sqrt(len(times)))
        print(f'Mean time per event = {times[0]}±{times[1]}')

    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool=True, pred_name:str='pred') -> Union[np.ndarray, Tensor, None]:
        if not isinstance(inputs, FoldYielder): return self.predict_array(inputs, as_np=as_np)
        self.predict_folds(inputs, pred_name)

    def get_weights(self) -> OrderedDict: return self.model.state_dict()

    def set_weights(self, weights:OrderedDict) -> None: self.model.load_state_dict(weights)

    def get_lr(self) -> float: return self.opt.param_groups[0]['lr']

    def set_lr(self, lr:float) -> None: self.opt.param_groups[0]['lr'] = lr

    def get_mom(self) -> float: return self.opt.param_groups[0]['lr']

    def set_mom(self, mom:float) -> None:
        if   'betas'    in self.opt.param_groups: self.opt.param_groups[0]['betas'][0] = mom
        elif 'momentum' in self.opt.param_groups: self.opt.param_groups[0]['lr']       = mom
    
    def save(self, name:str) -> None: torch.save({'model':self.model.state_dict(), 'opt':self.opt.state_dict()}, name)
        
    def load(self, name:str, model_builder:ModelBuilder=None) -> None:
        if model_builder is not None: self.model, self.opt, self.loss = model_builder.get_model()
        state = torch.load(name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        self.objective = self.model_builder.objective if model_builder is None else model_builder.objective

    def export2onnx(self, name:str, bs:int=1) -> None:
        warnings.warn("""ONNX export of LUMIN models has not been fully explored or sufficiently tested yet.
                         Please use with caution, and report any trouble""")
        if '.onnx' not in name: name += '.onnx'
        dummy_input = torch.rand(bs, self.model_builder.n_cont_in+self.model_builder.cat_embedder.n_cat_in)
        torch.onnx.export(self.model, dummy_input, name)
    
    def export2tfpb(self, name:str, bs:int=1) -> None:
        import onnx
        from onnx_tf.backend import prepare
        warnings.warn("""Tensorflow ProtocolBuffer export of LUMIN models (via ONNX) has not been fully explored or sufficiently tested yet.
                         Please use with caution, and report any trouble""")
        if '.' in name: name = name[:name.rfind('.')]
        self.export2onnx(name, bs)
        m = onnx.load(f'{name}.onnx')
        tf_rep = prepare(m)
        tf_rep.export_graph(f'{name}.pb')
           
    def get_feat_importance(self, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        return get_nn_feat_importance(self, fy, eval_metric)

    def get_out_size(self) -> int: return self.tail.get_out_size()
