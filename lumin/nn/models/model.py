import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
from collections import OrderedDict
from fastprogress import master_bar, progress_bar
import warnings
from fastcore.all import is_listy, partialler, Path
from random import shuffle
from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from .abs_model import AbsModel,FitParams
from .model_builder import ModelBuilder
from ..data.batch_yielder import BatchYielder
from ..callbacks.abs_callback import AbsCallback
from ..callbacks.cyclic_callbacks import AbsCyclicCallback
from ..callbacks.pred_handlers import PredHandler
from ..callbacks.monitors import MetricLogger
from ..data.fold_yielder import FoldYielder
from ..interpretation.features import get_nn_feat_importance
from ..metrics.eval_metric import EvalMetric
from ...plotting.plot_settings import PlotSettings
from ...utils.misc import to_np, to_device, is_partially

__all__ = ['Model']


class Model(AbsModel):
    r'''
    Wrapper class to handle training and inference of NNs created via a :class:`~lumin.nn.models.model_builder.ModelBuilder`.
    Note that saved models can be instantiated direcly via :meth:`~lumin.nn.models.model.Model.from_save` classmethod.

    # TODO: Improve mask description & user-friendlyness, change to indicate that 'masked' inputs are actually the ones which are used
    
    Arguments:
        model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` which will construct the network, loss, optimiser, and input mask

    Examples::
        >>> model = Model(model_builder)
    '''

    def __init__(self, model_builder:Optional[ModelBuilder]=None):
        self.model_builder,self.input_mask = model_builder,None
        if self.model_builder is not None:
            self.model, self.opt, self.loss, self.input_mask = self.model_builder.get_model()
            self.head, self.body, self.tail = self.model[0], self.model[1], self.model[2]
            self.objective = self.model_builder.objective
            self.n_out = self.tail.get_out_size()
            self.parameters = self.model.parameters

    def __repr__(self) -> str:
        return f'''Inputs:\n{self.head.n_cont_in} Continuous: {self.head.cont_feats}
                   \n{self.head.n_cat_in}  Categorical: {self.head.cat_feats}
                   \n{self.head.n_matrix_in}  Matrix elements: {self.head.matrix_feats}
                   \n\nModel:\n{self.model.parameters}
                   \n\nNumber of trainable parameters: {self.get_param_count()}
                   \n\nOptimiser:\n{self.opt}
                   \n\nLoss:\n{self.loss}'''

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
        r'''
        Instantiated a :class:`~lumin.nn.models.model.Model` and load saved state from file.
        
        Arguments:
            name: name of file containing saved state
            model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` which was used to construct the network
        
        Returns:
            Instantiated :class:`~lumin.nn.models.model.Model` with network weights, optimiser state, and input mask loaded from saved state
        
        Examples::
            >>> model = Model.from_save('weights/model.h5', model_builder)
        '''

        m = cls(model_builder)
        m.load(name)
        return m

    def get_param_count(self, trainable:bool=True) -> int:
        r'''
        Return number of parameters in model.

        Arguments:
            trainable: if true (default) only count trainable parameters

        Returns:
            NUmber of (trainable) parameters in model
        '''
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable) 

    def set_input_mask(self, mask:np.ndarray) -> None:
        r'''
        Mask input columns by only using input columns whose indeces are listed in mask

        Arguments:
            mask: array of column indeces to use from all input columns
        '''

        self.input_mask = mask

    def _fit_batch(self, x:Tensor, y:Tensor, w:Tensor) -> None:
        self.fit_params.x,self.fit_params.y,self.fit_params.w = x,y,w
        for c in self.fit_params.cbs: c.on_batch_begin()
        self.fit_params.y_pred = self.model(self.fit_params.x)
        if self.fit_params.state != 'test' and self.fit_params.loss_func is not None:
            if hasattr(self.fit_params.loss_func, 'weights'): self.fit_params.loss_func.weights = self.fit_params.w  # Proper weighting required
            else:                                             self.fit_params.loss_func.weight  = self.fit_params.w
            self.fit_params.loss_val = self.fit_params.loss_func(self.fit_params.y_pred, self.fit_params.y)
        for c in self.fit_params.cbs: c.on_forwards_end()
        if self.fit_params.state == 'train':
            self.fit_params.opt.zero_grad()
            for c in self.fit_params.cbs: c.on_backwards_begin()
            self.fit_params.loss_val.backward()
            for c in self.fit_params.cbs: c.on_backwards_end()
            self.fit_params.opt.step()
        for c in self.fit_params.cbs: c.on_batch_end()

    def fit(self, n_epochs:int, fy:FoldYielder, bs:int, bulk_move:bool=True, train_on_weights:bool=True, trn_idxs:Optional[List[int]]=None,
            val_idx:Optional[int]=None,  cbs:Optional[Union[AbsCallback,List[AbsCallback]]]=None, cb_savepath:Path=Path('train_weights'),
            model_bar:Optional[master_bar]=None, visible_bar:bool=True) -> List[AbsCallback]:
        r'''
        Fit network to training data according to the model's loss and optimiser.
        
        Training continues until:
        - All of the training folds are used n_epoch number of times;
        - Or a callback triggers training to stop, e.g. :class:`~lumin.nn.callbacks.cyclic_callbacks.OneCycle`,
            or :class:`~lumin.nn.callbacks.monitors.EarlyStopping`

        Arguments:
            n_epochs: number of epochs for which to train
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` containing training and validation data
            bs: Batch size
            bulk_move: if true, will optimise for speed by using more RAM and VRAM
            train_on_weights: whether to actually use data weights, if present
            trn_idxs: Fold indexes in `fy` to use for training. If not set, will use all folds except val_idx
            val_idx: Fold index in `fy` to use for validation. If not set, will not compute validation losses
            cbs: list of instantiated callbacks to adjust training. Will be called in order listed.
            cb_savepath: General save directory for any callbacks which require saving models and other information (accessible from `fit_params`),
            model_bar: Optional `master_bar` for aligning progress bars, i.e. if training multiple models

        Returns:
            List of all callbacks used during training
        '''
        
        if cbs is None: cbs = []
        elif not is_listy(cbs): cbs = [cbs]
        cyclic_cbs,loss_cbs,metric_log = [],[],None
        for c in cbs:
            if isinstance(c, AbsCyclicCallback): cyclic_cbs.append(c)  # CBs that might prevent a model from stopping training due to a hyper-param cycle
            if hasattr(c, "get_loss"): loss_cbs.append(c)  # CBs that produce alternative losses that should be considered
            if isinstance(c, MetricLogger): metric_log = c  # CB that logs losses and eval_metrics

        self.fit_params = FitParams(cbs=cbs, cyclic_cbs=cyclic_cbs, loss_cbs=loss_cbs, metric_log=metric_log, stop=False, n_epochs=n_epochs, fy=fy,
                                    val_idx=val_idx, bs=bs, bulk_move=bulk_move, train_on_weights=train_on_weights, cb_savepath=Path(cb_savepath),
                                    loss_func=self.loss, opt=self.opt, val_requires_grad=False)
        self.fit_params.cb_savepath.mkdir(parents=True, exist_ok=True)
        if is_partially(self.fit_params.loss_func): self.fit_params.loss_func = self.fit_params.loss_func()
        self.fit_params.partial_by = partialler(BatchYielder, objective=self.objective, use_weights=self.fit_params.train_on_weights,
                                                bulk_move=self.fit_params.bulk_move, input_mask=self.input_mask)

        if trn_idxs is None: trn_idxs = list(range(fy.n_folds))
        if val_idx is not None and val_idx in trn_idxs: trn_idxs.remove(val_idx)
        self.fit_params.trn_idxs,self.fit_params.val_idx = trn_idxs,val_idx
        if self.fit_params.val_idx is not None:
            if bulk_move:
                val_by = self.fit_params.partial_by(**self.fit_params.fy.get_fold(self.fit_params.val_idx), drop_last=False, shuffle=False,
                                                    bs=self.fit_params.fy.get_data_count(self.fit_params.val_idx) if bulk_move else self.fit_params.bs)
            else:
                val_by = partial(self.fit_params.partial_by, drop_last=False, shuffle=False,
                                 bs=self.fit_params.fy.get_data_count(self.fit_params.val_idx) if bulk_move else self.fit_params.bs)
        trn_by = partial(self.fit_params.partial_by, drop_last=True, bs=self.fit_params.bs, shuffle=True)

        def fit_epoch() -> None:
            self.model.train()
            self.fit_params.state = 'train'
            self.fit_params.epoch += 1
            shuffle(trn_idxs)
            self.fit_params.trn_idxs = trn_idxs
            for c in self.fit_params.cbs: c.on_epoch_begin()
            for self.fit_params.trn_idx in self.fit_params.trn_idxs:
                self.fit_params.sub_epoch += 1
                self.fit_params.by = trn_by(**self.fit_params.fy.get_fold(self.fit_params.trn_idx))
                for c in self.fit_params.cbs: c.on_fold_begin()
                for b in self.fit_params.by: self._fit_batch(*b)
                for c in self.fit_params.cbs: c.on_fold_end()
                if self.fit_params.stop: break
            for c in self.fit_params.cbs: c.on_epoch_end()

            if self.fit_params.val_idx is not None:
                self.model.eval()
                with torch.set_grad_enabled(self.fit_params.val_requires_grad):
                    self.fit_params.state = 'valid'
                    for c in self.fit_params.cbs: c.on_epoch_begin()
                    self.fit_params.by = val_by if bulk_move else val_by(**self.fit_params.fy.get_fold(self.fit_params.val_idx))
                    for c in self.fit_params.cbs: c.on_fold_begin()
                    for b in self.fit_params.by: self._fit_batch(*b)
                    for c in self.fit_params.cbs: c.on_fold_end()
                    for c in self.fit_params.cbs: c.on_epoch_end()

        try:
            for c in self.fit_params.cbs: c.set_model(self)
            for c in self.fit_params.cbs: c.on_train_begin()
            for e in progress_bar(range(self.fit_params.n_epochs), parent=model_bar, display=visible_bar):
                fit_epoch()
                if self.fit_params.stop: break
            if self.fit_params.val_idx is not None: del val_by
            for c in self.fit_params.cbs: c.on_train_end()
        finally:
            self.fit_params = None
            torch.cuda.empty_cache()
        return cbs

    def _predict_by(self, by:BatchYielder, pred_cb:PredHandler=PredHandler(), cbs:Optional[Union[AbsCallback,List[AbsCallback]]]=None) -> np.ndarray:
        if cbs is None: cbs = []
        elif not is_listy(cbs): cbs = [cbs]
        cbs.append(pred_cb)
        self.fit_params = FitParams(cbs=cbs, by=by, state='test', val_requires_grad=False)
        try:
            for c in self.fit_params.cbs: c.set_model(self)
            self.model.eval()
            with torch.set_grad_enabled(self.fit_params.val_requires_grad):
                for c in self.fit_params.cbs: c.on_pred_begin()
                for b in self.fit_params.by: self._fit_batch(*b)
                for c in self.fit_params.cbs: c.on_pred_end()
        finally:
            self.fit_params = None
            cbs.pop()  # Remove pred_cb to avoid mutating cbs
            torch.cuda.empty_cache()
        return pred_cb.get_preds()

    def _predict_array(self, inputs:Union[np.ndarray,pd.DataFrame,Tensor,Tuple], as_np:bool=True, pred_cb:PredHandler=PredHandler(),
                       cbs:Optional[List[AbsCallback]]=None, bs:Optional[int]=None) -> Union[np.ndarray, Tensor]:
        by = BatchYielder(inputs=inputs, bs=len(inputs) if bs is None else bs, objective=self.objective, shuffle=False, bulk_move=bs is None,
                          input_mask=self.input_mask, drop_last=False)
        preds = self._predict_by(by, pred_cb=pred_cb, cbs=cbs)
        if as_np:
            preds = to_np(preds)
            if 'multiclass' in self.objective: preds = np.exp(preds)
        return preds

    def evaluate(self, inputs:Union[np.ndarray,Tensor,Tuple,BatchYielder], targets:Optional[Union[np.ndarray,Tensor]]=None,
                 weights:Optional[Union[np.ndarray,Tensor]]=None, bs:Optional[int]=None) -> float:
        r'''
        Compute loss on provided data.

        Arguments:
            inputs: input data, or :class:`~lumin.nn.data.batch_yielder.BatchYielder` with input, target, and weight data
            targets: targets, not required if :class:`~lumin.nn.data.batch_yielder.BatchYielder` is passed to inputs
            weights: Optional weights, not required if :class:`~lumin.nn.data.batch_yielder.BatchYielder`, or no weights should be considered
            bs: batch size to use. If `None`, will evaluate all data at once

        Returns:
            (weighted) loss of model predictions on provided data
        '''

        # TODO: make this work with non-meaned losses

        if hasattr(self, 'fit_params') and self.fit_params is not None:
            raise ValueError('Evaluate will overwrite exisiting fit_params for this model. Most likely it is being called during training.')
        if not isinstance(inputs, BatchYielder): inputs = BatchYielder(inputs=inputs, targets=targets, weights=weights, bs=len(inputs) if bs is None else bs,
                                                                       objective=self.objective, shuffle=False, bulk_move=bs is None,
                                                                       input_mask=self.input_mask, drop_last=False)
        self.fit_params = FitParams(cbs=[], by=inputs, state='valid', loss_func=self.loss)
        if is_partially(self.fit_params.loss_func): self.fit_params.loss_func = self.fit_params.loss_func()
        self.model.eval()
        loss,cnt = 0,0
        try:
            for b in self.fit_params.by:
                self._fit_batch(*b)
                sz = len(b[0])
                loss += self.fit_params.loss_val.data.item()*sz
                cnt += sz
        finally:
            self.fit_params = None
        return loss/cnt

    def _predict_folds(self, fy:FoldYielder, pred_name:str='pred', pred_cb:PredHandler=PredHandler(), cbs:Optional[List[AbsCallback]]=None,
                       bs:Optional[int]=None) -> None:
        for fold_idx in progress_bar(range(len(fy))):
            if not fy.test_time_aug:
                pred = self._predict_array(fy.get_fold(fold_idx)['inputs'], pred_cb=pred_cb, cbs=cbs, bs=bs)
            else:
                tmpPred = []
                for aug in range(fy.aug_mult): tmpPred.append(self._predict_array(fy.get_test_fold(fold_idx, aug)['inputs'], pred_cb=pred_cb, cbs=cbs, bs=bs))
                pred = np.mean(tmpPred, axis=0)

            if self.n_out > 1: fy.save_fold_pred(pred, fold_idx, pred_name=pred_name)
            else: fy.save_fold_pred(pred[:, 0], fold_idx, pred_name=pred_name)

    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool=True, pred_name:str='pred', pred_cb:PredHandler=PredHandler(),
                cbs:Optional[List[AbsCallback]]=None, bs:Optional[int]=None) -> Union[np.ndarray, Tensor, None]:
        r'''
        Apply model to inputed data and compute predictions.
        
        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device, or :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            as_np: whether to return predictions as Numpy array (otherwise tensor) if inputs are a Numpy array, Pandas DataFrame, or tensor
            pred_name: name of group to which to save predictions if inputs are a :class:`~lumin.nn.data.fold_yielder.FoldYielder`
            pred_cb: :class:`~lumin.nn.callbacks.pred_handlers.PredHandler` callback to determin how predictions are computed.
                Default simply returns the model predictions. Other uses could be e.g. running argmax on a multiclass classifier
            cbs: list of any instantiated callbacks to use during prediction
            bs: if not `None`, will run prediction in batches of specified size to save of memory

        Returns:
            if inputs are a Numpy array, Pandas DataFrame, or tensor, will return predicitions as either array or tensor
        '''

        if isinstance(inputs, BatchYielder): return self._predict_by(inputs, pred_cb=pred_cb, cbs=cbs)
        if not isinstance(inputs, FoldYielder): return self._predict_array(inputs, as_np=as_np, pred_cb=pred_cb, cbs=cbs, bs=bs)
        self._predict_folds(inputs, pred_name, pred_cb=pred_cb, cbs=cbs, bs=bs)

    def get_weights(self) -> OrderedDict:
        r'''
        Get state_dict of weights for network

        Returns:
            state_dict of weights for network
        '''
        
        return self.model.state_dict()

    def set_weights(self, weights:OrderedDict) -> None:
        r'''
        Set state_dict of weights for network

        Arguments:
            weights: state_dict of weights for network
        '''
        
        self.model.load_state_dict(weights)

    def get_lr(self) -> float:
        r'''
        Get learning rate of optimiser

        Returns:
            learning rate of optimiser
        '''
        
        return self.opt.param_groups[0]['lr']

    def set_lr(self, lr:float) -> None:
        r'''
        set learning rate of optimiser

        Arguments:
            lr: learning rate of optimiser
        '''
        
        self.opt.param_groups[0]['lr'] = lr

    def get_mom(self) -> float:
        r'''
        Get momentum/beta_1 of optimiser

        Returns:
            momentum/beta_1 of optimiser
        '''
        
        if   'betas'    in self.opt.param_groups[0]: return self.opt.param_groups[0]['betas'][0]
        elif 'momentum' in self.opt.param_groups[0]: return self.opt.param_groups[0]['momentum']

    def set_mom(self, mom:float) -> None:
        r'''
        Set momentum/beta_1 of optimiser

        Arguments:
            mom: momentum/beta_1 of optimiser
        '''

        if   'betas'    in self.opt.param_groups[0]: self.opt.param_groups[0]['betas'] = (mom, self.opt.param_groups[0]['betas'][1])
        elif 'momentum' in self.opt.param_groups[0]: self.opt.param_groups[0]['momentum'] = mom
    
    def save(self, name:str) -> None:
        r'''
        Save model, optimiser, and input mask states to file

        Arguments:
            name: name of save file
        '''

        torch.save({'model':self.model.state_dict(), 'opt':self.opt.state_dict(), 'input_mask':self.input_mask}, str(name))
        
    def load(self, name:str, model_builder:ModelBuilder=None) -> None:
        r'''
        Load model, optimiser, and input mask states from file

        Arguments:
            name: name of save file
            model_builder: if :class:`~lumin.nn.models.model.Model` was not initialised with a :class:`~lumin.nn.models.model_builder.ModelBuilder`,
                you will need to pass one here
        '''

        # TODO: update map location when device choice is changable by user

        if model_builder is not None: self.model, self.opt, self.loss, self.input_mask = model_builder.get_model()
        state = torch.load(name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        self.input_mask = state['input_mask']
        self.objective = self.model_builder.objective if model_builder is None else model_builder.objective

    def export2onnx(self, name:str, bs:int=1) -> None:
        r'''
        Export network to ONNX format.
        Note that ONNX expects a fixed batch size (bs) which is the number of datapoints your wish to pass through the model concurrently.

        Arguments:
            name: filename for exported file
            bs: batch size for exported models
        '''

        # TODO: Pass FoldYielder to get example dummy input, or account for matrix inputs
        
        warnings.warn("""ONNX export of LUMIN models has not been fully explored or sufficiently tested yet.
                         Please use with caution, and report any trouble""")
        if '.onnx' not in name: name += '.onnx'
        dummy_input = to_device(torch.rand(bs, self.model_builder.n_cont_in+self.model_builder.cat_embedder.n_cat_in))
        torch.onnx.export(self.model, dummy_input, name)
    
    def export2tfpb(self, name:str, bs:int=1) -> None:
        r'''
        Export network to Tensorflow ProtocolBuffer format, via ONNX.
        Note that ONNX expects a fixed batch size (bs) which is the number of datapoints your wish to pass through the model concurrently.

        Arguments:
            name: filename for exported file
            bs: batch size for exported models
        '''

        import onnx
        from onnx_tf.backend import prepare
        warnings.warn("""Tensorflow ProtocolBuffer export of LUMIN models (via ONNX) has not been fully explored or sufficiently tested yet.
                         Please use with caution, and report any trouble""")
        self.export2onnx(name, bs)
        m = onnx.load(f'{name}.onnx')
        tf_rep = prepare(m)
        tf_rep.export_graph(f'{name}.pb')
           
    def get_feat_importance(self, fy:FoldYielder, bs:Optional[int]=None, eval_metric:Optional[EvalMetric]=None, savename:Optional[str]=None,
                            settings:PlotSettings=PlotSettings()) -> pd.DataFrame:
        r'''
        Call :meth:`~lumin.nn.interpretation.features.get_nn_feat_importance` passing this :class:`~lumin.nn.models.model.Model` and provided arguments

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data used to train model
            bs: If set, will evaluate model in batches of data, rather than all at once
            eval_metric: Optional :class:`~lumin.nn.metric.eval_metric.EvalMetric` to use to quantify performance in place of loss
            savename: Optional name of file to which to save the plot of feature importances
            settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
        '''

        return get_nn_feat_importance(self, fy=fy, bs=bs, eval_metric=eval_metric, savename=savename, settings=settings)

    def get_out_size(self) -> int:
        r'''
        Get number of outputs of model

        Returns:
            Number of outputs of model
        '''

        return self.tail.get_out_size()

    def freeze_layers(self) -> None:
        r'''
        Make parameters untrainable
        '''

        for p in self.model.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self) -> None:
        r'''
        Make parameters trainable
        '''

        for p in self.model.parameters(): p.requires_grad = True
