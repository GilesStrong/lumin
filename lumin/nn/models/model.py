import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
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

__all__ = ['Model']


class Model(AbsModel):
    r'''
    Wrapper class to handle training and inference of NNs created via a :class:`~lumin.nn.models.model_builder.ModelBuilder`.
    Note that saved models can be instantiated direcly via :meth:`~lumin.nn.models.model.Model.from_save` classmethod.
    
    Arguments:
        model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` which will construct the network, loss, and optimiser

    Examples::
        >>> model = Model(model_builder)
    '''

    # TODO: Improve mask description & user-friendlyness, change to indicate that 'masked' inputs are actually the ones which are used
    # TODO: Check if mask_inputs can be removed

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
        
    def fit(self, batch_yielder:BatchYielder, callbacks:Optional[List[AbsCallback]]=None, mask_inputs:bool=True) -> float:
        r'''
        Fit network for one complete iteration of a :class:`~lumin.nn.data.batch_yielder.BatchYielder`, i.e. one (sub-)epoch

        Arguments:
            batch_yielder: :class:`~lumin.nn.data.batch_yielder.BatchYielder` providing training data in form of tuple of inputs, targtes, and weights as tensors on device
            callbacks: list of :class:`~lumin.nn.callbacks.abs_callback.AbsCallback` to be used during training
            mask_inputs: whether to apply input mask if one has been set

        Returns:
            Loss on training data averaged across all minibatches
        '''

        self.model.train()
        self.stop_train = False
        losses = []
        if callbacks is None: callbacks = []
        for c in callbacks: c.on_epoch_begin(by=batch_yielder)
        if self.input_mask is not None and mask_inputs: batch_yielder.inputs = batch_yielder.inputs[:,self.input_mask]

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
              
    def evaluate(self, inputs:Union[Tensor,np.ndarray,Tuple[Tensor,Tensor],Tuple[np.ndarray,np.ndarray]], targets:Union[Tensor,np.ndarray],
                 weights:Optional[Union[Tensor,np.ndarray]]=None, callbacks:Optional[List[AbsCallback]]=None,
                 mask_inputs:bool=True) -> float:
        r'''
        Compute loss on provided data.

        Arguments:
            inputs: input data
            targets: targets
            weights: Optional weights
            callbacks: list of any callbacks to use during evaluation
            mask_inputs: whether to apply input mask if one has been set

        Returns:
            (weighted) loss of model predictions on provided data
        '''

        if callbacks is None: callbacks = []
        self.model.eval()
        if not isinstance(inputs, Tensor):
            if isinstance(inputs, tuple):
                if not isinstance(inputs[0], Tensor): inputs = (to_device(Tensor(inputs[0]).float()),to_device(Tensor(inputs[1]).float()))
            else:
                inputs = to_device(Tensor(inputs).float())
        for c in callbacks: c.on_eval_begin(inputs=inputs, targets=targets, weights=weights)
        if self.input_mask is not None and mask_inputs:
            if isinstance(inputs, tuple):
                inputs[0] = inputs[0][:,self.input_mask]
            else:
                inputs = inputs[:,self.input_mask]
        y_pred = self.model(inputs)

        if not isinstance(targets, Tensor): targets = to_device(Tensor(targets))
        if weights is not None and not isinstance(weights, Tensor): weights = to_device(Tensor(weights))

        if   'multiclass'     in self.objective and not isinstance(targets, torch.LongTensor):  targets = targets.long().squeeze()
        elif 'multiclass' not in self.objective and not isinstance(targets, torch.FloatTensor): targets = targets.float()

        loss = self.loss(weight=weights)(y_pred, targets) if weights is not None else self.loss()(y_pred, targets)
        for c in callbacks: c.on_eval_end(loss=loss)        
        return loss.data.item()

    def evaluate_from_by(self, by:BatchYielder, callbacks:Optional[List[AbsCallback]]=None) -> float:
        r'''
        Compute loss on provided data in batches provided by a `:class:~lumin.nn.data.batch_yielder.BatchYielder`.

        Arguments:
            by: `:class:~lumin.nn.data.batch_yielder.BatchYielder` with data
            callbacks: list of any callbacks to use during evaluation

        Returns:
            (weighted) loss of model predictions on provided data
        '''

        loss = 0
        for x, y, w in by: loss += self.evaluate(x, y, w, callbacks)*by.bs
        return loss/(len(by)*by.bs)

    def predict_array(self, inputs:Union[np.ndarray,pd.DataFrame,Tensor,Tuple], as_np:bool=True, mask_inputs:bool=True,
                      callbacks:Optional[List[AbsCallback]]=None, bs:Optional[int]=None) -> Union[np.ndarray, Tensor]:
        r'''
        Pass inputs through network and obtain predictions.

        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device
            as_np: whether to return predictions as Numpy array (otherwise tensor)
            mask_inputs: whether to apply input mask if one has been set
            callbacks: list of any callbacks to use during evaluation
            bs: if not `None`, will run prediction in batches of specified size to save of memory

        Returns:
            Model prediction(s) per datapoint
        '''

        def _get_preds(inputs, callbacks):
            for c in callbacks: c.on_pred_begin(inputs=inputs)
            if isinstance(inputs, pd.DataFrame): inputs = to_device(Tensor(inputs.values).float())
            if self.input_mask is not None and mask_inputs:
                if isinstance(inputs, tuple):
                    inputs[0] = inputs[0][:,self.input_mask]
                else:
                    inputs = inputs[:,self.input_mask]
            if not isinstance(inputs, Tensor):
                if isinstance(inputs, tuple):
                    if not isinstance(inputs[0], Tensor): inputs = (to_device(Tensor(inputs[0]).float()),to_device(Tensor(inputs[1]).float()))
                else:
                    inputs = to_device(Tensor(inputs).float())
            pred = self.model(inputs)
            for c in callbacks: c.on_pred_end(pred=pred)
            return to_np(pred)
        
        self.model.eval()
        if callbacks is None: callbacks = []
        if bs is None:
            pred = _get_preds(inputs, callbacks)
        else:
            pred = []
            if isinstance(inputs, tuple):
                if len(inputs[1]) > bs:
                    for i in range(0, len(inputs[1])-bs+1, bs): pred.append(_get_preds((inputs[0][i:i+bs], inputs[1][i:i+bs]), callbacks))
                    pred.append(_get_preds((inputs[0][i+bs:], inputs[1][i+bs:]), callbacks))
                else:
                    pred.append(_get_preds((inputs[0], inputs[1]), callbacks))
            else:
                if len(inputs) > bs:
                    for i in range(0, len(inputs)-bs+1, bs): pred.append(_get_preds(inputs[i:i+bs], callbacks))
                    pred.append(_get_preds(inputs[i+bs:], callbacks))
                else:
                    pred.append(_get_preds(inputs, callbacks))
            pred = np.vstack(pred)
        if as_np:
            if 'multiclass' in self.objective: return np.exp(pred)
            else:                              return pred
        else:
            return to_device(Tensor(pred))

    def predict_folds(self, fy:FoldYielder, pred_name:str='pred', callbacks:Optional[List[AbsCallback]]=None, verbose:bool=True,
                      bs:Optional[int]=None) -> None:
        r'''
        Apply model to all dataaccessed by a :class:`~lumin.nn.data.fold_yielder.FoldYielder` and save predictions as new group in fold file

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            pred_name: name of group to which to save predictions
            callbacks: list of any callbacks to use during evaluation
            verbose: whether to print average prediction timings
            bs: if not `None`, will run prediction in batches of specified size to save of memory
        '''

        times = []
        mb = master_bar(range(len(fy)))
        for fold_idx in mb:
            fold_tmr = timeit.default_timer()
            if not fy.test_time_aug:
                fold = fy.get_fold(fold_idx)['inputs']
                pred = self.predict_array(fold, callbacks=callbacks, bs=bs)
            else:
                tmpPred = []
                pb = progress_bar(range(fy.aug_mult), parent=mb)
                for aug in pb:
                    fold = fy.get_test_fold(fold_idx, aug)['inputs']
                    tmpPred.append(self.predict_array(fold, callbacks=callbacks, bs=bs))
                pred = np.mean(tmpPred, axis=0)

            times.append((timeit.default_timer()-fold_tmr)/len(fold))
            if self.n_out > 1: fy.save_fold_pred(pred, fold_idx, pred_name=pred_name)
            else: fy.save_fold_pred(pred[:, 0], fold_idx, pred_name=pred_name)
        times = uncert_round(np.mean(times), np.std(times, ddof=1)/np.sqrt(len(times)))
        if verbose: print(f'Mean time per event = {times[0]}±{times[1]}')

    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor, FoldYielder], as_np:bool=True, pred_name:str='pred',
                callbacks:Optional[List[AbsCallback]]=None, verbose:bool=True, bs:Optional[int]=None) -> Union[np.ndarray, Tensor, None]:
        r'''
        Apply model to inputed data and compute predictions.
        A compatability method to call :meth:`~lumin.nn.models.model.Model.predict_array` or meth:`~lumin.nn.models.model.Model.predict_folds`, depending on input type.

        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device, or :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            as_np: whether to return predictions as Numpy array (otherwise tensor) if inputs are a Numpy array, Pandas DataFrame, or tensor
            pred_name: name of group to which to save predictions if inputs are a :class:`~lumin.nn.data.fold_yielder.FoldYielder`
            callbacks: list of any callbacks to use during evaluation
            verbose: whether to print average prediction timings
            bs: if not `None`, will run prediction in batches of specified size to save of memory

        Returns:
            if inputs are a Numpy array, Pandas DataFrame, or tensor, will return predicitions as either array or tensor
        '''
        if not isinstance(inputs, FoldYielder): return self.predict_array(inputs, as_np=as_np, callbacks=callbacks, bs=bs)
        self.predict_folds(inputs, pred_name, callbacks=callbacks, verbose=verbose, bs=bs)

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
        
        if   'betas'    in self.opt.param_groups: return self.opt.param_groups[0]['betas'][0]
        elif 'momentum' in self.opt.param_groups: return self.opt.param_groups[0]['momentum']

    def set_mom(self, mom:float) -> None:
        r'''
        Set momentum/beta_1 of optimiser

        Arguments:
            mom: momentum/beta_1 of optimiser
        '''

        if   'betas'    in self.opt.param_groups: self.opt.param_groups[0]['betas'][0] = mom
        elif 'momentum' in self.opt.param_groups: self.opt.param_groups[0]['momentum'] = mom
    
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
            model_builder: if :class:`~lumin.nn.models.model.Model` was not initialised with a :class:`~lumin.nn.models.model_builder.ModelBuilder`, you will need to pass one here
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
           
    def get_feat_importance(self, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        r'''
        Call :meth:`~lumin.nn.interpretation.features.get_nn_feat_importance` passing this :class:`~lumin.nn.models.model.Model` and provided arguments

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data on which to evaluate importance
            eval_metric: Optional :class:`~lumin.nn.metric.eval_metric.EvalMetric` to use for quantifying performance
        '''

        return get_nn_feat_importance(self, fy, eval_metric)

    def get_out_size(self) -> int:
        r'''
        Get number of outputs of model

        Returns:
            Number of outputs of model
        '''

        return self.tail.get_out_size()
