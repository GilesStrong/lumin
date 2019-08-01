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
    r'''
    Wrapper class to handle training and inference of NNs created via a :class:ModelBuilder.
    Note that saved models can be instantiated direcly via :meth:from_save classmethod.
    
    Arguments:
        model_builder: :class:ModelBuilder which will construct the network, loss, and optimiser

    Examples::
        >>> model = Model(model_builder)
    '''

    # TODO: Improve mask description & user-friendlyness, change to indicate that 'masked' inputs are actually the ones which are used
    # TODO: Chek if mask_inputs can be removed

    def __init__(self, model_builder:Optional[ModelBuilder]=None):
        self.model_builder,self.input_mask = model_builder,None
        if self.model_builder is not None:
            self.model, self.opt, self.loss = self.model_builder.get_model()
            self.model = to_device(self.model)
            self.head, self.body, self.tail = self.model[0], self.model[1], self.model[2]
            self.objective = self.model_builder.objective
            self.n_out = self.tail.get_out_size()
            self.parameters = self.model.parameters

    def __repr__(self) -> str:
        return f'''Model:\n{self.model.parameters}
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
        Instantiated a :class:Model and load saved state from file.
        
        Arguments:
            name: name of file containing saved state
            model_builder: :class:ModelBuilder which was used to construct the network
        
        Returns:
            Instantiated :class:Model with network weights, optimiser state, and input mask loaded from saved state
        
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
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

    def set_input_mask(self, mask:np.ndarray) -> None:
        r'''
        Mask input columns by only using input columns whose indeces are listed in mask

        Arguments:
            mask: array of column indeces to use from all input columns
        '''

        self.input_mask = mask
        
    def fit(self, batch_yielder:BatchYielder, callbacks:Optional[List[AbsCallback]]=None) -> float:
        r'''
        Fit network for one complete iteration of a :class:BatchYielder, i.e. one (sub-)epoch

        Arguments:
            batch_yielder: :class:BatchYielder providing training data in form of tuple of inputs, targtes, and weights as tensors on device
            callbacks: list of :class:AbsCallback to be used during training

        Returns:
            Loss on training data averaged across all minibatches
        '''

        self.model.train()
        self.stop_train = False
        losses = []
        if callbacks is None: callbacks = []
        for c in callbacks: c.on_epoch_begin(by=batch_yielder)

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
              
    def evaluate(self, inputs:Tensor, targets:Tensor, weights:Optional[Tensor]=None, callbacks:Optional[List[AbsCallback]]=None,
                 mask_inputs:bool=True) -> float:
        r'''
        Compute loss on provided data.

        Arguments:
            inputs: input data as tensor on device
            targets: targets as tensor on device
            weights: Optional weights as tensor on device
            callbacks: list of any callbacks to use during evaluation
            mask_inputs: whether to apply input mask if one has been set

        Returns:
            (weighted) loss of model predictions on provided data
        '''

        if callbacks is None: callbacks = []
        self.model.eval()
        for c in callbacks: c.on_eval_begin(inputs=inputs, targets=targets, weights=weights)
        if self.input_mask is not None and mask_inputs: inputs = inputs[:,self.input_mask]
        y_pred = self.model(to_device(inputs.float()))
        if 'multiclass' in self.objective and not isinstance(targets, torch.LongTensor): targets = targets.long().squeeze()
        loss = self.loss(weight=to_device(weights))(y_pred, to_device(targets)) if weights is not None else self.loss()(y_pred, to_device(targets))
        for c in callbacks: c.on_eval_end(loss=loss)        
        return loss.data.item()

    def predict_array(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor], as_np:bool=True, mask_inputs:bool=True) -> Union[np.ndarray, Tensor]:
        r'''
        Pass inputs through network and obtain predictions.

        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device
            as_np: whether to return predictions as Numpy array (otherwise tensor)
            mask_inputs: whether to apply input mask if one has been set

        Returns:
            Model prediction(s) per datapoint
        '''
        
        self.model.eval()
        if self.input_mask is not None and mask_inputs: inputs = inputs[:,self.input_mask]
        if isinstance(inputs, pd.DataFrame): inputs = Tensor(inputs.values)
        if not isinstance(inputs, Tensor): inputs = Tensor(inputs)
        pred = self.model(to_device(inputs.float()))
        if as_np:
            if 'multiclass' in self.objective: return np.exp(to_np(pred))
            else:                              return to_np(pred)
        else:
            return pred

    def predict_folds(self, fy:FoldYielder, pred_name:str='pred') -> None:
        r'''
        Apply model to all dataaccessed by a :class:FoldYielder and save predictions as new group in fold file

        Arguments:
            fy: :class:FoldYielder interfacing to data
            pred_name: name of group to which to save predictions
        '''

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
        r'''
        Apply model to inputed data and compute predictions.
        A compatability method to call :meth:predict_array or :meth:predict_folds, depending on input type.

        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device, or :class:FoldYielder interfacing to data
            as_np: whether to return predictions as Numpy array (otherwise tensor) if inputs are a Numpy array, Pandas DataFrame, or tensor
            pred_name: name of group to which to save predictions if inputs are a :class:FoldYielder

        Returns:
            if inputs are a Numpy array, Pandas DataFrame, or tensor, will return predicitions as either array or tensor
        '''
        if not isinstance(inputs, FoldYielder): return self.predict_array(inputs, as_np=as_np)
        self.predict_folds(inputs, pred_name)

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
            model_builder: if :class:Model was not initialised with a :class:ModelBuilder, you will need to pass one here
        '''

        if model_builder is not None: self.model, self.opt, self.loss = model_builder.get_model()
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
        if '.' in name: name = name[:name.rfind('.')]
        self.export2onnx(name, bs)
        m = onnx.load(f'{name}.onnx')
        tf_rep = prepare(m)
        tf_rep.export_graph(f'{name}.pb')
           
    def get_feat_importance(self, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        r'''
        Call :meth:get_nn_feat_importance passing this :class:Model and provided arguments

        Arguments:
            fy: :class:FoldYielder interfacing to data on which to evaluate importance
            eval_metric: Optional :class:EvalMetric to use for quantifying performance
        '''

        return get_nn_feat_importance(self, fy, eval_metric)

    def get_out_size(self) -> int:
        r'''
        Get number of outputs of model

        Returns:
            Number of outputs of model
        '''

        return self.tail.get_out_size()
