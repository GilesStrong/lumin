from typing import Union, Tuple, Callable, Optional, List
import numpy as np
import json
from pathlib import Path

import torch
from torch import Tensor

from .callback import Callback
from ..data.batch_yielder import BatchYielder
from ..data.fold_yielder import FoldYielder
from ...utils.misc import to_np, to_device
from ..models.abs_model import AbsModel


class BinaryLabelSmooth(Callback):
    r'''
    Callback for applying label smoothing to binary classes, based on https://arxiv.org/abs/1512.00567
    Applies smoothing during both training and inference.

    Arguments:
        coefs: Smoothing coefficients: 0->coef[0] 1->1-coef[1]. if passed float, coef[0]=coef[1]
        model: not used, only for compatability

    Examples::
        >>> lbl_smooth = BinaryLabelSmooth(0.1)
        >>> lbl_smooth = BinaryLabelSmooth((0.1, 0.02))
    '''

    def __init__(self, coefs:Union[float,Tuple[float,float]]=0, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.coefs = coefs if isinstance(coefs, tuple) else (coefs, coefs)
    
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None:
        '''Apply smoothing at train-time'''
        by.targets = by.targets.astype(float)
        by.targets[by.targets == 0] = self.coefs[0]
        by.targets[by.targets == 1] = 1-self.coefs[1]
         
    def on_eval_begin(self, targets:Tensor, **kargs) -> None:
        '''Apply smoothing at test-time'''
        targets[targets == 0] = self.coefs[0]
        targets[targets == 1] = 1-self.coefs[1]


class SequentialReweight(Callback):
    r'''
    **Experiemntal proceedure**
    During ensemble training, sequentially reweight training data in last validation fold based on prediction performance of last trained model.
    Reweighting highlights data which are easier or more difficult to predict to the next model being trained.

    Arguments:
        reweight_func: callable function returning a tensor of same shape as targets, ideally quantifying model-prediction performance
        scale: multiplicative factor for rescaling returned tensor of reweight_func
        model: :class:Model to provide predictions, alternatively call :meth:set_model

    Examples::
        >>> seq_reweight = SequentialReweight(reweight_func=nn.BCELoss(reduction='none'), scale=0.1)
    '''

    def __init__(self, reweight_func:Callable[[Tensor,Tensor], Tensor], scale:float=1e-1, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.scale,self.reweight_func = scale,reweight_func

    def reweight_fold(self, fy:FoldYielder, fold_id:int) -> None:
        fld = fy.get_fold(fold_id)
        preds = self.model.predict_array(fld['inputs'], as_np=False)
        coefs = to_np(self.reweight_func(preds, to_device(Tensor(fld['targets']))))
        weight = np.sum(fld['weights'])
        fld['weights'] += self.scale*coefs*fld['weights']
        fld['weights'] *= weight/np.sum(fld['weights'])
        fy.foldfile[f'fold_{fold_id}/weights'][...] = fld['weights'].squeeze()
    
    def on_train_end(self, fy:FoldYielder, val_id:int, **kargs) -> None: self.reweight_fold(fy, val_id)


class SequentialReweightClasses(SequentialReweight):
    r'''
    **Experiemntal proceedure**
    Version of :class:SequentialReweight designed for classification, which renormalises class weights to original weight-sum after reweighting
    During ensemble training, sequentially reweight training data in last validation fold based on prediction performance of last trained model.
    Reweighting highlights data which are easier or more difficult to predict to the next model being trained.

    Arguments:
        reweight_func: callable function returning a tensor of same shape as targets, ideally quantifying model-prediction performance
        scale: multiplicative factor for rescaling returned tensor of reweight_func
        model: :class:Model to provide predictions, alternatively call :meth:set_model

    Examples::
        >>> seq_reweight = SequentialReweight(reweight_func=nn.BCELoss(reduction='none'), scale=0.1)
    '''

    def reweight_fold(self, fy:FoldYielder, fold_id:int) -> None:
        fld = fy.get_fold(fold_id)
        preds = self.model.predict_array(fld['inputs'], as_np=False)
        coefs = to_np(self.reweight_func(preds, to_device(Tensor(fld['targets']))))
        for c in set(fld['targets'].squeeze()):
            weight = np.sum(fld['weights'][fld['targets'] == c])
            fld['weights'][fld['targets'] == c] += self.scale*(coefs*fld['weights'])[fld['targets'] == c]
            fld['weights'][fld['targets'] == c] *= weight/np.sum(fld['weights'][fld['targets'] == c])
        fy.foldfile[f'fold_{fold_id}/weights'][...] = fld['weights'].squeeze()


class BootstrapResample(Callback):
    r'''
    Callback for bootstrap sampling new training datasets from original training data during (ensemble) training.

    Arguments:
        n_folds: the number of folds present in training :class:FoldYielder
        bag_each_time: whether to sample a new set for each sub-epoch or to use the same sample each time
        reweight: whether to reweight the sampleed data to mathch the weight sum (per class) of the original data
        model: not used, only for compatability

    Examples::
        >>> bs_resample BootstrapResample(n_folds=len(train_fy))
    '''

    def __init__(self, n_folds:int, bag_each_time:bool=False, reweight:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.n_trn_flds,self.bag_each_time,self.reweight = n_folds-1,bag_each_time,reweight
        
    def get_sample(self, length:int) -> np.ndarray: return np.random.choice(range(length), length, replace=True)
    
    def resample(self, sample:np.ndarray, inputs:Union[np.ndarray,Tensor], targets:Union[np.ndarray,Tensor],
                 weights:Union[np.ndarray,Tensor,None]) -> None:
        pkg = np if isinstance(weights, np.ndarray) else torch 
        # Get weight sums before resampling
        if weights is not None and self.reweight:
            if 'class' in self.objective:
                weight_sum = {}
                for c in pkg.unique(targets.squeeze()): weight_sum[c] = pkg.sum(weights[targets.squeeze() == c])
            else:
                weight_sum = pkg.sum(weights)
                    
        # Resample
        inputs[...] = inputs[sample]
        targets[...] = targets[sample]
        if weights is not None:
            weights[...] = weights[sample]
        
            # Reweight
            if self.reweight:
                if 'class' in self.objective:
                    for c in weight_sum: weights[targets.squeeze() == c] *= weight_sum[c]/pkg.sum(weights[targets.squeeze() == c])
                else: weights *= weight_sum/pkg.sum(weights)
        
    def on_train_begin(self, **kargs) -> None:
        self.iter,self.samples,self.objective = 0,[],None
        np.random.seed()  # Is this necessary?
    
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None:
        if self.bag_each_time or self.iter < self.n_trn_flds:
            sample = self.get_sample(len(by.targets))
            if not self.bag_each_time: self.samples.append(sample)
        else:
            sample = self.samples[self.iter % self.n_trn_flds]
        self.iter += 1
        if self.objective is None: self.objective = by.objective
        self.resample(sample, by.inputs, by.targets, by.weights)


class FeatureSubsample(Callback):
    r'''
    Callback for training a model on a random sub-sample of the range of possible input features.
    Only sub-samples continuous features. Number of continuous inputs infered from model.
    Associated :class:Model will automatically mask its inputs during inference; simply provide inputs with the same number of columns as trainig data. 

    Arguments:
        cont_feats: list of all continuous features in input data. Order must match.
        model: :class:Model being trained, alternatively call :meth:set_model        

    Examples::
        >>> feat_subsample = FeatureSubsample(cont_feats=['pT', 'eta', 'phi'])
    '''

    # TODO cont feat names no longer required only number; move to infer number of cont feats from model_builder and batch_yielder

    def __init__(self, cont_feats:List[str], model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.cont_feats = cont_feats
        
    def _sample_feats(self) -> None:
        cont_idxs = np.random.choice(range(len(self.cont_feats)), size=self.model.model_builder.n_cont_in, replace=False)
        self.feat_idxs = np.hstack((cont_idxs, len(self.cont_feats)+np.arange(self.model.model_builder.cat_embedder.n_cat_in)))
        self.feat_idxs.sort()
    
    def on_train_begin(self, model_num:int, savepath:Path, **kargs) -> None:
        self.model_num,self.savepath = model_num,savepath
        np.random.seed()  # Is this necessary?
        self._sample_feats()
        self.model.set_input_mask(self.feat_idxs)
        
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None: by.inputs = by.inputs[:,self.feat_idxs]
