from typing import Union, Tuple, Callable, Optional, List
import numpy as np
import pandas as pd
from fastcore.all import is_listy

import torch
from torch import Tensor

from .callback import Callback, OldCallback
from ..data.batch_yielder import BatchYielder
from ..data.fold_yielder import FoldYielder
from ...utils.misc import to_np, to_device
from ..models.abs_model import AbsModel, OldAbsModel

__all__ = ['BinaryLabelSmooth', 'BootstrapResample', 'ParametrisedPrediction']


class OldBinaryLabelSmooth(OldCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.data_callbacks.BinaryLabelSmooth`.
        It is a copy of the old `BinaryLabelSmooth` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, coefs:Union[float,Tuple[float,float]]=0, model:Optional[OldAbsModel]=None):
        super().__init__(model=model)
        self.coefs = coefs if isinstance(coefs, tuple) else (coefs, coefs)
    
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None:
        r'''
        Apply smoothing at train-time
        '''

        by.targets = by.targets.astype(float)
        by.targets[by.targets == 0] = self.coefs[0]
        by.targets[by.targets == 1] = 1-self.coefs[1]
         
    def on_eval_begin(self, targets:Tensor, **kargs) -> None:
        r'''
        Apply smoothing at test-time
        '''

        targets[targets == 0] = self.coefs[0]
        targets[targets == 1] = 1-self.coefs[1]


class BinaryLabelSmooth(Callback):
    r'''
    Callback for applying label smoothing to binary classes, based on https://arxiv.org/abs/1512.00567
    Applies smoothing during both training.

    Arguments:
        coefs: Smoothing coefficients: 0->coef[0] 1->1-coef[1]. if passed float, coef[0]=coef[1]

    Examples::
        >>> lbl_smooth = BinaryLabelSmooth(0.1)
        >>>
        >>> lbl_smooth = BinaryLabelSmooth((0.1, 0.02))
    '''

    def __init__(self, coefs:Union[float,Tuple[float,float]]=0):
        super().__init__()
        self.coefs = coefs if isinstance(coefs, tuple) else (coefs, coefs)
    
    def on_fold_begin(self) -> None:
        r'''
        Apply smoothing
        '''

        if self.model.fit_params.state != 'train': return
        self.model.fit_params.by.targets = self.model.fit_params.by.targets.astype(float)
        m = self.model.fit_params.self.model.fit_params.by.targets == 0
        self.model.fit_params.by.targets[m] = self.coefs[0]
        self.model.fit_params.by.targets[m] = 1-self.coefs[1]


class SequentialReweight(OldCallback):
    r'''
    .. Caution:: Experiemntal proceedure

    .. Attention:: This class is depreciated.
        It will be removed in V0.8

    During ensemble training, sequentially reweight training data in last validation fold based on prediction performance of last trained model.
    Reweighting highlights data which are easier or more difficult to predict to the next model being trained.

    Arguments:
        reweight_func: callable function returning a tensor of same shape as targets, ideally quantifying model-prediction performance
        scale: multiplicative factor for rescaling returned tensor of reweight_func
        model: :class:`~lumin.nn.models.model.Model` to provide predictions, alternatively call :meth:`~lumin.nn.models.Model.set_model`

    Examples::
        >>> seq_reweight = SequentialReweight(
        ...     reweight_func=nn.BCELoss(reduction='none'), scale=0.1)
    '''

    # XXX remove in V0.8

    def __init__(self, reweight_func:Callable[[Tensor,Tensor], Tensor], scale:float=1e-1, model:Optional[OldAbsModel]=None):
        super().__init__(model=model)
        self.scale,self.reweight_func = scale,reweight_func

    def _reweight_fold(self, fy:FoldYielder, fold_id:int, bs:Optional[int]=None) -> None:
        fld = fy.get_fold(fold_id)
        preds = self.model.predict_array(fld['inputs'], as_np=False, bs=bs)
        coefs = to_np(self.reweight_func(preds, to_device(Tensor(fld['targets']))))
        weight = np.sum(fld['weights'])
        fld['weights'] += self.scale*coefs*fld['weights']
        fld['weights'] *= weight/np.sum(fld['weights'])
        fy.foldfile[f'fold_{fold_id}/weights'][...] = fld['weights'].squeeze()
    
    def on_train_end(self, fy:FoldYielder, val_id:int, bs:Optional[int]=None, **kargs) -> None:
        r'''
        Reweighs the validation fold once training is finished

        Arguments:
            fy: FoldYielder providing the training and validation data
            fold_id: Fold index which was used for validation
        '''
        
        self._reweight_fold(fy, val_id, bs)


class SequentialReweightClasses(SequentialReweight):
    r'''
    .. Caution:: Experiemntal proceedure

    .. Attention:: This class is depreciated.
        It will be removed in V0.8

    Version of :class:`~lumin.nn.callbacks.data_callbacks.SequentialReweight` designed for classification, which renormalises class weights to original weight-sum after reweighting
    During ensemble training, sequentially reweight training data in last validation fold based on prediction performance of last trained model.
    Reweighting highlights data which are easier or more difficult to predict to the next model being trained.

    Arguments:
        reweight_func: callable function returning a tensor of same shape as targets, ideally quantifying model-prediction performance
        scale: multiplicative factor for rescaling returned tensor of reweight_func
        model: :class:`~lumin.nn.models.model.Model` to provide predictions, alternatively call :meth:`~lumin.nn.models.Model.set_model`

    Examples::
        >>> seq_reweight = SequentialReweight(
        ...     reweight_func=nn.BCELoss(reduction='none'), scale=0.1)
    '''

    # XXX remove in V0.8

    def _reweight_fold(self, fy:FoldYielder, fold_id:int) -> None:
        fld = fy.get_fold(fold_id)
        preds = self.model.predict_array(fld['inputs'], as_np=False)
        coefs = to_np(self.reweight_func(preds, to_device(Tensor(fld['targets']))))
        for c in set(fld['targets'].squeeze()):
            weight = np.sum(fld['weights'][fld['targets'] == c])
            fld['weights'][fld['targets'] == c] += self.scale*(coefs*fld['weights'])[fld['targets'] == c]
            fld['weights'][fld['targets'] == c] *= weight/np.sum(fld['weights'][fld['targets'] == c])
        fy.foldfile[f'fold_{fold_id}/weights'][...] = fld['weights'].squeeze()


class OldBootstrapResample(OldCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.data_callbacks.BootstrapResample`.
        It is a copy of the old `BootstrapResample` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, n_folds:int, bag_each_time:bool=False, reweight:bool=True, model:Optional[OldAbsModel]=None):
        super().__init__(model=model)
        self.n_trn_flds,self.bag_each_time,self.reweight = n_folds-1,bag_each_time,reweight
        
    def _get_sample(self, length:int) -> np.ndarray: return np.random.choice(range(length), length, replace=True)
    
    def _resample(self, sample:np.ndarray, inputs:Union[np.ndarray,Tensor], targets:Union[np.ndarray,Tensor],
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
        r'''
        Resets internal parameters to prepare for a new training
        '''

        self.iter,self.samples,self.objective = 0,[],None
        np.random.seed()  # Is this necessary?
    
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None:
        r'''
        Resamples training data for new epoch

        Arguments:
            by: BatchYielder providing data for the upcoming epoch
        '''

        if self.bag_each_time or self.iter < self.n_trn_flds:
            sample = self._get_sample(len(by.targets))
            if not self.bag_each_time: self.samples.append(sample)
        else:
            sample = self.samples[self.iter % self.n_trn_flds]
        self.iter += 1
        if self.objective is None: self.objective = by.objective
        self._resample(sample, by.inputs, by.targets, by.weights)


class BootstrapResample(Callback):
    r'''
    Callback for bootstrap sampling new training datasets from original training data during (ensemble) training.

    Arguments:
        n_folds: the number of folds present in training :class:`~lumin.nn.data.fold_yielder.FoldYielder`
        bag_each_time: whether to sample a new set for each sub-epoch or to use the same sample each time
        reweight: whether to reweight the sampleed data to mathch the weight sum (per class) of the original data

    Examples::
        >>> bs_resample BootstrapResample(n_folds=len(train_fy))
    '''

    def __init__(self, n_folds:int, bag_each_time:bool=False, reweight:bool=True):
        super().__init__()
        self.n_trn_flds,self.bag_each_time,self.reweight = n_folds-1,bag_each_time,reweight
        
    def _get_sample(self, length:int) -> np.ndarray: return np.random.choice(range(length), length, replace=True)
    
    def _resample(self, sample:np.ndarray, by:BatchYielder) -> None:
        # Get weight sums before resampling
        if by.weights is not None and self.reweight:
            if 'class' in self.model.objective:
                weight_sum = {}
                for c in torch.unique(by.targets.squeeze()): weight_sum[c] = torch.sum(by.weights[by.targets.squeeze() == c])
            else:
                weight_sum = torch.sum(by.weights)
                    
        # Resample
        by.inputs[...] = by.inputs[sample]
        by.targets[...] = by.targets[sample]
        if by.weights is not None:
            by.weights[...] = by.weights[sample]
        
            # Reweight
            if self.reweight:
                if 'class' in self.model.objective:
                    for c in weight_sum: by.weights[by.targets.squeeze() == c] *= weight_sum[c]/torch.sum(by.weights[by.targets.squeeze() == c])
                else: by.weights *= weight_sum/torch.sum(by.weights)
        
    def on_train_begin(self) -> None:
        r'''
        Resets internal parameters to prepare for a new training
        '''

        super().on_train_begin()
        self.iter,self.samples = 0,[]
        np.random.seed()  # Is this necessary?
    
    def on_fold_begin(self) -> None:
        r'''
        Resamples training data for new epoch
        '''

        if self.model.fit_params.state != 'train': return
        if self.bag_each_time or self.iter < self.n_trn_flds:
            sample = self._get_sample(len(self.model.fit_params.by.targets))
            if not self.bag_each_time: self.samples.append(sample)
        else:
            sample = self.samples[self.iter % self.n_trn_flds]
        self.iter += 1
        self._resample(sample, self.model.fit_params.by)


class OldParametrisedPrediction(OldCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.data_callbacks.ParametrisedPrediction`.
        It is a copy of the old `ParametrisedPrediction` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, feats:List[str], param_feat:Union[List[str],str], param_val:Union[List[float],float], model:Optional[AbsModel]=None):
        super().__init__(model=model)
        if not isinstance(param_feat, list): param_feat = [param_feat]
        if not isinstance(param_val, list):  param_val  = [param_val]
        self.feats,self.param_feat,self.param_val = feats,param_feat,param_val
        self.param_idx = [self.feats.index(f) for f in self.param_feat]
        
    def on_pred_begin(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor], **kargs) -> None:
        r'''
        Adjusts the data to be passed to the model by setting in place the parameterisation feature to the preset value
        '''

        if isinstance(inputs, pd.DataFrame):
            for f, v in zip(self.param_feat, self.param_val): inputs[f] = v
        else:
            for f, v in zip(self.param_idx, self.param_val):  inputs[:, f] = v


class ParametrisedPrediction(Callback):
    r'''
    Callback for running predictions for a parametersied network (https://arxiv.org/abs/1601.07913); one which has been trained using one of more inputs which
    represent e.g. different hypotheses for the classes such as an unknown mass of some new particle.
    In such a scenario, multiple signal datasets could be used for training, with background receiving a random mass. During prediction one then needs to set
    these parametrisation features all to the same values to evaluat the model's response for that hypothesis.
    This callback can be passed to the predict method of the model/ensemble to adjust the parametrisation features to the desired values.

    Arguments:
        feats: list of feature names used during training (in the same order)
        param_feat: the feature name which is to be adjusted, or a list of features to adjust
        param_val: the value to which to set the paramertisation feature, of the list of values to set the parameterisation features to

    Examples::
        >>> mass_param = ParametrisedPrediction(train_feats, 'res_mass', 300)
        >>> model.predict(fold_yeilder, pred_name=f'pred_mass_300', callbacks=[mass_param])
        >>>
        >>> mass_param = ParametrisedPrediction(train_feats, 'res_mass', 300)
        >>> spin_param = ParametrisedPrediction(train_feats, 'spin', 1)
        >>> model.predict(fold_yeilder, pred_name=f'pred_mass_300', callbacks=[mass_param, spin_param])

    '''

    def __init__(self, feats:List[str], param_feat:Union[List[str],str], param_val:Union[List[float],float]):
        super().__init__()
        if not is_listy(param_feat): param_feat = [param_feat]
        if not is_listy(param_val):  param_val  = [param_val]
        self.feats,self.param_feat,self.param_val = feats,param_feat,param_val
        self.param_idx = [self.feats.index(f) for f in self.param_feat]
        
    def on_pred_begin(self) -> None:
        r'''
        Adjusts the data to be passed to the model by setting in place the parameterisation feature to the preset value
        '''

        for f, v in zip(self.param_idx, self.param_val):  self.fit_params.by.inputs[:, f] = v
