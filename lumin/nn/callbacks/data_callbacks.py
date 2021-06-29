from typing import Union, Tuple, List, Optional
import numpy as np
from fastcore.all import is_listy, store_attr
from abc import abstractmethod, ABCMeta

import torch
from torch import Tensor

from .callback import Callback
from ..data.batch_yielder import BatchYielder

__all__ = ['BinaryLabelSmooth', 'BootstrapResample', 'ParametrisedPrediction', 'TargReplace']


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
        self.param_val = list(param_val)
        self.param_idx = [feats.index(f) for f in param_feat]
        
    def on_pred_begin(self) -> None:
        r'''
        Adjusts the data to be passed to the model by setting in place the parameterisation feature to the preset value
        '''

        for f, v in zip(self.param_idx, self.param_val):  self.model.fit_params.by.inputs[:, f] = v


class TargReplace(Callback):
    r'''
    Callback to replace target data with requested data from foldfile, allowing one to e.g. train two models simultaneously with the same inputs but different targets for e.g. adversarial training.
    At the end of validation epochs, the target data is swapped back to the original target data, to allow for the correct computation of any metrics
    
    Arguments:
        targ_feats: list of column names in foldfile to get and horizontally stack to replace target data in current :class:`~lumin.nn.data.batch_yielder.BatchYielder`
        
    Examples::
        >>> targ_replace = TargReplace(['is_fake'])
        >>> targ_replace = TargReplace(['class', 'is_fake'])
        
    '''
    
    def __init__(self, targ_feats:List[str]):
        store_attr()
        super().__init__()
        if not is_listy(self.targ_feats): self.targ_feats = list(self.targ_feats)

    def on_fold_begin(self) -> None:
        r'''
        Stack new target datasets and replace in target data in current :class:`~lumin.nn.data.batch_yielder.BatchYielder`
        '''
        
        targs = []
        idx = self.model.fit_params.trn_idx if self.model.fit_params.state == 'train' else self.model.fit_params.val_idx
        for t in self.targ_feats:
            targs.append(self.model.fit_params.fy.get_column(t, n_folds=1, fold_idx=idx, add_newaxis=True))
        self.model.fit_params.by.targets = np.hstack(targs)
        
    def on_epoch_end(self) -> None:
        r'''
        Swap target data back at the end of validation epochs
        '''
        
        if self.model.fit_params.state != 'valid': return
        self.model.fit_params.by.targets = self.model.fit_params.fy.get_column('targets', n_folds=1, fold_idx=self.model.fit_params.val_idx, add_newaxis=True)


class AbsWeightData(Callback, metaclass=ABCMeta):
    r'''
    Callback to weight folds of data accoridng to a function of the inputs or targets.
    Inherit and override the `weight_func` method according to your task.

    Arguments:
        on_eval: if true, also weight data during validation and testing
    '''

    def __init__(self, on_eval:bool):
        super().__init__()
        self.on_eval = on_eval
    
    @abstractmethod
    def weight_func(self, x:Union[np.ndarray,Tensor], mx:Optional[Union[np.ndarray,Tensor]], y:Union[np.ndarray,Tensor], w:Union[np.ndarray,Tensor]) \
        -> Union[np.ndarray,Tensor]: pass
    
    def on_fold_begin(self) -> None:
        r'''
        Weight all data in fold.
        '''

        if self.model.fit_params.state != 'train' and not self.on_eval: return
        self.model.fit_params.by.weights = self.weight_func(x=self.model.fit_params.by.inputs, mx=self.model.fit_params.by.matrix_inputs,
                                                            y=self.model.fit_params.by.targets, w=self.model.fit_params.by.weights)
