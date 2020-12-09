import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from fastcore.all import store_attr

import torch

from ..data.fold_yielder import FoldYielder
from ..callbacks.callback import Callback

__all__ = ['EvalMetric']


class OldEvalMetric(ABC):
    r'''
    Abstract class for evaluating performance of a model using some metric

    Arguments:
        targ_name: name of group in fold file containing regression targets
        wgt_name: name of group in fold file containing datapoint weights
    '''

    def __init__(self, targ_name:str='targets', wgt_name:Optional[str]=None): self.targ_name,self.wgt_name,self.lower_metric_better = targ_name,wgt_name,True

    def get_df(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> pd.DataFrame:
        r'''
        Returns a DataFrame for the given fold containing targets, weights, and predictions

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            DataFrame for the given fold containing targets, weights, and predictions
        '''

        df = pd.DataFrame()
        if self.wgt_name is not None: df['gen_weight'] = fy.get_column(column=self.wgt_name, n_folds=1, fold_idx=idx)
        
        targets = fy.get_column(column=self.targ_name, n_folds=1, fold_idx=idx)
        if len(targets.shape) > 1:
            for t in range(targets.shape[-1]): df[f'gen_target_{t}'] = targets[:,t]
        else:
            df['gen_target'] = targets
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            for p in range(y_pred.shape[-1]): df[f'pred_{p}'] = y_pred[:,p]
        else:
            df['pred'] = y_pred.squeeze()
        return df

    @abstractmethod
    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        r'''
        Evaluate the required metric for a given fold and set of predictions

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            metric value
        '''

        pass


class EvalMetric(Callback):
    r'''
    Abstract class for evaluating performance of a model using some metric

    Arguments:
        lower_metric_better: whether a lower metric value should be treated as representing better perofrmance
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted
    '''

    def __init__(self, lower_metric_better:bool, main_metric:bool=True): store_attr()

    def on_train_begin(self) -> None:
        r'''
        Ensures that only one main metric is used
        '''

        super().on_train_begin()
        self.metric = None
        if self.main_metric:
            for c in self.model.fit_params.cbs:
                if hasattr(c, 'main_metric'): c.main_metric = False
            self.main_metric = True

    def on_epoch_begin(self) -> None: self.preds,self.metric = [],None
    
    def on_forwards_end(self) -> None:
        if self.model.fit_params.state == 'valid': self.preds.append(self.model.fit_params.y_pred.cpu().detach())
    
    def on_epoch_end(self) -> None:
        self.preds = torch.cat(self.preds)
        self.targets = self.model.fit_params.by.targets
        self.weights = self.model.fit_params.by.weights if self.model.fit_params.by.use_weights else None
        self.metric = self.evaluate()
        del self.preds

    @abstractmethod
    def evaluate(self) -> float:
        r'''
        Evaluate the required metric for a given fold and set of predictions

        Returns:
            metric value
        '''

        pass

    def get_df(self) -> pd.DataFrame:
        r'''
        Returns a DataFrame for the given fold containing targets, weights, and predictions

        Returns:
            DataFrame for the given fold containing targets, weights, and predictions
        '''

        df = pd.DataFrame()
        if hasattr(self, 'wgt_name'):
            df['gen_weight'] = self.model.fit_params.fy.get_column(column=self.wgt_name, n_folds=1, fold_idx=self.model.fit_params.val_idx)
        else:
            df['gen_weight'] = self.weights
        
        if hasattr(self, 'targ_name'):
            targets = self.model.fit_params.fy.get_column(column=self.targ_name, n_folds=1, fold_idx=self.model.fit_params.val_idx)
        else:
            targets = self.targets
        
        if len(targets.shape) > 1:
            for t in range(targets.shape[-1]): df[f'gen_target_{t}'] = targets[:,t]
        else:
            df['gen_target'] = targets

        if len(self.preds.shape) > 1 and self.preds.shape[-1] > 1:
            for p in range(self.preds.shape[-1]): df[f'pred_{p}'] = self.preds[:,p]
        else:
            df['pred'] = self.preds.squeeze()
        return df
