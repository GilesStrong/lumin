import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

from ..data.fold_yielder import FoldYielder


class EvalMetric(ABC):
    '''Abstract class for evaluating performance of a model using some metric'''
    def __init__(self, targ_name:str, wgt_name:Optional[str]=None): self.targ_name,self.wgt_name,self.lower_metric_better = targ_name,wgt_name,True

    def get_df(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> pd.DataFrame:
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
    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float: pass
