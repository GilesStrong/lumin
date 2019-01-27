import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

from ..data.fold_yielder import FoldYielder


class EvalMetric(ABC):
    def __init__(self, targ_name:str='targets', weight_name:Optional[str]=None):
        self.targ_name,self.weight_name = targ_name,weight_name
        self.lower_better = True

    def get_df(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame()
        if self.weight_name is not None: df['gen_weight'] = data.get_column(column=self.weight_name, n_folds=1, fold_id=index)
        
        targets = data.get_column(column=self.targ_name, n_folds=1, fold_id=index)
        if len(targets.shape) > 1:
            for t in range(targets.shape[-1]):
                df[f'gen_target_{t}'] = targets[:,t]
        else:
            df['gen_target'] = targets

        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            for p in range(y_pred.shape[-1]):
                df[f'pred_{p}'] = y_pred[:,p]
        else:
            df['pred'] = y_pred.squeeze()
        return df

    @abstractmethod
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        pass
