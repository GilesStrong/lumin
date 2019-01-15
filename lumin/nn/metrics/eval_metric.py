import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from ..data.fold_yielder import FoldYielder


class EvalMetric(ABC):
    def __init__(self, targ_name, weight_name):
        self.targ_name,self.weight_name = targ_name,weight_name
        self.lower_better = True

    def get_df(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame()
        df['gen_weight'] = data.get_column(column=self.weight_name, n_folds=1, fold_id=index)
        df['gen_target'] = data.get_column(column=self.targ_name,   n_folds=1, fold_id=index)
        df['pred']       = y_pred
        return df

    @abstractmethod
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        pass
