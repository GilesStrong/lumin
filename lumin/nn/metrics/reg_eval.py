import numpy as np
from typing import Optional
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

from ...utils.statistics import bootstrap_stats
from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder


class RegPull(EvalMetric):
    def __init__(self, use_bs:bool=False, use_weights:bool=True, ret_mean=False, use_pull=True, targ_name:str='targets', weight_name:Optional[str]=None):
        super().__init__(targ_name=targ_name, weight_name=weight_name)
        self.use_bs,self.use_weights,self.ret_mean,self.use_pull = use_bs,use_weights,ret_mean,use_pull

    def compute(self, df:pd.DataFrame) -> float:
        df['diff'] = (df['pred']-df['gen_target'])
        if self.use_pull: df['diff'] /= df['gen_target']
        weights = df['gen_weight'].values.astype('float64')/df['gen_weight'].values.astype('float64').sum() if self.use_weights else None
        
        if self.use_bs:
            bs_args = {'data': df['diff'], 'mean': self.ret_mean, 'std': True, 'n':100}
            if self.use_weights and 'gen_weight' in df.columns: bs_args['weights'] = weights
            bs = bootstrap_stats(bs_args)
            return np.mean(bs['_mean']) if self.ret_mean else np.mean(bs['_std'])
        else:
            return np.average(df['diff'], weights=weights) if self.ret_mean else DescrStatsW(df['diff'].values, ddof=1, weights=weights*len(weights)).std
            
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, index, y_pred)
        return self.compute(df)


class RegAsProxyPull(RegPull):
    def __init__(self, func, use_bs:bool=False, use_weights:bool=True, ret_mean=False, use_pull=True, targ_name:str='targets', weight_name:Optional[str]=None):
        super().__init__(use_bs=use_bs, use_weights=use_weights, ret_mean=ret_mean, use_pull=use_pull, targ_name=targ_name, weight_name=weight_name)
        self.func = func
            
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, index, y_pred)
        self.func(df)
        return self.compute(df)
