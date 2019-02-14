import numpy as np
from typing import Optional
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

from ...utils.statistics import bootstrap_stats
from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder


class RegPull(EvalMetric):
    '''Compute mean or std of delta or pull of some feature which is being directly regressed to.
    Optionally, use bootstrap resampling on validation data.'''
    def __init__(self, use_bootstrap:bool=False, use_weights:bool=True, return_mean=False, use_pull=True, targ_name:str='targets', wgt_name:Optional[str]=None):
        super().__init__(targ_name=targ_name, wgt_name=wgt_name)
        self.use_bootstrap,self.use_weights,self.return_mean,self.use_pull = use_bootstrap,use_weights,return_mean,use_pull

    def compute(self, df:pd.DataFrame) -> float:
        df['diff'] = (df['pred']-df['gen_target'])
        if self.use_pull: df['diff'] /= df['gen_target']
        weights = df['gen_weight'].values.astype('float64')/df['gen_weight'].values.astype('float64').sum() if self.use_weights and 'gen_weight' in df.columns else None
        
        if self.use_bootstrap:
            bs_args = {'data': df['diff'], 'mean': self.return_mean, 'std': True, 'n':100}
            if self.use_weights and 'gen_weight' in df.columns: bs_args['weights'] = weights
            bs = bootstrap_stats(bs_args)
            return np.mean(bs['_mean']) if self.return_mean else np.mean(bs['_std'])
        else:
            return np.average(df['diff'], weights=weights) if self.return_mean else DescrStatsW(df['diff'].values, ddof=1, weights=weights*len(weights) if weights is not None else None).std
            
    def evaluate(self, data:FoldYielder, idx:int, y_pred:np.ndarray) -> float: return self.compute(self.get_df(data, idx, y_pred))


class RegAsProxyPull(RegPull):
    '''Compute mean or std of delta or pull of some feature which is being indirectly regressed to via a proxy function.
    Optionally, use bootstrap resampling on validation data.'''
    def __init__(self, proxy_func, use_bootstrap:bool=False, use_weights:bool=True, return_mean=False, 
                 use_pull=True, targ_name:str='targets', wgt_name:Optional[str]=None):
        super().__init__(use_bootstrap=use_bootstrap, use_weights=use_weights, return_mean=return_mean, use_pull=use_pull, targ_name=targ_name, wgt_name=wgt_name)
        self.proxy_func = proxy_func
            
    def evaluate(self, data:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, idx, y_pred)
        self.proxy_func(df)
        return self.compute(df)
