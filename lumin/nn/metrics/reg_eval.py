import numpy as np

from ...utils.statistics import bootstrap_stats
from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder


class RegPull(EvalMetric):
    def __init__(self, ret_mean=False, use_pull=True):
        super().__init__()
        self.ret_mean,self.use_pull = ret_mean,use_pull
            
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, index, y_pred)
        df['diff'] = (df['pred']-df['gen_target'])
        if self.use_pull: df['diff'] /= df['gen_target']
        bs = bootstrap_stats({'data': df['diff'], 'mean': self.ret_mean, 'std': True, 'n':100})
        return np.mean(bs['_mean']) if self.ret_mean else np.mean(bs['_std'])


class RegAsProxyPull(RegPull):
    def __init__(self, func, ret_mean=False, use_pull=True):
        super().__init__(ret_mean=ret_mean, use_pull=use_pull)
        self.func = func
            
    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, index, y_pred)
        self.func(df)
        df['diff'] = (df['pred']-df['gen_target'])
        if self.use_pull: df['diff'] /= df['gen_target']
        bs = bootstrap_stats({'data': df['diff'], 'mean': self.ret_mean, 'std': True, 'n':100})
        return np.mean(bs['_mean']) if self.ret_mean else np.mean(bs['_std'])
