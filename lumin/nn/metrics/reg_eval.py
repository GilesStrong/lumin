import numpy as np
from typing import Optional, Callable
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

from ...utils.statistics import bootstrap_stats
from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder

__all__ = ['RegPull', 'RegAsProxyPull']


class RegPull(EvalMetric):
    r'''
    Compute mean or standard deviation of delta or pull of some feature which is being directly regressed to.
    Optionally, use bootstrap resampling on validation data.

    Arguments:
        return_mean: whether to return the mean or the standard deviation
        use_bootstrap: whether to bootstrap resamples validation fold when computing statisitic
        use_weights: whether to actually use weights if wgt_name is set
        use_pull: whether to return the pull (differences / targets) or delta (differences)
        targ_name: name of group in fold file containing regression targets
        wgt_name: name of group in fold file containing datapoint weights

    Examples::
        >>> mean_pull  = RegPull(return_mean=True, use_bootstrap=True,
        ...                      use_pull=True)
        >>>
        >>> std_delta  = RegPull(return_mean=False, use_bootstrap=True,
        ...                      use_pull=False)
        >>>
        >>> mean_pull  = RegPull(return_mean=True, use_bootstrap=False,
        ...                      use_pull=True, wgt_name='weights')
    '''

    # TODO: Check how this handels multi-target regression, may need to adjust averaging axis & DescrStatsW may not handle multi-dimensional data well.
    # TODO: Remove use_weights and rely on whether wgt_name is None

    def __init__(self, return_mean:bool, use_bootstrap:bool=False, use_weights:bool=True, use_pull:bool=True, targ_name:str='targets',
                 wgt_name:Optional[str]=None):
        super().__init__(targ_name=targ_name, wgt_name=wgt_name)
        self.use_bootstrap,self.use_weights,self.return_mean,self.use_pull = use_bootstrap,use_weights,return_mean,use_pull

    def _compute(self, df:pd.DataFrame) -> float:
        df['diff'] = df['pred']-df['gen_target']
        if self.use_pull: df['diff'] /= df['gen_target']
        if self.use_weights and 'gen_weight' in df.columns:
            weights = df['gen_weight'].values.astype('float64')/df['gen_weight'].values.astype('float64').sum()
        else:
            weights = None
        
        if self.use_bootstrap:
            bs_args = {'data': df['diff'], 'mean': self.return_mean, 'std': True, 'n':100}
            if self.use_weights and 'gen_weight' in df.columns: bs_args['weights'] = weights
            bs = bootstrap_stats(bs_args)
            return np.mean(bs['_mean']) if self.return_mean else np.mean(bs['_std'])
        else:
            if self.return_mean:
                return np.average(df['diff'], weights=weights)
            else:
                return DescrStatsW(df['diff'].values, ddof=1, weights=weights*len(weights) if weights is not None else None).std
            
    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        r'''
        Compute statisitic on fold using provided predictions.

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            Statistic set in initialisation computed on the chsoen fold

        Examples::
            >>> mean = mean_pull.evaluate(train_fy, val_id, val_preds)
        '''

        return self._compute(self.get_df(fy, idx, y_pred))


class RegAsProxyPull(RegPull):
    r'''
    Compute mean or standard deviation of delta or pull of some feature which is being indirectly regressed to via a proxy function.
    Optionally, use bootstrap resampling on validation data.

    Arguments:
        proxy_func: function which acts on regression predictions and adds pred and gen_target columns to the Pandas DataFrame it is passed which contains
            prediction columns pred_{i}
        return_mean: whether to return the mean or the standard deviation
        use_bootstrap: whether to bootstrap resamples validation fold when computing statisitic
        use_weights: whether to actually use weights if wgt_name is set
        use_pull: whether to return the pull (differences / targets) or delta (differences)
        targ_name: name of group in fold file containing regression targets
        wgt_name: name of group in fold file containing datapoint weights

    Examples::
        >>> def reg_proxy_func(df):
        >>>     df['pred'] = calc_pair_mass(df, (1.77682, 1.77682),
        ...                                 {targ[targ.find('_t')+3:]:
        ...                                 f'pred_{i}' for i, targ
        ...                                 in enumerate(targ_feats)})
        >>>     df['gen_target'] = 125
        >>>    
        >>> std_delta = RegAsProxyPull(proxy_func=reg_proxy_func,
        ...                            return_mean=False, use_pull=False)
    '''

    def __init__(self, proxy_func:Callable[[pd.DataFrame],None], return_mean:bool, use_bootstrap:bool=False, use_weights:bool=True, 
                 use_pull:bool=True, targ_name:str='targets', wgt_name:Optional[str]=None):
        super().__init__(use_bootstrap=use_bootstrap, use_weights=use_weights, return_mean=return_mean, use_pull=use_pull, targ_name=targ_name,
                         wgt_name=wgt_name)
        self.proxy_func = proxy_func
            
    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        r'''
        Compute statisitic on fold using provided predictions.

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            Statistic set in initialisation computed on the chsoen fold

        Examples::
            >>> mean = mean_pull.evaluate(train_fy, val_id, val_preds)
        '''

        df = self.get_df(fy, idx, y_pred)
        self.proxy_func(df)
        return self._compute(df)
