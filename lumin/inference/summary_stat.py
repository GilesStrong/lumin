import numpy as np
import pandas as pd
from typing import List, Optional
from fastprogress import progress_bar

__all__ = ['bin_binary_class_pred']


def bin_binary_class_pred(df:pd.DataFrame, max_unc:float, consider_samples:Optional[List[str]]=None, step_sz:float=1e-3, pred_name:str='pred',
                          sample_name:str='gen_sample', compact_samples:bool=False, class_name:str='gen_target',
                          add_pure_signal_bin:bool=False, max_unc_pure_signal:float=0.1, verbose:bool=True) -> List[float]:
    r'''
    Define bin-edges for binning particle process samples as a function of event class prediction (signal | background) such that the statistical uncertainties on per bin yields are
    below max_unc for each considered sample.

    Arguments:
        df: DataFrame containing the data
        max_unc: maximum fractional statisitcal uncertainty to allow when defining bins
        consider_samples: if set, only listed samples are considered when defining bins
        step_sz: resolution of scan along event prediction
        pred_name: column to use as event class prediction
        sample_name: column to use as particle process fo reach event
        compact_samples: if true, will not consider samples when computing bin edges, only the class
        class_name: name of column to use as class indicator
        add_pure_signal_bin: if true will attempt to add a bin which oonly contains signal (class 1) if the fractional bin-fill uncertainty would be less than
            max_unc_pure_signal
        max_unc_pure_signal: maximum fractional statisitcal uncertainty to allow when defining pure-signal bins
        verbose: whether to show progress bar
        
    Returns:
        list of bin edges 
    '''

    # TODO: allow option for stepping through each event, rather than fixed resolution scan
    
    if consider_samples is None: consider_samples = set(df[sample_name])
    n_min = int((1/max_unc)**2)
    edges,ub,lb = [1],1,0
    if add_pure_signal_bin:
        max_zero = df.loc[df[class_name] == 0, pred_name].max()
        max_zero = (np.floor(max_zero/step_sz)+1)*step_sz
        if len(df[(df[class_name] == 1) & (df[pred_name] > max_zero)]) >= int((1/max_unc_pure_signal)**2):
            edges.append(max_zero)
            ub = max_zero

    for i in progress_bar(np.linspace(ub,lb+step_sz, (ub-lb)/step_sz), display=verbose):
        cut = (df[pred_name] > i) & (df[pred_name] <= edges[-1])
        pops = [len(df[(df[class_name] == c) & cut]) for c in df[class_name].unique()] if compact_samples \
            else [len(df[(df[sample_name] == s) & cut]) for s in consider_samples]
        if np.min(pops) >= n_min: edges.append(i)
    edges.append(0)
    return np.sort(edges)
    