import numpy as np
import pandas as pd
from typing import List, Optional
from fastprogress import progress_bar


def bin_binary_class_pred(df:pd.DataFrame, max_unc:float, consider_samples:Optional[List[str]]=None, step_sz:float=1e-3,
                          pred_name:str='pred', sample_name:str='gen_sample') -> List[float]:
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

    Returns:
        list of bin edges 
    '''

    # TODO: allow option for stepping through each event, rather than fixed resolution scan
    
    if consider_samples is None: consider_samples = set(df[sample_name])
    n_min = int((1/max_unc)**2)
    edges = [1]
    for i in progress_bar(np.linspace(1, step_sz, 1/step_sz)):
        cut = (df[pred_name] > i) & (df[pred_name] <= edges[-1])
        pops = [len(df[(df[sample_name] == s) & cut]) for s in consider_samples]
        if np.min(pops) >= n_min: edges.append(i)
    edges.append(0)
    return np.sort(edges)
    