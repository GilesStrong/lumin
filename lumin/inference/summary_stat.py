import numpy as np
import pandas as pd
from typing import List, Optional


def bin_binary_class_pred(df:pd.DataFrame, min_unc:float, consider:Optional[List[str]]=None, step_sz:float=1e-3,
                          pred_name:str='pred', sample_name:str='gen_sample') -> List[float]:
    """Bin samples such that uncertainties are below min_unc for each considered sample"""
    if consider is None: consider = set(df[sample_name])
    n_min = int((1/min_unc)**2)
    upper = [1]
    i = 1
    while i-step_sz >= 0:
        i -= step_sz
        cut = (df[pred_name] > i) & (df[pred_name] <= upper[-1])
        pops = [len(df[(df[sample_name] == s) & cut]) for s in consider]
        if np.min(pops) >= n_min: upper.append(i)
    edges = upper + [0]
    return np.sort(edges)
    