import numpy as np
import pandas as pd
from typing import Tuple
from fastprogress import progress_bar

import torch
from torch import Tensor


def calc_ams(s:float, b:float, br:float=0, unc_b:float=0) -> float:
    r'''
    Compute Approximate Median Significance (https://arxiv.org/abs/1007.1727)

    Arguments:
        s: signal weight
        b: background weight
        br: background offset bias
        unc_b: fractional systemtatic uncertainty on background

    Returns:
        Approximate Median Significance if b > 0 else -1
    '''

    if b == 0: return -1
    if not unc_b:
        radicand = 2*((s+b+br)*np.log(1.0+s/(b+br))-s)
    else:
        sigma_b_2 = np.square(unc_b*b)
        radicand = 2*(((s+b)*np.log((s+b)*(b+sigma_b_2)/((b**2)+((s+b)*sigma_b_2))))-(((b**2)/sigma_b_2)*np.log(1+((sigma_b_2*s)/(b*(b+sigma_b_2))))))
    return np.sqrt(radicand) if radicand > 0 else -1


def calc_ams_torch(s:Tensor, b:Tensor, br:float=0, unc_b:float=0) -> Tensor:
    r'''
    Compute Approximate Median Significance (https://arxiv.org/abs/1007.1727) using Tensor inputs

    Arguments:
        s: signal weight
        b: background weight
        br: background offset bias
        unc_b: fractional systemtatic uncertainty on background

    Returns:
        Approximate Median Significance if b > 0 else 1e-18 * s
    '''

    '''Compute Approximate Median Significance with torch for signal (background) weight s (b),
    fractional systemtatic uncertainty unc_b, and offset br'''
    if b == 0: return 1e-18*s
    if not unc_b:
        radicand = 2*((s+b+br)*torch.log(1.0+s/(b+br))-s)
    else:
        sigma_b_2 = torch.square(unc_b*b)
        radicand = 2*(((s+b)*torch.log((s+b)*(b+sigma_b_2)/((b**2)+((s+b)*sigma_b_2))))-(((b**2)/sigma_b_2)*torch.log(1+((sigma_b_2*s)/(b*(b+sigma_b_2))))))
    return torch.sqrt(radicand) if radicand > 0 else 1e-18*s


def ams_scan_quick(df:pd.DataFrame, wgt_factor:float=1, br:float=0, syst_unc_b:float=0,
                   pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight') -> Tuple[float,float]:
    r'''
    Scan accross a range of possible prediction thresholds in order to maximise the Approximate Median Significance (https://arxiv.org/abs/1007.1727).
    Note that whilst this method is quicker than :meth:ams_scan_slow, it sufferes from float precison - not recommended for final evaluation.
    
    Arguments:
        df: DataFrame containing prediction data
        wgt_factor: factor to reweight signal and background weights
        br: background offset bias
        syst_unc_b: fractional systemtatic uncertainty on background
        pred_name: column to use as predictions
        targ_name: column to use as truth labels for signal and background
        wgt_name: column to use as weights for signal and background events

    Returns:
        maximum AMS
        prediction threshold corresponding to maximum AMS
    '''

    max_ams, threshold = 0, 0.0
    df = df.sort_values(by=[pred_name])
    s = np.sum(df.loc[(df[targ_name] == 1), wgt_name])
    b = np.sum(df.loc[(df[targ_name] == 0), wgt_name])

    for i, cut in enumerate(df[pred_name]):
        ams = calc_ams(max(0, s*wgt_factor), max(0, b*wgt_factor), br, syst_unc_b)
        if ams > max_ams: max_ams, threshold = ams, cut
        if df[targ_name].values[i]: s -= df[wgt_name].values[i]
        else:                       b -= df[wgt_name].values[i]        
    return max_ams, threshold


def ams_scan_slow(df:pd.DataFrame, wgt_factor:float=1, br:float=0, syst_unc_b:float=0, 
                  use_stat_unc:bool=False, start_cut:float=0.9, min_events:int=10,
                  pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight', show_prog:bool=True) -> Tuple[float,float]:
    r'''
    Scan accross a range of possible prediction thresholds in order to maximise the Approximate Median Significance (https://arxiv.org/abs/1007.1727).
    Note that whilst this method is slower than :meth:ams_scan_quick, it does not suffer as much from float precison.
    Additionally it allows one to account for statistical uncertainty in AMS calculation.
    
    Arguments:
        df: DataFrame containing prediction data
        wgt_factor: factor to reweight signal and background weights
        br: background offset bias
        syst_unc_b: fractional systemtatic uncertainty on background
        use_stat_unc: whether to account for the statistical uncertainty on the background
        start_cut: minimum prediction to consider; useful for speeding up scan
        min_events: minimum number of background unscaled events required to pass threshold
        pred_name: column to use as predictions
        targ_name: column to use as truth labels for signal and background
        wgt_name: column to use as weights for signal and background events
        show_prog: whether to display progress and ETA of scan

    Returns:
        maximum AMS
        prediction threshold corresponding to maximum AMS
    '''
    
    max_ams, threshold = 0, 0.0
    sig, bkg = df[df[targ_name] == 1], df[df[targ_name] == 0]
    syst_unc_b2 = np.square(syst_unc_b)

    for i, cut in enumerate(progress_bar(df.loc[df[pred_name] >= start_cut, pred_name].values, display=show_prog, leave=show_prog)):
        bkg_pass = bkg.loc[(bkg[pred_name] >= cut), wgt_name]
        n_bkg = len(bkg_pass)
        if n_bkg < min_events: continue

        s = np.sum(sig.loc[(sig[pred_name] >= cut), wgt_name])
        b = np.sum(bkg_pass)
        if use_stat_unc: unc_b = np.sqrt(syst_unc_b2+(1/n_bkg))
        else:            unc_b = syst_unc_b

        ams = calc_ams(s*wgt_factor, b*wgt_factor, br, unc_b)
        if ams > max_ams: max_ams, threshold = ams, cut      
    return max_ams, threshold



