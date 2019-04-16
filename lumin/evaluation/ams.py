import numpy as np
import pandas as pd
from typing import Tuple
from fastprogress import progress_bar

import torch
from torch import Tensor


def calc_ams(s:float, b:float, br:float=0, unc_b:float=0) -> float:
    '''Compute Approximate Median Significance for signal (background) weight s (b),
    fractional systemtatic uncertainty unc_b, and offset br'''
    if b == 0: return -1
    if not unc_b:
        radicand = 2*((s+b+br)*np.log(1.0+s/(b+br))-s)
    else:
        sigma_b_2 = np.square(unc_b*b)
        radicand = 2*(((s+b)*np.log((s+b)*(b+sigma_b_2)/((b**2)+((s+b)*sigma_b_2))))-(((b**2)/sigma_b_2)*np.log(1+((sigma_b_2*s)/(b*(b+sigma_b_2))))))
    return np.sqrt(radicand) if radicand > 0 else -1


def calc_ams_torch(s:Tensor, b:Tensor, br:float=0, unc_b:float=0) -> Tensor:
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
    '''Determine optimum calc_ams and cut,
    wgt_factor used rescale weights to get comparable calc_amss
    sufferes from float precison - not recommended for final evaluation'''
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
    '''Determine optimum calc_ams and cut,
    wgt_factor used rescale weights to get comparable calc_amss
    slower than ams_scan_quick, but doesn't suffer from float precision'''
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



