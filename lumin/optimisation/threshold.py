import pandas as pd
import numpy as np
from typing import Tuple
import warnings

from ..evaluation.ams import calc_ams
from ..plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt


def binary_class_cut(df:pd.DataFrame, top_perc:float=5.0, min_pred:float=0.9,
                     wgt_factor:float=1.0, br:float=0.0, syst_unc_b:float=0.0,
                     pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight',
                     plot_settings:PlotSettings=PlotSettings()) -> Tuple[float,float,float]:
    r'''
    Depreciated: renamed to plot_rank_order_dendrogram
    '''
    
    # XXX Remove in v0.4
    warnings.warn('''binary_class_cut has been renamed to binary_class_cut_by_ams. binary_class_cut is now depreciated and will be removed in v0.4''')
    return binary_class_cut_by_ams(df=df, top_perc=top_perc, min_pred=min_pred,
                                   wgt_factor=wgt_factor, br=br, syst_unc_b=syst_unc_b,
                                   pred_name=pred_name, targ_name=targ_name, wgt_name=wgt_name,
                                   plot_settings=plot_settings)


def binary_class_cut_by_ams(df:pd.DataFrame, top_perc:float=5.0, min_pred:float=0.9,
                            wgt_factor:float=1.0, br:float=0.0, syst_unc_b:float=0.0,
                            pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight',
                            plot_settings:PlotSettings=PlotSettings()) -> Tuple[float,float,float]:
    r'''
    Optimise a cut on a signal-background classifier prediction by the Approximate Median Significance
    Cut which should generalise better by taking the mean class prediction of the top top_perc percentage of points as ranked by AMS

    Arguments:
        df: Pandas DataFrame containing data
        top_perc: top percentage of events to consider as ranked by AMS
        min_pred: minimum prediction to consider
        wgt_factor: single multiplicative coeficient for rescaling signal and background weights before computing AMS
        br: background offset bias
        syst_unc_b: fractional systemtatic uncertainty on background
        pred_name: column to use as predictions
        targ_name: column to use as truth labels for signal and background
        wgt_name: column to use as weights for signal and background events
        plot_settings: :class:PlotSettings class to control figure appearance

    Returns:
        Optimised cut
        AMS at cut
        Maximum AMS
    '''

    # TODO: Multithread AMS calculation
    
    sig, bkg = (df.gen_target == 1), (df.gen_target == 0)
    if 'ams' not in df.columns:
        df['ams'] = -1
        df.loc[df[pred_name] >= min_pred, 'ams'] = df[df[pred_name] >= min_pred].apply(
            lambda row: calc_ams(wgt_factor*np.sum(df.loc[(df[pred_name] >= row[pred_name]) & sig, wgt_name]),
                                 wgt_factor*np.sum(df.loc[(df[pred_name] >= row[pred_name]) & bkg, wgt_name]),
                                 br=br, unc_b=syst_unc_b), axis=1)
        
    sort = df.sort_values(by='ams', ascending=False)
    cuts = sort[pred_name].values[0:int(top_perc*len(sort)/100)]

    cut = np.mean(cuts)
    ams = calc_ams(wgt_factor*np.sum(sort.loc[(sort[pred_name] >= cut) & sig, 'gen_weight']),
                   wgt_factor*np.sum(sort.loc[(sort[pred_name] >= cut) & bkg, 'gen_weight']),
                   br=br, unc_b=syst_unc_b)
    
    print(f'Mean cut at {cut} corresponds to AMS of {ams}')
    print(f'Maximum AMS for data is {sort.iloc[0]["ams"]} at cut of {sort.iloc[0][pred_name]}')
    with sns.axes_style(plot_settings.style), sns.color_palette(plot_settings.cat_palette) as palette:
        plt.figure(figsize=(plot_settings.w_small, plot_settings.h_small))
        sns.distplot(cuts, label=f'Top {top_perc}%')
        plt.axvline(x=cut, label='Mean prediction', color=palette[1])
        plt.axvline(x=sort.iloc[0][pred_name], label='Max. AMS', color=palette[2])
        plt.legend(loc=plot_settings.leg_loc, fontsize=plot_settings.leg_sz)
        plt.xticks(fontsize=plot_settings.tk_sz, color=plot_settings.tk_col)
        plt.yticks(fontsize=plot_settings.tk_sz, color=plot_settings.tk_col)
        plt.xlabel('Class prediction', fontsize=plot_settings.lbl_sz, color=plot_settings.lbl_col)
        plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=plot_settings.lbl_sz, color=plot_settings.lbl_col)
        plt.show()
    return cut, ams, sort.iloc[0]["ams"]
