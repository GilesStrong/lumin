import pandas as pd
import numpy as np
from typing import Tuple

from ..evaluation.ams import calc_ams
from ..plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt

'''
Todo:
- Multithread ams calc
'''


def binary_class_cut(df:pd.DataFrame, top_perc:float=5.0, min_pred:float=0.9,
                     wgt_factor:float=1.0, br:float=0.0, syst_unc_b:float=0.0,
                     pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight',
                     plot_settings:PlotSettings=PlotSettings()) -> Tuple[float,float,float]:
    '''Find a fluctaution resiliant cut which should generalise better by 
    taking the mean class prediction of the top top_perc percentage of points
    as ranked by AMS'''
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
