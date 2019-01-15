import pandas as pd
import numpy as np

from ..evaluation.ams import calc_ams
from ..plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt

'''
Todo:
- Multithread ams calc
'''


def binary_class_cut(in_data:pd.DataFrame, top_perc:float=0.05, min_pred:float=0.9,
                     w_factor:float=1.0, br:float=0.0, syst_unc_b:float=0.0,
                     pred_name:str='pred', targ_name:str='gen_target', weight_name:str='gen_weight',
                     plot_settings:PlotSettings=PlotSettings()) -> float:
    '''Find a fluctaution resiliant cut which should generalise better by 
    taking the mean class prediction of the top top_perc percentage of points
    as ranked by AMS'''
    sig = (in_data.gen_target == 1)
    bkg = (in_data.gen_target == 0)
    if 'ams' not in in_data.columns:
        in_data['ams'] = -1
        in_data.loc[in_data[pred_name] >= min_pred, 'ams'] = in_data[in_data[pred_name] >= min_pred].apply(lambda row:
                                                                                                           calc_ams(w_factor*np.sum(in_data.loc[(in_data[pred_name] >= row[pred_name]) & sig, weight_name]),
                                                                                                                    w_factor*np.sum(in_data.loc[(in_data[pred_name] >= row[pred_name]) & bkg, weight_name]),
                                                                                                                    br=br, unc_b=syst_unc_b), axis=1)
        
    in_data.sort_values(by='ams', ascending=False, inplace=True)
    cuts = in_data[pred_name].values[0:int(top_perc * len(in_data))]

    cut = np.mean(cuts)
    ams = calc_ams(w_factor*np.sum(in_data.loc[(in_data[pred_name] >= cut) & sig, 'gen_weight']),
                   w_factor*np.sum(in_data.loc[(in_data[pred_name] >= cut) & bkg, 'gen_weight']),
                   br=br, unc_b=syst_unc_b)
    
    print(f'Mean cut at {cut} corresponds to AMS of {ams}')
    print(f'Maximum AMS for data is {in_data.iloc[0]["ams"]} at cut of {in_data.iloc[0][pred_name]}')
    with sns.axes_style(plot_settings.style), sns.color_palette(plot_settings.cat_palette) as palette:
        plt.figure(figsize=(plot_settings.w_small, plot_settings.h_small))
        sns.distplot(cuts, label=f'Top {top_perc}%')
        plt.axvline(x=cut, label='Mean prediction', color=palette[1])
        plt.axvline(x=in_data.iloc[0][pred_name], label='Max. AMS', color=palette[2])
        plt.legend(loc=plot_settings.leg_loc, fontsize=plot_settings.leg_sz)
        plt.xticks(fontsize=plot_settings.tk_sz, color=plot_settings.tk_col)
        plt.yticks(fontsize=plot_settings.tk_sz, color=plot_settings.tk_col)
        plt.xlabel('Class prediction', fontsize=plot_settings.lbl_sz, color=plot_settings.lbl_col)
        plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=plot_settings.lbl_sz, color=plot_settings.lbl_col)
        plt.show()
    return cut
