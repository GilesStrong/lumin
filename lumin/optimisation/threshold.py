import pandas as pd
import numpy as np
from scipy.constants import golden_ratio

from ..evaluation.ams import calc_ams

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

'''
Todo:
- Multithread ams calc
- Imporve plot colours
'''


def binary_class_cut(in_data:pd.DataFrame, top_perc:float=0.05, min_pred:float=0.9,
                     w_factor:float=1.0, br:float=0.0, syst_unc_b:float=0.0,
                     pred_name:str='pred', targ_name:str='gen_target', weight_name:str='gen_weight') -> float:
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
    plt.figure(figsize=(5*golden_ratio, 5))
    sns.distplot(cuts, label=f'Top {top_perc}%')
    plt.axvline(x=cut, label='Mean prediction', color='green')
    plt.axvline(x=in_data.iloc[0][pred_name], label='Max. AMS', color='red')
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel('Class prediction', fontsize=24, color='black')
    plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    plt.show()
    return cut
