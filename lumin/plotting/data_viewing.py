import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any

import seaborn as sns
import matplotlib.pyplot as plt
from .plot_settings import PlotSettings


# def plot_feat(in_data:Union[pd.DataFrame,List[pd.DataFrame]], feat:str, weight_name:Optional[str]=None, cuts:Optional[List[pd.Series]]=None,
#               labels:Optional[List[str]]=None, plot_bulk:bool=True, n_samples:int=100000, plot_params:Optional[List[Dict[str,Any]]]=None, moments=False,
#               title:Optional[str]=None, savename:Optional[str]=None) -> None:
#     if not isinstance(cuts,   list): cuts   = [cuts]
#     if not isinstance(labels, list): labels = [labels]    

#             if not isinstance(cuts, list): raise ValueError(f"{len(cuts)} plots requested, but no labels passed")
#             elif len(cuts) != len(labels): raise ValueError(f"{len(cuts)} plots requested, but {len(labels)} labels passed")
    
#     plt.figure(figsize=(16, 8))
#         for i in range(len(cuts)):
#             if isinstance(params, list):
#                 tmp_params = params[i]
#             else:
#                 tmp_params = params

#             if plot_bulk:  # Ignore tails for indicative plotting
#                 feat_range = np.percentile(in_data[feat], [1, 99])
#                 # feat_range = np.percentile(in_data.loc[cuts[i], feat], [1,99])
#                 if feat_range[0] == feat_range[1]: break
                
#                 cut = (cuts[i])
#                 cut = cut & (in_data[cut][feat] > feat_range[0]) & (in_data[cut][feat] < feat_range[1])
#                 if isinstance(weight_name, type(None)):
#                     plot_data = in_data.loc[cut, feat]
#                 else:
#                     plot_data = np.random.choice(in_data.loc[cut, feat], n_samples, p=in_data.loc[cut, weight_name] / np.sum(in_data.loc[cut, weight_name]))
                    
#             else:
#                 if isinstance(weight_name, type(None)):
#                     plot_data = in_data.loc[cuts[i], feat]
#                 else:
#                     plot_data = np.random.choice(in_data.loc[cuts[i], feat], n_samples, p=in_data.loc[cuts[i], weight_name] / np.sum(in_data.loc[cuts[i], weight_name]))
            
#             label = labels[i]
#             if moments:
#                 label += r', $\bar{x}=$' + str(np.mean(plot_data)) + r', $\sigma_x=$' + str(np.std(plot_data))

#             sns.distplot(plot_data, label=labels[i], **tmp_params)
#     else:
#         if plot_bulk:  # Ignore tails for indicative plotting
#             feat_range = np.percentile(in_data[feat], [1, 99])
#             if feat_range[0] == feat_range[1]: return -1
            
#             cut = (in_data[feat] > feat_range[0]) & (in_data[feat] < feat_range[1])
             
#             if isinstance(weight_name, type(None)):
#                 plot_data = in_data.loc[cut, feat]
#             else:
#                 plot_data = np.random.choice(in_data.loc[cut, feat], n_samples, p=in_data.loc[cut, weight_name] / np.sum(in_data.loc[cut, weight_name]))     
                
#         else:
#             if isinstance(weight_name, type(None)):
#                 plot_data = in_data[feat]
#             else:
#                 plot_data = np.random.choice(in_data[feat], n_samples, p=in_data[weight_name] / np.sum(in_data[weight_name]))
        
#         label = ''
#         if moments:
#             label += r', $\bar{x}=$' + str(np.mean(plot_data)) + r', $\sigma_x=$' + str(np.std(plot_data))
#         sns.distplot(plot_data, label=label, **params)

#     if loop or moments:
#         plt.legend(loc='best', fontsize=16)
#     plt.xticks(fontsize=16, color='black')
#     plt.yticks(fontsize=16, color='black')
#     plt.ylabel("Density", fontsize=24, color='black')
#     plt.xlabel(feat, fontsize=24, color='black')
#     plt.show()


def compare_events(events: list) -> None:
    with sns.axes_style('white'), sns.color_palette('coloublind'):
        fig, axs = plt.subplots(3, len(events), figsize=(9 * len(events), 27))
        for vector in [x[:-3] for x in events[0].columns if '_px' in x.lower()]:
            for i, in_data in enumerate(events):
                x = in_data[vector + '_px'].values[0]
                try: y = in_data[vector + '_py'].values[0]
                except KeyError: y = 0
                try: z = in_data[vector + '_pz'].values[0]
                except KeyError: z = 0
                axs[0, i].plot((0, x), (0, y), label=vector)
                axs[1, i].plot((0, z), (0, x), label=vector)
                axs[2, i].plot((0, z), (0, y), label=vector)
        circle = np.linspace(-np.pi, np.pi, 200)
        box = np.linspace(-1, 1, 80)
        for ax in axs[0]:
            ax.scatter(np.cos(circle), np.sin(circle), color='grey', alpha=0.5)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_x$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)  
        for ax in axs[1]:
            ax.scatter(2 * box, np.ones_like(box), color='grey', alpha=0.5)
            ax.scatter(2 * box, -np.ones_like(box), color='grey', alpha=0.5)
            ax.scatter(-2 * np.ones_like(box), box, color='grey', alpha=0.5)
            ax.scatter(2 * np.ones_like(box), box, color='grey', alpha=0.5)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_x$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        for ax in axs[2]: 
            ax.scatter(2 * box, np.ones_like(box), color='grey', alpha=0.5)
            ax.scatter(2 * box, -np.ones_like(box), color='grey', alpha=0.5)
            ax.scatter(-2 * np.ones_like(box), box, color='grey', alpha=0.5)
            ax.scatter(2 * np.ones_like(box), box, color='grey', alpha=0.5)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        fig.show()
