import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

import scipy
from scipy.cluster import hierarchy as hc

from .plot_settings import PlotSettings
from ..utils.statistics import uncert_round, get_moments

import seaborn as sns
import matplotlib.pyplot as plt


def plot_feat(df:pd.DataFrame, feat:str, wgt_name:Optional[str]=None, cuts:Optional[List[pd.Series]]=None,
              labels:Optional[List[str]]='', plot_bulk:bool=True, n_samples:int=100000,
              plot_params:List[Dict[str,Any]]={}, size='mid', show_moments=True, ax_labels={'y': 'Density', 'x': None},
              savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    '''A flexible function to provide indicative information about the 1D distribution of a feature.
    By default it will produce a weighted KDE+histogram for the [1,99] percentile of the data,
    as wellas compute the mean and standard deviation of the data in this region.
    By passing a list of cuts and labels, it will plot multiple distributions of the same feature for different cuts.
    Since it is designed to provide quick, indicative information, more specific functions (such as `plot_kdes_from_bs`)
    should be used to provide final results.'''
    if not isinstance(labels, list): labels = [labels]
    if not isinstance(cuts,   list): cuts   = [cuts]
    if len(cuts) != len(labels): raise ValueError(f"{len(cuts)} plots requested, but {len(labels)} labels passed")
    
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.str2sz(size, 'x'), settings.str2sz(size, 'y')))
        for i in range(len(cuts)):
            tmp_plot_params = plot_params[i] if isinstance(plot_params, list) else plot_params

            if plot_bulk:  # Ignore tails for indicative plotting
                feat_range = np.percentile(np.nan_to_num(df[feat]), [1, 99])
                if feat_range[0] == feat_range[1]: break
                cut = (df[feat] > feat_range[0]) & (df[feat] < feat_range[1])
                if cuts[i] is not None: cut = cut & (cuts[i])
                if wgt_name is None:
                    plot_data = np.nan_to_num(df.loc[cut, feat])
                else:
                    weights = df.loc[cut, wgt_name].values.astype('float64')
                    weights /= weights.sum()
                    plot_data = np.random.choice(np.nan_to_num(df.loc[cut, feat]), n_samples, p=weights)
            else:
                tmp_data = df if cuts[i] is not None else df.loc[cuts[i]]
                if wgt_name is None:
                    plot_data = np.nan_to_num(tmp_data[feat])
                else:
                    weights = tmp_data[wgt_name].values.astype('float64')
                    weights /= weights.sum()
                    plot_data = np.random.choice(np.nan_to_num(tmp_data[feat]), n_samples, p=weights)
            label = labels[i]
            if show_moments:
                moms = get_moments(plot_data)
                mean = uncert_round(moms[0], moms[1])
                std = uncert_round(moms[2], moms[3])
                label += r' $\bar{x}=$' + f'{mean[0]}±{mean[1]}' + r', $\sigma_x=$' + f'{std[0]}±{std[1]}'

            sns.distplot(plot_data, label=label, **tmp_plot_params)

        if len(cuts) > 1 or show_moments: plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.ylabel(ax_labels['y'], fontsize=settings.lbl_sz, color=settings.lbl_col)
        x_lbl = feat if ax_labels['x'] is None else ax_labels['x']
        plt.xlabel(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def compare_events(events: list) -> None:
    '''Compare at least two events side by side in their transverse and longitudinal projections'''
    with sns.axes_style('whitegrid'), sns.color_palette('tab10'):
        fig, axs = plt.subplots(3, len(events), figsize=(9*len(events), 18), gridspec_kw={'height_ratios': [1, 0.5, 0.5]})
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
        for ax in axs[0]:
            ax.add_artist(plt.Circle((0, 0), 1, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_x$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)  
        for ax in axs[1]:
            ax.add_artist(plt.Rectangle((-2, -1), 4, 2, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_x$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        for ax in axs[2]: 
            ax.add_artist(plt.Rectangle((-2, -1), 4, 2, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        fig.show()


def plot_dendrogram(df:pd.DataFrame, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    '''Plot dendrogram of features in df clustered via Spearman's rank correlation coefficient'''
    with sns.axes_style('white'), sns.color_palette(settings.cat_palette):
        corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
        corr_condensed = hc.distance.squareform(1-corr)
        z = hc.linkage(corr_condensed, method='average')

        plt.figure(figsize=(settings.w_large, (0.5*len(df.columns))))
        hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=settings.lbl_sz)
        plt.xlabel('Distance', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def plot_kdes_from_bs(arr:np.ndarray, bs_stats:List[Dict[str,Any]], name2args:Dict[str,Dict[str,Any]], 
                      feat:str, units:Optional[str]=None, moments=True,
                      savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    '''Plot KDEs computed via bootstrap_stats'''
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i, name in enumerate(name2args):
            if 'color' not in name2args[name]: name2args[name]['color'] = palette[i]
            if 'label' in name2args[name]:
                name2args[name]['condition'] = name2args[name]['label']
                name2args[name].pop('label')
            if 'condition' in name2args[name] and moments:
                mean, mean_unc = uncert_round(np.mean(bs_stats[f'{name}_mean']), np.std(bs_stats[f'{name}_mean'], ddof=1))
                std, std_unc = uncert_round(np.mean(bs_stats[f'{name}_std']), np.std(bs_stats[f'{name}_std'], ddof=1))
                name2args[name]['condition'] += r', $\overline{x}=' + r'{}\pm{}\ \sigma= {}\pm{}$'.format(mean, mean_unc, std, std_unc)
            sns.tsplot(data=bs_stats[f'{name}_kde'], time=arr, **name2args[name])

        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        y_lbl = r'$\frac{1}{N}\ \frac{dN}{d' + feat.replace('$','') + r'}$'
        if units is not None:
            x_lbl = feat + r'$\ [' + units + r']$'
            y_lbl += r'$\ [' + units + r'^{-1}]$'
        else:
            x_lbl = feat
        plt.xlabel(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(y_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()  
