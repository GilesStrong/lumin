import numpy as np
from typing import Optional, Any
import pandas as pd
from collections import OrderedDict
from pdpbox import pdp

from .plot_settings import PlotSettings
from ..utils.misc import to_np

import seaborn as sns
import matplotlib.pyplot as plt


def plot_fi(df:pd.DataFrame, feat_name:str='Feature', imp_name:str='Importance',  unc_name:str='Uncertainty',
            savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = plt.subplots(figsize=(settings.w_large, (0.5)*settings.lbl_sz))
        xerr = None if unc_name not in df else 'Uncertainty'
        df.plot(feat_name, imp_name, 'barh', ax=ax, legend=False, xerr=xerr, error_kw={'elinewidth': 3})
        ax.set_xlabel('Importance via feature permutation', fontsize=16, color='black')
        ax.set_ylabel('Feature', fontsize=16, color='black')
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def plot_embedding(emb:OrderedDict, feat:str, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    with sns.axes_style(settings.style):
        plt.figure(figsize=(settings.w_small, settings.h_small))
        sns.heatmap(to_np(emb['weight']), annot=True, linewidths=.5, cmap=settings.div_palette, annot_kws={'fontsize':settings.leg_sz})
        
        plt.xlabel("Embedding", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()

    
def plot_pdp(model:Any, df:pd.DataFrame, feat:str, sample_sz:Optional[int]=None, weights:Optional[np.ndarray]=None, n_clusters:int=10, n_points:int=20,
             savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    if sample_sz is not None: df = df.sample(sample_sz, weights=weights)
    iso = pdp.pdp_isolate(model, df, df.columns, feat, num_grid_points=20)
    
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = pdp.pdp_plot(iso, feat, plot_lines=True, cluster=n_clusters is not None or n_clusters != 0, n_cluster_centers=n_clusters,
                               plot_params={'title': None, 'subtitle': None}, figsize=(settings.w_mid, 2*settings.h_small))
        ax['title_ax'].remove()
        ax['pdp_ax'].set_xlabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        ax['pdp_ax'].set_ylabel("Partial dependence", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()
