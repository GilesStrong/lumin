import numpy as np
from typing import Optional, Any, Tuple, List, Dict
import pandas as pd
from collections import OrderedDict
from pdpbox import pdp
from pdpbox.pdp import PDPIsolate, PDPInteract
from sklearn.pipeline import Pipeline

from .plot_settings import PlotSettings
from ..utils.misc import to_np
from ..utils.mod_ver import check_pdpbox

import seaborn as sns
import matplotlib.pyplot as plt


def plot_importance(df:pd.DataFrame, feat_name:str='Feature', imp_name:str='Importance',  unc_name:str='Uncertainty',
                    savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot feature importances as computted via `get_nn_feat_importance`, `get_ensemble_feat_importance`, or `rf_rank_features`

    Arguments:
        df: DataFrame containing columns of features, importances and, optionally, uncertainties
        feat_name: column name for features
        imp_name: column name for importances
        unc_name: column name for uncertainties (if present)
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance
    '''

    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = plt.subplots(figsize=(settings.w_large, (0.75)*settings.lbl_sz))
        xerr = None if unc_name not in df else 'Uncertainty'
        df.plot(feat_name, imp_name, 'barh', ax=ax, legend=False, xerr=xerr, error_kw={'elinewidth': 3})
        ax.set_xlabel('Importance via feature permutation', fontsize=settings.lbl_sz, color=settings.lbl_col)
        ax.set_ylabel('Feature', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def plot_embedding(embed:OrderedDict, feat:str, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Visualise weights in provided categorical entity-embedding matrix

    Arguments:
        embed: state_dict of trained nn.Embedding
        feat: name of feature embedded
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance
    '''

    with sns.axes_style(settings.style):
        plt.figure(figsize=(settings.w_small, settings.h_small))
        sns.heatmap(to_np(embed['weight']), annot=True, linewidths=.5, cmap=settings.div_palette, annot_kws={'fontsize':settings.leg_sz})
        plt.xlabel("Embedding", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()

    
def plot_1d_partial_dependence(model:Any, df:pd.DataFrame, feat:str, ignore_feats:Optional[List[str]]=None, input_pipe:Pipeline=None, 
                               sample_sz:Optional[int]=None, wgt_name:Optional[str]=None,  n_clusters:Optional[int]=10, n_points:int=20,
                               pdp_isolate_kargs:Optional[Dict[str,Any]]=None, pdp_plot_kargs:Optional[Dict[str,Any]]=None,
                               savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Wrapper for PDPbox to plot 1D dependence of specified feature using provided NN or RF.
    If features have been preprocessed using an SK-Learn Pipeline, then that can be passed in order to rescale the x-axis back to its original values.

    Arguments:
        model: any trained model with a .predict method
        df: DataFrame containing training data
        feat: feature for which to evaluate the partial dependence of the model
        ignore_feats: features present in training data which were not used to train the model (necessary to correctly deprocess feature using input_pipe)
        input_pipe: SK-Learn Pipeline which was used to process the training data
        sample_sz: if set, will only compute partial dependence on a random sample with replacement of the training data, sampled according to weights (if set).
            Speeds up computation and allows weighted partial dependencies to computed.
        wgt_name: Optional column name to use as sampling weights
        n_points: number of points at which to evaluate the model output, passed to pdp_isolate as num_grid_points
        n_clusters: number of clusters in which to group dependency lines. Set to None to show all lines
        pdp_isolate_kargs: optional dictionary of keyword arguments to pass to pdp_isolate
        pdp_plot_kargs: optional dictionary of keyword arguments to pass to pdp_plot
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance
    '''

    if pdp_isolate_kargs is None: pdp_isolate_kargs = {}
    if pdp_plot_kargs    is None: pdp_plot_kargs    = {}

    if sample_sz is not None:
        if wgt_name is not None:
            weights = df[wgt_name].values.as_type('float64')
            weights *= 1/np.sum(weights)
        df = df.sample(sample_sz, weights=weights)
    elif sample_sz is None and wgt_name is not None:
        print('''A wgt_name has been specified, but sample_sz is None. Weights will be ignored.
                 Please set sample_sz if you wish to compute weighted partical dependcies''')

    iso = pdp.pdp_isolate(model, df, [f for f in df.columns if ignore_feats is None or f not in ignore_feats], feat, num_grid_points=n_points,
                          **pdp_isolate_kargs)
    if input_pipe is not None: _deprocess_iso(iso, input_pipe, feat, df.columns)

    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = pdp.pdp_plot(iso, feat, center=False,  plot_lines=True, cluster=n_clusters is not None, n_cluster_centers=n_clusters,
                               plot_params={'title': None, 'subtitle': None}, figsize=(settings.w_mid, settings.h_mid), **pdp_plot_kargs)
        ax['title_ax'].remove()
        ax['pdp_ax'].set_xlabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        ax['pdp_ax'].set_ylabel("Partial dependence", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def plot_2d_partial_dependence(model:Any, df:pd.DataFrame, feats:Tuple[str,str], ignore_feats:Optional[List[str]]=None, input_pipe:Pipeline=None,
                               sample_sz:Optional[int]=None, wgt_name:Optional[str]=None, n_points:Tuple[int,int]=[20,20],
                               pdp_interact_kargs:Optional[Dict[str,Any]]=None, pdp_interact_plot_kargs:Optional[Dict[str,Any]]=None,
                               savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Wrapper for PDPbox to plot 2D dependence of specified pair of features using provided NN or RF.
    If features have been preprocessed using an SK-Learn Pipeline, then that can be passed in order to rescale them back to their original values.

    Arguments:
        model: any trained model with a .predict method
        df: DataFrame containing training data
        feats: pair of features for which to evaluate the partial dependence of the model
        ignore_feats: features present in training data which were not used to train the model (necessary to correctly deprocess feature using input_pipe)
        input_pipe: SK-Learn Pipeline which was used to process the training data
        sample_sz: if set, will only compute partial dependence on a random sample with replacement of the training data, sampled according to weights (if set).
            Speeds up computation and allows weighted partial dependencies to computed.
        wgt_name: Optional column name to use as sampling weights
        n_points: pair of numbers of points at which to evaluate the model output, passed to pdp_interact as num_grid_points
        n_clusters: number of clusters in which to group dependency lines. Set to None to show all lines
        pdp_isolate_kargs: optional dictionary of keyword arguments to pass to pdp_isolate
        pdp_plot_kargs: optional dictionary of keyword arguments to pass to pdp_plot
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance
    '''    
    
    check_pdpbox()
    if pdp_interact_kargs      is None: pdp_interact_kargs      = {}
    if pdp_interact_plot_kargs is None: pdp_interact_plot_kargs = {}

    if sample_sz is not None:
        if wgt_name is not None:
            weights = df[wgt_name].values.as_type('float64')
            weights *= 1/np.sum(weights)
        df = df.sample(sample_sz, weights=weights)
    elif sample_sz is None and wgt_name is not None:
        print('''A wgt_name has been specified, but sample_sz is None. Weights will be ignored.
                 Please set sample_sz if you wish to compute weighted partical dependcies''')

    interact = pdp.pdp_interact(model, df, [f for f in df.columns if ignore_feats is None or f not in ignore_feats], feats, num_grid_points=n_points,
                                **pdp_interact_kargs)
    if input_pipe is not None: _deprocess_interact(interact, input_pipe, feats, df.columns)
            
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = pdp.pdp_interact_plot(interact, feats, figsize=(settings.h_large, settings.h_large),
                                        plot_params={'title': None, 'subtitle': None, 'cmap':settings.seq_palette}, **pdp_interact_plot_kargs)
        ax['title_ax'].remove()
        ax['pdp_inter_ax'].set_xlabel(feats[0], fontsize=settings.lbl_sz, color=settings.lbl_col)
        ax['pdp_inter_ax'].set_ylabel(feats[1], fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def _deprocess_iso(iso:PDPIsolate, input_pipe:Pipeline, feat:str, feats:List[str]) -> None:
    feat_id = np.argwhere(feats == feat)[0][0]
    try:               in_sz = input_pipe.steps[0][1].n_samples_seen_.shape[0]
    except IndexError: in_sz = input_pipe.steps[0][1].mean_.shape[0]
    if feat_id >= in_sz: return
    x = iso.feature_grids
    x = np.broadcast_to(x[:,None], (x.shape[0], in_sz))
    x = input_pipe.inverse_transform(x)[:,feat_id]
    iso.feature_grids = x
    iso.ice_lines.columns = x


def _deprocess_interact(interact:PDPInteract, input_pipe:Pipeline, feat_pair:Tuple[str,str], feats:List[str]) -> None:
    for i, feat in enumerate(feat_pair):
        feat_id = np.argwhere(feats == feat)[0][0]
        try:               in_sz = input_pipe.steps[0][1].n_samples_seen_.shape[0]
        except IndexError: in_sz = input_pipe.steps[0][1].mean_.shape[0]
        if feat_id > in_sz: continue
        x = interact.feature_grids[i]
        x = np.broadcast_to(x[:,None], (x.shape[0], in_sz))
        x = input_pipe.inverse_transform(x)[:,feat_id]
        interact.feature_grids[i] = x
