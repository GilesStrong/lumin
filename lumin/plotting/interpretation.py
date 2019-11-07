import numpy as np
from typing import Optional, Any, Tuple, List, Dict, Union
import pandas as pd
from collections import OrderedDict
from pdpbox import pdp
from pdpbox.pdp import PDPIsolate, PDPInteract
from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from .plot_settings import PlotSettings
from ..utils.misc import to_np, FowardHook
from ..utils.mod_ver import check_pdpbox
from ..nn.models.abs_model import AbsModel

__all__ = ['plot_importance', 'plot_embedding', 'plot_1d_partial_dependence', 'plot_2d_partial_dependence', 'plot_multibody_weighted_outputs',
           'plot_bottleneck_weighted_inputs']


def plot_importance(df:pd.DataFrame, feat_name:str='Feature', imp_name:str='Importance',  unc_name:str='Uncertainty', threshold:Optional[float]=None,
                    x_lbl:str='Importance via feature permutation', savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot feature importances as computted via `get_nn_feat_importance`, `get_ensemble_feat_importance`, or `rf_rank_features`

    Arguments:
        df: DataFrame containing columns of features, importances and, optionally, uncertainties
        feat_name: column name for features
        imp_name: column name for importances
        unc_name: column name for uncertainties (if present)
        threshold: if set, will draw a line at the threshold hold used for feature importance
        x_lbl: label to put on the x-axis
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        fig, ax = plt.subplots(figsize=(settings.w_large, (0.75)*settings.lbl_sz))
        xerr = None if unc_name not in df else 'Uncertainty'
        df.plot(feat_name, imp_name, 'barh', ax=ax, legend=False, xerr=xerr, error_kw={'elinewidth': 3}, color=palette[0])
        if threshold is not None:
            ax.axvline(x=threshold, label=f'Threshold {threshold}', color=palette[1], linestyle='--', linewidth=3)
            plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        ax.set_xlabel(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
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
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    with sns.axes_style(**settings.style):
        plt.figure(figsize=(settings.w_small, settings.h_small))
        sns.heatmap(to_np(embed['weight']), annot=True, fmt='.1f', linewidths=.5, cmap=settings.div_palette, annot_kws={'fontsize':settings.leg_sz})
        plt.xlabel("Embedding", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()

    
def plot_1d_partial_dependence(model:Any, df:pd.DataFrame, feat:str, train_feats:List[str], ignore_feats:Optional[List[str]]=None, input_pipe:Pipeline=None, 
                               sample_sz:Optional[int]=None, wgt_name:Optional[str]=None,  n_clusters:Optional[int]=10, n_points:int=20,
                               pdp_isolate_kargs:Optional[Dict[str,Any]]=None, pdp_plot_kargs:Optional[Dict[str,Any]]=None,
                               y_lim:Optional[Union[Tuple[float,float],List[float]]]=None, 
                               savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Wrapper for PDPbox to plot 1D dependence of specified feature using provided NN or RF.
    If features have been preprocessed using an SK-Learn Pipeline, then that can be passed in order to rescale the x-axis back to its original values.

    Arguments:
        model: any trained model with a .predict method
        df: DataFrame containing training data
        feat: feature for which to evaluate the partial dependence of the model
        train_feats: list of all training features including ones which were later ignored, i.e. input features considered when input_pipe was fitted
        ignore_feats: features present in training data which were not used to train the model (necessary to correctly deprocess feature using input_pipe)
        input_pipe: SK-Learn Pipeline which was used to process the training data
        sample_sz: if set, will only compute partial dependence on a random sample with replacement of the training data, sampled according to weights (if set).
            Speeds up computation and allows weighted partial dependencies to computed.
        wgt_name: Optional column name to use as sampling weights
        n_points: number of points at which to evaluate the model output, passed to pdp_isolate as num_grid_points
        n_clusters: number of clusters in which to group dependency lines. Set to None to show all lines
        pdp_isolate_kargs: optional dictionary of keyword arguments to pass to pdp_isolate
        pdp_plot_kargs: optional dictionary of keyword arguments to pass to pdp_plot
        y_lim: If set, will limit y-axis plot range to tuple
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    if pdp_isolate_kargs is None: pdp_isolate_kargs = {}
    if pdp_plot_kargs    is None: pdp_plot_kargs    = {}

    if sample_sz is not None or wgt_name is not None:
        if wgt_name is None:
            weights = None
        else:
            weights = df[wgt_name].values.astype('float64')
            weights *= 1/np.sum(weights)
        df = df.sample(len(df) if sample_sz is None else sample_sz, weights=weights, replace=True)

    iso = pdp.pdp_isolate(model, df, [f for f in train_feats if ignore_feats is None or f not in ignore_feats], feat, num_grid_points=n_points,
                          **pdp_isolate_kargs)
    if input_pipe is not None: _deprocess_iso(iso, input_pipe, feat, train_feats)

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        fig, ax = pdp.pdp_plot(iso, feat, center=False,  plot_lines=True, cluster=n_clusters is not None, n_cluster_centers=n_clusters,
                               plot_params={'title': None, 'subtitle': None}, figsize=(settings.w_mid, settings.h_mid), **pdp_plot_kargs)
        ax['title_ax'].remove()
        ax['pdp_ax'].set_xlabel(feat, fontsize=settings.lbl_sz, color=settings.lbl_col)
        ax['pdp_ax'].set_ylabel("Partial dependence", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if y_lim is not None: ax['pdp_ax'].set_ylim(y_lim)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()


def plot_2d_partial_dependence(model:Any, df:pd.DataFrame, feats:Tuple[str,str], train_feats:List[str], ignore_feats:Optional[List[str]]=None,
                               input_pipe:Pipeline=None, sample_sz:Optional[int]=None, wgt_name:Optional[str]=None, n_points:Tuple[int,int]=[20,20],
                               pdp_interact_kargs:Optional[Dict[str,Any]]=None, pdp_interact_plot_kargs:Optional[Dict[str,Any]]=None,
                               savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Wrapper for PDPbox to plot 2D dependence of specified pair of features using provided NN or RF.
    If features have been preprocessed using an SK-Learn Pipeline, then that can be passed in order to rescale them back to their original values.

    Arguments:
        model: any trained model with a .predict method
        df: DataFrame containing training data
        feats: pair of features for which to evaluate the partial dependence of the model
        train_feats: list of all training features including ones which were later ignored, i.e. input features considered when input_pipe was fitted
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
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''    
    
    check_pdpbox()
    if pdp_interact_kargs      is None: pdp_interact_kargs      = {}
    if pdp_interact_plot_kargs is None: pdp_interact_plot_kargs = {}

    if sample_sz is not None or wgt_name is not None:
        if wgt_name is None:
            weights = None
        else:
            weights = df[wgt_name].values.astype('float64')
            weights *= 1/np.sum(weights)
        df = df.sample(len(df) if sample_sz is None else sample_sz, weights=weights, replace=True)

    interact = pdp.pdp_interact(model, df, [f for f in train_feats if ignore_feats is None or f not in ignore_feats], feats, num_grid_points=n_points,
                                **pdp_interact_kargs)
    if input_pipe is not None: _deprocess_interact(interact, input_pipe, feats, train_feats)
            
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
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


def _deprocess_iso(iso:PDPIsolate, input_pipe:Pipeline, feat:str, feats:Union[np.ndarray,List[str]]) -> None:
    if not isinstance(feats, np.ndarray): feats = np.array(feats)
    feat_id = np.argwhere(feats == feat)[0][0]
    try:               in_sz = input_pipe.steps[0][1].n_samples_seen_.shape[0]
    except IndexError: in_sz = input_pipe.steps[0][1].mean_.shape[0]
    if feat_id >= in_sz: return
    x = iso.feature_grids
    x = np.broadcast_to(x[:,None], (x.shape[0], in_sz))
    x = input_pipe.inverse_transform(x)[:,feat_id]
    iso.feature_grids = x
    iso.ice_lines.columns = x


def _deprocess_interact(interact:PDPInteract, input_pipe:Pipeline, feat_pair:Tuple[str,str], feats:Union[np.ndarray,List[str]]) -> None:
    if not isinstance(feats, np.ndarray): feats = np.array(feats)
    for i, feat in enumerate(feat_pair):
        feat_id = np.argwhere(feats == feat)[0][0]
        try:               in_sz = input_pipe.steps[0][1].n_samples_seen_.shape[0]
        except IndexError: in_sz = input_pipe.steps[0][1].mean_.shape[0]
        if feat_id > in_sz: continue
        x = interact.feature_grids[i]
        x = np.broadcast_to(x[:,None], (x.shape[0], in_sz))
        x = input_pipe.inverse_transform(x)[:,feat_id]
        interact.feature_grids[i] = x


def plot_multibody_weighted_outputs(model:AbsModel, inputs:Union[np.ndarray,Tensor], block_names:Optional[List[str]]=None, use_mean:bool=False,
                                    savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Interpret how a model relies on the outputs of each block in a :class:MultiBlock by plotting the outputs of each block as weighted by the tail block.
    This function currently only supports models whose tail block contains a single neuron in the first dense layer.
    Input data is passed through the model and the absolute sums of the weighted block outputs are computed per datum, and optionally averaged over the number
    of block outputs.

    Arguments:
        model: model to interpret
        inputs: input data to use for interpretation
        block_names: names for each block to use when plotting
        use_mean: if True, will average the weighted outputs over the number of output neurons in each block
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    assert model.tail[0].weight.shape[0] == 1, 'This function currently only supports models whose tail block contains a single neuron in the first dense layer'
    if block_names is not None:
        assert len(block_names) == len(model.body.blocks), 'block_names passed, but number of names does not match number of blocks'
    else:
        block_names = [f'{i}' for i in range(len(model.body.blocks))]
    
    hook = FowardHook(model.tail[0])
    model.predict(inputs)
    
    y, itr = [], 0
    for b in model.body.blocks:
        o = hook.input[0][:,itr:itr+b.get_out_size()]
        w = model.tail[0].weight[0][itr:itr+b.get_out_size()]
        y.append(to_np(torch.abs(o@w)/b.get_out_size()) if use_mean else to_np(torch.abs(o@w)))
        itr += b.get_out_size()
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        sns.boxplot(x=block_names, y=y)
        plt.xlabel("Block", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r"Mean $|\bar{w}\cdot\bar{x}|$" if use_mean else r"$|\bar{w}\cdot\bar{x}|$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show() 


def plot_bottleneck_weighted_inputs(model:AbsModel, bottleneck_idx:int, inputs:Union[np.ndarray,Tensor], log_y:bool=True,
                                    savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Interpret how a single-neuron bottleneck in a :class:MultiBlock relies on input features by plotting the absolute values of the features times their
    associated weight for a given set of input data.

    Arguments:
        model: model to interpret
        bottleneck_idx: index of the bottleneck to interpret, i.e. model.body.bottleneck_blocks[bottleneck_idx]
        inputs: input data to use for interpretation
        log_y: whether to plot a log scale for the y-axis
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    body = model.body
    bn = body.bottleneck_blocks[bottleneck_idx]
    assert bn[0].weight.shape[0] == 1, 'This function currently only supports bottlenecks whose width is one neuron'
    
    hook = FowardHook(bn[0])
    model.predict(inputs)
    
    weighted_input = to_np(torch.abs(hook.input[0]*bn[0].weight[0]))
    rfm = {}
    for f in model.head.feat_map:
        if len(model.head.feat_map[f]) == 1:
            rfm[model.head.feat_map[f][0]] = f
        else:
            for i, idx in enumerate(model.head.feat_map[f]): rfm[idx] = f'{f}_{i}'
    y, x = [], []
    for i, f in enumerate(model.body.bottleneck_masks[bottleneck_idx]):
        x.append(rfm[f])
        y.append(weighted_input[:, i])
        
    x,y = np.array(x),np.array(y)
    order = np.argsort(y.mean(axis=1))
    x,y = list(x[order]),list(y[order])
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        sns.boxplot(x=x, y=y)
        plt.xlabel("Features", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r"$|w_i\times x_i|$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if log_y: plt.yscale('log', nonposy='clip')
        plt.xticks(rotation=70)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show() 