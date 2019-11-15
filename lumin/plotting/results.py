import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve 
from typing import List, Optional, Dict, Any, Union, Tuple
import multiprocessing as mp

from .plot_settings import PlotSettings
from ..utils.statistics import uncert_round
from ..utils.multiprocessing import mp_run

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['plot_roc', 'plot_binary_class_pred', 'plot_sample_pred']


def _bs_roc_auc(args:Dict[str,Any], out_q:mp.Queue) -> None:
    r'''
    Compute bootstrap roc score for a list of datasets simultaneously using multiprocessing
    '''

    out_dict,scores = {},[]
    if 'name' not in args: args['name'] = ''
    if 'n'    not in args: args['n']    = 100
    np.random.seed()
    for i in range(args['n']):
        points = np.random.choice(args['indeces'], len(args['indeces']), replace=True)
        if len(set(args['labels'].loc[points])) == 2: scores.append(roc_auc_score(args['labels'].loc[points], args['preds'].loc[points],
                                                                                  sample_weight=args['weights'].loc[points] if 'weights' in args else None))
    out_dict[f"{args['name']}_score"] = scores
    out_q.put(out_dict)


def plot_roc(data:Union[pd.DataFrame,List[pd.DataFrame]], pred_name:str='pred', targ_name:str='gen_target', wgt_name:Optional[str]=None, 
             labels:Optional[Union[str,List[str]]]=None, plot_params:Optional[Union[Dict[str,Any],List[Dict[str,Any]]]]=None, 
             n_bootstrap:int=0, log_x:bool=False, plot_baseline:bool=True,
             savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> Dict[str,Union[float,Tuple[float,float]]]:
    r'''
    Plot receiver operating characteristic curve(s), optionally using booststrap resampling

    Arguments:
        data: (list of) DataFrame(s) from which to draw predictions and targets 
        pred_name: name of column to use as predictions
        targ_name: name of column to use as targets
        wgt_name: optional name of column to use as sample weights
        labels: (list of) label(s) for plot legend
        plot_params: (list of) dictionar[y/ies] of argument(s) to pass to line plot
        n_bootstrap: if greater than 0, will bootstrap resample the data that many times when computing the ROC AUC. Currently, this does not affect the shape
            of the lines, which are based on computing the ROC for the entire dataset as is.
        log_x: whether to use a log scale for plotting the x-axis, useful for high AUC line
        plot_baseline: whether to plot a dotted line for AUC=0.5. Currently incompatable with log_x=True
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        Dictionary mapping data labels to aucs (and uncertainties if n_bootstrap > 0)
    '''

    # TODO: make plot show uncertainty bands
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        if isinstance(data, pd.DataFrame):
            data = [data]
        if labels      is None:             labels      = [f'{i}' for i in range(len(data))] if len(data) > 1 else ['' for i in range(len(data))]
        if plot_params is None:             plot_params = [{} for i in range(len(data))]
        elif isinstance(plot_params, dict): plot_params = [plot_params for i in range(len(data))]

        curves,mean_scores = [],[]
        if n_bootstrap > 1:
            auc_args = []
            for i in range(len(data)):
                auc_args.append({'n': n_bootstrap, 'labels': data[i][targ_name], 'preds': data[i][pred_name], 'name': i, 'indeces': data[i].index.tolist()})
                if wgt_name is not None:  auc_args[-1]['weights'] = data[i][wgt_name]
            res = mp_run(auc_args, _bs_roc_auc)
            for i in range(len(data)): mean_scores.append((np.mean(res[f'{i}_score']), np.std(res[f'{i}_score'], ddof=1)))

        else:
            for i in range(len(data)):
                if wgt_name is None: mean_scores.append(roc_auc_score(data[i][targ_name].values, data[i][pred_name]))
                else:                   mean_scores.append(roc_auc_score(data[i][targ_name].values, data[i][pred_name], sample_weight=data[i][wgt_name]))
        
        for i in range(len(data)):
            if wgt_name is None: curves.append(roc_curve(data[i][targ_name].values, data[i][pred_name].values)[:2])
            else:                   curves.append(roc_curve(data[i][targ_name].values, data[i][pred_name].values, sample_weight=data[i][wgt_name].values)[:2])

        aucs = {}
        plt.figure(figsize=(settings.h_mid, settings.h_mid))
        for i in range(len(data)):
            aucs[labels[i]] = mean_scores[i]
            if n_bootstrap > 0:
                mean_score = uncert_round(*mean_scores[i])
                plt.plot(*curves[i], label=f'{labels[i]} AUC = {mean_score[0]}±{mean_score[1]}', **plot_params[i])
            else:
                plt.plot(*curves[i], label=f'{labels[i]} AUC = {mean_scores[i]:.3f}', **plot_params[i])
        
        plt.xlabel(f'{settings.targ2class[0]} acceptance', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(f'{settings.targ2class[1]}  acceptance', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        if log_x:
            plt.xscale('log', nonposx='clip')
            plt.grid(True, which="both")
        elif plot_baseline:
            plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
        return aucs


def _get_samples(df:pd.DataFrame, sample_name:str, wgt_name:str):
    '''Returns set of samples present in df ordered by sum of weights''' 
    samples = set(df[sample_name])
    weights = [np.sum(df[df[sample_name] == sample][wgt_name]) for sample in samples]
    return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))]


def plot_binary_class_pred(df:pd.DataFrame, pred_name:str='pred', targ_name:str='gen_target', wgt_name:str=None, wgt_scale:float=1,
                           log_y:bool=False, lim_x:Tuple[float,float]=(0,1), density=True, 
                           savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Basic plotter for prediction distribution in a binary classification problem.
    Note that labels are set using the settings.targ2class dictionary, which by default is {0: 'Background', 1: 'Signal'}.

    Arguments:
        df: DataFrame with targets and predictions
        pred_name: name of column to use as predictions
        targ_name: name of column to use as targets
        wgt_name: optional name of column to use as sample weights
        wgt_scale: applies a global multiplicative rescaling to sample weights. Default 1 = no rescaling
        log_y: whether to use a log scale for the y-axis
        lim_x: limit for plotting on the x-axis
        density: whether to normalise each distribution to one, or keep set to sum of weights / datapoints
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for targ in sorted(set(df[targ_name])):
            cut = df[targ_name] == targ
            hist_kws = {} if wgt_name is None else {'weights': wgt_scale*df.loc[cut, wgt_name]}
            sns.distplot(df.loc[cut, pred_name], label=settings.targ2class[targ], hist_kws=hist_kws, norm_hist=density, kde=False)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.xlabel("Class prediction", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xlim(lim_x)
        if density:             plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        elif wgt_scale != 1:    plt.ylabel(str(wgt_scale) + r"$\times\frac{dN}{dp}$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        else:                   plt.ylabel(r"$\frac{dN}{dp}$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if log_y:
            plt.yscale('log', nonposy='clip')
            plt.grid(True, which="both")
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show() 


def plot_sample_pred(df:pd.DataFrame, pred_name:str='pred', targ_name:str='gen_target', wgt_name:str='gen_weight', sample_name:str='gen_sample',
                     wgt_scale:float=1, bins:Union[int,List[int]]=35, log_y:bool=True, lim_x:Tuple[float,float]=(0,1),
                     density=False, zoom_args:Optional[Dict[str,Any]]=None,
                     savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    More advanceed plotter for prediction distirbution in a binary class problem with stacked distributions for backgrounds and user-defined binning
    Can also zoom in to specified parts of plot
    Note that plotting colours can be controled by seeting the settings.sample2col dictionary
    
    Arguments:
        df: DataFrame with targets and predictions
        pred_name: name of column to use as predictions
        targ_name: name of column to use as targets
        wgt_name: name of column to use as sample weights
        sample_name: name of column to use as process names
        wgt_scale: applies a global multiplicative rescaling to sample weights. Default 1 = no rescaling        
        bins: either the number of bins to use for a uniform binning, or a list of bin edges for a variable-width binning
        log_y: whether to use a log scale for the y-axis
        lim_x: limit for plotting on the x-axis
        density: whether to normalise each distribution to one, or keep set to sum of weights / datapoints
        zoom_args: arguments to control the optional zoomed in section,
            e.g. {'x':(0.4,0.45), 'y':(0.2, 1500), 'anchor':(0,0.25,0.95,1), 'width_scale':1, 'width_zoom':4, 'height_zoom':3}
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''
    
    hist_params = {'range': lim_x, 'bins': bins, 'normed': density, 'alpha': 0.8, 'stacked':True, 'rwidth':1.0}
    sig,bkg = (df[targ_name] == 1),(df[targ_name] == 0)
    sig_samples = _get_samples(df[sig], sample_name, wgt_name)
    bkg_samples = _get_samples(df[bkg], sample_name, wgt_name)
    sample2col = {k: v for v, k in enumerate(bkg_samples)} if settings.sample2col is None else settings.sample2col
    if zoom_args is not None:
        width_scale = 1.6           if 'width_scale' not in zoom_args else zoom_args['width_scale']
        width_zoom  = 3             if 'width_zoom'  not in zoom_args else zoom_args['width_zoom']
        height_zoom = 5             if 'height_zoom' not in zoom_args else zoom_args['height_zoom']
        anchor      = (0,0,0.92,1)  if 'anchor'      not in zoom_args else zoom_args['anchor']
    else:
        width_scale = 1
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette, 1+max([sample2col[x] for x in sample2col])):
        fig, ax = plt.subplots(figsize=(settings.w_mid, settings.h_mid)) if zoom_args is None else plt.subplots(figsize=(width_scale*settings.w_mid, settings.h_mid))
        if zoom_args is not None: axins = inset_axes(ax, width_zoom, height_zoom, loc='right', bbox_to_anchor=anchor, bbox_transform=ax.figure.transFigure)
        ax.hist([df[df[sample_name] == sample][pred_name] for sample in bkg_samples],
                weights=[wgt_scale*df[df[sample_name] == sample][wgt_name] for sample in bkg_samples],
                label=bkg_samples, color=[sns.color_palette()[sample2col[s]] for s in bkg_samples], **hist_params)
        if zoom_args:
            axins.hist([df[df[sample_name] == sample][pred_name] for sample in bkg_samples],
                       weights=[wgt_scale*df[df[sample_name] == sample][wgt_name] for sample in bkg_samples],
                       label=None, color=[sns.color_palette()[sample2col[s]] for s in bkg_samples], **hist_params)
        
        for sample in sig_samples:
            ax.hist(df[df[sample_name] == sample][pred_name],
                    weights=wgt_scale*df[df[sample_name] == sample][wgt_name],
                    label=sample, histtype='step', linewidth='3', 
                    color='black', **hist_params)
            if zoom_args:
                axins.hist(df[df[sample_name] == sample][pred_name],
                           weights=wgt_scale*df[df[sample_name] == sample][wgt_name],
                           label=None, histtype='step', linewidth='3', 
                           color='black', **hist_params)
        
        if zoom_args:
            axins.set_xlim(zoom_args['x'])
            axins.set_ylim(zoom_args['y'])
            axins.tick_params(axis='x', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
            axins.tick_params(axis='y', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
            ax.xaxis.set_label_text("Class prediction", fontsize=settings.lbl_sz, color=settings.lbl_col)
            fig.legend(loc='right', fontsize=settings.leg_sz)
        else:
            ax.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        ax.set_xlim(*lim_x)
        ax.tick_params(axis='x', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
        ax.tick_params(axis='y', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
        ax.xaxis.set_label_text('Class prediction', fontsize=settings.lbl_sz, color=settings.lbl_col)
        if density: ax.yaxis.set_label_text(r"$\frac{1}{\mathcal{A}\sigma} \frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        else:       ax.yaxis.set_label_text(r"$\mathcal{L}_{\mathrm{int.}}\times\frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if log_y:
            ax.set_yscale('log', nonposy='clip')
            ax.grid(True, which="both")
            if zoom_args:
                axins.set_yscale('log', nonposy='clip')
                axins.grid(True, which="both")
        ax.set_title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        fig.show()
