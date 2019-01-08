import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve 
from typing import List, Optional, Dict, Any, Union, Tuple
import multiprocessing as mp

from ..utils.misc import uncert_round
from ..utils.multiprocessing import mp_run

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


def _bs_roc_auc(args:Dict[str,Any], out_q:mp.Queue) -> None:
    out_dict,scores = {},[]
    if 'name' not in args: args['name'] = ''
    if 'n'    not in args: args['n']    = 100
    if 'weights' in args: 
        for i in range(args['n']):
            points = np.random.choice(args['indeces'], len(args['indeces']), replace=True)
            scores.append(roc_auc_score(args['labels'].loc[points].values, args['preds'].loc[points].values, sample_weight=args['weights'].loc[points].values))
    else:
        for i in range(args['n']):
            points = np.random.choice(args['indeces'], len(args['indeces']), replace=True)
            scores.append(roc_auc_score(args['labels'].loc[points].values, args['preds'].loc[points].values))
    out_dict[f"{args['name']}_score"] = scores
    out_q.put(out_dict)


def plot_roc(in_data:Union[pd.DataFrame,List[pd.DataFrame]], pred_name:str='pred', targ_name:str='gen_target', weight_name:Optional[str]=None, 
             labels:Optional[List[str]]=None, targ2sample:Dict[int,str]={0: 'Background', 1: 'Signal'}, plot_params:Optional[List[Dict[str,Any]]]=None, 
             bootstrap:int=0, log_x:bool=False, plot_baseline:bool=True, title:Optional[str]=None, savename:Optional[str]=None) -> None:
    if isinstance(in_data, pd.DataFrame):
        in_data = [in_data]
        plot_params = [plot_params]
    if labels      is None: labels      = ['' for i in range(len(in_data))]
    if plot_params is None: plot_params = [{} for i in range(len(in_data))]

    curves,mean_scores = [],[]
    if bootstrap > 1:
        auc_args = []
        for i in range(len(in_data)):
            auc_args.append({'n': bootstrap, 'labels': in_data[i][targ_name], 'preds': in_data[i][pred_name], 'name': i, 'indeces': in_data[i].index.tolist()})
            if weight_name is not None:  auc_args[-1]['weights'] = in_data[i][weight_name]
        res = mp_run(auc_args, _bs_roc_auc)
        for i in range(len(in_data)): mean_scores.append((np.mean(res[f'{i}_score']), np.std(res[f'{i}_score'], ddof=1)))

    else:
        for i in range(len(in_data)):
            if weight_name is None: mean_scores.append(roc_auc_score(in_data[i][targ_name].values, in_data[i][pred_name]))
            else:                   mean_scores.append(roc_auc_score(in_data[i][targ_name].values, in_data[i][pred_name], sample_weight=in_data[i][weight_name]))
    
    for i in range(len(in_data)):
        if weight_name is None: curves.append(roc_curve(in_data[i][targ_name].values, in_data[i][pred_name].values)[:2])
        else:                   curves.append(roc_curve(in_data[i][targ_name].values, in_data[i][pred_name].values, sample_weight=in_data[i][weight_name].values)[:2])

    plt.figure(figsize=[8, 8])
    for i in range(len(in_data)):
        if bootstrap:
            mean_score = uncert_round(*mean_scores[i])
            plt.plot(*curves[i], label=f'{labels[i]} AUC = {mean_score[0]}±{mean_score[1]}')
        else:
            plt.plot(*curves[i], label=f'{labels[i]} AUC = {mean_scores[i]:.3f}', **plot_params[i])
    
    plt.xlabel(f'{targ2sample[0]} acceptance', fontsize=24, color='black')
    plt.ylabel(f'{targ2sample[1]}  acceptance', fontsize=24, color='black')
    plt.legend(loc='best', fontsize=16)
    if log_x:
        plt.xscale('log', nonposx='clip')
        plt.grid(True, which="both")
    elif plot_baseline:
        plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.title(title, fontsize=26, color='black', loc='left')
    if savename is not None: plt.savefig(savename)
    plt.show()


def _get_samples(in_data, sample_name, weight_name):
    samples = set(in_data[sample_name])
    weights = [np.sum(in_data[in_data[sample_name] == sample][weight_name]) for sample in samples]
    return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))]


def plot_binary_class_pred(in_data:pd.DataFrame, pred_name='pred', targ_name:str='gen_target', weight_name=None, weight_scale:float=1,
                           targ2sample:Dict[int,str]={0: 'Background', 1: 'Signal'}, log_y:bool=False, lim_x:Tuple[float,float]=(0,1), density=True, 
                           title:Optional[str]=None, savename:Optional[str]=None) -> None:
    plt.figure(figsize=(16, 8))
    for targ in sorted(set(in_data[targ_name])):
        cut = in_data[targ_name] == targ
        hist_kws = {} if weight_name is None else {'weights': weight_scale*in_data.loc[cut, weight_name]}
        sns.distplot(in_data.loc[cut, pred_name], label=targ2sample[targ], hist_kws=hist_kws, norm_hist=density, kde=False)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.xlim(lim_x)
    if density:             plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    elif weight_scale != 1: plt.ylabel(str(weight_scale) + r"$\times\frac{dN}{dp}$", fontsize=24, color='black')
    else:                   plt.ylabel(r"$\frac{dN}{dp}$", fontsize=24, color='black')
    if log_y:
        plt.yscale('log', nonposy='clip')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.title(title, fontsize=26, color='black', loc='left')
    if savename is not None: plt.savefig(savename)
    plt.show() 


def plot_sample_pred(in_data:pd.DataFrame, pred_name='pred', targ_name:str='gen_target', weight_name='gen_weight', sample_name='gen_sample',
                     weight_scale:float=1, bins:Union[int,List[int]]=35, log_y:bool=True, lim_x:Tuple[float,float]=(0,1),
                     pallet:str='tab10', sample2idx:Optional[Dict[str,int]]=None, desat=1, density=False, zoom_args:Optional[Dict[str,Any]]=None,
                     title:Optional[str]=None, savename:Optional[str]=None) -> None:
    '''Example zoom_args: {{'x':(0.95,1.0), 'y':(1, 1000)}, zoom_pos=10, zoom=2}'''
    hist_params = {'range': lim_x, 'bins': bins, 'normed': density, 'alpha': 0.8, 'stacked':True, 'rwidth':1.0}
    sig = (in_data[targ_name] == 1)
    bkg = (in_data[targ_name] == 0)
    sig_samples = _get_samples(in_data[sig], sample_name, weight_name)
    bkg_samples = _get_samples(in_data[bkg], sample_name, weight_name)
    if sample2idx is None: sample2idx = {k: v for v, k in enumerate(bkg_samples)}
    
    fig, ax = plt.subplots(figsize=(16, 8)) if zoom_args is None else plt.subplots(figsize=(24, 8))
    if zoom_args is not None: axins = inset_axes(ax, 3, 5, loc='right', bbox_to_anchor=(0,0,0.92,1), bbox_transform=ax.figure.transFigure)

    with sns.color_palette(pallet, len(sample2idx), desat=desat):
        ax.hist([in_data[in_data[sample_name] == sample][pred_name] for sample in bkg_samples],
                weights=[weight_scale*in_data[in_data[sample_name] == sample][weight_name] for sample in bkg_samples],
                label=bkg_samples, color=[sns.color_palette()[sample2idx[s]] for s in bkg_samples], **hist_params)
        if zoom_args:
            axins.hist([in_data[in_data[sample_name] == sample][pred_name] for sample in bkg_samples],
                       weights=[weight_scale*in_data[in_data[sample_name] == sample][weight_name] for sample in bkg_samples],
                       label=None, color=[sns.color_palette()[sample2idx[s]] for s in bkg_samples], **hist_params)
        
        for sample in sig_samples:
            ax.hist(in_data[in_data[sample_name] == sample][pred_name],
                    weights=weight_scale*in_data[in_data[sample_name] == sample][weight_name],
                    label='Signal', histtype='step', linewidth='3', 
                    color='black', **hist_params)
            if zoom_args:
                axins.hist(in_data[in_data[sample_name] == sample][pred_name],
                           weights=weight_scale*in_data[in_data[sample_name] == sample][weight_name],
                           label=None, histtype='step', linewidth='3', 
                           color='black', **hist_params)
        
        if zoom_args:
            axins.set_xlim(zoom_args['x'])
            axins.set_ylim(zoom_args['y'])
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
            ax.xaxis.set_label_text("Class prediction", fontsize=24, color='black')
            fig.legend(loc='right', fontsize=16)
        else:
            ax.legend(loc='best', fontsize=16)
        ax.set_xlim(*lim_x)
        if density: ax.yaxis.set_label_text(r"$\frac{1}{\mathcal{A}\sigma} \frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=24, color='black')
        else:       ax.yaxis.set_label_text(r"$\mathcal{L}_{\mathrm{int.}}\times\frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=24, color='black')
        if log_y:
            ax.set_yscale('log', nonposy='clip')
            ax.grid(True, which="both")
            if zoom_args:
                axins.set_yscale('log', nonposy='clip')
                axins.grid(True, which="both")
        ax.set_title(title, fontsize=26, color='black', loc='left')
        if savename is not None: fig.savefig(savename)
        fig.show() 