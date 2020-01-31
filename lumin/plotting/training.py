from typing import Optional, List, Dict, Union, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .plot_settings import PlotSettings
from ..nn.callbacks.opt_callbacks import LRFinder

__all__ = ['plot_train_history', 'plot_lr_finders']


def _lookup_name(name:str) -> str:
    if name == 'trn_loss': return 'Training'
    if name == 'val_loss': return 'Validation'
    if '_trn' in name:     return name[:name.find('_trn')] + 'Training'
    if '_val' in name:     return name[:name.find('_val')] + 'Validation'


def plot_train_history(histories:List[Dict[str,List[float]]], savename:Optional[str]=None, ignore_trn=True, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot histories object returned by :meth:`~lumin.nn.training.fold_train.fold_train_ensemble` showing the loss evolution over time per model trained.

    Arguments:
        histories: list of dictionaries mapping loss type to values at each (sub)-epoch
        savename: Optional name of file to which to save the plot of feature importances
        ignore_trn: whether to ignore training loss
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i, history in enumerate(histories):
            if i == 0:
                for j, l in enumerate(history):
                    if not('trn' in l and ignore_trn): plt.plot(history[l], color=palette[j], label=_lookup_name(l))
            else:
                for j, l in enumerate(history):
                    if not('trn' in l and ignore_trn): plt.plot(history[l], color=palette[j])

        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.xlabel("Epoch", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel("Loss", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if savename is not None: plt.savefig(f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_lr_finders(lr_finders:List[LRFinder], lr_range:Optional[Union[float,Tuple]]=None, loss_range:Optional[Union[float,Tuple,str]]='auto',
                    settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot mean loss evolution against learning rate for several :class:`~lumin.nn.callbacks.opt_callbacks.LRFinder callbacks as returned by
    :meth:`~lumin.nn.optimisation.hyper_param.fold_lr_find`.

    Arguments:
        lr_finders: list of :class:`~lumin.nn.callbacks.opt_callbacks.LRFinder callbacks used during training (e.g. as returned by
            :meth:`~lumin.nn.optimisation.hyper_param.fold_lr_find`)
        lr_range: limits the range of learning rates plotted on the x-axis: if float, maximum LR; if tuple, minimum & maximum LR
        loss_range: limits the range of losses plotted on the x-axis:
            if float, maximum loss;
            if tuple, minimum & maximum loss;
            if None, no limits;
            if 'auto', computes an upper limit automatically
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''
    
    df = pd.DataFrame()
    for lrf in lr_finders: df = df.append(lrf.get_df(), ignore_index=True)
    if lr_range is not None:
        if isinstance(lr_range, float): lr_range = (0, lr_range)
        df = df[(df.LR >= lr_range[0]) & (df.LR < lr_range[1])]

    if loss_range == 'auto':  # Max loss = 1.1 * max mean-loss at LR less than LR at min mean-loss
        agg = df.groupby(by='LR').agg(mean_loss=pd.NamedAgg(column='Loss', aggfunc='mean'))
        agg.reset_index(inplace=True)
        argmin_lr = agg.loc[agg.mean_loss.idxmin(), 'LR']
        loss_range = [0.8*agg.loc[agg.LR < argmin_lr, 'mean_loss'].min(), 1.2*agg.loc[agg.LR < argmin_lr, 'mean_loss'].max()]

    with sns.axes_style('whitegrid'), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        sns.lineplot(x='LR', y='Loss', data=df, ci='sd')
        plt.xscale('log')
        plt.grid(b=True, which="both", axis="both")
        if loss_range is not None: plt.ylim((0,loss_range) if isinstance(loss_range, float) else loss_range)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.xlabel("Learning rate", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel("Loss", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.show()
