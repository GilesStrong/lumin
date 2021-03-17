from typing import Optional, List, Union, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

from .plot_settings import PlotSettings
from ..nn.callbacks.abs_callback import AbsCallback

__all__ = ['plot_train_history', 'plot_lr_finders']


def plot_train_history(histories:List[OrderedDict], savename:Optional[str]=None, ignore_trn:bool=False, settings:PlotSettings=PlotSettings(),
                       show:bool=True, xlow:int=0, log_y:bool=False) -> None:
    r'''
    Plot histories object returned by :meth:`~lumin.nn.training.train.train_models` showing the loss evolution over time per model trained.

    Arguments:
        histories: list of dictionaries mapping loss type to values at each (sub)-epoch
        savename: Optional name of file to which to save the plot of feature importances
        ignore_trn: whether to ignore training loss
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
        show: whether or not to show the plot, or just save it
        xlow: if set, will cut out the first given number of epochs
        log_y: whether to plot the y-axis with a log scale
    '''

    if not isinstance(histories, list): histories = [histories]
    n_folds = len(histories[0][0]['Training'])//len(histories[0][0]['Validation'])

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i, history in enumerate(histories):
            for j, l in enumerate(history[0]):
                if j > 0 or not ignore_trn:
                    x = range(1,len(history[0][l])+1)[xlow*n_folds:] if j == 0 else range(n_folds,(n_folds*len(history[0][l]))+1,n_folds)[xlow:]
                    plt.plot(x, history[0][l][xlow:], color=palette[j], label=l if i == 0 else None)

        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.xlabel("Subepoch", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel("Loss", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if log_y:
            plt.yscale('log')
            plt.grid(b=True, which="both", axis="both")
        if savename is not None: plt.savefig(settings.savepath/f'{savename}_loss{settings.format}', bbox_inches='tight')
        if show: plt.show()

    for metric in history[1].keys():
        with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
            plt.figure(figsize=(settings.w_mid, settings.h_mid))
            for i, history in enumerate(histories):
                plt.plot(range(n_folds,(n_folds*len(history[1][metric]))+1,n_folds)[xlow:], history[1][metric][xlow:], color=palette[1])
            plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
            plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
            plt.xlabel("Subepoch", fontsize=settings.lbl_sz, color=settings.lbl_col)
            plt.ylabel(metric, fontsize=settings.lbl_sz, color=settings.lbl_col)
            if savename is not None: plt.savefig(settings.savepath/f'{savename}_{metric}{settings.format}', bbox_inches='tight')
            if show: plt.show()


def plot_lr_finders(lr_finders:List[AbsCallback], lr_range:Optional[Union[float,Tuple]]=None, loss_range:Optional[Union[float,Tuple,str]]='auto',
                    log_y:Union[str,bool]='auto', savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
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
        log_y: whether to plot y-axis as log. If 'auto', will set to log if maximal fractional difference in loss values is greater than 50
        savename: Optional name of file to which to save the plot
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
        if log_y == 'auto':
            if df.Loss.max()/df.Loss.min() > 50: plt.yscale('log')
        elif log_y:
            plt.yscale('log')
        plt.grid(b=True, which="both", axis="both")
        if loss_range is not None: plt.ylim((0,loss_range) if isinstance(loss_range, float) else loss_range)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.xlabel("Learning rate", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel("Loss", fontsize=settings.lbl_sz, color=settings.lbl_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}.png', bbox_inches='tight')
        plt.show()
