import numpy as np
from typing import Optional, List, Dict
import seaborn as sns
import matplotlib.pyplot as plt

from .plot_settings import PlotSettings
from ..nn.callbacks.opt_callbacks import LRFinder


def _lookup_name(name:str) -> str:
    if name == 'trn_loss': return 'Training'
    if name == 'val_loss': return 'Validation'
    if '_trn' in name:     return name[:name.find('_trn')] + 'Training'
    if '_val' in name:     return name[:name.find('_val')] + 'Validation'


def plot_train_history(histories:List[Dict[str,List[float]]], savename:Optional[str]=None, ignore_trn=True, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot histories object returned by :meth:fold_train_ensemble showing the loss evolution over time per model trained.

    Arguments:
        histories: list of dictionaries mapping loss type to values at each (sub)-epoch
        savename: Optional name of file to which to save the plot of feature importances
        ignore_trn: whether to ignore training loss
        settings: :class:PlotSettings class to control figure appearance
    '''
    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette) as palette:
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
        plt.show()
        if savename is not None: plt.savefig(savename, bbox_inches='tight')


def plot_lr_finders(lr_finders:List[LRFinder], loss='loss', cut=-10, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plot mean loss evolution against learning rate for several :class:LRFinder callbacks as returned by :meth:fold_lr_find.

    Arguments:
        lr_finders: list of :class:LRFinder callbacks used during training (e.g. as returned by :meth:fold_lr_find)
        loss: string representation of loss to plot
        cut: number of final iterations to cut
        settings: :class:PlotSettings class to control figure appearance
    '''

    with sns.axes_style(settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        min_len = np.min([len(lr_finders[x].history[loss][:cut]) for x in range(len(lr_finders))])
        sns.tsplot([lr_finders[x].history[loss][:min_len] for x in range(len(lr_finders))], time=lr_finders[0].history['lr'][:min_len], ci='sd')
        plt.xscale('log')
        plt.grid(True, which="both")
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.xlabel("Learning rate", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel("Loss", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.show()
