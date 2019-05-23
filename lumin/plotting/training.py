import numpy as np
from typing import Optional, List, Dict

from .plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt


def _lookup_name(name:str) -> str:
    if name == 'trn_loss': return 'Training'
    if name == 'val_loss': return 'Validation'
    if '_trn' in name:     return name[:name.find('_trn')] + 'Training'
    if '_val' in name:     return name[:name.find('_val')] + 'Validation'


def plot_train_history(histories:List[Dict[str,List[float]]], savename:Optional[str]=None, ignore_trn=True, settings:PlotSettings=PlotSettings()):
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


def plot_lr_finders(lr_finders, loss='loss', cut=-10, settings:PlotSettings=PlotSettings()):
    '''Get mean loss evolultion against learning rate for several LRFinder callbacks'''
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
