import numpy as np
from typing import Optional, List, Dict

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


def name_lookup(name:str) -> str:
    if name == 'trn_loss': return 'Training'
    if name == 'val_loss': return 'Validation'
    if '_trn' in name: return name[:name.find('_trn')] + 'Training'
    if '_val' in name: return name[:name.find('_val')] + 'Validation'


def plot_train_history(histories:List[Dict[str,List[float]]], savename:Optional[str]=None, ignore_trn=True):
    with sns.color_palette('colorblind') as color_palette:
        plt.figure(figsize=(16, 8))
        for i, history in enumerate(histories):
            if i == 0:
                for j, l in enumerate(history):
                    if not('trn' in l and ignore_trn): plt.plot(history[l], color=color_palette[j], label=name_lookup(l))
            else:
                for j, l in enumerate(history):
                    if not('trn' in l and ignore_trn): plt.plot(history[l], color=color_palette[j])

        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.xlabel("Epoch", fontsize=24, color='black')
        plt.ylabel("Loss", fontsize=24, color='black')
        plt.show()
        if savename is not None: plt.savefig(savename)


def plot_lr_finders(lr_finders, loss='loss', cut=-10):
    '''Get mean loss evolultion against learning rate for several LRFinder callbacks'''
    plt.figure(figsize=(16, 8))
    min_len = np.min([len(lr_finders[x].history[loss][:cut]) for x in range(len(lr_finders))])
    sns.tsplot([lr_finders[x].history[loss][:min_len] for x in range(len(lr_finders))], time=lr_finders[0].history['lr'][:min_len], ci='sd')
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()
