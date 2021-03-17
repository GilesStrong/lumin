import numpy as np
import math
from typing import Tuple, Optional
import pandas as pd

from .callback import Callback
from ...plotting.training import plot_lr_finders

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['LRFinder']


class LRFinder(Callback):
    r'''
    Callback class for Smith learning-rate range test (https://arxiv.org/abs/1803.09820)

    Arguments:
        nb: number of batches in a epoch
        lr_bounds: tuple of initial and final LR
    '''

    def __init__(self, lr_bounds:Tuple[float,float]=[1e-7, 10], nb:Optional[int]=None):
        super().__init__()
        self.lr_bounds,self.nb = lr_bounds,nb
        if self.nb is not None: self.lr_mult = (self.lr_bounds[1]/self.lr_bounds[0])**(1/self.nb)

    def on_train_begin(self) -> None:
        r'''
        Prepares variables and optimiser for new training
        '''

        super().on_train_begin()
        self.best,self.iter = math.inf,0
        self.model.set_lr(self.lr_bounds[0])
        self.history = {'loss': [], 'lr': []}

    def on_epoch_begin(self) -> None:
        r'''
        Gets number of batches total on first fold
        '''
        if self.model.fit_params.state != 'train': return
        if self.nb is None:
            self.nb = self.model.fit_params.n_epochs*np.sum([self.model.fit_params.fy.get_data_count(i)//self.model.fit_params.bs 
                                                             for i in self.model.fit_params.trn_idxs])
            self.lr_mult = (self.lr_bounds[1]/self.lr_bounds[0])**(1/self.nb)
        
    def _calc_lr(self): return self.lr_bounds[0]*(self.lr_mult**self.iter)
    
    def plot(self) -> None:
        r'''
        Plot the loss as a function of the LR.
        '''

        plot_lr_finders([self], loss_range='auto', settings=self.plot_settings, log_y='auto' if 'regress' in self.model.objective.lower() else False)
        
    def plot_lr(self) -> None:
        r'''
        Plot the LR as a function of iterations.
        '''

        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            plt.figure(figsize=(self.plot_settings.h_small, self.plot_settings.h_small))
            plt.plot(range(len(self.history['lr'])), self.history['lr'])
            plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.ylabel("Learning rate", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.xlabel("Iterations", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.show()

    def get_df(self) -> pd.DataFrame:
        r'''
        Returns a DataFrame of LRs and losses
        '''

        return pd.DataFrame({'LR': self.history['lr'], 'Loss': self.history['loss']})

    def on_batch_end(self) -> None:
        r'''
        Records loss and increments LR
        '''

        if self.model.fit_params.state != 'train': return
        loss = self.model.fit_params.loss_val.data.item()
        self.history['loss'].append(loss)
        self.history['lr'].append(self.model.opt.param_groups[0]['lr'])
        self.iter += 1
        lr = self._calc_lr()
        self.model.opt.param_groups[0]['lr'] = lr
        if math.isnan(loss) or loss > self.best*100 or lr > self.lr_bounds[1]: self.model.stop_train = True
        if loss < self.best and self.iter > 10: self.best = loss
