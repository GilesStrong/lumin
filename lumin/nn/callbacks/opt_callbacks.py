import numpy as np
import math
from typing import Tuple, Optional
import pandas as pd

from .callback import Callback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['LRFinder']


class LRFinder(Callback):
    r'''
    Callback class for Smith learning-rate range test (https://arxiv.org/abs/1803.09820)

    Arguments:
        nb: number of batches in a (sub-)epoch
        lr_bounds: tuple of initial and final LR
        model: :class:`~limin.nn.models.Model` to alter, alternatively call :meth:`set_model`
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    def __init__(self, nb:int, lr_bounds:Tuple[float,float]=[1e-7, 10], model:Optional[AbsModel]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        self.lr_bounds = lr_bounds
        self.lr_mult = (self.lr_bounds[1]/self.lr_bounds[0])**(1/nb)
        
    def on_train_begin(self, **kargs) -> None:
        r'''
        Prepares variables and optimiser for new training
        '''

        self.best,self.iter = math.inf,0
        self.model.set_lr(self.lr_bounds[0])
        self.history = {'loss': [], 'lr': []}
        
    def _calc_lr(self): return self.lr_bounds[0]*(self.lr_mult**self.iter)
    
    def plot(self, n_skip:int=0, n_max:Optional[int]=None, lim_y:Optional[Tuple[float,float]]=None) -> None:
        r'''
        Plot the loss as a function of the LR.

        Arguments:
            n_skip: Number of initial iterations to skip in plotting
            n_max: Maximum iteration number to plot
            lim_y: y-range for plotting
        '''

        # TODO: Decide on whether to keep this; could just pass to plot_lr_finders

        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
            plt.plot(self.history['lr'][n_skip:n_max], self.history['loss'][n_skip:n_max], label='Training loss', color='g')
            if np.log10(self.lr_bounds[1])-np.log10(self.lr_bounds[0]) >= 3: plt.xscale('log')
            plt.ylim(lim_y)
            plt.grid(True, which="both")
            plt.legend(loc=self.plot_settings.leg_loc, fontsize=self.plot_settings.leg_sz)
            plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.ylabel("Loss", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.xlabel("Learning rate", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.show()
        
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

    def on_batch_end(self, loss:float, **kargs) -> None:
        r'''
        Records loss and increments LR

        Arguments:
            loss: training loss for most recent batch
        '''

        self.history['loss'].append(loss)
        self.history['lr'].append(self.model.opt.param_groups[0]['lr'])
        self.iter += 1
        lr = self._calc_lr()
        self.model.opt.param_groups[0]['lr'] = lr
        if math.isnan(loss) or loss > self.best*100 or lr > self.lr_bounds[1]: self.model.stop_train = True
        if loss < self.best and self.iter > 10: self.best = loss
