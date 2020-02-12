from typing import Tuple, List, Optional
from IPython.display import display
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ...plotting.plot_settings import PlotSettings


__all__ = ['MetricLogger']


# TODO: Non-notebook version?


class MetricLogger():
    r'''
    Provides live feedback during training showing a variety of metrics to help highlight problems or test hyper-parameters without completing a full training.

    Arguments:
        loss_names: List of names of losses which will be passed to the logger in the order in which they will be passed.
            By convention the first name will be used as the training loss when computing the ratio of training to validation losses
        n_folds: Number of folds present in the training data.
            The logger assumes that one of these folds is for validation, and so 1 training epoch = (n_fold-1) folds.
        autolog_scale: Whether to automatically change the scale of the y-axis for loss to logarithmic when the current loss drops below one 50th of its
            starting value
        extra_detail: Whether to include extra detail plots (loss velocity and training validation ratio), slight slower but potentially useful.
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Examples::
        >>> metric_log = MetricLogger(loss_names=['Train', 'Validation'], n_folds=train_fy.n_folds)
        >>> val_losses = []
        >>> metric_log.reset()  # Initialises plots and variables
        >>> for epoch in epochs:
        >>>     for fold in train_folds:
        >>>         # train for one fold (subepoch)
        >>>         metric_log.update_vals([train_loss, val_loss], best=best_val_loss)
        >>>     metric_log.update_plot()
        >>> plt.clf()
    '''

    def __init__(self, loss_names:List[str], n_folds:int, autolog_scale:bool=True, extra_detail:bool=True, plot_settings:PlotSettings=PlotSettings()):
        self.loss_names,self.n_folds,self.autolog_scale,self.extra_detail,self.settings = loss_names,n_folds,autolog_scale,extra_detail,plot_settings

    def add_loss_name(self, name:str) -> None:
        r'''
        Adds an additional loss name to the loss names displayed. The associated losses will be set to zero for any prior subepochs which have elapsed already.

        Arguments:
            name: name of loss to be added
        '''

        self.loss_names.append(name)
        self.loss_vals.append(list(np.zeros_like(self.loss_vals[0])))
        self.vel_vals.append(list(np.zeros_like(self.vel_vals[0])))
        self.gen_vals.append(list(np.zeros_like(self.gen_vals[0])))
        self.mean_losses.append(None)

    def update_vals(self, vals:List[float]) -> None:
        r'''
        Appends values to the losses. This is interpreted as one subepoch having elapsed (i.e. one training fold).

        Arguments:
            vals: loss values from the last subepoch in the order of `loss_names`
        '''

        for i, v in enumerate(vals):
            self.loss_vals[i].append(v)
            if not self.log and self.autolog_scale:
                if self.loss_vals[i][0]/self.loss_vals[i][-1] > 50: self.log = True
        self.subepochs.append(self.subepochs[-1]+1)
        if self.extra_detail:
            self.count += 1
            if self.count >= self.n_folds:
                self.count = 1
                self.epochs.append(self.epochs[-1]+1)
                for i, v in enumerate(self.loss_vals):
                    vel, self.mean_losses[i] = self._get_vel(v, self.mean_losses[i])
                    self.vel_vals[i].append(vel)
                    if i > 0: self.gen_vals[i-1].append(self._get_gen_err(v))

    def update_plot(self, best:Optional[float]=None) -> None:
        r'''
        Updates the plot(s), Optionally showing the user-chose best loss achieved.

        Arguments:
            best: the value of the best loss achieved so far
        '''

        # Loss
        self.loss_ax.clear()
        with sns.axes_style(**self.settings.style), sns.color_palette(self.settings.cat_palette):
            for v,m in zip(self.loss_vals,self.loss_names): self.loss_ax.plot(self.subepochs[1:], v, label=m)
        if best is not None: self.loss_ax.plot(self.subepochs[1:], np.ones_like(self.subepochs[1:])*best, label=f'Best = {best:.3E}', linestyle='--')
        if self.log:
            self.loss_ax.set_yscale('log', nonposy='clip')
            self.loss_ax.tick_params(axis='y', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')
        self.loss_ax.grid(True, which="both")
        self.loss_ax.legend(loc='upper right', fontsize=0.8*self.settings.leg_sz)
        self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
        self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)

        if self.extra_detail:
            # Velocity
            self.vel_ax.clear()
            self.vel_ax.tick_params(axis='y', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')
            self.vel_ax.grid(True, which="both")
            with sns.color_palette(self.settings.cat_palette):
                for v,m in zip(self.vel_vals,self.loss_names): self.vel_ax.plot(self.epochs[1:], v, label=f'{m} {v[-1]:.2E}')
            self.vel_ax.legend(loc='lower right', fontsize=0.8*self.settings.leg_sz)
            self.vel_ax.set_ylabel(r'$\Delta \bar{L}\ /$ Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)

            # Generalisation
            self.gen_ax.clear()
            self.gen_ax.grid(True, which="both")
            with sns.color_palette(self.settings.cat_palette) as palette:
                for i, (v,m) in enumerate(zip(self.gen_vals,self.loss_names[1:])):
                    self.gen_ax.plot(self.epochs[1:], v, label=f'{m} {v[-1]:.2f}', color=palette[i+1])
            self.gen_ax.legend(loc='upper left', fontsize=0.8*self.settings.leg_sz)
            self.gen_ax.set_xlabel('Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
            self.gen_ax.set_ylabel('Validation / Train', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
            if len(self.epochs) > 5:
                self.epochs = self.epochs[1:]
                for i in range(len(self.vel_vals)): self.vel_vals[i] = self.vel_vals[i][1:]
                for i in range(len(self.gen_vals)):  self.gen_vals[i]  = self.gen_vals[i][1:]
            
            self.display.update(self.fig)
        else:
            self.display.update(self.loss_ax.figure)

    def _get_vel(self, losses:List[float], old_mean:Optional[float]=None) -> Tuple[float,float]:
        mean = np.mean(losses[1-self.n_folds:])
        if old_mean is None: old_mean = losses[0]
        return mean-old_mean, mean

    def _get_gen_err(self, losses:List[float]) -> float:
        trn = np.mean(self.loss_vals[0][1-self.n_folds:])
        return (np.mean(losses[1-self.n_folds:]))/trn

    def reset(self) -> None:
        r'''
        Resets/initialises the logger's values and plots, and produces a placeholder plot. Should be called prior to `update_vals` or `update_plot`.
        '''

        self.loss_vals, self.vel_vals, self.gen_vals = [[] for _ in self.loss_names], [[] for _ in self.loss_names], [[] for _ in range(len(self.loss_names)-1)]
        self.mean_losses = [None for _ in self.loss_names]
        self.subepochs, self.epochs = [0], [0]
        self.count,self.log = 1,False

        with sns.axes_style(**self.settings.style):
            if self.extra_detail:
                self.fig = plt.figure(figsize=(self.settings.w_mid, self.settings.h_mid), constrained_layout=True)
                gs = self.fig.add_gridspec(2, 3)
                self.loss_ax = self.fig.add_subplot(gs[:,:-1])
                self.vel_ax = self.fig.add_subplot(gs[:1,2:])
                self.gen_ax  = self.fig.add_subplot(gs[1:2,2:])
                for ax in [self.loss_ax, self.vel_ax, self.gen_ax]:
                    ax.tick_params(axis='x', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col)
                    ax.tick_params(axis='y', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col)
                self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.vel_ax.set_ylabel(r'$\Delta \bar{L}\ /$ Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.gen_ax.set_xlabel('Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.gen_ax.set_ylabel('Validation / Train', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.display = display(self.fig, display_id=True)
            else:
                self.fig, self.loss_ax = plt.subplots(1, figsize=(self.settings.w_mid, self.settings.h_mid))
                self.loss_ax.tick_params(axis='x', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col)
                self.loss_ax.tick_params(axis='y', labelsize=0.8*self.settings.tk_sz, labelcolor=self.settings.tk_col)
                self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.settings.lbl_sz, color=self.settings.lbl_col)
                self.display = display(self.loss_ax.figure, display_id=True)
        

