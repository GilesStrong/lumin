from fastcore.all import store_attr
import math
from typing import Optional
import numpy as np
from fastprogress.fastprogress import IN_NOTEBOOK
from IPython.display import display
from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns

from .callback import Callback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings

__all__ = ['EarlyStopping', 'SaveBest', 'MetricLogger']


class EarlyStopping(Callback):
    r'''
    Tracks validation loss during training and terminates training if loss doesn't decrease after `patience` number of epochs.
    Losses are assumed to be averaged and will be re-averaged over the epoch unless `loss_is_meaned` is false.

    Arguments:
        patience: number of epochs to wait without improvement before stopping training
        loss_is_meaned: if the batch loss value has been averaged over the number of elements in the batch, this should be true;
            average loss will be computed over all elements in batch.
            If the batch loss is not an average value, then the average will be computed over the number of batches.
        model: :class:`~lumin.nn.models.model.Model` to provide predictions, alternatively call :meth:`~lumin.nn.models.Model.set_model`           
    '''
    
    def __init__(self, patience:int, loss_is_meaned:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        store_attr(but=['model'])
        self._reset()

    def _reset(self) -> None: self.epochs,self.min_loss = 0,math.inf

    def on_train_begin(self) -> None:
        self._reset()
        self.cyclic_cb = None if len(self.model.fit_params.cyclic_cbs) == 0 else self.model.fit_params.cyclic_cbs[-1]
        self.improve_in_cycle = False

    def on_epoch_begin(self) -> None:
        if self.model.fit_params.state != 'valid': return
        self.cnt = 0
        self.loss = [0] + [0 for _ in self.model.fit_params.loss_cbs]  # Consider all losses e.g. SWA loss

    def on_forwards_end(self) -> None:
        if self.model.fit_params.state != 'valid': return
        sz = len(self.model.fit_params.x) if self.loss_is_meaned else 1
        self.loss[0] += self.model.fit_params.loss_val.data.item()*sz
        for i,c in enumerate(self.model.fit_params.loss_cbs): self.loss[i+1] += c.get_loss()*sz
        self.cnt += sz

    def on_epoch_end(self) -> None:
        if self.model.fit_params.state != 'valid': return
        loss = np.min(self.loss)/self.cnt
        if loss <= self.min_loss:
            self.min_loss = loss
            self.epochs = 0
            self.improve_in_cycle = True
            if self.cyclic_cb is not None and self.cyclic_cb.cycle_end: self.improve_in_cycle = False
        elif self.cyclic_cb is not None:
            if self.cyclic_cb.cycle_end:
                if self.improve_in_cycle:
                    self.epochs = 0
                    self.improve_in_cycle = False
                else:
                    self.epochs += 1
        else:
            self.epochs += 1
        if self.epochs >= self.patience:
            print('Early stopping')
            self.model.fit_params.stop = True


class SaveBest(Callback):
    r'''
    Tracks validation loss during training and automatically saves a copy of the weights to indicated file whenever validation loss decreases.
    Losses are assumed to be averaged and will be re-averaged over the epoch unless `loss_is_meaned` is false.
    '''

    def __init__(self, auto_reload:bool=True, loss_is_meaned:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        store_attr(but=['model'])
        self._reset()

    def _reset(self) -> None: self.min_loss = math.inf
    def on_train_begin(self) -> None: self._reset()

    def on_epoch_begin(self) -> None:
        if self.model.fit_params.state != 'valid': return
        self.cnt = 0
        self.loss = [0] + [0 for _ in self.model.fit_params.loss_cbs]  # Consider all losses e.g. SWA loss

    def on_forwards_end(self) -> None:
        if self.model.fit_params.state != 'valid': return
        sz = len(self.model.fit_params.x) if self.loss_is_meaned else 1
        self.loss[0] += self.model.fit_params.loss_val.data.item()*sz
        for i,c in enumerate(self.model.fit_params.loss_cbs): self.loss[i+1] += c.get_loss()*sz
        self.cnt += sz

    def on_epoch_end(self) -> None:
        if self.model.fit_params.state != 'valid': return
        loss = np.array(self.loss)/self.cnt
        lm = np.min(loss)
        if lm < self.min_loss:
            self.min_loss = lm
            lam  = np.argmin(loss)
            m = self.model
            if lam > 0: m = self.model.fit_params.loss_cbs[lam-1].test_model
            m.save(self.model.fit_params.cb_savepath/'best.h5')

    def on_train_end(self) -> None:
        print(f'Loading best model with loss {self.min_loss:.3E}')
        self.model.load(self.model.fit_params.cb_savepath/'best.h5')


class MetricLogger(Callback):
    r'''        
    '''

    def __init__(self, show_plots:bool=IN_NOTEBOOK, extra_detail:bool=True, loss_is_meaned:bool=True, model:Optional[AbsModel]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        store_attr(but=['model','plot_settings'])

    def on_train_begin(self) -> None:
        self.reset()
        for c in self.model.fit_params.loss_cbs: self._add_loss_name(type(c).__name__)

    def on_epoch_begin(self) -> None: self.loss,self.cnt = 0,0
    def on_fold_begin(self) -> None: self.on_epoch_begin()

    def on_fold_end(self) -> None:
        if self.model.fit_params.state != 'train': return
        self.loss_vals[0].append(self.loss/self.cnt)

    def on_epoch_end(self) -> None:
        if self.model.fit_params.state != 'valid': return
        self.epochs.append(self.epochs[-1]+1)
        self.loss_vals[1].append(self.loss/self.cnt)
        for i,c in enumerate(self.model.fit_params.loss_cbs):  self.loss_vals[i+2].append(c.get_loss())

        if self.show_plots:
            for i, v in enumerate(self.loss_vals[1:]):
                if len(self.loss_vals[1]) > 1 and self.extra_detail:
                    self.vel_vals[i].append(v[-1]-v[-2])
                    self.gen_vals[i].append(v[-1]/self.loss_vals[0][-1])
                if self.loss_vals[i+1][-1] <= self.best_loss: self.best_loss = self.loss_vals[i+1][-1]
            if not self.log:
                if self.loss_vals[0][0]/self.loss_vals[0][-1] > 50: self.log = True
            self.update_plot()
        else:
            self.print_losses()

    def on_forwards_end(self) -> None:
        sz = len(self.model.fit_params.x) if self.loss_is_meaned else 1
        self.loss += self.model.fit_params.loss_val.data.item()*sz
        self.cnt += sz

    def _add_loss_name(self, name:str) -> None:
        self.loss_names.append(name)
        self.loss_vals.append(list(np.zeros_like(self.loss_vals[1])))
        self.vel_vals.append(list(np.zeros_like(self.vel_vals[0])))
        self.gen_vals.append(list(np.zeros_like(self.gen_vals[0])))

    def print_losses(self) -> None:
        p = f'Epoch {len(self.loss_vals[1])+1}: Training = {np.mean(self.loss_vals[0][-self.n_trn_flds:]):.2E}'
        for v,m in zip(self.loss_vals[1:],self.loss_names[1:]): p += f' {m} = {v[-1]:.2E}'
        print(p)

    def update_plot(self) -> None:
        r'''
        Updates the plot(s).
        '''

        # TODO: make this faster

        # Loss
        self.loss_ax.clear()
        with sns.axes_style(**self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            self.loss_ax.plot(range(1,len(self.loss_vals[0])+1), self.loss_vals[0], label=self.loss_names[0])
            x = range(self.n_trn_flds, self.n_trn_flds*len(self.loss_vals[1])+1, self.n_trn_flds)
            for v,m in zip(self.loss_vals[1:],self.loss_names[1:]):
                self.loss_ax.plot(x, v, label=m)
        self.loss_ax.plot([1,x[-1]], [self.best_loss,self.best_loss], label=f'Best = {self.best_loss:.3E}', linestyle='--')
        if self.log:
            self.loss_ax.set_yscale('log', nonposy='clip')
            self.loss_ax.tick_params(axis='y', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col, which='both')
        self.loss_ax.grid(True, which="both")
        self.loss_ax.legend(loc='upper right', fontsize=0.8*self.plot_settings.leg_sz)
        self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
        self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)

        if self.extra_detail and len(self.loss_vals[1]) > 1:
            # Velocity
            self.vel_ax.clear()
            self.vel_ax.tick_params(axis='y', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col, which='both')
            self.vel_ax.grid(True, which="both")
            with sns.color_palette(self.plot_settings.cat_palette) as palette:
                for i, (v,m) in enumerate(zip(self.vel_vals,self.loss_names[1:])):
                    self.vel_ax.plot(self.epochs[2:], v, label=f'{m} {v[-1]:.2E}', color=palette[i+1])
            self.vel_ax.legend(loc='lower right', fontsize=0.8*self.plot_settings.leg_sz)
            self.vel_ax.set_ylabel(r'$\Delta \bar{L}\ /$ Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)

            # Generalisation
            self.gen_ax.clear()
            self.gen_ax.grid(True, which="both")
            with sns.color_palette(self.plot_settings.cat_palette) as palette:
                for i, (v,m) in enumerate(zip(self.gen_vals,self.loss_names[1:])):
                    self.gen_ax.plot(self.epochs[2:], v, label=f'{m} {v[-1]:.2f}', color=palette[i+1])
            self.gen_ax.legend(loc='upper left', fontsize=0.8*self.plot_settings.leg_sz)
            self.gen_ax.set_xlabel('Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            self.gen_ax.set_ylabel('Validation / Train', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            if len(self.epochs) > 5:
                self.epochs = self.epochs[1:]
                for i in range(len(self.vel_vals)): self.vel_vals[i],self.gen_vals[i] = self.vel_vals[i][1:],self.gen_vals[i][1:]
            
            self.display.update(self.fig)
        else:
            self.display.update(self.loss_ax.figure)

    def reset(self) -> None:
        r'''
        Resets/initialises the logger's values and plots, and produces a placeholder plot. Should be called prior to `update_vals` or `update_plot`.
        '''

        self.loss_names = ['Training', 'Validation']
        self.loss_vals = [[] for _ in self.loss_names]
        self.vel_vals, self.gen_vals = [[] for _ in range(len(self.loss_names)-1)], [[] for _ in range(len(self.loss_names)-1)]
        self.n_trn_flds = len(self.model.fit_params.trn_idxs)
        self.log,self.best_loss,self.epochs = False,math.inf,[0]

        if self.show_plots:
            with sns.axes_style(**self.plot_settings.style):
                if self.extra_detail:
                    self.fig = plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid), constrained_layout=True)
                    gs = self.fig.add_gridspec(2, 3)
                    self.loss_ax = self.fig.add_subplot(gs[:,:-1])
                    self.vel_ax  = self.fig.add_subplot(gs[:1,2:])
                    self.gen_ax  = self.fig.add_subplot(gs[1:2,2:])
                    for ax in [self.loss_ax, self.vel_ax, self.gen_ax]:
                        ax.tick_params(axis='x', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                        ax.tick_params(axis='y', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                    self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.vel_ax.set_ylabel(r'$\Delta \bar{L}\ /$ Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.gen_ax.set_xlabel('Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.gen_ax.set_ylabel('Validation / Train', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.display = display(self.fig, display_id=True)
                else:
                    self.fig, self.loss_ax = plt.subplots(1, figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
                    self.loss_ax.tick_params(axis='x', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                    self.loss_ax.tick_params(axis='y', labelsize=0.8*self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                    self.loss_ax.set_xlabel('Sub-Epoch', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.loss_ax.set_ylabel('Loss', fontsize=0.8*self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    self.display = display(self.loss_ax.figure, display_id=True)

    def get_loss_history(self) -> OrderedDict:
        history = OrderedDict()
        for v,m in zip(self.loss_vals,self.loss_names): history[m] = v
        return history
