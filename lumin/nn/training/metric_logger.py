from typing import Tuple, List, Optional
from IPython.display import clear_output, display, HTML
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ...plotting.plot_settings import PlotSettings


__all__ = ['MetricLogger']


# TODO: Non-notebook version?

# class MetricLogger():
#     def __init__(self, metrics:List[str], autolog_scale:bool=True, plot_settings:PlotSettings=PlotSettings()):
#         self.metrics,self.autolog_scale,self.settings = metrics,autolog_scale,plot_settings

#     def add_metric(self, metric:str) -> None:
#         self.metrics.append(metric)
#         self.vals.append(list(np.zeros_like(self.vals[0])))

#     def update(self, vals:List[float], best:Optional[float]=None) -> None:
#         for i, v in enumerate(vals):
#             self.vals[i].append(v)
#             if not self.log and self.autolog_scale:
#                 if self.vals[i][0]/self.vals[i][-1] > 10: self.log = True

#         self.ax.clear()
#         with sns.axes_style(**self.settings.style), sns.color_palette(self.settings.cat_palette):
#             for v,m in zip(self.vals,self.metrics): self.ax.plot(self.x, v, label=m)
#         if best is not None: self.ax.plot(self.x, np.ones_like(self.x)*best, label=f'Best = {best:.3E}', linestyle='--')
#         if self.log:
#             self.ax.set_yscale('log', nonposy='clip')
#             self.ax.grid(True, which="both")
#             self.ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')

#         self.ax.legend(loc='upper right', fontsize=self.settings.leg_sz)
#         self.ax.set_xlabel('Sub-Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
#         self.ax.set_ylabel('Loss', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
#         self.display.update(self.ax.figure)
#         self.x.append(self.x[-1]+1)

#     def reset(self) -> None:
#         self.fig, self.ax = plt.subplots(1, figsize=(self.settings.w_mid, self.settings.h_mid))
#         self.ax.tick_params(axis='x', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
#         self.ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
#         self.display = display(self.ax.figure, display_id=True)
#         self.vals = [[] for _ in self.metrics]
#         self.x = [1]
#         self.log = False

class MetricLogger():
    def __init__(self, metrics:List[str], n_folds:int, autolog_scale:bool=True, plot_settings:PlotSettings=PlotSettings()):
        self.metrics,self.n_folds,self.autolog_scale,self.settings = metrics,n_folds,autolog_scale,plot_settings

    def add_metric(self, metric:str) -> None:
        self.metrics.append(metric)
        self.loss_vals.append(list(np.zeros_like(self.loss_vals[0])))
        self.accl_vals.append(list(np.zeros_like(self.accl_vals[0])))

    def update(self, vals:List[float], best:Optional[float]=None) -> None:
        for i, v in enumerate(vals):
            self.loss_vals[i].append(v)
            if not self.log and self.autolog_scale:
                if self.loss_vals[i][0]/self.loss_vals[i][-1] > 50: self.log = True
        self.count += 1

        # Subepoch tick
        self.loss_ax.clear()
        with sns.axes_style(**self.settings.style), sns.color_palette(self.settings.cat_palette):
            for v,m in zip(self.loss_vals,self.metrics): self.loss_ax.plot(self.subepochs, v, label=m)
        if best is not None: self.loss_ax.plot(self.subepochs, np.ones_like(self.subepochs)*best, label=f'Best = {best:.3E}', linestyle='--')
        if self.log:
            self.loss_ax.set_yscale('log', nonposy='clip')
            self.loss_ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')
        self.loss_ax.grid(True, which="both")
        self.loss_ax.legend(loc='upper right', fontsize=self.settings.leg_sz)
        self.loss_ax.set_xlabel('Sub-Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
        self.loss_ax.set_ylabel('Loss', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
        self.subepochs.append(self.subepochs[-1]+1)

        # Epoch tick
        if self.count >= self.n_folds:
            self.count = 0
            for i, v in enumerate(self.loss_vals):
                self.accl_vals[i].append(self._get_accl(v))
                if i > 0: self.gen_vals[i-1].append(self._get_gen_err(v))

            self.accl_ax.clear()
            self.accl_ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')
            self.accl_ax.grid(True, which="both")
            with sns.color_palette(self.settings.cat_palette):
                for v,m in zip(self.accl_vals,self.metrics): self.accl_ax.plot(self.epochs, v, label=f'{m} {v[-1]:.2E}')
            self.accl_ax.legend(loc='lower right', fontsize=0.8*self.settings.leg_sz)
            self.accl_ax.set_ylabel(r'$\Delta L\ /$ Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)

            self.gen_ax.clear()
            self.gen_ax.grid(True, which="both")
            with sns.color_palette(self.settings.cat_palette):
                for v,m in zip(self.gen_vals,self.metrics[1:]): self.gen_ax.plot(self.epochs, v, label=f'{m} {v[-1]:.2f}')
            self.gen_ax.legend(loc='upper left', fontsize=0.8*self.settings.leg_sz)
            self.gen_ax.set_xlabel('Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.gen_ax.set_ylabel('Validation / Train', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.epochs.append(self.epochs[-1]+1)
            if len(self.epochs) > 5:
                self.epochs = self.epochs[1:]
                for i in range(len(self.accl_vals)): self.accl_vals[i] = self.accl_vals[i][1:]
                for i in range(len(self.gen_vals)):  self.gen_vals[i]  = self.gen_vals[i][1:]
                
        self.display.update(self.fig)

    def _get_accl(self, losses:List[float]) -> float: return (losses[-1]-losses[-self.n_folds])/self.n_folds

    def _get_gen_err(self, losses:List[float]) -> float:
        trn = np.mean(self.loss_vals[0][-self.n_folds:])
        return (np.mean(losses[-self.n_folds:]))/trn

    def reset(self) -> None:
        with sns.axes_style(**self.settings.style):
            self.fig = plt.figure(figsize=(self.settings.w_mid, self.settings.h_mid), constrained_layout=True)
            gs = self.fig.add_gridspec(2, 3)
            self.loss_ax = self.fig.add_subplot(gs[:,:-1])
            self.accl_ax = self.fig.add_subplot(gs[:1,2:])
            self.gen_ax  = self.fig.add_subplot(gs[1:2,2:])
            for ax in [self.loss_ax, self.accl_ax, self.gen_ax]:
                ax.tick_params(axis='x', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
                ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
            self.loss_ax.set_xlabel('Sub-Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.loss_ax.set_ylabel('Loss', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.accl_ax.set_ylabel(r'$\Delta L\ /$ Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.gen_ax.set_xlabel('Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            self.gen_ax.set_ylabel('Validation / Train', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
        self.display = display(self.fig, display_id=True)
        self.loss_vals, self.accl_vals, self.gen_vals = [[] for _ in self.metrics], [[] for _ in self.metrics], [[] for _ in range(len(self.metrics)-1)]
        self.subepochs, self.epochs = [1], [1]
        self.count = 0
        self.log = False
