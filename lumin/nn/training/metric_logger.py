from typing import Tuple, List, Optional
from IPython.display import clear_output, display, HTML
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ...plotting.plot_settings import PlotSettings


__all__ = ['MetricLogger']


# TODO: Non-notebook version?

class MetricLogger():
    def __init__(self, metrics:List[str], autolog_scale:bool=True, plot_settings:PlotSettings=PlotSettings()):
        self.metrics,self.autolog_scale,self.settings = metrics,autolog_scale,plot_settings

    def add_metric(self, metric:str) -> None:
        self.metrics.append(metric)
        self.vals.append(list(np.zeros_like(self.vals[0])))

    def update(self, vals:List[float], best:Optional[float]=None) -> None:
        for i, v in enumerate(vals):
            self.vals[i].append(v)
            if not self.log and self.autolog_scale:
                if self.vals[i][0]/self.vals[i][-1] > 10: self.log = True

        self.ax.clear()
        with sns.axes_style(**self.settings.style), sns.color_palette(self.settings.cat_palette):
            for v,m in zip(self.vals,self.metrics): self.ax.plot(self.x, v, label=m)
        if best is not None: self.ax.plot(self.x, np.ones_like(self.x)*best, label=f'Best = {best:.3E}', linestyle='--')
        if self.log:
            self.ax.set_yscale('log', nonposy='clip')
            self.ax.grid(True, which="both")
            self.ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col, which='both')

        self.ax.legend(loc='upper right', fontsize=self.settings.leg_sz)
        self.ax.set_xlabel('Sub-Epoch', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
        self.ax.set_ylabel('Loss', fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
        self.display.update(self.ax.figure)
        self.x.append(self.x[-1]+1)

    def reset(self) -> None:
        self.fig, self.ax = plt.subplots(1, figsize=(self.settings.w_mid, self.settings.h_mid))
        self.ax.tick_params(axis='x', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
        self.ax.tick_params(axis='y', labelsize=self.settings.tk_sz, labelcolor=self.settings.tk_col)
        self.display = display(self.ax.figure, display_id=True)
        self.vals = [[] for _ in self.metrics]
        self.x = [1]
        self.log = False
