import numpy as np
import math
from typing import Tuple, Optional, Dict, Any

from .callback import Callback
from ..models.model import Model

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


class LRFinder(Callback):
    def __init__(self, nb:int, lr_bounds:Tuple[float,float]=[1e-7, 10], model:Optional[Model]=None):
        super().__init__(model=model)
        self.lr_bounds = lr_bounds
        self.lr_mult = (self.lr_bounds[1]/self.lr_bounds[0])**(1/nb)
        
    def on_train_begin(self, logs:Dict[str,Any]={}):
        self.best = 1e9
        self.iter = 0
        self.model.set_lr(self.lr_bounds[0])
        self.history = {}
        self.history['loss'] = []
        self.history['lr'] = []
        
    def calc_lr(self):
        return self.lr_bounds[0]*(self.lr_mult**self.iter)
    
    def plot(self, n_skip=0, n_max:Optional[int]=None, ylim=None):
        plt.figure(figsize=(16, 8))
        plt.plot(self.history['lr'][n_skip:n_max], self.history['loss'][n_skip:n_max], label='Training loss', color='g')
        if np.log10(self.lr_bounds[1])-np.log10(self.lr_bounds[0]) >= 3: plt.xscale('log')
        plt.ylim(ylim)
        plt.grid(True, which="both")
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.ylabel("Loss", fontsize=24, color='black')
        plt.xlabel("Learning rate", fontsize=24, color='black')
        plt.show()
        
    def plot_lr(self):
        plt.figure(figsize=(4, 4))
        plt.xlabel("Iterations", fontsize=24, color='black')
        plt.ylabel("Learning rate", fontsize=24, color='black')
        plt.plot(range(len(self.history['lr'])), self.history['lr'])
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()

    def on_batch_end(self, logs:Dict[str,Any]={}):
        loss = logs['loss']
        self.history['loss'].append(loss)
        self.history['lr'].append(self.model.opt.param_groups[0]['lr'])
        self.iter += 1
        lr = self.calc_lr()
        self.model.opt.param_groups[0]['lr'] = lr
        if math.isnan(loss) or loss > self.best*10 or lr > self.lr_bounds[1]: self.model.stop_train = True
        if loss < self.best and self.iter > 10: self.best = loss
