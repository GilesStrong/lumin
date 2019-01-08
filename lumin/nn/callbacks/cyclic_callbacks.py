import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

from .callback import Callback
from ..models.model import Model

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


class AbsCyclicCallback(Callback):
    def __init__(self, interp:str, param_range:Tuple[float,float], cycle_mult:int=1, decrease:bool=False, scale:int=1, model:Optional[Model]=None, nb:Optional[int]=None):
        super().__init__(model=model)
        self.param_range,self.cycle_mult,self.decrease,self.scale = param_range,cycle_mult,decrease,scale
        self.interp = interp.lower()
        if nb is not None: self.nb = self.scale*nb
        self.cycle_iter = 0
        self.cycle_count = 0
        self.cycle_end = False
        self.hist = []

    def set_nb(self, nb:int) -> None:
        self.nb = self.scale*nb

    def incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True

    def plot(self) -> None:
        plt.figure(figsize=(16, 8))
        plt.xlabel("Iterations", fontsize=24, color='black')
        plt.ylabel(self.param_name, fontsize=24, color='black')
        plt.plot(range(len(self.hist)), self.hist)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()

    def calc_param(self) -> float:
        if   self.interp == 'cosine':
            x = np.cos(np.pi*(self.cycle_iter)/self.nb)+1
            dx = (self.param_range[1]-self.param_range[0])*x/2
            return self.param_range[1]-dx if not self.decrease else dx+self.param_range[0]
        elif self.interp == 'linear':
            x = np.abs((self.cycle_iter/self.nb)-1)
            dx = (self.param_range[1]-self.param_range[0])*np.max([0, 1-x])
            return self.param_range[1]-dx if self.decrease else dx+self.param_range[0]
        else:
            raise ValueError(f"Interpolation mode {self.interp} not implemented")

    def on_epoch_begin(self, logs:Dict[str,Any]={}) -> None:
        self.cycle_end = False

    def on_batch_end(self, logs:Dict[str,Any]={}) -> None:
        self.incr_cycle()

    def on_batch_begin(self, logs:Dict[str,Any]={}) -> float:
        param = self.calc_param()
        self.hist.append(param)
        return param


class CycleLR(AbsCyclicCallback):
    def __init__(self, lr_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease:Union[str,bool]='auto', scale:int=1, model:Optional[Model]=None, nb:Optional[int]=None):
        if decrease == 'auto': decrease = True if interp == 'cosine' else False
        super().__init__(interp=interp, param_range=lr_range, cycle_mult=cycle_mult, decrease=decrease, scale=scale, model=model, nb=nb)
        self.param_name = 'Learning Rate'
        
    def on_batch_begin(self, logs:Dict[str,Any]={}) -> None:
        lr = super().on_batch_begin(logs)
        self.model.set_lr(lr)


class CycleMom(AbsCyclicCallback):
    def __init__(self, mom_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease:Union[str,bool]='auto', scale:int=1, model:Optional[Model]=None, nb:Optional[int]=None):
        if decrease == 'auto': decrease = False if interp == 'cosine' else True
        super().__init__(interp=interp, param_range=mom_range, cycle_mult=cycle_mult, decrease=decrease, scale=scale, model=model, nb=nb)
        self.param_name = 'Momentum'
        
    def on_batch_begin(self, logs:Dict[str,Any]={}) -> None:
        mom = super().on_batch_begin(logs)
        self.model.set_mom(mom) 


class OneCycle(AbsCyclicCallback):
    def __init__(self, lengths:Tuple[int,int], lr_range:Tuple[float,float], mom_range:Tuple[float,float], interp:str='cosine', model:Optional[Model]=None, nb:Optional[int]=None):
        super().__init__(interp=interp, param_range=None, cycle_mult=1, scale=lengths[0], model=model, nb=nb)
        self.lengths,self.lr_range,self.mom_range = lengths,lr_range,mom_range
        self.hist = {'lr': [], 'mom': []}

    def on_batch_begin(self, logs:Dict[str,Any]={}) -> None:
        self.decrease = self.cycle_count % 1 != 0
        self.param_range = self.lr_range
        lr = self.calc_param()
        self.hist['lr'].append(lr)
        self.model.set_lr(lr)

        self.decrease = self.cycle_count % 1 == 0
        self.param_range = self.mom_range
        mom = self.calc_param()
        self.hist['mom'].append(mom)
        self.model.set_mom(mom)              

    def incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb = self.lengths[1]*self.nb/self.lengths[0]
            self.cycle_count += 0.5
            self.cycle_end = self.cycle_count % 1 == 0
            self.lr_range[0] = 0
        if self.cycle_count == 1:
            self.model.stop_train = True

    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(16, 6))
        axs[1].set_xlabel("Iterations", fontsize=20, color='black')
        axs[0].set_ylabel("Learning Rate", fontsize=20, color='black')
        axs[1].set_ylabel("Momentum", fontsize=20, color='black')
        axs[0].plot(range(len(self.hist['lr'])), self.hist['lr'])
        axs[1].plot(range(len(self.hist['mom'])), self.hist['mom'])
        plt.show()
