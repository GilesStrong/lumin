import numpy as np
from typing import Optional, Tuple, Union, List

from .callback import Callback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


class AbsCyclicCallback(Callback):
    '''Abstract class for callbacks affecting lr or mom'''
    def __init__(self, interp:str, param_range:Tuple[float,float], cycle_mult:int=1, decrease_param:bool=False, scale:int=1,
                 model:Optional[AbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        self.param_range,self.cycle_mult,self.decrease_param,self.scale = param_range,cycle_mult,decrease_param,scale
        self.interp,self.cycle_iter,self.cycle_count,self.cycle_end,self.hist = interp.lower(),0,0,False,[]
        if nb is not None: self.nb = self.scale*nb

    def set_nb(self, nb:int) -> None: self.nb = self.scale*nb

    def incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True

    def plot(self) -> None:
        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
            plt.xlabel("Iterations", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.ylabel(self.param_name, fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.plot(range(len(self.hist)), self.hist)
            plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.show()

    def calc_param(self) -> float:
        if self.interp == 'cosine':
            x = np.cos(np.pi*(self.cycle_iter)/self.nb)+1
            dx = (self.param_range[1]-self.param_range[0])*x/2
            return self.param_range[1]-dx if not self.decrease_param else dx+self.param_range[0]
        elif self.interp == 'linear':
            x = np.abs((2*self.cycle_iter/self.nb)-1)
            dx = (self.param_range[1]-self.param_range[0])*np.max([0, 1-x])
            return self.param_range[1]-dx if self.decrease_param else dx+self.param_range[0]
        else:
            raise ValueError(f"Interpolation mode {self.interp} not implemented")

    def on_epoch_begin(self, **kargs) -> None: self.cycle_end = False
    
    def on_batch_end(self,   **kargs) -> None: self.incr_cycle()

    def on_batch_begin(self, **kargs) -> float:
        param = self.calc_param()
        self.hist.append(param)
        return param


class CycleLR(AbsCyclicCallback):
    '''Cycle lr during training, either:
    cosine interpolation for SGDR https://arxiv.org/abs/1608.03983
    or linear interpolation for Smith cycling https://arxiv.org/abs/1506.01186'''
    def __init__(self, lr_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 model:Optional[AbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        if decrease_param == 'auto': decrease_param = True if interp == 'cosine' else False
        super().__init__(interp=interp, param_range=lr_range, cycle_mult=cycle_mult,
                         decrease_param=decrease_param, scale=scale, model=model, nb=nb, plot_settings=plot_settings)
        self.param_name = 'Learning Rate'
        
    def on_batch_begin(self, **kargs) -> None:
        lr = super().on_batch_begin(**kargs)
        self.model.set_lr(lr)


class CycleMom(AbsCyclicCallback):
    '''Cycle momentum (beta_1) during training, either:
    cosine interpolation for SGDR https://arxiv.org/abs/1608.03983
    or linear interpolation for Smith cycling https://arxiv.org/abs/1506.01186'''
    def __init__(self, mom_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 model:Optional[AbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        if decrease_param == 'auto': decrease_param = False if interp == 'cosine' else True
        super().__init__(interp=interp, param_range=mom_range, cycle_mult=cycle_mult,
                         decrease_param=decrease_param, scale=scale, model=model, nb=nb, plot_settings=plot_settings)
        self.param_name = 'Momentum'
        
    def on_batch_begin(self, **kargs) -> None:
        mom = super().on_batch_begin(**kargs)
        self.model.set_mom(mom) 


class OneCycle(AbsCyclicCallback):
    '''Smith 1-cycle evolution for lr and momentum (beta_1) https://arxiv.org/abs/1803.09820
    Default interpolation uses fastai-style cosine function'''
    def __init__(self, lengths:Tuple[int,int], lr_range:List[float], mom_range:Tuple[float,float]=(0.85, 0.95), interp:str='cosine',
                 model:Optional[AbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(interp=interp, param_range=None, cycle_mult=1, scale=lengths[0], model=model, nb=nb, plot_settings=plot_settings)
        self.lengths,self.lr_range,self.mom_range,self.hist = lengths,lr_range,mom_range,{'lr': [], 'mom': []}

    def on_batch_begin(self, **kargs) -> None:
        self.decrease_param = self.cycle_count % 1 != 0
        self.param_range = self.lr_range
        lr = self.calc_param()
        self.hist['lr'].append(lr)
        self.model.set_lr(lr)

        self.decrease_param = self.cycle_count % 1 == 0
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
        if self.cycle_count == 1: self.model.stop_train = True

    def plot(self):
        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            fig, axs = plt.subplots(2, 1, figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
            axs[1].set_xlabel("Iterations", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            axs[0].set_ylabel("Learning Rate", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            axs[1].set_ylabel("Momentum", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            axs[0].plot(range(len(self.hist['lr'])), self.hist['lr'])
            axs[1].plot(range(len(self.hist['mom'])), self.hist['mom'])
            for ax in axs:
                ax.tick_params(axis='x', labelsize=self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                ax.tick_params(axis='y', labelsize=self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
            plt.show()
