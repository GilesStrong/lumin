import numpy as np
from typing import Optional, Tuple, Union, List
from fastcore.all import store_attr

from .callback import Callback, OldCallback
from ..models.abs_model import OldAbsModel
from ...plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['AbsCyclicCallback', 'CycleLR', 'CycleMom', 'OneCycle']


class OldAbsCyclicCallback(OldCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.cyclic_callbacks.AbsCyclicCallback`.
        It is a copy of the old `AbsCyclicCallback` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, interp:str, param_range:Tuple[float,float], cycle_mult:int=1, decrease_param:bool=False, scale:int=1,
                 model:Optional[OldAbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        self.param_range,self.cycle_mult,self.decrease_param,self.scale = param_range,cycle_mult,decrease_param,scale
        self.interp,self.cycle_iter,self.cycle_count,self.cycle_end,self.hist = interp.lower(),0,0,False,[]
        if nb is not None: self.nb = self.scale*nb

    def set_nb(self, nb:int) -> None:
        r'''
        Sets the callback's internal number of iterations per cycle equal to `nb*scale`

        Arguments:
            nb: number of minibatches per epoch
        '''
        
        self.nb = self.scale*nb

    def _incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True

    def plot(self) -> None:
        r'''
        Plots the history of the parameter evolution as a function of iterations
        '''

        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
            plt.xlabel("Iterations", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.ylabel(self.param_name, fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.plot(range(len(self.hist)), self.hist)
            plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.show()

    def _calc_param(self) -> float:
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

    def on_epoch_begin(self, **kargs) -> None:
        r'''
        Ensures the `cycle_end` flag is false when the epoch starts
        '''
        
        self.cycle_end = False
    
    def on_batch_end(self,   **kargs) -> None:
        r'''
        Increments the callback's progress through the cycle
        '''
        
        self._incr_cycle()

    def on_batch_begin(self, **kargs) -> float:
        r'''
        Computes the new value for the optimiser parameter and returns it

        Returns:
            new value for optimiser parameter
        '''

        param = self._calc_param()
        self.hist.append(param)
        return param


class AbsCyclicCallback(Callback):
    r'''
    Abstract class for callbacks affecting lr or mom

    Arguments:
        interp: string representation of interpolation function. Either 'linear' or 'cosine'.
        param_range: minimum and maximum values for parameter
        cycle_mult: multiplicative factor for adjusting the cycle length after each cycle.
            E.g `cycle_mult=1` keeps the same cycle length, `cycle_mult=2` doubles the cycle length after each cycle.
        decrease_param: whether to begin by decreasing the parameter, otherwise begin by increasing it
        scale: multiplicative factor for setting the initial number of epochs per cycle.
            E.g `scale=1` means 1 epoch per cycle, `scale=5` means 5 epochs per cycle.
        cycle_save: if true will save a copy of the model at the end of each cycle. Used for building ensembles from single trainings (e.g. snapshot ensembles)
        nb: number of minibatches (iterations) to expect per epoch
    '''

    def __init__(self, interp:str, param_range:Tuple[float,float], cycle_mult:int=1, decrease_param:bool=False, scale:int=1, cycle_save:bool=False):
        super().__init__()
        if not isinstance(cycle_mult, int):
            print('Coercing cycle_mult to int')
            cycle_mult = int(cycle_mult)
        if not isinstance(scale, int):
            print('Coercing scale to int')
            scale = int(scale)
        store_attr(but=['model','plot_settings','interp'])
        self.interp = interp.lower()
        self._reset()

    def _reset(self) -> None: self.cycle_iter,self.cycle_count,self.cycle_end,self.hist,self.cycle_losses,self.nb = 0,0,False,[],[],None

    def _save_cycle(self) -> None:
        self.model.save(self.model.fit_params.cb_savepath/f'cycle_{self.cycle_count}.h5')
        self.cycle_losses.append(self.model.fit_params.loss_val.data.item())

    def _incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter >= self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.cycle_save: self._save_cycle()
            self.cycle_count += 1
            self.cycle_end = True

    def plot(self) -> None:
        r'''
        Plots the history of the parameter evolution as a function of iterations
        '''

        with sns.axes_style(self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
            plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
            plt.xlabel("Iterations", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.ylabel(self.param_name, fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
            plt.plot(range(len(self.hist)), self.hist)
            plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
            plt.show()

    def _calc_param(self) -> float:
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
    
    def on_train_begin(self) -> None:
        super().on_train_begin()
        self._reset()

    def on_epoch_begin(self) -> None:
        r'''
        Ensures the `cycle_end` flag is false when the epoch starts
        '''
        
        if self.model.fit_params.state != 'train': return
        if self.nb is None:
            self.nb = self.scale*np.sum([self.model.fit_params.fy.get_data_count(i)//self.model.fit_params.bs for i in self.model.fit_params.trn_idxs])
        self.cycle_end = False
    
    def on_batch_end(self) -> None:
        r'''
        Increments the callback's progress through the cycle
        '''
        
        if self.model.fit_params.state != 'train': return
        self._incr_cycle()

    def _set_param(self, param:float) -> None: pass

    def on_batch_begin(self) -> None:
        r'''
        Computes the new value for the optimiser parameter and passes it to `_set_param` method
        '''

        if self.model.fit_params.state != 'train': return
        param = self._calc_param()
        self.hist.append(param)
        self._set_param(param)


class OldCycleLR(OldAbsCyclicCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.cyclic_callbacks.CycleLR`.
        It is a copy of the old `CycleLR` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, lr_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 model:Optional[OldAbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        if decrease_param == 'auto': decrease_param = True if interp == 'cosine' else False
        super().__init__(interp=interp, param_range=lr_range, cycle_mult=cycle_mult,
                         decrease_param=decrease_param, scale=scale, model=model, nb=nb, plot_settings=plot_settings)
        self.param_name = 'Learning Rate'
        
    def on_batch_begin(self, **kargs) -> None:
        r'''
        Computes the new lr and assignes it to the optimiser
        '''

        lr = super().on_batch_begin(**kargs)
        self.model.set_lr(lr)


class CycleLR(AbsCyclicCallback):
    r'''
    Callback to cycle learning rate during training according to either:
    cosine interpolation for SGDR https://arxiv.org/abs/1608.03983
    or linear interpolation for Smith cycling https://arxiv.org/abs/1506.01186

    Arguments:
        lr_range: tuple of initial and final LRs
        interp: 'cosine' or 'linear' interpolation
        cycle_mult: Multiplicative constant for altering the cycle length after each complete cycle
        decrease_param: whether to increase or decrease the LR (effectively reverses lr_range order), 'auto' selects according to interp
        scale: Multiplicative constant for altering the length of a cycle. 1 corresponds to one cycle = one epoch
        cycle_save: if true will save a copy of the model at the end of each cycle. Used for building ensembles from single trainings (e.g. snapshot ensembles)
        nb: Number of batches in a epoch

    Examples::
        >>> cosine_lr = CycleLR(lr_range=(0, 2e-3), cycle_mult=2, scale=1,
        ...                     interp='cosine', nb=100)
        >>>
        >>> cyclical_lr = CycleLR(lr_range=(2e-4, 2e-3), cycle_mult=1, scale=5,
                                  interp='linear', nb=100)
    '''

    # TODO sort lr-range or remove decrease_param

    def __init__(self, lr_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 cycle_save:bool=False):
        if decrease_param == 'auto': decrease_param = True if interp == 'cosine' else False
        super().__init__(interp=interp, param_range=lr_range, cycle_mult=cycle_mult, decrease_param=decrease_param, scale=scale, cycle_save=cycle_save)
        self.param_name = 'Learning Rate'

    def _set_param(self, param:float) -> None: self.model.set_lr(param)


class OldCycleMom(OldAbsCyclicCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.cyclic_callbacks.CycleMom`.
        It is a copy of the old `CycleMom` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, mom_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 model:Optional[OldAbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        if decrease_param == 'auto': decrease_param = False if interp == 'cosine' else True
        super().__init__(interp=interp, param_range=mom_range, cycle_mult=cycle_mult,
                         decrease_param=decrease_param, scale=scale, model=model, nb=nb, plot_settings=plot_settings)
        self.param_name = 'Momentum'
        
    def on_batch_begin(self, **kargs) -> None:
        r'''
        Computes the new momentum and assignes it to the optimiser
        '''

        mom = super().on_batch_begin(**kargs)
        self.model.set_mom(mom) 


class CycleMom(AbsCyclicCallback):
    r'''
    Callback to cycle momentum (beta 1) during training according to either:
    cosine interpolation for SGDR https://arxiv.org/abs/1608.03983
    or linear interpolation for Smith cycling https://arxiv.org/abs/1506.01186
    By default is set to evolve in opposite direction to learning rate, a la https://arxiv.org/abs/1803.09820

    Arguments:
        mom_range: tuple of initial and final momenta
        interp: 'cosine' or 'linear' interpolation
        cycle_mult: Multiplicative constant for altering the cycle length after each complete cycle
        decrease_param: whether to increase or decrease the momentum (effectively reverses mom_range order), 'auto' selects according to interp
        scale: Multiplicative constant for altering the length of a cycle. 1 corresponds to one cycle = one epoch
        cycle_save: if true will save a copy of the model at the end of each cycle. Used for building ensembles from single trainings (e.g. snapshot ensembles)
        nb: Number of batches in a epoch

    Examples::
        >>> cyclical_mom = CycleMom(mom_range=(0.85 0.95), cycle_mult=1,
        ...                         scale=5, interp='linear', nb=100)
    '''

    # TODO sort lr-range or remove decrease_param

    def __init__(self, mom_range:Tuple[float,float], interp:str='cosine', cycle_mult:int=1, decrease_param:Union[str,bool]='auto', scale:int=1,
                 cycle_save:bool=False):
        if decrease_param == 'auto': decrease_param = False if interp == 'cosine' else True
        super().__init__(interp=interp, param_range=mom_range, cycle_mult=cycle_mult, decrease_param=decrease_param, scale=scale, cycle_save=cycle_save)
        self.param_name = 'Momentum'

    def _set_param(self, param:float) -> None: self.model.set_mom(param)


class OldOneCycle(OldAbsCyclicCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.cyclic_callbacks.OneCycle`.
        It is a copy of the old `OneCycle` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, lengths:Tuple[int,int], lr_range:List[float], mom_range:Tuple[float,float]=(0.85, 0.95), interp:str='cosine',
                 model:Optional[OldAbsModel]=None, nb:Optional[int]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(interp=interp, param_range=None, cycle_mult=1, scale=lengths[0], model=model, nb=nb, plot_settings=plot_settings)
        self.lengths,self.lr_range,self.mom_range,self.hist = lengths,lr_range,mom_range,{'lr': [], 'mom': []}
        if len(self.lr_range) == 2: self.lr_range.append(0)

    def on_batch_begin(self, **kargs) -> None:
        r'''
        Computes the new lr and momentum and assignes them to the optimiser
        '''

        self.decrease_param = self.cycle_count % 1 != 0
        self.param_range = self.lr_range
        lr = self._calc_param()
        self.hist['lr'].append(lr)
        self.model.set_lr(lr)

        self.decrease_param = self.cycle_count % 1 == 0
        self.param_range = self.mom_range
        mom = self._calc_param()
        self.hist['mom'].append(mom)
        self.model.set_mom(mom)

    def _incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb = self.lengths[1]*self.nb/self.lengths[0]
            self.cycle_count += 0.5
            self.cycle_end = self.cycle_count % 1 == 0
            self.lr_range[0] = self.lr_range[2]
        if self.cycle_count == 1: self.model.stop_train = True

    def plot(self):
        r'''
        Plots the history of the lr and momentum evolution as a function of iterations
        '''

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


class OneCycle(AbsCyclicCallback):
    r'''
    Callback implementing Smith 1-cycle evolution for lr and momentum (beta_1) https://arxiv.org/abs/1803.09820
    Default interpolation uses fastai-style cosine function.
    Automatically triggers early stopping on cycle completion.

    Arguments:
        lengths: tuple of number of epochs in first and second stages of cycle
        lr_range: list of initial and max LRs and optionally a final LR. If only two LRs supplied, then final LR will be zero.
        mom_range: tuple of initial and final momenta
        interp: 'cosine' or 'linear' interpolation
        cycle_ends_training: whether to stop training once the cycle finishes, or continue running at the last LR and momentum

    Examples::
        >>> onecycle = OneCycle(lengths=(15, 30), lr_range=[1e-4, 1e-2],
        ...                     mom_range=(0.85, 0.95), interp='cosine', nb=100)
    '''

    def __init__(self, lengths:Tuple[int,int], lr_range:List[float], mom_range:Tuple[float,float]=(0.85, 0.95), interp:str='cosine',
                 cycle_ends_training:bool=True):
        super().__init__(interp=interp, param_range=None, cycle_mult=1, scale=lengths[0])
        self.lengths,self.lr_range,self.mom_range,self.cycle_ends_training,self.hist = lengths,lr_range,mom_range,cycle_ends_training,{'lr': [], 'mom': []}
        if len(self.lr_range) == 2: self.lr_range.append(0)

    def _reset(self) -> None: self.cycle_iter,self.cycle_count,self.cycle_end,self.hist,self.cycle_losses,self.nb = 0,0,False,{'lr': [], 'mom': []},[],None

    def on_batch_begin(self) -> None:
        r'''
        Computes the new lr and momentum and assignes them to the optimiser
        '''

        if self.model.fit_params.state != 'train' or self.cycle_count == 1: return

        self.decrease_param = self.cycle_count % 1 != 0
        self.param_range = self.lr_range
        lr = self._calc_param()
        self.hist['lr'].append(lr)
        self.model.set_lr(lr)

        self.decrease_param = self.cycle_count % 1 == 0
        self.param_range = self.mom_range
        mom = self._calc_param()
        self.hist['mom'].append(mom)
        self.model.set_mom(mom)

    def _incr_cycle(self) -> None:
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb = self.lengths[1]*self.nb/self.lengths[0]
            self.cycle_count += 0.5
            self.cycle_end = self.cycle_count % 1 == 0
            self.lr_range[0] = self.lr_range[2]
        if self.cycle_count == 1 and self.cycle_ends_training: self.model.stop_train = True

    def plot(self):
        r'''
        Plots the history of the lr and momentum evolution as a function of iterations
        '''

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
