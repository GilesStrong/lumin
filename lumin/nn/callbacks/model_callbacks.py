import numpy as np
from typing import Optional, Dict
from abc import abstractmethod
import copy

import torch.tensor as Tensor

from ..models.abs_model import AbsModel
from .callback import Callback
from .cyclic_callbacks import AbsCyclicCallback
from ...utils.misc import to_tensor
from ...plotting.plot_settings import PlotSettings

__all__ = ['SWA', 'AbsModelCallback']


class AbsModelCallback(Callback):
    r'''
    Abstract class for callbacks which provide alternative models during training
    
    Arguments:
        model: :class:`~lumin.nn.models.model.Model` to provide parameters, alternatively call :meth:`~lumin.nn.models.Model.set_model`
        val_fold: Dictionary containing inputs, targets, and weights (or None) as Numpy arrays
        cyclic_callback: Optional for any cyclical callback which is running
        update_on_cycle_end: Whether to lock in to the cyclic callback and only update at the end of a cycle. Default yes, if cyclic callback present.
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    def __init__(self, model:Optional[AbsModel]=None, val_fold:Optional[Dict[str,np.ndarray]]=None,
                 cyclic_callback:Optional[AbsCyclicCallback]=None, update_on_cycle_end:Optional[bool]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        self.val_fold,self.cyclic_callback,self.update_on_cycle_end,self.active = val_fold,cyclic_callback,update_on_cycle_end,False

    def set_val_fold(self, val_fold:Dict[str,np.ndarray]) -> None:
        r'''
        Sets the validation fold used for evaluating new models
        '''
        
        self.val_fold = val_fold
    
    def set_cyclic_callback(self, cyclic_callback:AbsCyclicCallback) -> None:
        r'''
        Sets the cyclical callback to lock into for updating new models
        '''

        if cyclic_callback is not None:
            self.cyclic_callback = cyclic_callback
            if self.update_on_cycle_end is None: self.update_on_cycle_end = True

    @abstractmethod
    def get_loss(self) -> float: pass


class SWA(AbsModelCallback):
    r'''
    Callback providing Stochastic Weight Averaging based on (https://arxiv.org/abs/1803.05407)
    This adapted version allows the tracking of a pair of average models in order to avoid having to hardcode a specific start point for averaging:
    
    - Model average x0 will begin to be tracked start_epoch (sub-)epochs/cycles after training begins.
    - `cycle_since_replacement` is set to 1
    - Renewal_period (sub-)epochs/cycles later, a second average x1 will be tracked.
    - At the next renewal period, the performance of x0 and x1 will be compared on data contained in val_fold.
        
        - If x0 is better than x1:
            - x1 is replaced by a copy of the current model
            - cycle_since_replacement is increased by 1
            - renewal_period is multiplied by cycle_since_replacement
        - Else:
            - x0 is replaced by x1
            - x1 is replaced by a copy of the current model
            - cycle_since_replacement is set to 1
            - renewal_period is set back to its original value

    Additonally, will optionally (default True) lock-in to any cyclical callbacks to only update at the end of a cycle.

    Arguments:
        start_epoch: (sub-)epoch/cycle to begin averaging
        renewal_period: How often to check performance of averages, and renew tracking of least performant
        model: :class:`~lumin.nn.models.model.Model` to provide parameters, alternatively call :meth:`~lumin.nn.models.Model.set_model`
        val_fold: Dictionary containing inputs, targets, and weights (or None) as Numpy arrays
        cyclic_callback: Optional for any cyclical callback which is running
        update_on_cycle_end: Whether to lock in to the cyclic callback and only update at the end of a cycle. Default yes, if cyclic callback present.
        verbose: Whether to print out update information for testing and operation confirmation
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Examples::
        >>> swa = SWA(start_epoch=5, renewal_period=5)
    '''

    def __init__(self, start_epoch:int, renewal_period:int=-1, model:Optional[AbsModel]=None, val_fold:Optional[Dict[str,np.ndarray]]=None,
                 cyclic_callback:Optional[AbsCyclicCallback]=None, update_on_cycle_end:Optional[bool]=None,
                 verbose:bool=False, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, val_fold=val_fold, cyclic_callback=cyclic_callback, update_on_cycle_end=update_on_cycle_end, plot_settings=plot_settings)
        self.start_epoch,self.renewal_period,self.verbose = start_epoch,renewal_period,verbose
        self.weights,self.loss = None,None
        
    def on_train_begin(self, **kargs) -> None:
        r'''
        Initialises model variables to begin tracking new model averages
        '''

        if self.weights is None:
            self.weights = copy.deepcopy(self.model.get_weights())
            self.weights_new = copy.deepcopy(self.model.get_weights())
            self.test_model = copy.deepcopy(self.model)
            self.epoch,self.swa_n,self.n_since_renewal,self.first_completed,self.cycle_since_replacement,self.active = 0,0,0,False,1,False
            
    def on_epoch_begin(self, **kargs) -> None:
        r'''
        Resets loss to prepare for new epoch
        '''
        
        self.loss = None

    def on_epoch_end(self, **kargs) -> None:
        r'''
        Checks whether averages should be updated (or reset) and increments counters
        '''

        if self.epoch >= self.start_epoch and ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end):
            if self.swa_n == 0 and not self.active:
                if self.verbose: print("SWA beginning")
                self.active = True
            elif self.update_on_cycle_end and self.cyclic_callback.cycle_mult > 1:
                if self.verbose: print("Updating average")
                self.active = True
            self._update_average_model()
            self.swa_n += 1
            
            if self.swa_n > self.renewal_period:
                self.first_completed = True
                self.n_since_renewal += 1
                if self.n_since_renewal > self.cycle_since_replacement*self.renewal_period and self.renewal_period > 0: self._compare_averages()
            
        if (not self.update_on_cycle_end) or self.cyclic_callback.cycle_end: self.epoch += 1
        if self.active and not ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end or self.cyclic_callback.cycle_mult == 1): self.active = False
            
    def _update_average_model(self) -> None:
        if self.verbose: print(f"Model is {self.swa_n} epochs old")
        c_weights = self.model.get_weights()
        for param in self.weights:
            self.weights[param] *= self.swa_n
            self.weights[param] += c_weights[param]
            self.weights[param] /= self.swa_n+1
        
        if self.swa_n > self.renewal_period and self.first_completed and self.renewal_period > 0:
            if self.verbose: print(f"New model is {self.n_since_renewal} epochs old")
            for param in self.weights_new:
                self.weights_new[param] *= self.n_since_renewal
                self.weights_new[param] += c_weights[param]
                self.weights_new[param] /= (self.n_since_renewal+1)
            
    def _compare_averages(self) -> None:
        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(self.val_fold['inputs'], self.val_fold['targets'], self.val_fold['weights'])
        self.test_model.set_weights(self.weights_new)
        new_loss = self.test_model.evaluate(self.val_fold['inputs'], self.val_fold['targets'], self.val_fold['weights'])
        
        if self.verbose: print(f"Checking renewal of swa model, current model: {self.loss}, new model: {new_loss}")
        if new_loss < self.loss:
            if self.verbose: print("New model better, replacing\n____________________\n\n")
            self.loss = new_loss
            self.swa_n = self.n_since_renewal
            self.n_since_renewal = 1
            self.weights = copy.deepcopy(self.weights_new)
            self.weights_new = copy.deepcopy(self.model.get_weights())
            self.cycle_since_replacement = 1

        else:
            if self.verbose: print("Current model better, keeping\n____________________\n\n")
            self.weights_new = copy.deepcopy(self.model.get_weights())
            self.n_since_renewal = 1
            self.test_model.set_weights(self.weights)
            self.cycle_since_replacement += 1
                
    def get_loss(self) -> float:
        r'''
        Evaluates SWA model and returns loss

        Returns:
            Loss on validation fold for oldest SWA average
        '''

        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(self.val_fold['inputs'], self.val_fold['targets'], self.val_fold['weights'])
        return self.loss
        