from typing import Optional
import copy
from distutils.version import LooseVersion
from fastcore.all import store_attr

import torch

from ..models.model import Model
from .callback import Callback

__all__ = ['SWA']
        

class SWA(Callback):
    r'''
    Callback providing Stochastic Weight Averaging based on (https://arxiv.org/abs/1803.05407)
    This adapted version allows the tracking of a pair of average models in order to avoid having to hardcode a specific start point for averaging:
    
    - Model average #0 will begin to be tracked start_epoch epochs/cycles after training begins.
    - `cycle_since_replacement` is set to 1
    - Renewal_period epochs/cycles later, a second average #1 will be tracked.
    - At the next renewal period, the performance of #0 and #1 will be compared on data contained in val_fold.
        
        - If #0 is better than #1:
            - #1 is replaced by a copy of the current model
            - cycle_since_replacement is increased by 1
            - renewal_period is multiplied by cycle_since_replacement
        - Else:
            - #0 is replaced by #1
            - #1 is replaced by a copy of the current model
            - cycle_since_replacement is set to 1
            - renewal_period is set back to its original value

    Additonally, will optionally (default True) lock-in to any cyclical callbacks to only update at the end of a cycle.

    Arguments:
        start_epoch: epoch/cycle to begin averaging
        renewal_period: How often to check performance of averages, and renew tracking of least performant. If None, will not track a second average.
        update_on_cycle_end: Whether to lock in to the cyclic callback and only update at the end of a cycle. Default yes, if cyclic callback present.
        verbose: Whether to print out update information for testing and operation confirmation

    Examples::
        >>> swa = SWA(start_epoch=5, renewal_period=5)
    '''

    def __init__(self, start_epoch:int, renewal_period:Optional[int]=None, update_on_cycle_end:Optional[bool]=None, verbose:bool=False):
        super().__init__()
        if not isinstance(start_epoch, int):
            print('Coercing start_epoch to int')
            start_epoch = int(start_epoch)
        if not (isinstance(renewal_period, int) or renewal_period is None):
            print('Coercing renewal_period to int')
            renewal_period = int(renewal_period)
        store_attr(but=['model','plot_settings'])
        self.weights,self.loss = None,None
        self.true_div = True if LooseVersion(torch.__version__) >= LooseVersion("1.6") else False  # Integer division changed in PyTorch 1.6
        
    def on_train_begin(self) -> None:
        r'''
        Initialises model variables to begin tracking new model averages
        '''
        
        super().on_train_begin()
        self.cyclic_callback = None if len(self.model.fit_params.cyclic_cbs) == 0 else self.model.fit_params.cyclic_cbs[-1]
        self.epoch,self.swa_n,self.n_since_renewal,self.first_completed,self.cycle_since_replacement,self.active = 0,0,0,False,1,False
        
    def _create_weights(self) -> None:
        self.weights = copy.deepcopy(self.model.get_weights())
        self.weights_new = copy.deepcopy(self.model.get_weights())
        self.test_model = Model(self.model.model_builder)  # Can't deep copy model since fit_params contains SWA callback
        self.test_model.loss = copy.deepcopy(self.model.loss)  # In case user has manually changed the loss function
            
    def on_epoch_begin(self) -> None:
        r'''
        Resets loss to prepare for new epoch
        '''
        
        self.loss = None

    def on_epoch_end(self) -> None:
        r'''
        Checks whether averages should be updated (or reset) and increments counters
        '''

        if self.model.fit_params.state != 'train': return
        if self.epoch >= self.start_epoch and ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end):
            if self.swa_n == 0 and not self.active:
                if self.verbose: print("SWA beginning")
                self.active = True
                self._create_weights()
            elif self.update_on_cycle_end and self.cyclic_callback.cycle_mult > 1:
                if self.verbose: print("Updating average")
                self.active = True
            self._update_average_model()
            self.swa_n += 1
            
            if self.swa_n > self.renewal_period:
                self.first_completed = True
                self.n_since_renewal += 1
                if self.n_since_renewal > self.cycle_since_replacement*self.renewal_period and self.renewal_period is not None: self._compare_averages()
            
        if (not self.update_on_cycle_end) or self.cyclic_callback.cycle_end: self.epoch += 1
        if self.active and not ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end or self.cyclic_callback.cycle_mult == 1): self.active = False
            
    def _update_average_model(self) -> None:
        if self.verbose: print(f"Model is {self.swa_n} epochs old")
        c_weights = self.model.get_weights()
        for param in self.weights:
            self.weights[param] *= self.swa_n
            self.weights[param] += c_weights[param]
            if self.true_div: self.weights[param] = torch.true_divide(self.weights[param], self.swa_n+1)
            else:             self.weights[param] /= self.swa_n+1
        
        if self.swa_n > self.renewal_period and self.first_completed and self.renewal_period is not None:
            if self.verbose: print(f"New model is {self.n_since_renewal} epochs old")
            for param in self.weights_new:
                self.weights_new[param] *= self.n_since_renewal
                self.weights_new[param] += c_weights[param]
                if self.true_div: self.weights_new[param] = torch.true_divide(self.weights_new[param], self.n_since_renewal+1)
                else:             self.weights_new[param] /= self.n_since_renewal+1
            
    def _compare_averages(self) -> None:
        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(self.model.fit_params.by)
        self.test_model.set_weights(self.weights_new)
        new_loss = self.test_model.evaluate(self.model.fit_params.by)
        
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
        '''

        if self.epoch <= self.start_epoch: return self.model.fit_params.loss_val.data.item()
        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(self.model.fit_params.by)
        return self.loss
        