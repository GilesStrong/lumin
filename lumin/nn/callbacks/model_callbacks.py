import numpy as np
from typing import Optional
from abc import abstractmethod
import copy

import torch.tensor as Tensor

from ..models.abs_model import AbsModel
from .callback import Callback
from .cyclic_callbacks import AbsCyclicCallback
from ...utils.misc import to_tensor
from ...plotting.plot_settings import PlotSettings


class AbsModelCallback(Callback):
    '''Abstract class for callbacks which provide alternative models during training'''
    def __init__(self, model:Optional[AbsModel]=None, val_fold:Optional[np.ndarray]=None,
                 cyclic_callback:Optional[AbsCyclicCallback]=None, update_on_cycle_end:Optional[bool]=None, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, plot_settings=plot_settings)
        self.val_fold,self.cyclic_callback,self.update_on_cycle_end,self.active = val_fold,cyclic_callback,update_on_cycle_end,False

    def set_val_fold(self, val_fold:np.ndarray) -> None: self.val_fold = val_fold
    
    def set_cyclic_callback(self, cyclic_callback:AbsCyclicCallback) -> None:
        if cyclic_callback is not None:
            self.cyclic_callback = cyclic_callback
            if self.update_on_cycle_end is None: self.update_on_cycle_end = True

    @abstractmethod
    def get_loss(self) -> float: pass


class SWA(AbsModelCallback):
    '''Track weight-avergaed model during training using Stochastic Weight Averaging https://arxiv.org/abs/1803.05407'''
    def __init__(self, start_epoch:int, renewal_period:int=-1, model:Optional[AbsModel]=None, val_fold:Optional[np.ndarray]=None,
                 cyclic_callback:Optional[AbsCyclicCallback]=None, update_on_cycle_end:Optional[bool]=None,
                 verbose=False, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, val_fold=val_fold, cyclic_callback=cyclic_callback, update_on_cycle_end=update_on_cycle_end, plot_settings=plot_settings)
        self.start_epoch,self.renewal_period,self.verbose = start_epoch,renewal_period,verbose
        self.weights,self.loss = None,None
        
    def on_train_begin(self, **kargs) -> None:
        if self.weights is None:
            self.weights = copy.deepcopy(self.model.get_weights())
            self.weights_new = copy.deepcopy(self.model.get_weights())
            self.test_model = copy.deepcopy(self.model)
            self.epoch,self.swa_n,self.n_since_renewal,self.first_completed,self.cycle_since_replacement,self.active = 0,0,0,False,1,False
            
    def on_epoch_begin(self, **kargs) -> None: self.loss = None

    def on_epoch_end(self, **kargs) -> None:
        if self.epoch >= self.start_epoch and ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end):
            if self.swa_n == 0 and not self.active:
                if self.verbose: print("SWA beginning")
                self.active = True
            elif self.update_on_cycle_end and self.cyclic_callback.cycle_mult > 1:
                if self.verbose: print("Updating average")
                self.active = True
            self.update_average_model()
            self.swa_n += 1
            
            if self.swa_n > self.renewal_period:
                self.first_completed = True
                self.n_since_renewal += 1
                if self.n_since_renewal > self.cycle_since_replacement*self.renewal_period and self.renewal_period > 0: self.compare_averages()
            
        if (not self.update_on_cycle_end) or self.cyclic_callback.cycle_end: self.epoch += 1
        if self.active and not ((not self.update_on_cycle_end) or self.cyclic_callback.cycle_end or self.cyclic_callback.cycle_mult == 1): self.active = False
            
    def update_average_model(self) -> None:
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
            
    def compare_averages(self) -> None:
        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(Tensor(self.val_fold['inputs']), Tensor(self.val_fold['targets']), to_tensor(self.val_fold['weights']))
        self.test_model.set_weights(self.weights_new)
        new_loss = self.test_model.evaluate(Tensor(self.val_fold['inputs']), Tensor(self.val_fold['targets']), to_tensor(self.val_fold['weights']))
        
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
        if self.loss is None:
            self.test_model.set_weights(self.weights)
            self.loss = self.test_model.evaluate(Tensor(self.val_fold['inputs']), Tensor(self.val_fold['targets']), to_tensor(self.val_fold['weights']))
        return self.loss
