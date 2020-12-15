import numpy as np
from typing import Optional, Dict, List
from abc import abstractmethod
import copy
from distutils.version import LooseVersion
from fastcore.all import store_attr

import torch

from ..models.abs_model import AbsModel, OldAbsModel
from ..models.model import Model
from .callback import Callback, OldCallback
from .cyclic_callbacks import AbsCyclicCallback
from .abs_callback import AbsCallback
from ...plotting.plot_settings import PlotSettings
from ..data.batch_yielder import BatchYielder

__all__ = ['SWA']


class OldAbsModelCallback(OldCallback):
    r'''
    .. Attention:: This class is depreciated.
        It is a copy of the old `AbsModelCallback` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

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
    def get_loss(self, bs:Optional[int]=None, use_weights:bool=True, callbacks:Optional[List[AbsCallback]]=None) -> float: pass


class OldSWA(OldAbsModelCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.model_callbacks.SWA`.
        It is a copy of the old `SWA` class used in lumin<=0.6.
        It will be removed in V0.8

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
    
    # XXX remove in V0.8

    def __init__(self, start_epoch:int, renewal_period:int=-1, model:Optional[OldAbsModel]=None, val_fold:Optional[Dict[str,np.ndarray]]=None,
                 cyclic_callback:Optional[AbsCyclicCallback]=None, update_on_cycle_end:Optional[bool]=None,
                 verbose:bool=False, plot_settings:PlotSettings=PlotSettings()):
        super().__init__(model=model, val_fold=val_fold, cyclic_callback=cyclic_callback, update_on_cycle_end=update_on_cycle_end, plot_settings=plot_settings)
        self.start_epoch,self.renewal_period,self.verbose = start_epoch,renewal_period,verbose
        self.weights,self.loss = None,None
        self.true_div = True if LooseVersion(torch.__version__) >= LooseVersion("1.6") else False  # Integer division changed in PyTorch 1.6
        
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
            if self.true_div: self.weights[param] = torch.true_divide(self.weights[param], self.swa_n+1)
            else:             self.weights[param] /= self.swa_n+1
        
        if self.swa_n > self.renewal_period and self.first_completed and self.renewal_period > 0:
            if self.verbose: print(f"New model is {self.n_since_renewal} epochs old")
            for param in self.weights_new:
                self.weights_new[param] *= self.n_since_renewal
                self.weights_new[param] += c_weights[param]
                if self.true_div: self.weights_new[param] = torch.true_divide(self.weights_new[param], self.n_since_renewal+1)
                else:             self.weights_new[param] /= self.n_since_renewal+1
            
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
                
    def get_loss(self, bs:Optional[int]=None, use_weights:bool=True, callbacks:Optional[List[AbsCallback]]=None) -> float:
        r'''
        Evaluates SWA model and returns loss

        Arguments:
            bs: If not None, will evaluate loss in batches, rather than loading whole fold onto device
            use_weights: Whether to compute weighted loss if weights are present
            callbacks: list of any callbacks to use during evaluation

        Returns:
            Loss on validation fold for oldest SWA average
        '''

        if self.loss is None:
            self.test_model.set_weights(self.weights)
            if bs is None:
                self.loss = self.test_model.evaluate(self.val_fold['inputs'], self.val_fold['targets'], self.val_fold['weights'], callbacks=callbacks)
            else:
                by = BatchYielder(**self.val_fold, objective=self.model.objective,
                                  bs=bs, use_weights=use_weights, shuffle=False, bulk_move=False)
                self.loss = self.test_model.evaluate_from_by(by, callbacks=callbacks)
        return self.loss
        

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
        