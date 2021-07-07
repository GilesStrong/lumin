from typing import List, Optional, Callable
from fastcore.all import is_listy, store_attr
import timeit

from torch import nn, Tensor

from .callback import Callback
from .data_callbacks import TargReplace
from ..models.model_builder import ModelBuilder
from ..models.model import Model
from ...utils.misc import is_partially

__all__ = ['PivotTraining']


class PivotTraining(Callback):
    r'''
    Callback implementation of "Learning to Pivot with Adversarial Networks" (Louppe, Kagan, & Cranmer, 2016) 
    (https://papers.nips.cc/paper/2017/hash/48ab2f9b45957ab574cf005eb8a76760-Abstract.html).
    The default target data in the :class:`~lumin.nn.data.fold_yielder.FoldYielder` should be the target data for the main model,
    and it should contain additional columns for target data for the adversary (names should be passed to the `adv_targets` argument.)
    
    Once training begins, both the main model and the adversary will be pretrained in isolation.
    Further training of the main model then starts, with the frozen adversary providing a bonus to the loss value if the adversary cannot predict well its
    targets based on the prediction of the main model.
    At a set interval (multiples of per batch/fold/epoch), the adversary is refined for 1 epoch with the main model frozen (if per batch, this can take a long
    time with no progression indicated to the user).
    States of the model and the adversary are saved to the savepath after both pretraining and further training.
    
    Arguments:
        n_pretrain_main: number of epochs to pretrain the main model 
        n_pretrain_adv: number of epochs to pretrain the adversary
        adv_coef: relative weighting for the adversarial bonus (lambda in the paper),
            code assumes a positive value and subtracts adversarial loss from the main loss
        adv_model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` defining the adversary (note that this should not define main_model+adversary) 
        adv_targets: list of column names in foldfile to use as targets for the adversary
        adv_update_freq: sets how often the adversary is refined (e.g. once every `adv_update_freq` ticks)
        adv_update_on:str defines the tick for refining the adversary, can be batch, fold, or epoch. The paper refines once for every batch of training data.
        main_pretrain_cb_partials: Optional list of partial callbacks to use when pretraining the main model
        adv_pretrain_cb_partials: Optional list of partial callbacks to use when pretraining the adversary model
        adv_train_cb_partials: Optional list of partial callbacks to use when refining the adversary model
    '''

    def __init__(self, n_pretrain_main:int, n_pretrain_adv:int,
                 adv_coef:float, adv_model_builder:ModelBuilder, adv_targets:List[str], adv_update_freq:int, adv_update_on:str,
                 main_pretrain_cb_partials:Optional[List[Callable[[],Callback]]]=None,
                 adv_pretrain_cb_partials:Optional[List[Callable[[],Callback]]]=None,
                 adv_train_cb_partials:Optional[List[Callable[[],Callback]]]=None):
        store_attr(but='adv_update_on')
        adv_update_on = adv_update_on.lower()
        if adv_update_on not in ['batch','fold','epoch']: raise ValueError("adv_update_on must be one of ['batch','fold','epoch']")
        self.adv_update_on = adv_update_on
        if not is_listy(self.adv_targets): adv_targets = [adv_targets]
        if self.main_pretrain_cb_partials is None: self.main_pretrain_cb_partials = []
        if not is_listy(self.main_pretrain_cb_partials): self.main_pretrain_cb_partials = [self.main_pretrain_cb_partials]
        if self.adv_pretrain_cb_partials is None: self.adv_pretrain_cb_partials = []
        if not is_listy(self.adv_pretrain_cb_partials): self.adv_pretrain_cb_partials = [self.adv_pretrain_cb_partials]
        if self.adv_train_cb_partials is None: self.adv_train_cb_partials = []
        if not is_listy(self.adv_train_cb_partials): self.adv_train_cb_partials = [self.adv_train_cb_partials]

    def on_train_begin(self) -> None:
        r'''
        Pretrains main model and adversary, then prepares for further training.
        Adds prepends training callbacks with a :class:`~lumin.nn.callbacks.data_callbacks.TargReplace` instance to grab both the target and pivot data
        '''
        
        super().on_train_begin()
        for c in self.model.fit_params.cbs:
            if isinstance(c, TargReplace): return  # Don't run again (on_train_begin prepends callback to cbs)
        # Pretrain models
        print("Pretraining main model")
        main = Model(self.model.model_builder)
        cbs = []
        for c in self.main_pretrain_cb_partials: cbs.append(c())
        model_tmr = timeit.default_timer()
        main.fit(n_epochs=self.n_pretrain_main, fy=self.model.fit_params.fy, bs=self.model.fit_params.bs,
                 bulk_move=self.model.fit_params.bulk_move, train_on_weights=self.model.fit_params.train_on_weights,
                 trn_idxs=self.model.fit_params.trn_idxs, cbs=cbs, cb_savepath=self.model.fit_params.cb_savepath)
        print(f"pretraining main model took {timeit.default_timer()-model_tmr:.3f}s\n")
        main.save(self.model.fit_params.cb_savepath/'pretrain_main.h5')
        self.model.set_weights(main.get_weights())
        
        print("Pretraining adversary")
        self.adv = Model(self.adv_model_builder)
        self.adv.model = nn.Sequential(self.model.model,self.adv.model)
        self.adv.opt = self.adv_model_builder._build_opt(self.adv.model)
        self.model.freeze_layers()
        cbs = [TargReplace(self.adv_targets)]
        for c in self.adv_pretrain_cb_partials: cbs.append(c())
        model_tmr = timeit.default_timer()
        self.adv.fit(n_epochs=self.n_pretrain_adv, fy=self.model.fit_params.fy, bs=self.model.fit_params.bs,
                     bulk_move=self.model.fit_params.bulk_move, train_on_weights=self.model.fit_params.train_on_weights,
                     trn_idxs=self.model.fit_params.trn_idxs, cbs=cbs, cb_savepath=self.model.fit_params.cb_savepath)
        print(f"pretraining adversary took {timeit.default_timer()-model_tmr:.3f}s\n")
        self.adv.save(self.model.fit_params.cb_savepath/'pretrain_adv.h5')
        
        # prep for combined training
        self.adv_loss_func = self.adv_model_builder.loss
        if is_partially(self.adv_loss_func): self.adv_loss_func = self.adv_loss_func()
        self.model.fit_params.cbs.insert(0, TargReplace(['targets']+self.adv_targets))
        self.model.fit_params.cbs[0].set_model(self.model)
        self.count = -1
        self.adv.freeze_layers()
        self.model.unfreeze_layers()
        
    def on_train_end(self) -> None:
        r'''
        Save final version of adversary
        '''
        
        self.adv.save(self.model.fit_params.cb_savepath/'adv.h5')
        
    def _increment(self) -> None:
        r'''
        Increments tick and refines adversary if required
        '''
        
        self.count += 1
        if self.count >= self.adv_update_freq:
            self.count = 0
            self.adv.unfreeze_layers()
            self.model.freeze_layers()
            cbs = [TargReplace(self.adv_targets)]
            for c in self.adv_train_cb_partials: cbs.append(c())
            self.adv.fit(n_epochs=1, fy=self.model.fit_params.fy, bs=self.model.fit_params.bs,
                         bulk_move=self.model.fit_params.bulk_move, train_on_weights=self.model.fit_params.train_on_weights,
                         trn_idxs=self.model.fit_params.trn_idxs, cbs=cbs, cb_savepath=self.model.fit_params.cb_savepath, visible_bar=False)
            self.adv.freeze_layers()
            self.model.unfreeze_layers()
        
    def on_batch_begin(self) -> None:
        r'''
        Slices off adversarial and main-model targets. Increments tick if required.
        '''
        
        self.adv_y = self.model.fit_params.y[:,-len(self.adv_targets):]
        self.model.fit_params.y = self.model.fit_params.y[:,:-len(self.adv_targets)]
        if self.model.fit_params.state == 'train' and self.adv_update_on == 'batch': self._increment()
            
    def on_fold_begin(self)  -> None:
        r'''
        Increments tick if required.
        '''
        
        if self.model.fit_params.state == 'train' and self.adv_update_on == 'fold':  self._increment()
            
    def on_epoch_begin(self) -> None:
        r'''
        Increments tick if required.
        '''
        
        if self.model.fit_params.state == 'train' and self.adv_update_on == 'epoch': self._increment()
        
    def _compute_adv_loss(self) -> Tensor:
        r'''
        Computes (weighted) adversarial loss value
        '''
        
        adv_p = self.adv.model(self.model.fit_params.x)
        if hasattr(self.adv_loss_func, 'weights'): self.adv_loss_func.weights = self.model.fit_params.w  # Proper weighting required
        else:                                      self.adv_loss_func.weight  = self.model.fit_params.w
        return self.adv_coef*self.adv_loss_func(adv_p, self.adv_y)
    
    def on_forwards_end(self) -> None:
        r'''
        Applies adversarial bonus to main-model loss
        '''
        
        if self.model.fit_params.state == 'test': return
        elif self.model.fit_params.state == 'valid': self.adv.model.eval()
        else:                                        self.adv.model.train()
        self.model.fit_params.loss_val -= self._compute_adv_loss()  # Move to maxima
