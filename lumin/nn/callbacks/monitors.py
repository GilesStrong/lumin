from fastcore.all import store_attr, Path
import math
from typing import Optional, Union
import numpy as np

from .callback import Callback
from ..models.abs_model import AbsModel

__all__ = ['EarlyStopping']


class EarlyStopping(Callback):
    r'''
    Tracks validation loss during training and terminates training if loss doesn't decrease after `patience` number of epochs.
    Losses are assumed to be averaged and will be re-averaged over the epoch unless `loss_is_meaned` is false.
    '''
    
    def __init__(self, patience:int, loss_is_meaned:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        store_attr(but=['model'])
        self._reset()

    def _reset(self) -> None: self.epochs,self.min_loss = 0,math.inf

    def on_train_begin(self) -> None:
        self._reset()
        self.cyclic_cb = None if len(self.model.fit_params.cyclic_cbs) == 0 else self.model.fit_params.cyclic_cbs[-1]
        self.improve_in_cycle = False

    def on_epoch_begin(self) -> None:
        if self.model.state != 'valid': return
        self.cnt = 0
        self.loss = [0] + [0 for _ in self.model.fit_params.loss_cbs]  # Consider all losses e.g. SWA loss

    def on_forwards_end(self) -> None:
        if self.model.state != 'valid': return
        sz = len(self.model.x) if self.loss_is_meaned else 1
        self.loss[0] += self.model.loss_val.data.item()*sz
        for i in range(len(self.model.fit_params.loss_cbs)): self.loss[i+1] += self.model.fit_params.loss_cbs[i].get_loss()
        self.cnt += sz

    def on_epoch_end(self) -> None:
        if self.model.state != 'valid': return
        loss = np.min(self.loss)/self.cnt
        if loss <= self.min_loss:
            self.min_loss = loss
            self.epochs = 0
            self.improve_in_cycle = True
        elif self.cyclic_cb is not None:
            if self.cyclic_cb.cycle_end:
                if self.improve_in_cycle:
                    self.epochs = 0
                    self.improve_in_cycle = False
                else:
                    self.epochs += 1
        else:
            self.epochs += 1
        if self.epochs >= self.patience:
            print('Early stopping')
            self.model.stop = True


class SaveBest(Callback):
    r'''
    Tracks validation loss during training and automatically saves a copy of the weights to indicated file whenever validation loss decreases.
    Losses are assumed to be averaged and will be re-averaged over the epoch unless `loss_is_meaned` is false.
    '''

    def __init__(self, auto_reload:bool=True, loss_is_meaned:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        store_attr(but=['model'])
        self._reset()

    def _reset(self) -> None: self.min_loss = math.inf
    def on_train_begin(self) -> None: self._reset()

    def on_epoch_begin(self) -> None:
        if self.model.state != 'valid': return
        self.cnt = 0
        self.loss = [0] + [0 for _ in self.model.fit_params.loss_cbs]  # Consider all losses e.g. SWA loss

    def on_forwards_end(self) -> None:
        if self.model.state != 'valid': return
        sz = len(self.model.x) if self.loss_is_meaned else 1
        self.loss += self.model.loss_val.data.item()*sz
        self.cnt += sz

    def on_epoch_end(self) -> None:
        if self.model.state != 'valid': return
        loss = np.array(self.loss)/self.cnt
        lm = np.min(loss)
        if lm < self.min_loss:
            self.min_loss = lm
            lam  = np.argmin(loss)
            m = self.model
            if lam >= 0: m = self.model.fit_params.loss_cbs[lam-1].test_model
            m.save(self.model.fit_params.cb_savepath/'best.h5')

    def on_train_end(self) -> None:
        print(f'Loading best model with loss {self.min_loss}')
        self.model.load(self.model.fit_params.cb_savepath/'best.h5')
