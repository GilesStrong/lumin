from typing import Union, Tuple, Callable, Optional
import numpy as np

from torch import Tensor

from .callback import Callback
from ..data.batch_yielder import BatchYielder
from ..data.fold_yielder import FoldYielder
from ...utils.misc import to_np, to_device
from ..models.abs_model import AbsModel


class BinaryLabelSmooth(Callback):
    '''Apply label smoothing to binary classes, based on arXiv:1512.00567'''
    def __init__(self, coefs:Union[float,Tuple[float,float]]=0, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.coefs = coefs if isinstance(coefs, tuple) else (coefs, coefs)
    
    def on_epoch_begin(self, batch_yielder:BatchYielder, **kargs) -> None:
        '''Apply smoothing at train-time'''
        batch_yielder.targets = batch_yielder.targets.astype(float)
        batch_yielder.targets[batch_yielder.targets == 0] = self.coefs[0]
        batch_yielder.targets[batch_yielder.targets == 1] = 1-self.coefs[1]
         
    def on_eval_begin(self, targets:Tensor, **kargs) -> None:
        '''Apply smoothing at test-time'''
        targets[targets == 0] = self.coefs[0]
        targets[targets == 1] = 1-self.coefs[1]


class DynamicReweight(Callback):
    def __init__(self, reweight:Callable[[Tensor], Tensor], scale:float=1e-2, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.scale,self.reweight = scale,reweight
    
    def on_train_end(self, fy:FoldYielder, val_id:int, **kargs) -> None:
        val_fld = fy.get_fold(val_id)
        preds = self.model.predict_array(val_fld['inputs'], as_np=False)
        coefs = to_np(self.reweight(preds, to_device(Tensor(val_fld['targets']))))
        start_sum = np.sum(val_fld['weights'])
        val_fld['weights'] += self.scale*coefs*val_fld['weights']
        val_fld['weights'] *= start_sum/np.sum(val_fld['weights'])
        fy.foldfile[f'fold_{val_id}/weights'][...] = val_fld['weights'].squeeze()