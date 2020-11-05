import numpy as np
from typing import Optional

import torch

from .callback import Callback
from ..models.abs_model import AbsModel


class PredHandler(Callback):
    r'''
    Default callback for predictions. Collects predictions over batches and returns them as stacked array
    '''
    
    def __init__(self, model:Optional[AbsModel]):
        super().__init__(model=model)
        self._reset()

    def _reset(self) -> None: self.preds = []
    def on_pred_begin(self) -> None: self.reset()
    def on_pred_end(self) -> None: self.preds = torch.stack(self.preds)
    def get_preds(self) -> np.ndarray: return self.preds
    def on_forwards_end(self) -> None:
        if self.wrapper.state == 'test': self.preds.append(self.wrapper.y_pred)
