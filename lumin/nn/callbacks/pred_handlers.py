import numpy as np
from typing import Optional

import torch

from .callback import Callback
from ..models.abs_model import AbsModel


class PredHandler(Callback):
    r'''
    Default callback for predictions. Collects predictions over batches and returns them as stacked array
    '''
    
    def __init__(self, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.reset()

    def reset(self) -> None: self.preds = []
    def on_pred_begin(self) -> None: self.reset()
    def on_pred_end(self) -> None: self.preds = torch.cat(self.preds)
    def get_preds(self) -> np.ndarray: return self.preds
    def on_forwards_end(self) -> None:
        if self.model.fit_params.state == 'test': self.preds.append(self.model.fit_params.y_pred)
