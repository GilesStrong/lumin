import numpy as np

import torch

from .callback import Callback


class PredHandler(Callback):
    r'''
    Default callback for predictions. Collects predictions over batches and returns them as stacked array
    '''

    def on_pred_begin(self) -> None:
        super().on_pred_begin()
        self.preds = []

    def on_pred_end(self) -> None: self.preds = torch.cat(self.preds)
    def get_preds(self) -> np.ndarray: return self.preds
    def on_forwards_end(self) -> None:
        if self.model.fit_params.state == 'test': self.preds.append(self.model.fit_params.y_pred.detach().cpu())
