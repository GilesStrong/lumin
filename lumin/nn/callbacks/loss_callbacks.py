from typing import Optional

import torch.nn as nn

from .callback import Callback
from ..models.abs_model import AbsModel


class GradClip(Callback):
    def __init__(self, clip:float, clip_norm:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.clip = clip
        self.func = nn.utils.clip_grad_norm_ if clip_norm else nn.utils.clip_grad_value_
        
    def on_backwards_end(self, **kargs) -> None:
        if self.clip > 0: self.func(self.model.parameters(), self.clip)
            