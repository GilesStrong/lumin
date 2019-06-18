from typing import Optional

import torch.nn as nn

from .callback import Callback
from ..models.abs_model import AbsModel


class GradClip(Callback):
    r'''
    Callback for clipping gradients by norm or value.

    Arguments:
        clip: value to clip at
        clip_norm: whether to clip according to norm (nn.utils.clip_grad_norm_) or value (nn.utils.clip_grad_value_)
         model: :class:Model with parameters to clip gradients, alternatively call :meth:set_model

    Examples::
        >>> grad_clip = GradClip(1e-5)
    '''

    def __init__(self, clip:float, clip_norm:bool=True, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.clip = clip
        self.func = nn.utils.clip_grad_norm_ if clip_norm else nn.utils.clip_grad_value_
        
    def on_backwards_end(self, **kargs) -> None:
        if self.clip > 0: self.func(self.model.parameters(), self.clip)
            