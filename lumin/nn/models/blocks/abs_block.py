from typing import Callable, Optional

from torch import Tensor
import torch.nn as nn

from ..initialisations import lookup_normal_init


class AbsBlock(nn.Module):
    def __init__(self, lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        self.lookup_init,self.freeze = lookup_init,freeze
        super().__init__()

    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def freeze_layers(self) -> None:
        r'''
        Make parameters untrainable
        '''

        for p in self.parameters(): p.requires_grad = False
    
    def unfreeze_layers(self) -> None:
        r'''
        Make parameters trainable
        '''

        for p in self.parameters(): p.requires_grad = True

    def forward(self, x:Tensor) -> Tensor: pass

    def get_out_size(self) -> int: pass
