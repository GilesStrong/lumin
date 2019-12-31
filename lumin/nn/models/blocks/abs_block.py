from typing import Callable, Optional
from abc import abstractmethod

from torch import Tensor
import torch.nn as nn

from ..initialisations import lookup_normal_init

__all__ = []


class AbsBlock(nn.Module):
    def __init__(self, lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init, freeze:bool=False):
        self.lookup_init,self.freeze = lookup_init,freeze
        super().__init__()

    def __getitem__(self, key:int) -> nn.Module: return self.layers[key]

    def get_param_count(self, trainable:bool=True) -> int:
        r'''
        Return number of parameters in block.

        Arguments:
            trainable: if true (default) only count trainable parameters

        Returns:
            NUmber of (trainable) parameters in block
        '''
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

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
    
    @abstractmethod
    def forward(self, x:Tensor) -> Tensor:
        r'''
        Pass tensor through block

        Arguments:
            x: input tensor
        
        Returns
            Resulting tensor
        '''

        pass

    @abstractmethod
    def get_out_size(self) -> int:
        r'''
        Get size width of output layer

        Returns:
            Width of output layer
        '''

        pass
