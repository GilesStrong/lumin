import numpy as np
import pandas as pd
from typing import Union
from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from ....utils.misc import to_np

__all__ = ['AbsEndcap']


class AbsEndcap(nn.Module):
    r'''
    Abstract class for constructing post training layer which performs further calculation on NN outputs.
    Used when NN was trained to some proxy objective

    Arguments:
        model: trained :class:`~lumin.nn.models.model.Model` to wrap
    '''

    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
    
    @abstractmethod
    def func(self, x:Tensor) -> Tensor:
        r'''
        Transformation functio to apply to model outputs

        Arguements:
            x: model output tensor

        Returns:
            Resulting tensor
        '''
        
        pass
        
    def forward(self, x:Tensor) -> Tensor:
        r'''
        Pass tensor through endcap and compute function

        Arguments:
            x: model output tensor
        
        Returns
            Resulting tensor
        '''

        return self.func(x)
    
    def predict(self, inputs:Union[np.ndarray, pd.DataFrame, Tensor], as_np:bool=True) -> Union[np.ndarray, Tensor]:
        r'''
        Evaluate model on input tensor, and comput function of model outputs
        
        Arguments:
            inputs: input data as Numpy array, Pandas DataFrame, or tensor on device
            as_np: whether to return predictions as Numpy array (otherwise tensor)

        Returns:
            model predictions pass through endcap function
        '''

        # TODO add mask

        x = self.model.predict(inputs, as_np=False)
        x = self.func(x)
        return to_np(x) if as_np else x
