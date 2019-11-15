import numpy as np
from typing import Union, List, Tuple, Optional
import pandas as pd
import sympy

from sklearn.utils import resample

from torch.tensor import Tensor
import torch
import torch.nn as nn

__all__ = ['to_np', 'to_device', 'to_tensor', 'str2bool', 'to_binary_class', 'ids2unique', 'FowardHook', 'subsample_df']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # TODO: make device choosable by user


def to_np(x:Tensor) -> np.ndarray:
    r'''
    Convert Tensor x to a Numpy array

    Arguments:
        x: Tensor to convert

    Returns:
        x as a Numpy array
    '''
    
    return x.cpu().detach().numpy()


def to_device(x:Union[Tensor,List[Tensor]], device:torch.device=device) -> Union[Tensor,List[Tensor]]:
    r'''
    Recursively place Tensor(s) onto device

    Arguments:
        x: Tensor(s) to place on device

    Returns:
        Tensor(s) on device
    '''

    if x is None: return x
    if isinstance(x, list): return [to_device(o, device) for o in x]
    return x.to(device)


def to_tensor(x:Union[np.ndarray,None]) -> Union[Tensor, None]: 
    r'''
    Convert Numpy array to Tensor with possibility of a None being passed

    Arguments:
        x: Numpy array or None

    Returns:
        x as Tensor or None
    '''
    
    return Tensor(x) if x is not None else None


def str2bool(string:Union[str,bool]) -> bool:
    r'''
    Convert string representation of Boolean to bool

    Arguments:
        string: string representation of Boolean (or a Boolean)

    Returns:
        bool if bool was passed else, True if lowercase string matches is in ("yes", "true", "t", "1")
    '''

    if isinstance(string, bool): return string
    else:                        return string.lower() in ("yes", "true", "t", "1")


def to_binary_class(df:pd.DataFrame, zero_preds:List[str], one_preds:List[str]) -> None:
    r'''
    Map class precitions back to a binary prediction.
    The maximum prediction for features listed in zero_preds is treated as the prediction for class 0, vice versa for one_preds.
    The binary prediction is added to df in place as column 'pred'
    
    Arguments:
        df: DataFrame containing prediction features
        zero_preds: list of column names for predictions associated with class 0
        one_preds: list of column names for predictions associated with class 0
    '''

    zero = df[zero_preds].max(axis=1)[:, None]
    one = df[one_preds].max(axis=1)[:, None]
    tup = np.hstack((zero, one))
    predargs = np.argmax(tup, axis=1)
    preds = np.max(tup, axis=1)
    preds[predargs == 0] = 1-preds[predargs == 0]
    df['pred'] = preds


def ids2unique(ids: Union[List[int], np.ndarray]) -> np.ndarray:
    r'''
    Map a permutaion of integers to a unique number, or a 2D array of integers to unique numbers by row.
    Returned numbers are unique for a given permutation of integers.
    This is achieved by computing the product of primes raised to powers equal to the integers. Beacause of this, it can be easy to produce numbers which are
    too large to be stored if many (large) integers are passed.

    Arguments:
        ids: (array of) permutation(s) of integers to map

    Returns:
        (Array of) unique id(s) for given permutation(s)
    '''

    if not isinstance(ids, np.ndarray): ids = np.array(ids)[:,None]
    primes = np.broadcast_to(np.array([sympy.prime(i) for i in range(1, 1+ids.shape[1])]), ids.shape)
    return (primes**ids).prod(axis=-1)


class FowardHook():
    r'''
    Create a hook for performing an action based on the forward pass thorugh a nn.Module

    Arguments:
        module: nn.Module to hook
        hook_fn: Optional function to perform. Default is to record input and output of module

    Examples::
        >>> hook = ForwardHook(model.tail.dense)
        >>> model.predict(inputs)
        >>> print(hook.inputs)
    '''
    def __init__(self, module:nn.Module, hook_fn:Optional=None):
        self.input,self.output = None,None
        if hook_fn is not None: self.hook_fn = hook_fn
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module:nn.Module, input:Union[Tensor,Tuple[Tensor]], output:Union[Tensor,Tuple[Tensor]]) -> None:
        r'''
        Default hook function records inputs and outputs of module

        Arguments:
            module: nn.Module to hook
            input: input tensor
            output: output tensor of module
        '''

        self.input,self.output = input,output
        
    def remove(self) -> None:
        r'''
        Call when finished to remove hook
        '''

        self.hook.remove()


def subsample_df(df:pd.DataFrame, objective:str, targ_name:str, n_samples:Optional[int]=None, replace:bool=False, strat_key:Optional[str]=None,
                 wgt_name:Optional[str]=None) -> pd.DataFrame:
    r'''
    Subsamples, or samples with replacement, a DataFrame.
    Will automatically reweight data such that weight sums remain the same as the original DataFrame (per class)

    Arguments:
        df: DataFrame to sample
        objective: string representation of objective: either 'classification' or 'regression'
        targ_name: name of column containing target data
        n_samples: If set, will sample that number of data points, otherwise will sample with replacement a new DataFRame of the same size as the original
        replace: whether to sample with replacement
        strat_key: column name to use for stratified subsampling, if desired
        wgt_name: name of column containing weight data. If set, will reweight subsampled data, otherwise will not
    '''

    tmp_df = df.loc[resample(df.index, replace=replace, n_samples=n_samples, stratify=None if strat_key is None else df[strat_key])]
    
    # Reweight resampled data
    if wgt_name is not None:
        if 'class' in objective.lower():
            for c in tmp_df[targ_name].unique():
                tmp_df.loc[tmp_df[targ_name] == c, wgt_name] *= df.loc[df[targ_name] == c, wgt_name].sum() / tmp_df.loc[tmp_df[targ_name] == c, wgt_name].sum()
        else:
            tmp_df[wgt_name] *= df[wgt_name].sum() / tmp_df[wgt_name].sum()
    return tmp_df
