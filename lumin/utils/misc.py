import numpy as np
from typing import Union, List
import pandas as pd
import sympy

from torch.tensor import Tensor
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_np(x:Tensor) -> np.ndarray: return x.cpu().detach().numpy()


def to_device(x:Union[Tensor,List[Tensor]], device:torch.device=device):
    if x is None: return x
    if isinstance(x, list): return [to_device(o, device) for o in x]
    return x.to(device)


def to_tensor(x:np.ndarray) -> Union[Tensor, None]: return Tensor(x) if x is not None else None


def str2bool(string:str) -> bool:
    if isinstance(string, bool): return string
    else:                        return string.lower() in ("yes", "true", "t", "1")


def to_binary_class(df:pd.DataFrame, zero_preds:List[str], one_preds:List[str]) -> None:
    '''Map class precitions back to a binary prediction'''
    zero = df[zero_preds].max(axis=1)[:, None]
    one = df[one_preds].max(axis=1)[:, None]
    tup = np.hstack((zero, one))
    predargs = np.argmax(tup, axis=1)
    preds = np.max(tup, axis=1)
    preds[predargs == 0] = 1-preds[predargs == 0]
    df['pred'] = preds


def ids2unique(ids: Union[List, np.ndarray]) -> np.ndarray:
    '''Map list of integers to a unique number'''
    if not isinstance(ids, np.ndarray): ids = np.array(ids)[:,None]
    primes = np.broadcast_to(np.array([sympy.prime(i) for i in range(1, 1+ids.shape[1])]), ids.shape)
    return (primes**ids).prod(axis=-1)
