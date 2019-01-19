import numpy as np
from typing import Union, List
import pandas as pd

from torch.tensor import Tensor


def to_np(x:Tensor) -> np.ndarray:
    return x.detach().numpy()


def to_tensor(x:np.ndarray) -> Union[Tensor, None]:
    return Tensor(x) if x is not None else None


def str2bool(x:str) -> bool:
    if isinstance(x, bool): return x
    else: return x.lower() in ("yes", "true", "t", "1")


def to_binary_class(df:pd.DataFrame, zero_preds:List[str], one_preds:List[str]) -> None:
    zero = df[zero_preds].max(axis=1)[:, None]
    one = df[one_preds].max(axis=1)[:, None]
    tup = np.hstack((zero, one))
    predargs = np.argmax(tup, axis=1)
    preds = np.max(tup, axis=1)
    preds[predargs == 0] = 1-preds[predargs == 0]
    df['pred'] = preds
