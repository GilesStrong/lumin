import numpy as np
from typing import Union

from torch.tensor import Tensor


def to_np(x:Tensor) -> np.ndarray:
    return x.detach().numpy()


def to_tensor(x:np.ndarray) -> Union[Tensor, None]:
    return Tensor(x) if x is not None else None


def uncert_round(value, uncert):
    if uncert == 0: return value, uncert
    
    factor = 1.0
    while uncert / factor > 1: factor *= 10.0
    value /= factor
    uncert /= factor   
    i = 0    
    while uncert * (10**i) <= 1: i += 1
    
    round_uncert = factor * round(uncert, i)
    round_value = factor * round(value, i)
    if int(round_uncert) == round_uncert:
        round_uncert = int(round_uncert)
        round_value = int(round_value)
    return round_value, round_uncert
