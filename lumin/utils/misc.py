import numpy as np
from typing import Union, Tuple

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


def str2bool(x:str) -> bool:
    if isinstance(x, bool): return x
    else: return x.lower() in ("yes", "true", "t", "1")


def get_moments(x:np.ndarray) -> Tuple[float,float,float,float]:
    n = len(x)
    m = np.mean(x)
    m_4 = np.mean((x-m)**4)
    s = np.std(x, ddof=1)
    s4 = s**4
    se_s2 = ((m_4-(s4*(n-3)/(n-1)))/n)**0.25
    se_s = se_s2/(2*s)
    return m, s/np.sqrt(n), s, se_s
