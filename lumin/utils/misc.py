import numpy as np
from typing import Union

from torch.tensor import Tensor


def to_np(x:Tensor) -> np.ndarray:
    return x.detach().numpy()


def to_tensor(x:np.ndarray) -> Union[Tensor, None]:
    return Tensor(x) if x is not None else None


def str2bool(x:str) -> bool:
    if isinstance(x, bool): return x
    else: return x.lower() in ("yes", "true", "t", "1")
