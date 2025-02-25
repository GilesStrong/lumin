from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from ..data.fold_yielder import FoldYielder
from ..models.abs_model import AbsModel

__all__ = []


class AbsEnsemble(metaclass=ABCMeta):
    """Abstract classs for ensembles"""

    def __init__(self):
        self.models, self.weights, self.size = [], [], 0

    def __repr__(self) -> str:
        return f"Ensemble of size {self.size}\nWeights:\n{self.weights}\nModels:\n{self.models}"

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Tuple[float, AbsModel]:
        for w, m in zip(self.weights, self.models):
            yield w, m

    def __getitem__(self, idx: int) -> Tuple[float, AbsModel]:
        return self.weights[idx], self.models[idx]

    def __setitem__(self, idx: int, value: Tuple[float, AbsModel]) -> None:
        self.weights[idx], self.models[idx] = value

    def append(self, value: Tuple[float, AbsModel]) -> None:
        self.weights.append(value[0])
        self.models.append(value[1])
        self.size = len(self.models)

    def pop(self, idx: int = -1) -> Tuple[float, AbsModel]:
        w = self.weights.pop(idx)
        m = self.models.pop(idx)
        self.size = len(self.models)
        return w, m

    @abstractmethod
    def predict(
        self,
        in_data: Union[np.ndarray, FoldYielder, List[np.ndarray]],
        n_models: Optional[int] = None,
        pred_name: str = "pred",
    ) -> Union[None, np.ndarray]:
        pass
