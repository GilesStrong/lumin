from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from ...utils.misc import to_device

__all__ = ["BatchYielder", "TorchGeometricBatchYielder"]


class BatchYielder:
    r"""
    Yields minibatches to model during training. Iteration provides one minibatch as tuple of tensors of inputs, targets, and weights.

    TODO: Improve this/change to dataloader

    Arguments:
        inputs: input array for (sub-)epoch
        targets: target array for (sub-)epoch
        bs: batchsize, number of data to include per minibatch
        objective: 'classification', 'multiclass classification', or 'regression'. Used for casting target dtype.
        weights: Optional weight array for (sub-)epoch
        shuffle: whether to shuffle the data at the beginning of an iteration
        use_weights: if passed weights, whether to actually pass them to the model
        bulk_move: whether to move all data to device at once. Default is true (saves time), but if device has low memory you can set to False.
        input_mask: optionally only use Boolean-masked inputs
        drop_last: whether to drop the last batch if it does not contain `bs` elements
    """

    def __init__(
        self,
        inputs: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        bs: int,
        objective: str,
        targets: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        shuffle: bool = True,
        use_weights: bool = True,
        bulk_move: bool = True,
        input_mask: Optional[np.ndarray] = None,
        drop_last: bool = True,
    ):
        (
            self.inputs,
            self.targets,
            self.weights,
            self.bs,
            self.objective,
            self.shuffle,
            self.use_weights,
            self.bulk_move,
            self.input_mask,
            self.drop_last,
        ) = (inputs, targets, weights, bs, objective, shuffle, use_weights, bulk_move, input_mask, drop_last)
        if isinstance(self.inputs, tuple):
            self.inputs, self.matrix_inputs = self.inputs
        else:
            self.matrix_inputs = None
        if isinstance(self.inputs, pd.DataFrame):
            self.inputs = self.inputs.values
        if self.input_mask is not None:
            self.inputs = self.inputs[:, self.input_mask]

    def __iter__(self) -> List[Tensor]:
        r"""
        Iterate through data in batches.

        Returns:
            tuple of batches of inputs, targets, and weights as tensors on device
        """

        full_idxs = np.arange(len(self.inputs))
        upper = len(full_idxs)
        if self.drop_last:
            upper -= self.bs - 1
        if self.shuffle:
            np.random.shuffle(full_idxs)

        if self.bulk_move:
            inputs = to_device(Tensor(self.inputs))
            if self.targets is not None:
                if "multiclass" in self.objective:
                    targets = to_device(Tensor(self.targets).long().squeeze(-1))
                else:
                    targets = to_device(Tensor(self.targets))
            if self.weights is not None and self.use_weights:
                weights = to_device(Tensor(self.weights))
            else:
                weights = None
            if self.matrix_inputs is not None:
                matrix_inputs = to_device(Tensor(self.matrix_inputs))
            else:
                matrix_inputs = None

            for i in range(0, upper, self.bs):
                idxs = full_idxs[i : i + self.bs]
                x = inputs[idxs] if matrix_inputs is None else (inputs[idxs], matrix_inputs[idxs])
                y = None if self.targets is None else targets[idxs]
                w = None if weights is None else weights[idxs]
                yield x, y, w

        else:
            for i in range(0, upper, self.bs):
                idxs = full_idxs[i : i + self.bs]
                if self.targets is not None:
                    if "multiclass" in self.objective:
                        y = to_device(Tensor(self.targets[idxs]).long().squeeze(-1))
                    else:
                        y = to_device(Tensor(self.targets[idxs]))
                else:
                    y = None
                if self.matrix_inputs is None:
                    x = to_device(Tensor(self.inputs[idxs]))
                else:
                    x = (to_device(Tensor(self.inputs[idxs])), to_device(Tensor(self.matrix_inputs[idxs])))
                w = to_device(Tensor(self.weights[idxs])) if self.weights is not None and self.use_weights else None
                yield x, y, w

    def __len__(self):
        return len(self.inputs) // self.bs if self.drop_last else math.ceil(len(self.inputs) / self.bs)

    def get_inputs(self, on_device: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Returns all data.

        Arguments:
            on_device: whether to place tensor on device

        Returns:
            tuple of inputs, targets, and weights as tensors on device
        """

        if on_device:
            if self.matrix_inputs is None:
                return to_device(Tensor(self.inputs))
            else:
                return (to_device(Tensor(self.inputs)), to_device(Tensor(self.matrix_inputs)))
        else:
            if self.matrix_inputs is None:
                return self.inputs
            else:
                return (self.inputs, self.matrix_inputs)


class TorchGeometricBatchYielder(BatchYielder):
    r"""
    :class:`~lumin.nn.data.batch_yielder.BatchYielder` for PyTorch Geometric data. kwargs for compatibility only.

    Arguments:
        inputs: PyTorch Geometric Dataset containing inputs, weights, and targets
        bs: batchsize, number of data to include per minibatch
        shuffle: whether to shuffle the data at the beginning of an iteration
        exclude_keys: data keys to exclude from inputs
    """

    def __init__(
        self,
        inputs: PyGDataset,
        bs: int,
        shuffle: bool = True,
        exclude_keys: Optional[List[str]] = None,
        use_weights: bool = True,
        **kwargs: Any,
    ):

        self.loader = PyGDataLoader(inputs, batch_size=bs, shuffle=shuffle, exclude_keys=exclude_keys)
        self.use_weights = use_weights

    def __iter__(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        r"""
        Iterate through data in batches.

        Returns:
            tuple of batches of inputs, targets, and weights as dictionaries of tensors on device
        """

        for batch in self.loader:
            batch = to_device(batch)
            x = {k: batch[k] for k in batch.keys if k not in ["y", "ptr"]}
            y = {"y": batch.y, "batch": batch.batch}
            w = {"weight": batch.weight, "batch": batch.batch} if "weight" in batch.keys and self.use_weights else None
            yield x, y, w

    def __len__(self):
        return len(self.loader)

    def get_inputs(self, on_device: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""
        Returns all data.

        Arguments:
            on_device: whether to place tensor on device

        Returns:
            tuple of inputs, targets, and weights as dictionaries of tensors on device
        """

        if on_device:
            x = {k: to_device(self.loader.dataset[k]) for k in self.loader.dataset.keys if k not in ["y", "ptr"]}
        else:
            x = {k: self.loader.dataset[k] for k in self.loader.dataset.keys if k not in ["y", "ptr"]}
        return x
