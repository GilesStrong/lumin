from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastcore.all import store_attr

from ...utils.misc import to_np
from ..callbacks.callback import Callback
from ..data.fold_yielder import FoldYielder
from ..models.abs_model import AbsModel, FitParams

__all__ = ["EvalMetric", "TorchGeometricEvalMetric"]


class EvalMetric(Callback, metaclass=ABCMeta):
    r"""
    Abstract class for evaluating performance of a model using some metric

    Arguments:
        name: optional name for metric, otherwise will be inferred from class
        lower_metric_better: whether a lower metric value should be treated as representing better perofrmance
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted
    """

    def __init__(self, name: Optional[str], lower_metric_better: bool, main_metric: bool = True):
        store_attr(but=["name"])
        self.name = type(self).__name__ if name is None else name

    def on_train_begin(self) -> None:
        r"""
        Ensures that only one main metric is used
        """

        super().on_train_begin()
        self.metric = None
        if self.main_metric:
            for c in self.model.fit_params.cbs:
                if hasattr(c, "main_metric"):
                    c.main_metric = False
            self.main_metric = True

    def on_epoch_begin(self) -> None:
        r"""
        Resets prediction tracking
        """

        self.preds, self.metric = [], None

    def on_forwards_end(self) -> None:
        r"""
        Save predictions from batch
        """

        if self.model.fit_params.state == "valid":
            self.preds.append(self.model.fit_params.y_pred.cpu().detach())

    def on_epoch_end(self) -> None:
        r"""
        Compute metric using saved predictions
        """

        if self.model.fit_params.state != "valid":
            return
        self.preds = to_np(torch.cat(self.preds)).squeeze()
        if "multiclass" in self.model.objective:
            self.preds = np.exp(self.preds)
        self.targets = self.model.fit_params.by.targets.squeeze()
        self.weights = self.model.fit_params.by.weights if self.model.fit_params.by.use_weights else None
        if self.weights is not None:
            self.weights = self.weights.squeeze()
        self.metric = self.evaluate()
        del self.preds

    def get_metric(self) -> float:
        r"""
        Returns metric value

        Returns:
            metric value
        """

        return self.metric

    @abstractmethod
    def evaluate(self) -> float:
        r"""
        Evaluate the required metric for a given fold and set of predictions

        Returns:
            metric value
        """

        pass

    def evaluate_model(
        self,
        model: AbsModel,
        fy: FoldYielder,
        fold_idx: int,
        inputs: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
        bs: Optional[int] = None,
    ) -> float:
        r"""
        Gets model predicitons and computes metric value. fy and fold_idx arguments necessary in case the metric requires extra information beyond inputs,
        tragets, and weights.

        Arguments:
            model: model to evaluate
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` containing data
            fold_idx: fold index of corresponding data
            inputs: input data
            targets: target data
            weights: optional weights
            bs: optional batch size

        Returns:
            metric value
        """

        self.model = model
        preds = self.model.predict(inputs, bs=bs)
        return self.evaluate_preds(fy=fy, fold_idx=fold_idx, preds=preds, targets=targets, weights=weights)

    def evaluate_preds(
        self,
        fy: FoldYielder,
        fold_idx: int,
        preds: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Computes metric value from predictions. fy and fold_idx arguments necessary in case the metric requires extra information beyond inputs,
        tragets, and weights.

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` containing data
            fold_idx: fold index of corresponding data
            inputs: input data
            targets: target data
            weights: optional weights
            bs: optional batch size

        Returns:
            metric value
        """

        class MockModel:
            def __init__(self):
                pass

        if not hasattr(self, "model") or self.model is None:
            self.model = MockModel()
        self.model.fit_params = FitParams(val_idx=fold_idx, fy=fy)
        self.preds, self.targets, self.weights = preds.squeeze(), targets.squeeze(), weights
        if self.weights is not None:
            self.weights = weights.squeeze()
        self.model.fit_params = FitParams(val_idx=fold_idx, fy=fy)  # predict reset fit_params to None
        return self.evaluate()

    def get_df(self) -> pd.DataFrame:
        r"""
        Returns a DataFrame for the given fold containing targets, weights, and predictions

        Returns:
            DataFrame for the given fold containing targets, weights, and predictions
        """

        df = pd.DataFrame()
        if hasattr(self, "wgt_name"):
            df["gen_weight"] = self.model.fit_params.fy.get_column(
                column=self.wgt_name, n_folds=1, fold_idx=self.model.fit_params.val_idx
            )

        if hasattr(self, "targ_name") and self.targ_name is not None:
            targets = self.model.fit_params.fy.get_column(
                column=self.targ_name, n_folds=1, fold_idx=self.model.fit_params.val_idx
            )
        else:
            targets = self.targets

        if len(targets.shape) > 1:
            for t in range(targets.shape[-1]):
                df[f"gen_target_{t}"] = targets[:, t]
        else:
            df["gen_target"] = targets

        if len(self.preds.shape) > 1 and self.preds.shape[-1] > 1:
            for p in range(self.preds.shape[-1]):
                df[f"pred_{p}"] = self.preds[:, p]
        else:
            df["pred"] = self.preds.squeeze()
        return df


class TorchGeometricEvalMetric(EvalMetric):
    r"""
    Abstract class for evaluating performance of a model using some metric and PyTorch Geometric data

    Arguments:
        name: optional name for metric, otherwise will be inferred from class
        lower_metric_better: whether a lower metric value should be treated as representing better perofrmance
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted
    """

    def on_epoch_begin(self) -> None:
        r"""
        Resets prediction tracking
        """

        self.preds, self.targets, self.batches, self.weights, self.metric = [], [], [], [], None
        self.batch_cnt = 0

    def on_forwards_end(self) -> None:
        r"""
        Save predictions from batch
        """

        if self.model.fit_params.state == "valid":
            self.preds.append(self.model.fit_params.y_pred.cpu().detach())
            self.targets.append(self.model.fit_params.y["y"].cpu().detach())
            self.batches.append(self.model.fit_params.y["batch"].cpu().detach() + self.batch_cnt)
            self.batch_cnt = self.batches[-1].max()
            if self.model.fit_params.w is not None:
                self.weights.append(self.model.fit_params.w.cpu().detach())

    def on_epoch_end(self) -> None:
        r"""
        Compute metric using saved predictions
        """

        if self.model.fit_params.state != "valid":
            return
        self.preds = torch.cat(self.preds, dim=0)
        if "multiclass" in self.model.objective:
            self.preds = torch.exp(self.preds)
        self.targets = torch.cat(self.targets, dim=0)
        self.batches = torch.cat(self.batches, dim=0)
        self.weights = torch.cat(self.weights, dim=0) if len(self.weights) > 0 else None
        self.metric = self.evaluate()
        del self.preds
        del self.targets
        del self.batches
        del self.weights
