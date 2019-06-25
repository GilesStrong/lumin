from fastprogress import master_bar, progress_bar
import numpy as np
import pandas as pd
import sklearn.utils
from typing import Optional

from ...utils.misc import to_tensor
from ...utils.statistics import bootstrap_stats
from ...utils.multiprocessing import mp_run
from ...plotting.interpretation import plot_importance
from ..models.abs_model import AbsModel
from ..ensemble.abs_ensemble import AbsEnsemble
from ..data.fold_yielder import FoldYielder
from ..metrics.eval_metric import EvalMetric
from ...plotting.plot_settings import PlotSettings

from torch import Tensor


def get_nn_feat_importance(model:AbsModel, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None, pb_parent:master_bar=None,
                           plot:bool=True, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> pd.DataFrame:
    r'''
    Compute permutation importance of features used by a :class:Model on provided data using either loss or an :class:EvalMetric to quantify performance.
    Returns bootstrapped mean importance from sample constructed by computing importance for each fold in fy.

    Arguments:
        model: :class:Model to use to evaluate feature importance
        fy: :class:FoldYielder interfacing to data used to train model
        eval_metric: Optional :class:EvalMetric to use to quantify performance in place of loss
        pb_parent: Not used if calling method directly
        plot: whetehr to plot resulting feature importances
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance

    Returns:
        Pandas DataFrame containing mean importance and associated uncertainty for each feature

    Examples::
        >>> fi = get_nn_feat_importance(model, train_fy)
        >>> fi = get_nn_feat_importance(model, train_fy, savename='feat_import')
        >>> fi = get_nn_feat_importance(model, train_fy, eval_metric=AMS(n_total=100000))
    '''

    feats = fy.cont_feats + fy.cat_feats
    scores = []
    fold_bar = progress_bar(range(fy.n_folds), parent=pb_parent)
    for fold_idx in fold_bar:  # Average over folds
        val_fold = fy.get_fold(fold_idx)
        if val_fold['weights'] is not None: val_fold['weights'] /= val_fold['weights'].sum()
        targs = Tensor(val_fold['targets'])
        weights = to_tensor(val_fold['weights'])
        if eval_metric is None: nom = model.evaluate(Tensor(val_fold['inputs']), targs, weights=weights)
        else:                   nom = eval_metric.evaluate(fy, fold_idx, model.predict(Tensor(val_fold['inputs'])))
        tmp = []
        for i in range(len(feats)):
            x = val_fold['inputs'].copy()
            x[:,i] = sklearn.utils.shuffle(x[:,i])
            if eval_metric is None: tmp.append(model.evaluate(Tensor(x), targs, weights=weights))
            else:                   tmp.append(eval_metric.evaluate(fy, fold_idx, model.predict(Tensor(x))))

        if eval_metric is None: tmp = (np.array(tmp)-nom)/nom
        else:                   tmp = (np.array(tmp)-nom)/nom if eval_metric.lower_metric_better else (nom-np.array(tmp))/nom
        scores.append(tmp)

    # Bootstrap over folds
    scores = np.array(scores)
    bs = mp_run([{'data':scores[:,i], 'mean': True, 'std': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    fi = pd.DataFrame({'Feature':feats, 
                       'Importance':  [np.mean(bs[f'{i}_mean']) for i in range(len(feats))], 
                       'Uncertainty': [np.mean(bs[f'{i}_std'])  for i in range(len(feats))]})

    if plot:
        tmp_fi = fi.sort_values('Importance', ascending=False).reset_index(drop=True)
        print("Top ten most important features:\n", tmp_fi[:min(len(tmp_fi), 10)])
        plot_importance(tmp_fi, savename=savename, settings=settings)
    return fi


def get_ensemble_feat_importance(ensemble:AbsEnsemble, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None,
                                 savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> pd.DataFrame:
    r'''
    Compute permutation importance of features used by an :class:Ensemble on provided data using either loss or an :class:EvalMetric to quantify performance.
    Returns bootstrapped mean importance from sample constructed by computing importance for each :class:Model in ensemble.

    Arguments:
        ensemble: :class:Ensemble to use to evaluate feature importance
        fy: :class:FoldYielder interfacing to data used to train models in ensemble
        eval_metric: Optional :class:EvalMetric to use to quantify performance in place of loss
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:PlotSettings class to control figure appearance

    Returns:
        Pandas DataFrame containing mean importance and associated uncertainty for each feature

    Examples::
        >>> fi = get_ensemble_feat_importance(ensemble, train_fy)
        >>> fi = get_ensemble_feat_importance(ensemble, train_fy, savename='feat_import')
        >>> fi = get_ensemble_feat_importance(ensemble, train_fy, eval_metric=AMS(n_total=100000))
    '''

    mean_fi = []
    std_fi = []
    feats = fy.cont_feats + fy.cat_feats
    model_bar = master_bar(ensemble.models)

    for m, model in enumerate(model_bar):  # Average over models per fold
        fi = get_nn_feat_importance(model, fy, eval_metric=eval_metric, plot=False, pb_parent=model_bar)
        mean_fi.append(fi.Importance.values)
        std_fi.append(fi.Uncertainty.values)
    
    mean_fi = np.array(mean_fi)
    std_fi = np.array(std_fi)
    bs_mean = mp_run([{'data': mean_fi[:,i], 'mean': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    bs_std  = mp_run([{'data': std_fi[:,i],  'mean': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    
    fi = pd.DataFrame({'Feature':feats, 
                       'Importance':  [np.mean(bs_mean[f'{i}_mean']) for i in range(len(feats))], 
                       'Uncertainty': [np.mean(bs_std[f'{i}_mean'])  for i in range(len(feats))]}).sort_values('Importance', ascending=False).reset_index(drop=True)
    print("Top ten most important features:\n", fi[:min(len(fi), 10)])
    plot_importance(fi, savename=savename, settings=settings)
    return fi
