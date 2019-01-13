from fastprogress import master_bar, progress_bar
import numpy as np
import pandas as pd
import sklearn.utils

from ...utils.misc import to_tensor, bootstrap_stats
from ...utils.multiprocessing import mp_run
from ...plotting.interpretation import plot_fi
# from ..models.model import Model
from ..data.fold_yielder import FoldYielder
# from ..inference.ensemble import Ensemble

from torch import Tensor


def get_nn_feat_importance(model, fold_yielder:FoldYielder, pb_parent:master_bar=None, plot:bool=True) -> pd.DataFrame:
    feats = fold_yielder.input_feats
    scores = []
    fold_bar = progress_bar(range(fold_yielder.n_folds), parent=pb_parent)
    for fold_id in fold_bar:  # Average over folds
        val_fold = fold_yielder.get_fold(fold_id)
        val_fold['weights'] /= val_fold['weights'].sum()
        targs = Tensor(val_fold['targets'])
        weights = to_tensor(val_fold['weights'])
        nom = model.evaluate(Tensor(val_fold['inputs']), targs, weights=weights)

        tmp = []
        for i in range(len(feats)):
            x = val_fold['inputs'].copy()
            x[:,i] = sklearn.utils.shuffle(x[:,i])
            tmp.append(model.evaluate(Tensor(x), targs, weights=weights))
        tmp = np.array(tmp)
        scores.append((tmp-nom)/nom)

    # Bootstrap over folds
    scores = np.array(scores)
    bs = mp_run([{'data':scores[:,i], 'mean': True, 'std': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    fi = pd.DataFrame({'Feature':feats, 
                       'Importance':  [bs[f'{i}_mean'] for i in range(len(feats))], 
                       'Uncertainty': [bs[f'{i}_std']  for i in range(len(feats))]})

    if plot:
        tmp_fi = fi.sort_values('Importance', ascending=False).reset_index(drop=True)
        print("Top ten most important features:\n", tmp_fi[:min(len(tmp_fi), 10)])
        plot_fi(tmp_fi)
    return fi


def get_ensemble_feat_importance(ensemble, fold_yielder:FoldYielder) -> pd.DataFrame:
    mean_fi = []
    std_fi = []
    feats = fold_yielder.input_feats
    model_bar = master_bar(ensemble.models)

    for m, model in enumerate(model_bar):  # Average over models per fold
        fi = get_nn_feat_importance(model, fold_yielder, plot=False, pb_parent=model_bar)
        mean_fi.append(fi.Importance.values)
        std_fi.append(fi.Uncertainty.values)
    
    mean_fi = np.array(mean_fi)
    std_fi = np.array(std_fi)
    bs_mean = mp_run([{'data': mean_fi[:,i], 'mean': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    bs_std  = mp_run([{'data': std_fi[:,i],  'mean': True, 'name': i, 'n':100} for i in range(len(feats))], bootstrap_stats)
    
    fi = pd.DataFrame({'Feature':feats, 
                       'Importance':  [bs_mean[f'{i}_mean'] for i in range(len(feats))], 
                       'Uncertainty': [bs_std[f'{i}_mean']  for i in range(len(feats))]}).sort_values('Importance', ascending=False).reset_index(drop=True)
    print("Top ten most important features:\n", fi[:min(len(fi), 10)])
    plot_fi(fi)
    return fi
