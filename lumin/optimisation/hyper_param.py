from typing import List, Tuple, Dict
from fastprogress import master_bar, progress_bar
import timeit
import numpy as np
from collections import OrderedDict

from sklearn.ensemble.forest import ForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ..nn.data.fold_yielder import FoldYielder
from ..nn.data.batch_yielder import BatchYielder
from ..nn.models.model_builder import ModelBuilder
from ..nn.models.model import Model
from ..nn.callbacks.opt_callbacks import LRFinder
from ..plotting.training import plot_lr_finders
from ..plotting.plot_settings import PlotSettings

import matplotlib.pyplot as plt


def fold_lr_find(fold_yielder:FoldYielder, model_builder:ModelBuilder, bs:int,
                 train_on_weights:bool=True, shuffle_fold:bool=True, n_folds:int=-1, lr_bounds:Tuple[float,float]=[1e-5, 10],
                 plot_settings:PlotSettings=PlotSettings()) -> List[LRFinder]:
    start = timeit.default_timer()
    indeces = range(fold_yielder.n_folds) if n_folds < 1 else range(min(n_folds, fold_yielder.n_folds))
    lr_finders = []
    for trn_id in progress_bar(indeces):
        model = Model(model_builder)
        trn_fold = fold_yielder.get_fold(trn_id)
        nb = len(trn_fold['targets'])//bs
        lr_finder = LRFinder(nb=nb, lr_bounds=lr_bounds, model=model)
        lr_finder.on_train_begin()
        batch_yielder = BatchYielder(**fold_yielder.get_fold(trn_id), bs=bs, use_weights=train_on_weights, shuffle=shuffle_fold)
        model.fit(batch_yielder, [lr_finder])
        lr_finders.append(lr_finder)
        
    print("LR finder took {:.3f}s ".format(timeit.default_timer()-start))
    if n_folds != 1:
        plot_lr_finders(lr_finders, loss='loss', cut=-2, settings=plot_settings)
    else:
        lr_finders[0].plot_lr()    
        lr_finders[0].plot(n_skip=5)
    return lr_finders


def get_opt_rf_params(X_trn:np.ndarray, y_trn:np.ndarray, w_trn:np.ndarray,
                      X_val:np.ndarray, y_val:np.ndarray, w_val:np.ndarray, objective:str,
                      params=OrderedDict({'min_samples_leaf': [1,3,5,10,25,50,100], 'max_features': [0.3,0.5,0.7,0.9]}),
                      verbose=True) -> Tuple[Dict[str,float],ForestRegressor]:
    rf = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
    
    best_params = {'n_estimators': 40, 'n_jobs': -1, 'max_features':'sqrt'}
    best_scores = []
    scores = []
    mb = master_bar(params)
    mb.names = ['Best', 'Scores']
    if verbose: mb.update_graph([[[],[]], [[], []]])
    for param in mb:
        for value in progress_bar(params[param], parent=mb):
            m = rf(**{**best_params, param: value})
            m.fit(X_trn, y_trn, w_trn)
            scores.append(m.score(X_val, y_val, w_val))
            if len(best_scores) == 0 or scores[-1] > best_scores[-1]:
                best_scores.append(scores[-1])
                best_params[param] = value
                if verbose: print(f'Better score schieved: {param} @ {value} = {best_scores[-1]:.4f}')
                best_m = m
            else:
                best_scores.append(best_scores[-1])
            if verbose: mb.update_graph([[range(len(best_scores)), best_scores], [range(len(scores)), scores]])
    
    if verbose: delattr(mb, 'fig')
    if verbose: plt.clf()
    return best_params, best_m
