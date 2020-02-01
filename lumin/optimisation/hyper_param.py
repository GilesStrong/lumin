from typing import Tuple, Dict, List, Optional
from fastprogress import master_bar, progress_bar
import numpy as np
from collections import OrderedDict
import timeit
from functools import partial

from sklearn.ensemble.forest import ForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ..nn.data.fold_yielder import FoldYielder
from ..nn.data.batch_yielder import BatchYielder
from ..nn.models.model_builder import ModelBuilder
from ..nn.models.model import Model
from ..nn.callbacks.opt_callbacks import LRFinder
from ..nn.callbacks.cyclic_callbacks import AbsCyclicCallback
from ..nn.callbacks.model_callbacks import AbsModelCallback

from ..plotting.training import plot_lr_finders
from ..plotting.plot_settings import PlotSettings

import matplotlib.pyplot as plt

__all__ = ['get_opt_rf_params', 'fold_lr_find']


def get_opt_rf_params(x_trn:np.ndarray, y_trn:np.ndarray, x_val:np.ndarray, y_val:np.ndarray, objective:str,
                      w_trn:Optional[np.ndarray]=None, w_val:Optional[np.ndarray]=None,
                      params:Optional[OrderedDict]=None, n_estimators:int=40, verbose=True) -> Tuple[Dict[str,float],ForestRegressor]:
    r'''
    Use an ordered parameter-scan to roughly optimise Random Forest hyper-parameters.

    Arguments:
        x_trn: training input data
        y_trn: training target data
        x_val: validation input data
        y_val: validation target data
        objective: string representation of objective: either 'classification' or 'regression'
        w_trn: training weights
        w_val: validation weights
        params: ordered dictionary mapping parameters to optimise to list of values to cosnider
        n_estimators: number of trees to use in each forest
        verbose: Print extra information and show a live plot of model performance

    Returns:
        params: dictionary mapping parameters to their optimised values
        rf: best performing Random Forest
    '''
    if params is None: params = OrderedDict({'min_samples_leaf': [1,3,5,10,25,50,100], 'max_features': [0.3,0.5,0.7,0.9]})
    rf = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
    best_params = {'n_estimators': n_estimators, 'n_jobs': -1, 'max_features':'sqrt'}
    best_scores = []
    scores = []
    mb = master_bar(params)
    mb.names = ['Best', 'Scores']
    if verbose: mb.update_graph([[[],[]], [[], []]])
    for param in mb:
        pb = progress_bar(params[param], parent=mb)
        pb.comment = f'{param} = {params[param][0]}'
        for i, value in enumerate(pb):
            pb.comment = f'{param} = {params[param][min(i+1, len(params[param])-1)]}'
            m = rf(**{**best_params, param: value})
            m.fit(x_trn, y_trn, w_trn)
            scores.append(m.score(x_val, y_val, w_val))
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


def fold_lr_find(fy:FoldYielder, model_builder:ModelBuilder, bs:int,
                 train_on_weights:bool=True, shuffle_fold:bool=True, n_folds:int=-1, lr_bounds:Tuple[float,float]=[1e-5, 10],
                 callback_partials:Optional[List[partial]]=None, plot_settings:PlotSettings=PlotSettings(),
                 bulk_move:bool=True) -> List[LRFinder]:
    r'''
    Wrapper function for training using :class:`~lumin.nn.callbacks.opt_callbacks.LRFinder` which runs a Smith LR range test (https://arxiv.org/abs/1803.09820)
    using folds in :class:`~lumin.nn.data.fold_yielder.FoldYielder`.
    Trains models for 1 fold, interpolating LR between set bounds. This repeats for each fold in :class:`~lumin.nn.data.fold_yielder.FoldYielder`, and loss
    evolution is averaged.

    Arguments:
        fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` providing training data
        model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` providing networks and optimisers
        bs: batch size
        train_on_weights: If weights are present, whether to use them for training
        shuffle_fold: whether to shuffle data in folds
        n_folds: if >= 1, will only train n_folds number of models, otherwise will train one model per fold
        lr_bounds: starting and ending LR values
        callback_partials: optional list of functools.partial, each of which will a instantiate :class:`~lumin.nn.callbacks.callback.Callback` when called        
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        List of :class:`~lumin.nn.callbacks.opt_callbacks.LRFinder` which were used for each model trained
    '''

    if callback_partials is None: callback_partials = []
    idxs = range(fy.n_folds) if n_folds < 1 else range(min(n_folds, fy.n_folds))
    lr_finders = []
    tmr = timeit.default_timer()
    for trn_id in progress_bar(idxs):
        model = Model(model_builder)
        trn_fold = fy.get_fold(trn_id)
        nb = len(trn_fold['targets'])//bs
        lr_finder = LRFinder(nb=nb, lr_bounds=lr_bounds, model=model)
        cyclic_callback,callbacks = None,[]
        for c in callback_partials: callbacks.append(c(model=model))
        for c in callbacks:
            if isinstance(c, AbsCyclicCallback): c.set_nb(nb)
        for c in callbacks:
            if isinstance(c, AbsModelCallback): c.set_cyclic_callback(cyclic_callback)
        for c in callbacks:
            c.on_train_begin()
        lr_finder.on_train_begin()
        batch_yielder = BatchYielder(**fy.get_fold(trn_id), objective=model_builder.objective, bs=bs, use_weights=train_on_weights, shuffle=shuffle_fold,
                                     bulk_move=bulk_move)
        model.fit(batch_yielder, callbacks+[lr_finder])
        lr_finders.append(lr_finder)
        
    print("LR finder took {:.3f}s ".format(timeit.default_timer()-tmr))
    plot_lr_finders(lr_finders, loss_range='auto', settings=plot_settings)
    return lr_finders

