from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import IN_NOTEBOOK
import pickle
import timeit
import numpy as np
import os
from fastcore.all import is_listy

from ..data.fold_yielder import FoldYielder
from ..models.model_builder import ModelBuilder
from ..models.model import Model
from ..callbacks.callback import Callback
from ..callbacks.monitors import EarlyStopping, SaveBest, MetricLogger
from ...utils.statistics import uncert_round
from ..metrics.eval_metric import EvalMetric
from ...plotting.training import plot_train_history
from ...plotting.plot_settings import PlotSettings

import matplotlib.pyplot as plt

__all__ = ['train_models']


def train_models(fy:FoldYielder, n_models:int, bs:int, model_builder:ModelBuilder, n_epochs:int, patience:Optional[int]=None, loss_is_meaned:bool=True,
                 cb_partials:Optional[List[Callable[[],Callback]]]=None, metric_partials:Optional[List[Callable[[],EvalMetric]]]=None, save_best:bool=True,
                 train_on_weights:bool=True, bulk_move:bool=True, start_model_id:int=0,
                 excl_idxs:Optional[List[int]]=None, unique_trn_idxs:bool=False,
                 live_fdbk:bool=IN_NOTEBOOK, live_fdbk_first_only:bool=False, live_fdbk_extra:bool=True, live_fdbk_extra_first_only:bool=False,
                 savepath:Path=Path('train_weights'), plot_settings:PlotSettings=PlotSettings()) \
        -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]],List[Dict[str,float]]]:
    r'''
    Main training method for :class:`~lumin.nn.models.model.Model`.
    Trains a specified numer of models created by a :class:`~lumin.nn.models.model_builder.ModelBuilder` on data provided by a
    :class:`~lumin.nn.data.fold_yielder.FoldYielder`, and saves them to `savepath`.

    Note, this does not return trained models, instead they are saved and must be loaded later. Instead this method returns results of model training.
    Each :class:`~lumin.nn.models.model.Model` is trained on N-1 folds, for a :class:`~lumin.nn.data.fold_yielder.FoldYielder` with N folds, and the remaining
    fold is used as validation data.
        
    Depending on the live_fdbk arguments, live plots of losses and other metrics may be shown during training, if running in Jupyter.
    Showing the live plot slightly slows down the training, but can help highlight problems without having to wait to the end.
    If not running in Jupyter, then losses are printed to the terminal.

    Once training is finished, the state with the lowest validation loss is loaded, evaluated, and saved.

    Arguments:
        fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing ot training data
        n_models: number of models to train
        bs: batch size. Number of data points per iteration
        model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` creating the networks to train
        n_epochs: maximum number of epochs for which to train
        patience: if not `None`, sets the number of epochs or cycles to train without decrease in validation loss before ending training (early stopping)
        loss_is_meaned: if the batch loss value has been averaged over the number of elements in the batch, this should be true
        cb_partials: optional list of functools.partial, each of which will instantiate a :class:`~lumin.nn.callbacks.callback.Callback` when called
        metric_partials: optional list of functools.partial, each of which will a instantiate :class:`~lumin.nn.metric.eval_metric.EvalMetric`,
            used to compute additional metrics on validation data after each epoch.
            :class:`~lumin.nn.callbacks.monitors.SaveBest` and :class:`~lumin.nn.callbacks.monitors.EarlyStopping` will also act on the (first) metric set to 
            `main_metric` instead of loss, except when another callback produces an alternative loss and model
            (like :class:`~lumin.nn.callbacks.model_callbacks.SWA`).
        save_best: if true, will save the best performing model as the final model, otherwise will save the model state as per the end of training.
            A copy of the best model will still be saved anyway.
        train_on_weights: If weights are present in training data, whether to pass them to the loss function during training
        bulk_move: if true, will optimise for speed by using more RAM and VRAM
        start_model_id: model ID at whcih to start training,
            i.e. if training was interupted, this can be set to resume training form the last model which was trained
        excl_idxs: optional list of fold indeces to exclude from training and validation
        unique_trn_idxs: if false, then fold indeces can be shared, 
            e.g. if `fy` contains 10 folds and five models are requested, each model will be trained on 9 folds.
            if true, each model will every model will be trained on different folds,
            e.g. if `fy` contains 10 folds and five models are requested, each model will be trained on 2 folds and no same fold is used to train more than one model
            This is useful when the amount of training data exceeds the amount required to train a single model:
            it can be split into a large number of folds and a set of decorellated models trained.
        live_fdbk: whether or not to show any live feedback at all during training (slightly slows down training, but helps spot problems)
        live_fdbk_first_only: whether to only show live feedback for the first model trained (trade off between time and problem spotting)
        live_fdbk_extra: whether to show extra information live feedback (further slows training)
        live_fdbk_extra_first_only: whether to only show extra live feedback information for the first model trained (trade off between time and information)
        savepath: path to to which to save model weights and results
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        - results list of validation losses and other eval_metrics results, ordered by model training. Can be used to create an :class:`~lumin.nn.ensemble.ensemble.Ensemble`.
        - histories list of loss histories, ordered by model training
        - cycle_losses if an :class:`~lumin.nn.callbacks.cyclic_callbacks.AbsCyclicCallback` was passed, lists validation losses at the end of each cycle, ordered by model training. Can be passed to :class:`~lumin.nn.ensemble.ensemble.Ensemble`.
    '''

    results,histories,cycle_losses,savepath = [],[],[],Path(savepath)
    if cb_partials is None: cb_partials = []
    if not is_listy(cb_partials): cb_partials = [cb_partials]
    if metric_partials is None: metric_partials = []
    if not is_listy(metric_partials): metric_partials = [metric_partials]
    if excl_idxs is None: excl_idxs = []
    if not is_listy(excl_idxs): excl_idxs = [excl_idxs]
    idxs = [i for i in range(fy.n_folds) if i not in excl_idxs]
    trn_idx_sets,val_idxs = [],[]
    if unique_trn_idxs:
        n = len(idxs)//n_models
        if n == 0: raise ValueError(f"{len(idxs)} training folds are not enough to train {n_models} with unique data.")
        for i in range(n_models):
            val_idxs.append(idxs[i])
            trn_idx_sets.append([j for j in idxs if j != idxs[i] and j not in np.array(trn_idx_sets).flatten()][:n])

    model_rng = range(start_model_id, n_models)
    for i in model_rng: os.system(f"rm -r {savepath}/model_id_{i}")
    model_bar = master_bar(model_rng) if IN_NOTEBOOK else progress_bar(model_rng)
    train_tmr = timeit.default_timer()
    for model_num in (model_bar):    
        if IN_NOTEBOOK: model_bar.show()
        val_idx  = val_idxs[model_num]     if unique_trn_idxs else idxs[model_num % len(idxs)]
        trn_idxs = trn_idx_sets[model_num] if unique_trn_idxs else [j for j in idxs if j != val_idx]
        print(f"Training model {model_num+1} / {n_models}, Valid Index = {val_idx}, Train indices= {trn_idxs}")
        if model_num == 1:
            if live_fdbk_first_only: live_fdbk = False  # Only show fdbk for first training
            elif live_fdbk_extra_first_only: live_fdbk_extra = False  # Only show full fdbk info for first training

        model_dir = savepath/f'model_id_{model_num}'
        model_dir.mkdir(parents=True)
        model = Model(model_builder)

        cbs = []
        for c in cb_partials: cbs.append(c())
        for c in metric_partials: cbs.append(c())
        metric_log = MetricLogger(show_plots=live_fdbk, extra_detail=live_fdbk_extra, loss_is_meaned=loss_is_meaned)
        cbs += [metric_log, SaveBest(auto_reload=save_best, loss_is_meaned=loss_is_meaned)]
        if patience is not None: cbs.append(EarlyStopping(patience=patience, loss_is_meaned=loss_is_meaned))
        for c in cbs: c.set_plot_settings(plot_settings)

        model_tmr = timeit.default_timer()
        model.fit(n_epochs=n_epochs, fy=fy, bs=bs, bulk_move=bulk_move, train_on_weights=train_on_weights, trn_idxs=trn_idxs, val_idx=val_idx,
                  cbs=cbs, cb_savepath=model_dir)
        print(f"Model took {timeit.default_timer()-model_tmr:.3f}s\n")
        model.save(model_dir/f'train_{model_num}.h5')

        histories.append(metric_log.get_loss_history())
        cycle_losses.append([])
        for c in cbs:
            if hasattr(c, 'cycle_save') and c.cycle_save: cycle_losses[-1] = c.cycle_losses
        results.append(metric_log.get_results(save_best=save_best))
        print(f"Scores are: {results[-1]}")
        results[-1]['path'] = model_dir
        with open(savepath/'results_file.pkl', 'wb') as fout: pickle.dump(results, fout)
        with open(savepath/'cycle_file.pkl', 'wb') as fout: pickle.dump(cycle_losses, fout)
        
        plt.clf()

    print("\n______________________________________")
    print("Training finished")
    print(f"Cross-validation took {timeit.default_timer()-train_tmr:.3f}s ")
    plot_train_history(histories, savepath/'loss_history', settings=plot_settings, show=IN_NOTEBOOK, log_y='regress' in model_builder.objective)
    for score in results[0]:
        if score == 'path': continue
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))  # SHould this be ndof=1 for std?
        print(f"Mean {score} = {mean[0]}±{mean[1]}")
    print("______________________________________\n")
    return results, histories, cycle_losses
