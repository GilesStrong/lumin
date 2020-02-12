from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from fastprogress import master_bar, progress_bar
import pickle
import timeit
import numpy as np
import os
import sys
from random import shuffle
from collections import OrderedDict
import math
from functools import partial
import warnings

import torch.tensor as Tensor

from ..data.fold_yielder import FoldYielder
from ..data.batch_yielder import BatchYielder
from ..models.model_builder import ModelBuilder
from ..models.model import Model
from ..callbacks.cyclic_callbacks import AbsCyclicCallback
from ..callbacks.model_callbacks import AbsModelCallback
from ...utils.misc import to_tensor, to_device
from ...utils.statistics import uncert_round
from ..metrics.eval_metric import EvalMetric
from ...plotting.training import plot_train_history
from ...plotting.plot_settings import PlotSettings
from .metric_logger import MetricLogger

import matplotlib.pyplot as plt

__all__ = ['fold_train_ensemble']


def _get_folds(val_idx, n_folds, shuffle_folds:bool=True):
    r'''
    Return (shuffled) list of fold indeces which does not include the validation index
    '''

    folds = [x for x in range(n_folds) if x != val_idx]
    if shuffle_folds: shuffle(folds)
    return folds


def fold_train_ensemble(fy:FoldYielder, n_models:int, bs:int, model_builder:ModelBuilder,
                        callback_partials:Optional[List[partial]]=None, eval_metrics:Optional[Dict[str,EvalMetric]]=None,
                        train_on_weights:bool=True, eval_on_weights:bool=True, patience:int=10, max_epochs:int=200,
                        shuffle_fold:bool=True, shuffle_folds:bool=True, bulk_move:bool=True,
                        live_fdbk:bool=True, live_fdbk_first_only:bool=True, live_fdbk_extra:bool=True, live_fdbk_extra_first_only:bool=False,
                        savepath:Path=Path('train_weights'), verbose:bool=False, log_output:bool=False,
                        plot_settings:PlotSettings=PlotSettings(), plots:Optional[Any]=None) -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]],List[Dict[str,float]]]:
    r'''
    Main training method for :class:`~lumin.nn.models.model.Model`.
    Trains a specified numer of models created by a :class:`~lumin.nn.models.model_builder.ModelBuilder` on data provided by a
    :class:`~lumin.nn.data.fold_yielder.FoldYielder`, and save them to savepath.
    Note, this does not return trained models, instead they are saved and must be loaded later. Instead this method returns results of model training.
    Each :class:`~lumin.nn.models.model.Model` is trained on N-1 folds, for a :class:`~lumin.nn.data.fold_yielder.FoldYielder` with N folds, and the remaining
    fold is used as validation data.
    Training folds are loaded iteratively, and model evaluation takes place after each fold use (a sub-epoch), rather than after ever use of all folds (epoch).
    Training continues until:
        - All of the training folds are used max_epoch number of times;
        - Or validation loss does not decrease for patience number of training folds;
          (or cycles, if using an :class:`~lumin.nn.callbacks.cyclic_callbacks.AbsCyclicCallback`);
        - Or a callback triggers trainign to stop, e.g. :class:`~lumin.nn.callbacks.cyclic_callbacks.OneCycle`
        
    Once training is finished, the state with the lowest validation loss is loaded, evaluated, and saved.

    Arguments:
        fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing ot training data
        n_models: number of models to train
        bs: batch size. Number of data points per iteration
        model_builder: :class:`~lumin.nn.models.model_builder.ModelBuilder` creating the networks to train
        callback_partials: optional list of functools.partial, each of which will a instantiate :class:`~lumin.nn.callbacks.callback.Callback` when called
        eval_metrics: list of instantiated :class:`~lumin.nn.metric.eval_metric.EvalMetric`.
            At the end of training, validation data and model predictions will be passed to each, and the results printed and saved
        train_on_weights: If weights are present in training data, whether to pass them to the loss function during training
        eval_on_weights: If weights are present in validation data, whether to pass them to the loss function during validation
        patience: number of folds (sub-epochs) or cycles to train without decrease in validation loss before ending training (early stopping)
        max_epochs: maximum number of epochs for which to train
        live_fdbk: whether or not to show any live feedback at all during training (slightly slows down training, but helps spot problems)
        live_fdbk_first_only: whether to only show live feedback for the first model trained (trade off between time and problem spotting)
        live_fdbk_extra: whether to show extra information live feedback (further slows training)
        live_fdbk_extra_first_only: whether to only show extra live feedback information for the first model trained (trade off between time and information)
        shuffle_fold: whether to tell :class:`~lumin.nn.data.batch_yielder.BatchYielder` to shuffle data
        shuffle_folds: whether to shuffle the order of the trainign folds
        bulk_move: whether to pass all training data to device at once, or by minibatch. Bulk moving will be quicker, but may not fit in memory.
        savepath: path to to which to save model weights and results
        verbose: whether to print out extra information during training
        log_output: whether to save printed results to a log file rather than printing them
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
        plots: Depreciated: loss history will always be shown,
            lr history will no longer be shown separately,
            and live feedback is now controlled by `live_fdbk` argument

    Returns:
        - results list of validation losses and other eval_metrics results, ordered by model training. Can be used to create an :class:`~lumin.nn.ensemble.ensemble.Ensemble`.
        - histories list of loss histories, ordered by model training
        - cycle_losses if an :class:`~lumin.nn.callbacks.cyclic_callbacks.AbsCyclicCallback` was passed, list of validation losses at the end of each cycle, ordered by model training. Can be passed to :class:`~lumin.nn.ensemble.ensemble.Ensemble`.
    '''
    # TODO: fix returns paret of doc string

    os.makedirs(savepath, exist_ok=True)
    os.system(f"rm {savepath}/*.h5 {savepath}/*.json {savepath}/*.pkl {savepath}/*.png {savepath}/*.log")
    if callback_partials is None: callback_partials = []
    
    if log_output:
        old_stdout = sys.stdout
        log_file = open(savepath/'training_log.log', 'w')
        sys.stdout = log_file

    if plots is not None:
        warnings.warn("The plots argument is now depreciated and ignored. Loss history will always be shown, lr history will no longer be shown separately, \
                       and live feedback is now controlled by the four live_fdbk arguments. This argument will be removed in V0.6.")

    train_tmr = timeit.default_timer()
    results,histories,cycle_losses = [],[],[]
    nb = len(fy.foldfile['fold_0/targets'])//bs

    if live_fdbk:
        metric_log = MetricLogger(loss_names=['Train', 'Validation'], n_folds=fy.n_folds, extra_detail=live_fdbk_extra or live_fdbk_extra_first_only,
                                  plot_settings=plot_settings)
    
    model_bar = master_bar(range(n_models))
    for model_num in (model_bar):
        val_id = model_num % fy.n_folds
        print(f"Training model {model_num+1} / {n_models}, Val ID = {val_id}")
        if model_num == 1:
            if live_fdbk_first_only: live_fdbk = False  # Only show fdbk for first training
            elif live_fdbk_extra_first_only: metric_log.extra_detail = False
        if live_fdbk: metric_log.reset()
        model_tmr = timeit.default_timer()
        os.system(f"rm {savepath}/best.h5")
        best_loss,epoch_counter,subEpoch,stop = math.inf,0,0,False
        loss_history = OrderedDict({'trn_loss': [], 'val_loss': []})
        cycle_losses.append({})
        trn_ids = _get_folds(val_id, fy.n_folds, shuffle_folds)
        model = Model(model_builder)
        val_fold = fy.get_fold(val_id)
        if not eval_on_weights: val_fold['weights'] = None

        cyclic_callback,callbacks,loss_callbacks = None,[],[]
        for c in callback_partials: callbacks.append(c(model=model))
        for c in callbacks:
            if isinstance(c, AbsCyclicCallback):
                c.set_nb(nb)
                cyclic_callback = c
        for c in callbacks:
            if isinstance(c, AbsModelCallback):
                c.set_val_fold(val_fold)
                c.set_cyclic_callback(cyclic_callback)
                if getattr(c, "get_loss", None):
                    loss_callbacks.append(c)
                    if live_fdbk: metric_log.add_loss_name(type(c).__name__)
                    loss_history[f'{type(c).__name__}_val_loss'] = []
        for c in callbacks: c.on_train_begin(model_num=model_num, savepath=savepath)

        # Validation data
        if bulk_move:
            if fy.has_matrix and fy.yield_matrix: val_x = (to_device(Tensor(val_fold['inputs'][0]).float()), to_device(Tensor(val_fold['inputs'][1]).float())) 
            else:                                 val_x =  to_device(Tensor(val_fold['inputs']).float())
            val_y = to_device(Tensor(val_fold['targets'])) if bulk_move else Tensor(val_fold['targets'])
            if train_on_weights: val_w = to_device(to_tensor(val_fold['weights'])) if bulk_move else to_tensor(val_fold['weights'])
            else:                val_w = None
            if 'multiclass' in model_builder.objective: val_y = val_y.long().squeeze()
            else:                                       val_y = val_y.float()

        epoch_pb = progress_bar(range(max_epochs), leave=True)
        if live_fdbk: model_bar.show()
        for epoch in epoch_pb:
            for trn_id in trn_ids:
                subEpoch += 1
                batch_yielder = BatchYielder(**fy.get_fold(trn_id), objective=model_builder.objective,
                                             bs=bs, use_weights=train_on_weights, shuffle=shuffle_fold, bulk_move=bulk_move)
                loss_history['trn_loss'].append(model.fit(batch_yielder, callbacks))
                del batch_yielder

                if bulk_move:
                    val_loss = model.evaluate(val_x, val_y, weights=val_w, callbacks=callbacks)
                else:
                    batch_yielder = BatchYielder(**val_fold, objective=model_builder.objective,
                                                 bs=bs, use_weights=train_on_weights, shuffle=shuffle_fold, bulk_move=bulk_move)
                    val_loss = model.evaluate_from_by(batch_yielder, callbacks=callbacks)
                    del batch_yielder

                loss_history['val_loss'].append(val_loss)
                loss_callback_idx = None
                loss = val_loss
                for i, lc in enumerate(loss_callbacks):
                    l = lc.get_loss()
                    if l < loss: loss, loss_callback_idx = l, i
                    if verbose: print(f'{subEpoch} {type(lc).__name__} loss {l}, default loss {val_loss}')
                    l = loss if l is None or not lc.active else l
                    loss_history[f'{type(lc).__name__}_val_loss'].append(l)

                if cyclic_callback is not None and cyclic_callback.cycle_end:
                    if verbose: print(f"Saving snapshot {cyclic_callback.cycle_count}")
                    cycle_losses[-1][cyclic_callback.cycle_count] = val_loss
                    model.save(str(savepath/f"{model_num}_cycle_{cyclic_callback.cycle_count}.h5"))

                if loss <= best_loss:
                    best_loss = loss
                    epoch_pb.comment = f'Epoch {subEpoch}, best loss: {best_loss:.4E}'
                    if verbose: print(epoch_pb.comment)
                    epoch_counter = 0
                    if loss_callback_idx is not None: loss_callbacks[loss_callback_idx].test_model.save(savepath/"best.h5")
                    else: model.save(savepath/"best.h5")
                elif cyclic_callback is not None:
                    if cyclic_callback.cycle_end: epoch_counter += 1
                else:
                    epoch_counter += 1

                if live_fdbk: metric_log.update_vals([loss_history[l][-1] for l in loss_history])
                if epoch_counter >= patience or model.stop_train:  # Early stopping
                    print('Early stopping after {} epochs'.format(subEpoch))
                    stop = True; break
            if live_fdbk: metric_log.update_plot(best_loss)
            if stop: break

        model.load(savepath/"best.h5")
        model.save(savepath/f'train_{model_num}.h5')
        for c in callbacks: c.on_train_end(fy=fy, val_id=val_id, bs=bs if not bulk_move else None)

        histories.append({})
        histories[-1] = loss_history
        results.append({})
        results[-1]['loss'] = best_loss
        if eval_metrics is not None and len(eval_metrics) > 0:
            y_pred = model.predict(val_fold['inputs'], bs=bs if not bulk_move else None)
            for m in eval_metrics: results[-1][m] = eval_metrics[m].evaluate(fy, val_id, y_pred)
        print(f"Scores are: {results[-1]}")
        with open(savepath/'results_file.pkl', 'wb') as fout: pickle.dump(results, fout)
        with open(savepath/'cycle_file.pkl', 'wb') as fout: pickle.dump(cycle_losses, fout)
        
        plt.clf()
        print(f"Fold took {timeit.default_timer()-model_tmr:.3f}s\n")

    print("\n______________________________________")
    print("Training finished")
    print(f"Cross-validation took {timeit.default_timer()-train_tmr:.3f}s ")
    plot_train_history(histories, savepath/'loss_history', settings=plot_settings)
    for score in results[0]:
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print(f"Mean {score} = {mean[0]}±{mean[1]}")
    print("______________________________________\n")
    if log_output:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories, cycle_losses
    