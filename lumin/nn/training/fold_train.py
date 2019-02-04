from typing import Dict, Any, List, Tuple
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

import torch.tensor as Tensor

from ..data.fold_yielder import FoldYielder
from ..data.batch_yielder import BatchYielder
from ..models.model_builder import ModelBuilder
from ..models.model import Model
from ..callbacks.cyclic_callbacks import AbsCyclicCallback
from ..callbacks.model_callbacks import AbsModelCallback
from ...utils.misc import to_tensor
from ...utils.statistics import uncert_round
from ..metrics.eval_metric import EvalMetric
from ...plotting.training import plot_train_history
from ...plotting.plot_settings import PlotSettings

import matplotlib.pyplot as plt


def get_folds(n, n_splits, shuffle_folds:bool=True,):
    train = [x for x in range(n_splits) if x != n]
    if shuffle_folds: shuffle(train)
    return train


def fold_train_ensemble(fold_yielder:FoldYielder, n_models:int, bs:int, model_builder:ModelBuilder,
                        use_callbacks:Dict[str,Dict[str,Any]]={}, eval_metrics:Dict[str,EvalMetric]={},
                        train_on_weights:bool=True, eval_on_weights:bool=True, patience:int=10, max_epochs:int=200,
                        plots:List[str]=['history'], shuffle_fold:bool=True, shuffle_folds:bool=True,
                        saveloc:Path=Path('train_weights'), verbose:bool=False, log_output:bool=False,
                        plot_settings:PlotSettings=PlotSettings()) -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]],List[Dict[str,float]]]:
    
    os.makedirs(saveloc, exist_ok=True)
    os.system(f"rm {saveloc}/*.h5 {saveloc}/*.json {saveloc}/*.pkl {saveloc}/*.png {saveloc}/*.log")
    
    if log_output:
        old_stdout = sys.stdout
        log_file = open(saveloc/'training_log.log', 'w')
        sys.stdout = log_file

    start = timeit.default_timer()
    results = []
    histories = []
    cycle_losses = []
    n_folds = fold_yielder.n_folds
    nb = len(fold_yielder.source['fold_0/targets'])//bs

    model_bar = master_bar(np.random.choice(range(n_folds), size=n_models, replace=False))
    model_bar.names = ['Best', 'Train', 'Validation']
    for model_num, val_id in enumerate(model_bar):
        print(f"Training model {model_num+1} / {n_models}")
        model_start = timeit.default_timer()
        os.system(f"rm {saveloc}/best.h5")
        best_loss = math.inf
        epoch_counter = 0
        subEpoch = 0
        stop = False
        loss_history = OrderedDict({'trn_loss': [], 'val_loss': []})
        cycle_losses.append({})
        trn_ids = get_folds(val_id, n_folds, shuffle_folds)
        model = Model(model_builder)
        val_fold = fold_yielder.get_fold(val_id)
        if not eval_on_weights: val_fold['weights'] = None

        cyclic_callback = None
        callbacks = []
        loss_callbacks = []
        for c in use_callbacks: callbacks.append(c['callback'](**c['kargs']))
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
                    model_bar.names.append(type(c).__name__)
                    loss_history[f'{type(c).__name__}_val_loss'] = []
        for c in callbacks:
            c.set_model(model)
            c.on_train_begin()

        model_bar.update_graph([[0, 0] for i in range(len(model_bar.names))])
        epoch_pb = progress_bar(range(max_epochs))
        for epoch in epoch_pb:
            for trn_id in trn_ids:
                subEpoch += 1
                batch_yielder = BatchYielder(**fold_yielder.get_fold(trn_id), objective=model_builder.objective, bs=bs, use_weights=train_on_weights, shuffle=shuffle_fold)
                loss_history['trn_loss'].append(model.fit(batch_yielder, callbacks))

                val_loss = model.evaluate(Tensor(val_fold['inputs']), Tensor(val_fold['targets']), weights=to_tensor(val_fold['weights']))
                loss_history['val_loss'].append(val_loss)
                loss_callback_idx = None
                loss = val_loss
                for i, lc in enumerate(loss_callbacks):
                    l = lc.get_loss()
                    if l < loss:
                        loss = l
                        loss_callback_idx = i
                    if verbose: print(f'{subEpoch} {type(lc).__name__} loss {l}, default loss {val_loss}')
                    l = loss if l is None or not lc.active else l
                    loss_history[f'{type(lc).__name__}_val_loss'].append(l)

                if cyclic_callback is not None and cyclic_callback.cycle_end:
                    if verbose: print(f"Saving snapshot {cyclic_callback.cycle_count}")
                    cycle_losses[-1][cyclic_callback.cycle_count] = val_loss
                    model.save(str(saveloc/f"{model_num}_cycle_{cyclic_callback.cycle_count}.h5"))

                if loss <= best_loss:
                    best_loss = loss
                    epoch_pb.comment = f'Epoch {subEpoch}, best loss: {best_loss:.4E}'
                    if verbose: print(epoch_pb.comment)
                    epoch_counter = 0
                    if loss_callback_idx is not None: loss_callbacks[loss_callback_idx].test_model.save(saveloc/"best.h5")
                    else: model.save(saveloc/"best.h5")
                elif cyclic_callback is not None:
                    if cyclic_callback.cycle_end:
                        epoch_counter += 1
                else:
                    epoch_counter += 1

                x = np.arange(len(loss_history['val_loss']))
                model_bar.update_graph([[x, best_loss*np.ones_like(x)]] + [[x, loss_history[l]] for l in loss_history])

                if epoch_counter >= patience or model.stop_train:  # Early stopping
                    print('Early stopping after {} epochs'.format(subEpoch))
                    stop = True
                    break
            if stop: break
        for c in callbacks: c.on_train_end

        model.load(saveloc/"best.h5")
        model.save(saveloc/f'train_{model_num}.h5')
        histories.append({})
        histories[-1] = loss_history
        results.append({})
        results[-1]['loss'] = best_loss
        if len(eval_metrics) > 0:
            y_pred = model.predict(Tensor(val_fold['inputs']))
            for m in eval_metrics: results[-1][m] = eval_metrics[m].evaluate(fold_yielder, val_id, y_pred)
        print(f"Scores are: {results[-1]}")
        with open(saveloc/'results_file.pkl', 'wb') as fout: pickle.dump(results, fout)
        with open(saveloc/'cycle_file.pkl', 'wb') as fout: pickle.dump(cycle_losses, fout)

        delattr(model_bar, 'fig')
        plt.clf()
        if 'cycle' in plots and cyclic_callback is not None: cyclic_callback.plot()
        print(f"Fold took {timeit.default_timer()-model_start:.3f}s\n")

    print("\n______________________________________")
    print("Training finished")
    print(f"Cross-validation took {timeit.default_timer()-start:.3f}s ")
    if 'history' in plots: plot_train_history(histories, saveloc/'loss_history.png', settings=plot_settings)
    for score in results[0]:
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print(f"Mean {score} = {mean[0]}±{mean[1]}")
    print("______________________________________\n")                
    if log_output:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories, cycle_losses