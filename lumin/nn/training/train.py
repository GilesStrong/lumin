from typing import Dict, List, Tuple, Optional
from pathlib import Path
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import IN_NOTEBOOK
import pickle
import timeit
import numpy as np
import os
import sys
from random import shuffle
from collections import OrderedDict
import math
from functools import partial
from fastcore.all import is_listy

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

__all__ = ['train_models']


def train_models(fy:FoldYielder, n_models:int, bs:int, model_builder:ModelBuilder, n_epochs:int, patience:Optional[int]=None, 
                 cb_partials:Optional[List[partial]]=None, eval_metrics:Optional[Dict[str,EvalMetric]]=None,
                 train_on_weights:bool=True, eval_on_weights:bool=True,
                 bulk_move:bool=True,
                 live_fdbk:bool=True, live_fdbk_first_only:bool=True, live_fdbk_extra:bool=True, live_fdbk_extra_first_only:bool=False,
                 savepath:Path=Path('train_weights'), opt:Optional[Callable[[Generator],optim.Optimizer]]=None,
                 loss:Optional[Callable[[],Callable[[Tensor,Tensor],Tensor]]]=None
                 plot_settings:PlotSettings=PlotSettings()) -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]],List[Dict[str,float]]]:
    r'''
    '''
    #     cbs = []
    #     for c in cb_partials: cbs.append(c(model=model))
    #     lrf = LRFinder(lr_bounds=lr_bounds, nb=nb, model=model)
    #     model.fit(n_epochs=n_epochs, fy=fy, bs=bs, bulk_move=bulk_move, train_on_weights=train_on_weights, trn_idxs=[trn_idx], cbs=cbs+[lrf], opt=opt,
    #               loss=loss)
    #     if nb is None: nb = lrf.nb  # Ensure all LR FInders follow same LR history
    #     lr_finders.append(lrf)
    # del model

    os.makedirs(savepath, exist_ok=True)
    os.system(f"rm {savepath}/*.h5 {savepath}/*.json {savepath}/*.pkl {savepath}/*.png")

    if cb_partials is None: cb_partials = []
    if not is_listy(cb_partials): cb_partials = [cb_partials]
    results,histories,cycle_losses = [],[],[]

    # if not IN_NOTEBOOK: live_fdbk = False
    # if live_fdbk:
    #     metric_log = MetricLogger(loss_names=['Train', 'Validation'], n_folds=fy.n_folds, extra_detail=live_fdbk_extra or live_fdbk_extra_first_only,
    #                               plot_settings=plot_settings)

    model_bar = master_bar(range(n_models)) if IN_NOTEBOOK else progress_bar(range(n_models))
    train_tmr = timeit.default_timer()
    for model_num in (model_bar):    
        if IN_NOTEBOOK: model_bar.show()
        val_idx = model_num % fy.n_folds
        print(f"Training model {model_num+1} / {n_models}, Val ID = {val_idx}")

        # if model_num == 1:
        #     if live_fdbk_first_only: live_fdbk = False  # Only show fdbk for first training
        #     elif live_fdbk_extra_first_only: metric_log.extra_detail = False
        # if live_fdbk: metric_log.reset()
        # if live_fdbk: metric_log.add_loss_name(type(c).__name__)
        # loss_history[f'{type(c).__name__}_val_loss'] = []
        # if live_fdbk: model_bar.show()
        #  if live_fdbk: metric_log.update_vals([loss_history[l][-1] for l in loss_history])
        # if live_fdbk: metric_log.update_plot(best_loss)

        model = Model(model_builder)
        cbs = []
        for c in cb_partials: cbs.append(c(model=model))

        model_tmr = timeit.default_timer()
        model.fit(n_epochs=n_epochs, fy=fy, bs=bs, bulk_move=bulk_move, train_on_weights=train_on_weights, val_idx=val_idx, cbs=cbs, cb_savepath=savepath,
                  opt=opt, loss=loss)
        print(f"Model took {timeit.default_timer()-model_tmr:.3f}s\n")

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

    print("\n______________________________________")
    print("Training finished")
    print(f"Cross-validation took {timeit.default_timer()-train_tmr:.3f}s ")
    plot_train_history(histories, savepath/'loss_history', settings=plot_settings, show=IN_NOTEBOOK)
    for score in results[0]:
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print(f"Mean {score} = {mean[0]}±{mean[1]}")
    print("______________________________________\n")
    if log_output:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories, cycle_losses
