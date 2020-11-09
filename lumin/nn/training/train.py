from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import IN_NOTEBOOK
import pickle
import timeit
import numpy as np
import os
from functools import partial
from fastcore.all import is_listy

from ..data.fold_yielder import FoldYielder
from ..models.model_builder import ModelBuilder
from ..models.model import Model
from ..callbacks.pred_handlers import PredHandler
from ..callbacks.monitors import EarlyStopping, SaveBest, MetricLogger
from ...utils.statistics import uncert_round
from ..metrics.eval_metric import EvalMetric
from ...plotting.training import plot_train_history
from ...plotting.plot_settings import PlotSettings

import matplotlib.pyplot as plt

__all__ = ['train_models']


def train_models(fy:FoldYielder, n_models:int, bs:int, model_builder:ModelBuilder, n_epochs:int, patience:Optional[int]=None, loss_is_meaned:bool=True,
                 cb_partials:Optional[List[partial]]=None, eval_metrics:Optional[Dict[str,EvalMetric]]=None, pred_cb:Callable[[],PredHandler]=PredHandler,
                 train_on_weights:bool=True, eval_on_weights:bool=True,
                 bulk_move:bool=True,
                 live_fdbk:bool=True, live_fdbk_first_only:bool=True, live_fdbk_extra:bool=True, live_fdbk_extra_first_only:bool=False,
                 savepath:Path=Path('train_weights'), 
                 plot_settings:PlotSettings=PlotSettings()) -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]],List[Dict[str,float]]]:
    r'''
    '''

    results,histories,cycle_losses,savepath = [],[],[],Path(savepath)
    if cb_partials is None: cb_partials = []
    if not is_listy(cb_partials): cb_partials = [cb_partials]
    if patience is not None: cb_partials.append(partial(EarlyStopping, patience=patience, loss_is_meaned=loss_is_meaned))

    model_bar = master_bar(range(n_models)) if IN_NOTEBOOK else progress_bar(range(n_models))
    train_tmr = timeit.default_timer()
    for model_num in (model_bar):    
        if IN_NOTEBOOK: model_bar.show()
        val_idx = model_num % fy.n_folds
        print(f"Training model {model_num+1} / {n_models}, Val ID = {val_idx}")
        if model_num == 1 and live_fdbk_first_only: live_fdbk_extra = False  # Only show fdbk for first training

        model_dir = savepath/f'model_id_{model_num}'
        model_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"rm {model_dir}/*.h5 {model_dir}/*.json {model_dir}/*.pkl {model_dir}/*.png")
        model = Model(model_builder)
        cbs = []
        for c in cb_partials: cbs.append(c(model=model))
        save_best = SaveBest(auto_reload=True, loss_is_meaned=loss_is_meaned)
        metric_log = MetricLogger(extra_detail=live_fdbk_extra, plot_settings=plot_settings)
        cbs += [save_best,metric_log]

        model_tmr = timeit.default_timer()
        model.fit(n_epochs=n_epochs, fy=fy, bs=bs, bulk_move=bulk_move, train_on_weights=train_on_weights, val_idx=val_idx, cbs=cbs, cb_savepath=model_dir)
        print(f"Model took {timeit.default_timer()-model_tmr:.3f}s\n")
        model.save(model_dir/f'train_{model_num}.h5')

        histories.append(metric_log.get_loss_history())
        cycle_losses.append([])
        for c in cbs:
            if hasattr(c, 'cycle_save') and c.cycle_save: cycle_losses[-1] = c.cycle_losses
        results.append({})
        results[-1]['loss'] = save_best.min_loss
        if eval_metrics is not None and len(eval_metrics) > 0:
            y_pred = model.predict(fy[val_idx]['inputs'], bs=bs if not bulk_move else None)
            for m in eval_metrics: results[-1][m] = eval_metrics[m].evaluate(fy, val_idx, y_pred)
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
    return results, histories, cycle_losses
