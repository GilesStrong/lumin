from typing import List, Tuple
from fastprogress import progress_bar
import timeit

from ..nn.data.fold_yielder import FoldYielder
from ..nn.data.batch_yielder import BatchYielder
from ..nn.models.model_builder import ModelBuilder
from ..nn.models.model import Model
from ..nn.callbacks.opt_callbacks import LRFinder
from ..plotting.training import plot_lr_finders
from ..plotting.plot_settings import PlotSettings


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
