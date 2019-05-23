import numpy as np
import pandas as pd
import os
import pickle
import glob
import warnings
from fastprogress import progress_bar, master_bar
from pathlib import Path
import timeit
from typing import Dict, Union, Any, List, Optional
from sklearn.pipeline import Pipeline

from torch.tensor import Tensor

from .abs_ensemble import AbsEnsemble
from ..models.model import Model
from ..models.model_builder import ModelBuilder
from ..data.fold_yielder import FoldYielder
from ..interpretation.features import get_ensemble_feat_importance
from ..metrics.eval_metric import EvalMetric
from ...utils.statistics import uncert_round


class Ensemble(AbsEnsemble):
    '''Standard class for building an ensemble object'''
    def __init__(self, input_pipe:Optional[Pipeline]=None, output_pipe:Optional[Pipeline]=None, model_builder:Optional[ModelBuilder]=None):
        super().__init__()
        self.input_pipe,self.output_pipe,self.model_builder = input_pipe,output_pipe,model_builder
        
    def add_input_pipe(self, pipe:Pipeline) -> None: self.input_pipe = pipe

    def add_output_pipe(self, pipe:Pipeline) -> None: self.output_pipe = pipe
    
    @staticmethod
    def load_trained_model(model_idx:int, model_builder:ModelBuilder, name:str='train_weights/train_') -> Model: 
        model = Model(model_builder)
        model.load(f'{name}{model_idx}.h5')
        return model
    
    @staticmethod
    def _get_weights(value:float, metric:str, weighting='reciprocal') -> float:
        if   weighting == 'reciprocal': return 1/value
        elif weighting == 'uniform':    return 1
        else: raise ValueError("No other weighting currently supported")

    @classmethod
    def from_save(cls, name:str) -> AbsEnsemble:
        ensemble = cls()
        ensemble.load(name)
        return ensemble

    @classmethod
    def from_results(cls,  results:List[Dict[str,float]], size:int, model_builder:ModelBuilder,
                     metric:str='loss', weighting:str='reciprocal', higher_metric_better:bool=False, snapshot_args:Optional[Dict[str,Any]]=None,
                     location:Path=Path('train_weights'), verbose:bool=True) -> AbsEnsemble:
        ensemble = cls()
        ensemble.build_ensemble(results=results, size=size, model_builder=model_builder,
                                metric=metric, weighting=weighting, higher_metric_better=higher_metric_better, snapshot_args=snapshot_args,
                                location=location, verbose=verbose)
        return ensemble
                
    def build_ensemble(self, results:List[Dict[str,float]], size:int, model_builder:ModelBuilder,
                       metric:str='loss', weighting:str='reciprocal', higher_metric_better:bool=False, snapshot_args:Optional[Dict[str,Any]]=None,
                       location:Path=Path('train_weights'), verbose:bool=True) -> None:
        self.model_builder = model_builder
        cycle_losses     = None if snapshot_args is None or 'cycle_losses'     not in snapshot_args else snapshot_args['cycle_losses']
        n_cycles         = None if snapshot_args is None or 'n_cycles'         not in snapshot_args else snapshot_args['n_cycles']
        load_cycles_only = None if snapshot_args is None or 'load_cycles_only' not in snapshot_args else snapshot_args['load_cycles_only']
        patience         = 2    if snapshot_args is None or 'patience'         not in snapshot_args else snapshot_args['patience']
        weighting_pwr    = 0    if snapshot_args is None or 'weighting_pwr'    not in snapshot_args else snapshot_args['weighting_pwr']    
    
        if (cycle_losses is not None and n_cycles is None) or (cycle_losses is None and n_cycles is not None):
            warnings.warn("Warning: cycle ensembles requested, but not enough information passed")
        if cycle_losses is not None and n_cycles is not None and metric != 'loss':
            warnings.warn("Warning: Setting ensemble metric to loss")
            metric = 'loss'
        if cycle_losses is not None and n_cycles is not None and weighting != 'uniform':
            warnings.warn("Warning: Setting model weighting to uniform")
            weighting = 'uniform'
    
        if verbose: print(f"Choosing ensemble by {metric}")
        values = np.sort(np.array([(i, result[metric] if not higher_metric_better else 1/result[metric]) for i, result in enumerate(results)],
                                  dtype=[('model', int), ('result', float)]), order=['result'])

        self.models, weights = [], []
        for i in progress_bar(range(min([size, len(results)]))):
            if not (load_cycles_only and n_cycles):
                self.models.append(self.load_trained_model(values[i]['model'], self.model_builder, name=location/'train_'))
                weights.append(self._get_weights(values[i]['result'], metric, weighting))
                if verbose:
                    print(f"Model {i} is {values[i]['model']} with {metric} = {values[i]['result'] if not higher_metric_better else 1/values[i]['result']}")

            if n_cycles:
                end_cycle = len(cycle_losses[values[i]['model']])-patience-1
                if load_cycles_only: end_cycle += 1
                for n, c in enumerate(range(end_cycle, max(0, end_cycle-n_cycles), -1)):
                    self.models.append(self.load_trained_model(c, self.model_builder, name=location/f'{values[i]["model"]}_cycle_'))
                    weights.append((n+1 if load_cycles_only else n+2)**weighting_pwr)
                    if verbose: print(f"Model {i} cycle {c} has {metric} = {cycle_losses[values[i]['model']][c]} and weight {weights[-1]}")
        
        weights = np.array(weights)
        self.weights = weights/weights.sum()
        self.size = len(self.models)
        self.n_out = self.models[0].get_out_size()
        self.results = results
        
    def predict_array(self, arr:np.ndarray, n_models:Optional[int]=None, parent_bar:Optional[master_bar]=None) -> np.ndarray:
        pred = np.zeros((len(arr), self.n_out))
        
        n_models = len(self.models) if n_models is None else n_models
        models = self.models[:n_models]
        weights = self.weights[:n_models]
        weights = weights/weights.sum()
        
        arr = Tensor(arr)
        for i, m in enumerate(progress_bar(models, parent=parent_bar, display=bool(parent_bar))):
            tmp_pred = m.predict(arr)
            if self.output_pipe is not None: tmp_pred = self.output_pipe.inverse_transform(tmp_pred)
            pred += weights[i]*tmp_pred
        return pred
    
    def predict_folds(self, fy:FoldYielder, n_models:Optional[int]=None, pred_name:str='pred') -> None:
        n_models = len(self.models) if n_models is None else n_models
        times = []
        mb = master_bar(range(len(fy.foldfile)))
        for fold_idx in mb:
            fold_tmr = timeit.default_timer()
            if not fy.test_time_aug:
                fold = fy.get_fold(fold_idx)['inputs']
                pred = self.predict_array(fold, n_models, mb)
            else:
                tmpPred = []
                pb = progress_bar(range(fy.aug_mult), parent=mb)
                for aug in pb:
                    fold = fy.get_test_fold(fold_idx, aug)['inputs']
                    tmpPred.append(self.predict_array(fold, n_models))
                pred = np.mean(tmpPred, axis=0)

            times.append((timeit.default_timer()-fold_tmr)/len(fold))
            if self.n_out > 1: fy.save_fold_pred(pred, fold_idx, pred_name=pred_name)
            else: fy.save_fold_pred(pred[:, 0], fold_idx, pred_name=pred_name)
        times = uncert_round(np.mean(times), np.std(times, ddof=1)/np.sqrt(len(times)))
        print(f'Mean time per event = {times[0]}±{times[1]}')

    def predict(self, inputs:Union[np.ndarray,FoldYielder,List[np.ndarray]], n_models:Optional[int]=None, pred_name:str='pred') -> Union[None,np.ndarray]:
        if not isinstance(inputs, FoldYielder): return self.predict_array(inputs, n_models)
        self.predict_folds(inputs, n_models, pred_name)
    
    def save(self, name:str, feats:Any=None, overwrite:bool=False) -> None:
        if (len(glob.glob(f"{name}*.json")) or len(glob.glob(f"{name}*.h5")) or len(glob.glob(f"{name}*.pkl"))) and not overwrite:
            raise FileExistsError("Ensemble already exists with that name, call with overwrite=True to force save")
        else:
            os.makedirs(name[:name.rfind('/')], exist_ok=True)
            os.system(f"rm {name}*.json {name}*.h5 {name}*.pkl")
            for i, model in enumerate(progress_bar(self.models)): model.save(f'{name}_{i}.h5')    
            with open(f'{name}_weights.pkl', 'wb')         as fout: pickle.dump(self.weights, fout)
            with open(f'{name}_results.pkl', 'wb')         as fout: pickle.dump(self.results, fout)
            with open(f'{name}_builder.pkl', 'wb')         as fout: pickle.dump(self.model_builder, fout)
            if self.input_pipe  is not None: 
                with open(f'{name}_input_pipe.pkl', 'wb')  as fout: pickle.dump(self.input_pipe, fout)
            if self.output_pipe is not None: 
                with open(f'{name}_output_pipe.pkl', 'wb') as fout: pickle.dump(self.output_pipe, fout)
            if feats            is not None: 
                with open(f'{name}_feats.pkl', 'wb')       as fout: pickle.dump(feats, fout)
                    
    def load(self, name:str) -> None:
        with open(f'{name}_builder.pkl', 'rb') as fin: self.model_builder = pickle.load(fin)
        names = glob.glob(f'{name}_*.h5')
        self.models = []
        for n in progress_bar(sorted(names)):
            m = Model(self.model_builder)
            m.load(n)
            self.models.append(m)
        self.size = len(self.models)
        self.n_out = self.models[0].get_out_size()
        with     open(f'{name}_weights.pkl', 'rb')     as fin: self.weights     = pickle.load(fin)
        try: 
            with open(f'{name}_input_pipe.pkl', 'rb')  as fin: self.input_pipe  = pickle.load(fin)
        except FileNotFoundError: pass
        try: 
            with open(f'{name}_output_pipe.pkl', 'rb') as fin: self.output_pipe = pickle.load(fin)
        except FileNotFoundError: pass
        try: 
            with open(f'{name}_feats.pkl', 'rb')       as fin: self.feats       = pickle.load(fin)
        except FileNotFoundError: pass

    def export2onnx(self, base_name:str, bs:int=1) -> None:
        for i, m in enumerate(self.models): m.export2onnx(f'{base_name}_{i}', bs)

    def export2tfpb(self, base_name:str, bs:int=1) -> None:
        for i, m in enumerate(self.models): m.export2tfpb(f'{base_name}_{i}', bs)

    def get_feat_importance(self, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        return get_ensemble_feat_importance(self, fy, eval_metric)
