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
    r'''
    Standard class for building an ensemble of collection of trained networks producedd by :meth:fold_train_ensemble
    Input and output pipelines can be added. to provide easy saving and loaded of exported ensembles.
    Currently, the input pipeline is not used, so input data is expected to be preprocessed.
    However the output pipeline will be used to deprocess model predictions.


    Once instanciated, :meth:build_ensemble or :meth:load should be called. Alternatively, class_methods :meth:from_save or :meth:from_results may be used.

    Arguments:
        input_pipe: Optional input pipeline, alternatively call :meth:add_input_pipe
        output_pipe: Optional output pipeline, alternatively call :meth:add_input_pipe
        model_builder: Optional :class:ModelBuilder for constructing models from saved weights.

    Examples::
        >>> ensemble = Ensemble()
        >>> ensemble = Ensemble(input_pipe, output_pipe, model_builder)
    '''

    # TODO: check whether model_builder is necessary here
    # TODO: Standardise pipeline treatment: currently inputs not processed, but outputs are

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
        r'''
        Instantiate :class:Ensemble from a saved :class:Ensemble

        Arguments:
            name: base filename of ensemble

        Returns:
            Loaded :class:Ensemble

        Examples::
            >>> ensemble = Ensemble.from_save('weights/ensemble')
        '''

        ensemble = cls()
        ensemble.load(name)
        return ensemble

    @classmethod
    def from_results(cls,  results:List[Dict[str,float]], size:int, model_builder:ModelBuilder,
                     metric:str='loss', weighting:str='reciprocal', higher_metric_better:bool=False, snapshot_args:Optional[Dict[str,Any]]=None,
                     location:Path=Path('train_weights'), verbose:bool=True) -> AbsEnsemble:
        r'''
        Instantiate :class:Ensemble from a outputs of :meth:fold_train_ensemble.
        If cycle models are loaded, then only uniform weighting between models is supported.

        Arguments:
            results: results saved/returned by :meth:fold_train_ensemble
            size: number of models to load as ranked by metric
            model_builder: :class:ModelBuilder used for building :class:Model from saved models
            metric: metric name listed in results to use for ranking and weighting trained models
            weighting: 'reciprocal' or 'uniform' how to weight model predictions during predicition.
                'reciprocal' = models weighted by 1/metric
                'uniform' = models treated with equal weighting
            higher_metric_better: whether metric should be maximised or minimised
            snapshot_args: Dictionary potentially containing:
                'cycle_losses': returned/save by :meth:fold_train_ensemble when using an :class:AbsCyclicCallback
                'patience': patience value that was passed to :meth:fold_train_ensemble
                'n_cycles': number of cycles to load per model
                'load_cycles_only': whether to only load cycles, or also the best performing model
                'weighting_pwr': weight cycles according to (n+1)**weighting_pwr, where n is the number of cycles loaded so far.
                    Models are loaded youngest to oldest
            location: Path to save location passed to :meth:fold_train_ensemble
            verbose: whether to print out information of models loaded

        Returns:
            Built :class:Ensemble

        Examples::
            >>> ensemble = Ensemble.from_results(results, 10, model_builder, location=Path('train_weights'))
            >>> ensemble = Ensemble.from_results(results, 1,  model_builder, location=Path('train_weights'),
                                                 snapshot_args={'cycle_losses':cycle_losses,
                                                                'patience':patience,
                                                                'n_cycles':8,
                                                                'load_cycles_only':True,
                                                                'weighting_pwr':0})
        '''

        ensemble = cls()
        ensemble.build_ensemble(results=results, size=size, model_builder=model_builder,
                                metric=metric, weighting=weighting, higher_metric_better=higher_metric_better, snapshot_args=snapshot_args,
                                location=location, verbose=verbose)
        return ensemble
                
    def build_ensemble(self, results:List[Dict[str,float]], size:int, model_builder:ModelBuilder,
                       metric:str='loss', weighting:str='reciprocal', higher_metric_better:bool=False, snapshot_args:Optional[Dict[str,Any]]=None,
                       location:Path=Path('train_weights'), verbose:bool=True) -> None:
        r'''
        Load up an instantiated :class:Ensemble with outputs of :meth:fold_train_ensemble

        Arguments:
            results: results saved/returned by :meth:fold_train_ensemble
            size: number of models to load as ranked by metric
            model_builder: :class:ModelBuilder used for building :class:Model from saved models
            metric: metric name listed in results to use for ranking and weighting trained models
            weighting: 'reciprocal' or 'uniform' how to weight model predictions during predicition.
                'reciprocal' = models weighted by 1/metric
                'uniform' = models treated with equal weighting
            higher_metric_better: whether metric should be maximised or minimised
            snapshot_args: Dictionary potentially containing:
                'cycle_losses': returned/save by :meth:fold_train_ensemble when using an :class:AbsCyclicCallback
                'patience': patience value that was passed to :meth:fold_train_ensemble
                'n_cycles': number of cycles to load per model
                'load_cycles_only': whether to only load cycles, or also the best performing model
                'weighting_pwr': weight cycles according to (n+1)**weighting_pwr, where n is the number of cycles loaded so far.
                    Models are loaded youngest to oldest
            location: Path to save location passed to :meth:fold_train_ensemble
            verbose: whether to print out information of models loaded

        Examples::
            >>> ensemble.build_ensemble(results, 10, model_builder, location=Path('train_weights'))
            >>> ensemble.build_ensemble(results, 1,  model_builder, location=Path('train_weights'),
                                        snapshot_args={'cycle_losses':cycle_losses,
                                                       'patience':patience,
                                                       'n_cycles':8,
                                                       'load_cycles_only':True,
                                                       'weighting_pwr':0})
        '''

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
        
    def predict_array(self, arr:np.ndarray, n_models:Optional[int]=None, parent_bar:Optional[master_bar]=None, display:bool=True) -> np.ndarray:
        r'''
        Apply ensemble to Numpy array and get predictions. If an output pipe has been added to the ensemble, then the predictions will be deprocessed.
        Inputs are expected to be preprocessed; i.e. any input pipe added to the ensemble is not used.

        Arguments:
            arr: input data
            n_models: number of models to use in predictions as ranked by the metric which was used when constructing the :class:Ensemble.
                By default, entire ensemble is used.
            parent_bar: not used when calling the method directly
            display: whether to display a progress bar for model evaluations

        Returns:
            Numpy array of predictions

        Examples::
            >>> preds = ensemble.predict_array(inputs)
        '''

        pred = np.zeros((len(arr), self.n_out))
        n_models = len(self.models) if n_models is None else n_models
        models = self.models[:n_models]
        weights = self.weights[:n_models]
        weights = weights/weights.sum()
        
        arr = Tensor(arr)
        for i, m in enumerate(progress_bar(models, parent=parent_bar, display=display)):
            tmp_pred = m.predict(arr)
            if self.output_pipe is not None: tmp_pred = self.output_pipe.inverse_transform(tmp_pred)
            pred += weights[i]*tmp_pred
        return pred
    
    def predict_folds(self, fy:FoldYielder, n_models:Optional[int]=None, pred_name:str='pred') -> None:
        r'''
        Apply ensemble to data accessed by a :class:FoldYielder and save predictions as a new group per fold in the foldfile.
        If an output pipe has been added to the ensemble, then the predictions will be deprocessed.
        Inputs are expected to be preprocessed; i.e. any input pipe added to the ensemble is not used.
        If foldyielder has test-time augmentation, then predictions will be averaged over all augmentated forms of the data.

        Arguments:
            fy: :class:FoldYielder interfacing with the input data
            n_models: number of models to use in predictions as ranked by the metric which was used when constructing the :class:Ensemble.
                By default, entire ensemble is used.
            pred_name: name for new group of predictions

        Examples::
            >>> ensemble.predict_array(test_fy, pred_name='pred_tta')
        '''

        n_models = len(self.models) if n_models is None else n_models
        times = []
        mb = master_bar(range(len(fy.foldfile)))
        for fold_idx in mb:
            fold_tmr = timeit.default_timer()
            if not fy.test_time_aug:
                fold = fy.get_fold(fold_idx)['inputs']
                pred = self.predict_array(fold, n_models, mb, display=True)
            else:
                tmpPred = []
                pb = progress_bar(range(fy.aug_mult), parent=mb)
                for aug in pb:
                    fold = fy.get_test_fold(fold_idx, aug)['inputs']
                    tmpPred.append(self.predict_array(fold, n_models, display=False))
                pred = np.mean(tmpPred, axis=0)

            times.append((timeit.default_timer()-fold_tmr)/len(fold))
            if self.n_out > 1: fy.save_fold_pred(pred, fold_idx, pred_name=pred_name)
            else: fy.save_fold_pred(pred[:, 0], fold_idx, pred_name=pred_name)
        times = uncert_round(np.mean(times), np.std(times, ddof=1)/np.sqrt(len(times)))
        print(f'Mean time per event = {times[0]}±{times[1]}')

    def predict(self, inputs:Union[np.ndarray,FoldYielder,List[np.ndarray]], n_models:Optional[int]=None, pred_name:str='pred') -> Union[None,np.ndarray]:
        r'''
        Compatability method for predicting data contained in either a Numpy array or a :class:FoldYielder
        Will either pass inputs to :meth:predict_array or :meth:predict_folds.

        Arguments:
            inputs: either a :class:FoldYielder interfacing with the input data, or the input data as an array
            n_models: number of models to use in predictions as ranked by the metric which was used when constructing the :class:Ensemble.
                By default, entire ensemble is used.
            pred_name: name for new group of predictions if passed a :class:FoldYielder

        Returns:
            If passed a Numpy array will return predictions.

        Examples::
            >>> preds = ensemble.predict(input_array)
            >>> ensemble.predict(test_fy)
        '''
        
        if not isinstance(inputs, FoldYielder): return self.predict_array(inputs, n_models, display=True)
        self.predict_folds(inputs, n_models, pred_name)
    
    def save(self, name:str, feats:Optional[Any]=None, overwrite:bool=False) -> None:
        r'''
        Save ensemble and associated objects

        Arguments:
            name: base name for saved objects
            feats: optional list of input features
            overwrite: if existing objects are found, whether to overwrite them
        
        Examples::
            >>> ensemble.save('weights/ensemble')
            >>> ensemble.save('weights/ensemble', ['pt','eta','phi'])
        '''

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
        r'''
        Load an instantiated :class:Ensemble with weights and :class:Model from save.

        Arguments;
            name: base name for saved objects

        Examples::
            >>> ensemble.load('weights/ensemble') 
        '''

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
        r'''
        Export all :class:Model contained in :class:Ensemble to ONNX format.
        Note that ONNX expects a fixed batch size (bs) which is the number of datapoints your wish to pass through the model concurrently.

        Arguments:
            base_name: Exported models will be called {base_name}_{model_num}.onnx
            bs: batch size for exported models
        '''
        
        for i, m in enumerate(self.models): m.export2onnx(f'{base_name}_{i}', bs)

    def export2tfpb(self, base_name:str, bs:int=1) -> None:
        r'''
        Export all :class:Model contained in :class:Ensemble to Tensorflow ProtocolBuffer format, via ONNX.
        Note that ONNX expects a fixed batch size (bs) which is the number of datapoints your wish to pass through the model concurrently.

        Arguments:
            base_name: Exported models will be called {base_name}_{model_num}.pb
            bs: batch size for exported models
        '''

        for i, m in enumerate(self.models): m.export2tfpb(f'{base_name}_{i}', bs)

    def get_feat_importance(self, fy:FoldYielder, eval_metric:Optional[EvalMetric]=None) -> pd.DataFrame:
        r'''
        Call :meth:get_ensemble_feat_importance passing this :class:Ensemble and provided arguments

        Arguments:
            fy: :class:FoldYielder interfacing to data on which to evaluate importance
            eval_metric: Optional :class:EvalMetric to use for quantifying performance
        '''
        
        return get_ensemble_feat_importance(self, fy, eval_metric)
