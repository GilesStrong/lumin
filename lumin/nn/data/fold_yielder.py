import numpy as np
import pandas as pd
import h5py
from typing import Dict, Optional, Union, List
import pickle
import warnings
from pathlib import Path
from collections import OrderedDict
import json

from sklearn.pipeline import Pipeline

__all__ = ['FoldYielder', 'HEPAugFoldYielder']


class FoldYielder:
    r'''
    Interface class for accessing data from foldfiles created by :meth:`~lumin.data_processing.file_proc.df2foldfile`

    Arguments:
        foldfile: filename of hdf5 file or opened hdf5 file
        cont_feats: list of names of continuous features present in input data, not required if foldfile contains meta data already
        cat_feats: list of names of categorical features present in input data, not required if foldfile contains meta data already
        ignore_feats: optional list of input features which should be ignored
        input_pipe: optional Pipeline, or filename for pickled Pipeline, which was used for processing the inputs
        output_pipe: optional Pipeline, or filename for pickled Pipeline, which was used for processing the targets
        yield_matrix: whether to actually yield matrix data if present
        matrix_pipe: preprocessing pipe for matrix data


    Examples::
        >>> fy = FoldYielder('train.h5')
        >>>
        >>> fy = FoldYielder('train.h5', ignore_feats=['phi'], input_pipe='input_pipe.pkl')
        >>>
        >>> fy = FoldYielder('train.h5', input_pipe=input_pipe, matrix_pipe=matrix_pipe)
        >>>
        >>> fy = FoldYielder('train.h5', input_pipe=input_pipe, yield_matrix=False)
    '''

    # TODO: Matrix example

    def __init__(self, foldfile:Union[str,Path,h5py.File], cont_feats:Optional[List[str]]=None, cat_feats:Optional[List[str]]=None,
                 ignore_feats:Optional[List[str]]=None, input_pipe:Optional[Union[str,Pipeline,Path]]=None, output_pipe:Optional[Union[str,Pipeline,Path]]=None,
                 yield_matrix:bool=True, matrix_pipe:Optional[Union[str,Pipeline,Path]]=None):
        self.cont_feats,self.cat_feats,self.input_pipe,self.output_pipe = cont_feats,cat_feats,input_pipe,output_pipe
        self.yield_matrix,self.matrix_pipe = yield_matrix,matrix_pipe
        self.augmented,self.aug_mult,self.train_time_aug,self.test_time_aug = False,0,False,False
        self._set_foldfile(foldfile)
        self.input_feats = self.cont_feats + self.cat_feats
        self.orig_cont_feats,self.orig_cat_feat,self._ignore_feats = self.cont_feats,self.cat_feats,[]
        if isinstance(self.input_pipe, str) or isinstance(self.input_pipe, Path): self.add_input_pipe_from_file(self.input_pipe)
        if isinstance(self.output_pipe, str) or isinstance(self.input_pipe, Path): self.add_output_pipe_from_file(self.output_pipe)
        if isinstance(self.matrix_pipe, str) or isinstance(self.matrix_pipe, Path): self.add_matrix_pipe_from_file(self.matrix_pipe)
        if ignore_feats is not None: self.add_ignore(ignore_feats)

    def __repr__(self) -> str: return f'FoldYielder with {self.n_folds} folds, containing {self.columns()}'
    
    def __len__(self) -> int: return self.n_folds
    
    def __getitem__(self, idx:int) -> Dict[str,np.ndarray]: return self.get_fold(idx)
    
    def __iter__(self) -> Dict[str,np.ndarray]:
        for i in range(self.n_folds): yield self.get_fold(i)

    def columns(self) -> List[str]:
        r'''
        Returns list of columns present in foldfile

        Returns:
            list of columns present in foldfile
        '''

        return [k for k in self.foldfile["fold_0"].keys()]

    def add_ignore(self, feats:List[str]) -> None:
        r'''
        Add features to ignored features.

        Arguments:
            feats: list of feature names to ignore
        '''

        self._ignore_feats += feats
        self.cont_feats = [f for f in self.cont_feats if f not in self._ignore_feats]
        self.cat_feats  = [f for f in self.cat_feats  if f not in self._ignore_feats]
    
    def get_ignore(self) -> List[str]:
        r'''
        Returns list of ignored features

        Returns:
            Features removed from training data
        '''
        
        return self._ignore_feats

    def get_use_cont_feats(self) -> List[str]:
        r'''
        Returns list of continuous features which will be present in training data, accounting for ignored features.

        Returns:
            List of continuous features
        '''

        return [f for f in self.cont_feats if f not in self._ignore_feats]

    def get_use_cat_feats(self) -> List[str]:
        r'''
        Returns list of categorical features which will be present in training data, accounting for ignored features.

        Returns:
            List of categorical features
        '''

        return [f for f in self.cat_feats if f not in self._ignore_feats]

    def _set_foldfile(self, foldfile:Union[str,Path,h5py.File]) -> None:
        r'''
        Sets the file from which to access data

        Arguments:
            foldfile: filename of h5py file or opened h5py file
        '''
        
        if not isinstance(foldfile,  h5py.File): foldfile = h5py.File(foldfile, "r+")
        self.foldfile, self.n_folds = foldfile, len([f for f in foldfile if 'fold_' in f])
        self.has_matrix = 'matrix_inputs' in self.columns()
        if 'meta_data' in self.foldfile: self._load_meta_data()

    def _load_meta_data(self) -> None:
        if self.cont_feats is not None:
            warnings.warn("Fold file contains meta data information, explicit passing of continuous and categorical feature lists is no longer required.")
        self.cont_feats = json.loads(self.foldfile['meta_data/cont_feats'][()])
        self.cat_feats  = json.loads(self.foldfile['meta_data/cat_feats'][()])
        self.targ_feats = json.loads(self.foldfile['meta_data/targ_feats'][()])
        if 'wgt_feat' in self.foldfile['meta_data']: self.wgt_feat = json.loads(self.foldfile['meta_data/wgt_feat'][()])
        if 'cat_maps' in self.foldfile['meta_data']: self.cat_maps = OrderedDict(json.loads(self.foldfile['meta_data/cat_maps'][()]))
        if self.has_matrix:
            self.matrix_feats = json.loads(self.foldfile['meta_data/matrix_feats'][()])
            self.matrix_feats['missing'] = np.array(self.matrix_feats['missing'], dtype=np.bool)

    def close(self) -> None:
        r'''
        Closes the foldfile
        '''

        self.foldfile.close()

    def add_input_pipe(self, input_pipe:Union[str,Pipeline]) -> None:
        r'''
        Adds an input pipe to the FoldYielder for use when deprocessing data

        Arguments:
            input_pipe: Pipeline which was used for preprocessing the input data or name of pkl file containing Pipeline
        '''

        if isinstance(input_pipe, str) or isinstance(input_pipe, Path): self.add_input_pipe_from_file(input_pipe)
        else:                                                           self.input_pipe = input_pipe
    
    def add_matrix_pipe(self, matrix_pipe:Union[str,Pipeline]) -> None:
        r'''
        Adds an matrix pipe to the FoldYielder for use when deprocessing data

        .. Warning:: Deprocessing matrix data is not yet implemented

        Arguments:
            matrix_pipe: Pipeline which was used for preprocessing the input data or name of pkl file containing Pipeline
        '''
        
        if isinstance(matrix_pipe, str) or isinstance(matrix_pipe, Path): self.add_matrix_pipe_from_file(matrix_pipe)
        else:                                                             self.matrix_pipe = matrix_pipe

    def add_output_pipe(self, output_pipe:Union[str,Pipeline]) -> None:
        r'''
        Adds an output pipe to the FoldYielder for use when deprocessing data

        Arguments:
            output_pipe: Pipeline which was used for preprocessing the target data or name of pkl file containing Pipeline
        '''
        
        if isinstance(output_pipe, str) or isinstance(output_pipe, Path): self.add_output_pipe_from_file(output_pipe)
        else:                                                             self.output_pipe = output_pipe

    def add_input_pipe_from_file(self, name:str) -> None:
        r'''
        Adds an input pipe from a pkl file to the FoldYielder for use when deprocessing data

        Arguments:
            name: name of pkl file containing Pipeline which was used for preprocessing the input data
        '''

        with open(name, 'rb') as fin: self.input_pipe = pickle.load(fin)

    def add_matrix_pipe_from_file(self, name:str) -> None:
        r'''
        Adds an matrix pipe from a pkl file to the FoldYielder for use when deprocessing data

        Arguments:
            name: name of pkl file containing Pipeline which was used for preprocessing the matrix data
        '''

        with open(name, 'rb') as fin: self.matrix_pipe = pickle.load(fin)

    def add_output_pipe_from_file(self, name:str) -> None:
        r'''
        Adds an output pipe from a pkl file to the FoldYielder for use when deprocessing data

        Arguments:
            name: name of pkl file containing Pipeline which was used for preprocessing the target data
        '''

        with open(name, 'rb') as fin: self.output_pipe = pickle.load(fin)

    def get_fold(self, idx:int) -> Dict[str,np.ndarray]:
        r'''
        Get data for single fold. Data consists of dictionary of inputs, targets, and weights.
        Accounts for ignored features.
        Inputs are passed through np.nan_to_num to deal with nans and infs.

        Arguments:
            idx: fold index to load

        Returns:
            tuple of inputs, targets, and weights as Numpy arrays
        '''

        def _append_matrix(data):
            data['inputs'] = (data['inputs'],np.nan_to_num(self.get_column('matrix_inputs', n_folds=1, fold_idx=idx)))
            return data

        data = self.get_data(n_folds=1, fold_idx=idx)
        if len(self._ignore_feats) == 0:
            return _append_matrix(data) if self.has_matrix and self.yield_matrix else data
        else:
            inputs = pd.DataFrame(np.array(self.foldfile[f'fold_{idx}/inputs']), columns=self.input_feats)
            inputs = inputs[[f for f in self.input_feats if f not in self._ignore_feats]]
            data['inputs'] = np.nan_to_num(inputs.values)
            return _append_matrix(data) if self.has_matrix and self.yield_matrix else data

    def get_column(self, column:str, n_folds:Optional[int]=None, fold_idx:Optional[int]=None, add_newaxis:bool=False) -> Union[np.ndarray, None]:
        r'''
        Load column (h5py group) from foldfile. Used for getting arbitrary data which isn't automatically grabbed by other methods.

        Arguments:
            column: name of h5py group to get
            n_folds: number of folds to get data from. Default all folds. Not compatable with fold_idx
            fold_idx: Only load group from a single, specified fold. Not compatable with n_folds
            add_newaxis: whether expand shape of returned data if data shape is ()

        Returns:
            Numpy array of column data
        '''

        if column not in self.columns(): return None

        if fold_idx is None:
            data = []
            for i, fold in enumerate([f for f in self.foldfile if 'fold_' in f]):
                if n_folds is not None and i >= n_folds: break
                data.append(np.array(self.foldfile[f'{fold}/{column}']))
            data = np.concatenate(data)
        else:
            if f'fold_{fold_idx}' not in self.foldfile: raise IndexError(f"Fold {fold_idx} does not exist")
            data = np.array(self.foldfile[f'fold_{fold_idx}/{column}'])
        return data[:, None] if data[0].shape is () and add_newaxis else data

    def get_data(self, n_folds:Optional[int]=None, fold_idx:Optional[int]=None) -> Dict[str,np.ndarray]:
        r'''
        Get data for single, specified fold or several of folds. Data consists of dictionary of inputs, targets, and weights.
        Does not accounts for ignored features.
        Inputs are passed through np.nan_to_num to deal with nans and infs.

        Arguments:
            n_folds: number of folds to get data from. Default all folds. Not compatable with fold_idx
            fold_idx: Only load group from a single, specified fold. Not compatable with n_folds

        Returns:
            tuple of inputs, targets, and weights as Numpy arrays
        '''

        return {'inputs':  np.nan_to_num(self.get_column('inputs',  n_folds=n_folds, fold_idx=fold_idx)),
                'targets':               self.get_column('targets', n_folds=n_folds, fold_idx=fold_idx, add_newaxis=True),
                'weights':               self.get_column('weights', n_folds=n_folds, fold_idx=fold_idx, add_newaxis=True)}

    def get_df(self, pred_name:str='pred', targ_name:str='targets', wgt_name:str='weights', n_folds:Optional[int]=None, fold_idx:Optional[int]=None,
               inc_inputs:bool=False, inc_ignore:bool=False, deprocess:bool=False, verbose:bool=True, suppress_warn:bool=False,
               nan_to_num:bool=False, inc_matrix:bool=False) -> pd.DataFrame:
        r'''
        Get a Pandas DataFrameof the data in the foldfile. Will add columns for inputs (if requested), targets, weights, and predictions (if present)

        Arguments:
            pred_name: name of prediction group
            targ_name: name of target group
            wgt_name: name of weight group
            n_folds: number of folds to get data from. Default all folds. Not compatable with fold_idx
            fold_idx: Only load group from a single, specified fold. Not compatable with n_folds
            inc_inputs: whether to include input data
            inc_ignore: whether to include ignored features
            deprocess: whether to deprocess inputs and targets if pipelines have been
            verbose: whether to print the number of datapoints loaded
            suppress_warn: whether to supress the warning about missing columns
            nan_to_num: whether to pass input data through `np.nan_to_num`
            inc_matrix: whether to include flattened matrix data in output, if present

        Returns:
            Pandas DataFrame with requested data
        '''

        # TODO Decide how to handle deprocessing matrix data: option for object by object, flattened out?

        if inc_inputs:
            inputs = self.get_column('inputs', n_folds=n_folds, fold_idx=fold_idx)
            if deprocess and self.input_pipe is not None:
                try:
                    inputs = np.hstack((self.input_pipe.inverse_transform(inputs[:,:len(self.orig_cont_feats)]), inputs[:,len(self.orig_cont_feats):]))
                except ValueError:
                    if self.has_matrix:
                        print('Deprocessing of flat data failed, possible due to the input_pipe expecting to also transform matrix data. Deprocessing of matrix'
                              'is not currently implemented, and deprocessing of flat data using an input_pipe which expects matrix data as well is difficult '
                              'due to loss of variable ordering. In future please use separate pipes to preprocess flat data and matrix data. Returning inputs '
                              'as processed.')
                    else:
                        print('Deprocessing of flat data failed, returning inputs as processed.')
                        
            if nan_to_num: inputs = np.nan_to_num(inputs)
            data = pd.DataFrame(inputs, columns=self.input_feats)
            if len(self._ignore_feats) > 0 and not inc_ignore: data = data[[f for f in self.input_feats if f not in self._ignore_feats]]
            if self.has_matrix and inc_matrix:
                mat = self.get_column('matrix_inputs', n_folds=n_folds, fold_idx=fold_idx).reshape(len(inputs),np.multiply(*self.matrix_feats['shape']))
                mat = mat[:,np.logical_not(self.matrix_feats['missing'])]
                # if deprocess and self.matrix_pipe is not None: mat = self.matrix_pipe.inverse_transform(mat)
                if nan_to_num: mat = np.nan_to_num(mat)
                data = data.join(pd.DataFrame(mat, columns=self.matrix_feats['present_feats']))
        else:
            data = pd.DataFrame()

        targets = self.get_column(targ_name, n_folds=n_folds, fold_idx=fold_idx)
        if deprocess and self.output_pipe is not None: targets = self.output_pipe.inverse_transform(targets)
        if targets is not None and len(targets.shape) > 1:
            for t in range(targets.shape[-1]): data[f'gen_target_{t}'] = targets[:,t]
        elif targets is None and not suppress_warn:
            warnings.warn(f"{targ_name} not found in file")
        else:
            data['gen_target'] = targets

        weights = self.get_column(wgt_name, n_folds=n_folds, fold_idx=fold_idx)
        if weights is not None and weights is not None and len(weights.shape) > 1:
            for w in range(weights.shape[-1]): data[f'gen_weight_{w}'] = weights[:,w]
        elif weights is None and not suppress_warn:
            warnings.warn(f"{wgt_name} not found in file")
        else:
            data['gen_weight'] = weights

        preds = self.get_column(pred_name, n_folds=n_folds, fold_idx=fold_idx)
        if deprocess and self.output_pipe is not None: preds = self.output_pipe.inverse_transform(preds)
        if preds is not None and len(preds.shape) > 1:
            for p in range(preds.shape[-1]): data[f'pred_{p}'] = preds[:,p]
        elif preds is not None:
            data['pred'] = preds
        elif not suppress_warn:
            warnings.warn(f'{pred_name} not found in foldfile file')
        if verbose: print(f'{len(data)} datapoints loaded')
        return data

    def save_fold_pred(self, pred:np.ndarray, fold_idx:int, pred_name:str='pred') -> None:
        r'''
        Save predictions for given fold as a new column in the foldfile

        Arguments:
            pred: array of predictions in the same order as data appears in the file
            fold_idx: index for fold
            pred_name: name of column to save predictions under
        '''

        try: self.foldfile.create_dataset(f'fold_{fold_idx}/{pred_name}', shape=pred.shape, dtype='float32')
        except RuntimeError: pass
        self.foldfile[f'fold_{fold_idx}/{pred_name}'][...] = pred


class HEPAugFoldYielder(FoldYielder):
    r'''
    Specialised version of :class:`~lumin.nn.data.fold_yielder.FoldYielder` providing HEP specific data augmetation at train and test time.

    Arguments:
        foldfile: filename of hdf5 file or opened hdf5 file
        cont_feats: list of names of continuous features present in input data, not required if foldfile contains meta data already
        cat_feats: list of names of categorical features present in input data, not required if foldfile contains meta data already
        ignore_feats: optional list of input features which should be ignored
        targ_feats: optional list of target features to also be transformed
        rot_mult: number of rotations of event in phi to make at test-time (currently must be even).
                  Greater than zero will also apply random rotations during train-time
        random_rot: whether test-time rotation angles should be random or in steps of 2pi/rot_mult
        reflect_x: whether to reflect events in x axis at train and test time
        reflect_y: whether to reflect events in y axis at train and test time
        reflect_z: whether to reflect events in z axis at train and test time
        train_time_aug: whether to apply augmentations at train time
        test_time_aug: whether to apply augmentations at test time
        input_pipe: optional Pipeline, or filename for pickled Pipeline, which was used for processing the inputs
        output_pipe: optional Pipeline, or filename for pickled Pipeline, which was used for processing the targets
        yield_matrix: whether to actually yield matrix data if present
        matrix_pipe: preprocessing pipe for matrix data

    Examples::
        >>> fy = HEPAugFoldYielder('train.h5',
        ...                        cont_feats=['pT','eta','phi','mass'],
        ...                        rot_mult=2, reflect_y=True, reflect_z=True,
        ...                        input_pipe='input_pipe.pkl')
    '''

    def __init__(self, foldfile:Union[str,Path,h5py.File], cont_feats:Optional[List[str]]=None, cat_feats:Optional[List[str]]=None,
                 ignore_feats:Optional[List[str]]=None, targ_feats:Optional[List[str]]=None,
                 rot_mult:int=2, random_rot:bool=False,
                 reflect_x:bool=False, reflect_y:bool=True, reflect_z:bool=True,
                 train_time_aug:bool=True, test_time_aug:bool=True,
                 input_pipe:Optional[Pipeline]=None, output_pipe:Optional[Pipeline]=None,
                 yield_matrix:bool=True, matrix_pipe:Optional[Union[str,Pipeline]]=None):
        super().__init__(foldfile=foldfile, cont_feats=cont_feats, cat_feats=cat_feats,
                         ignore_feats=ignore_feats, input_pipe=input_pipe, output_pipe=output_pipe,
                         yield_matrix=yield_matrix, matrix_pipe=matrix_pipe)

        if rot_mult > 0 and not random_rot and rot_mult % 2 != 0:
            warnings.warn('Warning: rot_mult must currently be even for fixed rotations, adding an extra rotation multiplicity')
            rot_mult += 1
        self.rot_mult,self.random_rot,self.reflect_x,self.reflect_y,self.reflect_z,self.train_time_aug,self.test_time_aug,self.targ_feats = \
            rot_mult,random_rot,reflect_x,reflect_y,reflect_z,train_time_aug,test_time_aug,targ_feats
        self.augmented,self.reflect_axes,self.aug_mult = True,[],1
        self.vectors = [x[:-3] for x in self.cont_feats if '_px' in x]
        if self.targ_feats is not None: self.targ_vectors = [x[:-3] for x in self.targ_feats if '_px' in x]

        if self.rot_mult:
            print("Augmenting via phi rotations")
            self.aug_mult = self.rot_mult
            if self.reflect_y:
                print("Augmenting via y flips")
                self.reflect_axes += ['_py']
                self.aug_mult *= 2
            if self.reflect_z:
                print("Augmenting via longitunidnal flips")
                self.reflect_axes += ['_pz']
                self.aug_mult *= 2
        else:
            if self.reflect_x:
                print("Augmenting via x flips")
                self.reflect_axes += ['_px']
                self.aug_mult *= 2
            if self.reflect_y:
                print("Augmenting via y flips")
                self.reflect_axes += ['_py']
                self.aug_mult *= 2
            if self.reflect_z:
                print("Augmenting via longitunidnal flips")
                self.reflect_axes += ['_pz']
                self.aug_mult *= 2
        print(f'Total augmentation multiplicity is {self.aug_mult}')
    
    def _rotate(self, df:pd.DataFrame, vecs:List[str]) -> None:
        for vec in vecs:
            df.loc[:, f'{vec}_pxtmp'] = df.loc[:, f'{vec}_px']*np.cos(df.loc[:, 'aug_angle'])-df.loc[:, f'{vec}_py']*np.sin(df.loc[:, 'aug_angle'])
            df.loc[:, f'{vec}_py']    = df.loc[:, f'{vec}_py']*np.cos(df.loc[:, 'aug_angle'])+df.loc[:, f'{vec}_px']*np.sin(df.loc[:, 'aug_angle'])
            df.loc[:, f'{vec}_px']    = df.loc[:, f'{vec}_pxtmp']
    
    def _reflect(self, df:pd.DataFrame, vectors:List[str]) -> None:
        for vector in vectors:
            for coord in self.reflect_axes:
                try:
                    cut = (df[f'aug{coord}'] == 1)
                    df.loc[cut, f'{vector}{coord}'] = -df.loc[cut, f'{vector}{coord}']
                except KeyError:
                    pass
            
    def get_fold(self, idx:int) -> Dict[str,np.ndarray]:
        r'''
        Get data for single fold applying random train-time data augmentaion. Data consists of dictionary of inputs, targets, and weights.
        Accounts for ignored features.
        Inputs are passed through np.nan_to_num to deal with nans and infs.

        Arguments:
            idx: fold index to load

        Returns:
            tuple of inputs, targets, and weights as Numpy arrays
        '''

        def _append_matrix(data):
            data['inputs'] = (data['inputs'],np.nan_to_num(self.get_column('matrix_inputs', n_folds=1, fold_idx=idx)))
            return data

        data = self.get_data(n_folds=1, fold_idx=idx)
        if not self.augmented: return data
        inputs = pd.DataFrame(np.array(self.foldfile[f'fold_{idx}/inputs']), columns=self.input_feats)
        if self.targ_feats is not None: targets = pd.DataFrame(np.array(self.foldfile[f'fold_{idx}/targets']), columns=self.targ_feats)
            
        if self.rot_mult:
            inputs['aug_angle'] = (2*np.pi*np.random.random(size=len(inputs)))-np.pi
            self._rotate(inputs, self.vectors)
            if self.targ_feats is not None:
                targets['aug_angle'] = inputs['aug_angle']
                self._rotate(targets, self.targ_vectors)
            
        for coord in self.reflect_axes:
            inputs[f'aug{coord}'] = np.random.randint(0, 2, size=len(inputs))
            if self.targ_feats is not None: targets[f'aug{coord}'] = inputs[f'aug{coord}']
        self._reflect(inputs, self.vectors)
        if self.targ_feats is not None: self._reflect(targets, self.targ_vectors)

        inputs = inputs[[f for f in self.input_feats if f not in self._ignore_feats]]
        data['inputs'] = np.nan_to_num(inputs.values)
        if self.targ_feats is not None:
            targets = targets[self.targ_feats]
            data['targets'] = np.nan_to_num(targets.values)
        return _append_matrix(data) if self.has_matrix and self.yield_matrix else data

    def _get_ref_idx(self, aug_idx:int) -> str:
        n_axes = len(self.reflect_axes)
        div = self.rot_mult if self.rot_mult else 1
        if   n_axes == 3: return '{0:03b}'.format(int(aug_idx/div))
        elif n_axes == 2: return '{0:02b}'.format(int(aug_idx/div))
        elif n_axes == 1: return '{0:01b}'.format(int(aug_idx/div))
    
    def get_test_fold(self, idx:int, aug_idx:int) -> Dict[str, np.ndarray]:
        r'''
        Get test data for single fold applying test-time data augmentaion. Data consists of dictionary of inputs, targets, and weights.
        Accounts for ignored features.
        Inputs are passed through np.nan_to_num to deal with nans and infs.

        Arguments:
            idx: fold index to load
            aug_idx: index for the test-time augmentaion (ignored if random test-time augmentation requested)

        Returns:
            tuple of inputs, targets, and weights as Numpy arrays
        '''

        def _append_matrix(data):
            data['inputs'] = (data['inputs'],np.nan_to_num(self.get_column('matrix_inputs', n_folds=1, fold_idx=idx)))
            return data

        if aug_idx >= self.aug_mult: raise ValueError(f"Invalid augmentation idx passed {aug_idx}")
        data = self.get_data(n_folds=1, fold_idx=idx)
        if not self.augmented: return data
        
        inputs = pd.DataFrame(np.array(self.foldfile[f'fold_{idx}/inputs']), columns=self.input_feats)
        if len(self.reflect_axes) > 0 and self.rot_mult > 0:
            rot_idx = aug_idx % self.rot_mult
            ref_idx = self._get_ref_idx(aug_idx)
            if self.random_rot: inputs['aug_angle'] = (2*np.pi*np.random.random(size=len(inputs)))-np.pi
            else:               inputs['aug_angle'] = np.linspace(0, 2*np.pi, (self.rot_mult)+1)[rot_idx]
            self._rotate(inputs, self.vectors)            

            for i, coord in enumerate(self.reflect_axes): inputs[f'aug{coord}'] = int(ref_idx[i])
            self._reflect(inputs, self.vectors)
            
        elif len(self.reflect_axes) > 0:
            ref_idx = self._get_ref_idx(aug_idx)
            for i, coord in enumerate(self.reflect_axes): inputs[f'aug{coord}'] = int(ref_idx[i])
            self._reflect(inputs, self.vectors)
            
        elif self.rot_mult:
            if self.random_rot: inputs['aug_angle'] = (2*np.pi*np.random.random(size=len(inputs)))-np.pi
            else:               inputs['aug_angle'] = np.linspace(0, 2*np.pi, (self.rot_mult)+1)[aug_idx]
            self._rotate(inputs, self.vectors)
            
        inputs = inputs[[f for f in self.input_feats if f not in self._ignore_feats]]
        data['inputs'] = np.nan_to_num(inputs.values)
        
        if self.targ_feats is not None:
            targets = pd.DataFrame(np.array(self.foldfile[f'fold_{idx}/targets']), columns=self.targ_feats)
            if len(self.reflect_axes) > 0 and self.rot_mult > 0:
                rot_idx = aug_idx % self.rot_mult
                ref_idx = self._get_ref_idx(aug_idx)
                if self.random_rot: targets['aug_angle'] = (2*np.pi*np.random.random(size=len(targets)))-np.pi
                else:               targets['aug_angle'] = np.linspace(0, 2*np.pi, (self.rot_mult)+1)[rot_idx]
                self._rotate(targets, self.targ_vectors)            

                for i, coord in enumerate(self.reflect_axes): targets[f'aug{coord}'] = int(ref_idx[i])
                self._reflect(targets, self.targ_vectors)

            elif len(self.reflect_axes) > 0:
                ref_idx = self._get_ref_idx(aug_idx)
                for i, coord in enumerate(self.reflect_axes): targets[f'aug{coord}'] = int(ref_idx[i])
                self._reflect(targets, self.targ_vectors)

            elif self.rot_mult:
                if self.random_rot: targets['aug_angle'] = (2*np.pi*np.random.random(size=len(targets)))-np.pi
                else:               targets['aug_angle'] = np.linspace(0, 2*np.pi, (self.rot_mult)+1)[aug_idx]
                self._rotate(targets, self.targ_vectors)
            
            targets = targets[self.targ_feats]
            data['targets'] = np.nan_to_num(targets.values)
        return _append_matrix(data) if self.has_matrix and self.yield_matrix else data
