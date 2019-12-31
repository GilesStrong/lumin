import h5py
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
import os
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold

__all__ = ['save_to_grp', 'fold2foldfile', 'df2foldfile']


def save_to_grp(arr:np.ndarray, grp:h5py.Group, name:str) -> None:
    r'''
    Save Numpy array as a dataset in an h5py Group
    
    Arguments:
        arr: array to be saved
        grp: group in which to save arr
        name: name of dataset to create
    '''

    # TODO Option for string length

    ds = grp.create_dataset(name, shape=arr.shape, dtype=arr.dtype.name if arr.dtype.name != 'object' else 'S16')
    ds[...] = arr if arr.dtype.name != 'object' else arr.astype('S16')


def fold2foldfile(df:pd.DataFrame, out_file:h5py.File, fold_idx:int,
                  cont_feats:List[str], cat_feats:List[str], targ_feats:Union[str,List[str]], targ_type:Any,
                  misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None,
                  matrix_feats:Optional[Union[List[List[str]],np.ndarray]]=None) -> None:
    r'''
    Save fold of data into an h5py Group

    Arguments:
        df: Dataframe from which to save data
        out_file: h5py file to save data in
        fold_idx: ID for the fold; used name h5py group according to 'fold_{fold_idx}'
        cont_feats: list of columns in df to save as continuous variables
        cat_feats: list of columns in df to save as discreet variables
        targ_feats: (list of) column(s) in df to save as target feature(s)
        targ_type: type of target feature, e.g. int,'float32'
        misc_feats: any extra columns to save
        wgt_feat: column to save as data weights
        matrix_feats: 2-D Array of feature values to store as a matrix in relevant positions. Features listed but not present in df will be replaced with NaN.
    '''

    # TODO infer target type automatically

    grp = out_file.create_group(f'fold_{fold_idx}')
    save_to_grp(np.hstack((df[cont_feats].values.astype('float32'), df[cat_feats].values.astype('float32'))), grp, 'inputs')
    save_to_grp(df[targ_feats].values.astype(targ_type), grp, 'targets')
    if wgt_feat is not None: save_to_grp(df[wgt_feat].values.astype('float32'), grp, 'weights')
    if misc_feats is not None:
        for f in misc_feats: save_to_grp(df[f].values, grp, f)

    if matrix_feats is not None:
        if isinstance(matrix_feats, list): matrix_feats = np.array(matrix_feats)
        x = df[matrix_feats.flatten()].values
        mat = np.zeros((len(df),len(matrix_feats),len(matrix_feats[0])))
        k = 0 
        for i in range(len(matrix_feats)):
            for j,f in enumerate(matrix_feats[i]):
                if f in df.columns:
                    mat[:,i,j] = x[:,k]
                    k += 1
                else:
                    mat[:,i,j] = np.NaN
        save_to_grp(mat, grp, 'matrix_inputs')


def df2foldfile(df:pd.DataFrame, n_folds:int, cont_feats:List[str], cat_feats:List[str],
                targ_feats:Union[str,List[str]], savename:Union[Path,str], targ_type:str,
                strat_key:Optional[str]=None, misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None,
                matrix_feats:Optional[Union[List[List[str]],np.ndarray]]=None) -> None:
    r'''
    Convert dataframe into h5py file by splitting data into sub-folds to be accessed by a :class:`~lumin.nn.data.fold_yielder.FoldYielder`
    
    Arguments:
        df: Dataframe from which to save data
        n_folds: number of folds to split df into
        cont_feats: list of columns in df to save as continuous variables
        cat_feats: list of columns in df to save as discreet variables
        targ_feats (list of) column(s) in df to save as target feature(s)
        savename: name of h5py file to create (.h5py extension not required)
        targ_type: type of target feature, e.g. int,'float32'
        strat_key (optional): column to use for stratified splitting
        misc_feats (optional): any extra columns to save
        wgt_feat (optional): column to save as data weights
        matrix_feats: 2-D Array of feature values to store as a matrix in relevant positions. Features listed but not present in df will be replaced with NaN.
    '''

    savename = str(savename)
    os.system(f'rm {savename}.hdf5')
    os.makedirs(savename[:savename.rfind('/')], exist_ok=True)
    out_file = h5py.File(f'{savename}.hdf5', "w")

    if strat_key is None:
        kf = KFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df, df[strat_key])
    for fold_idx, (_, fold) in enumerate(folds):
        print(f"Saving fold {fold_idx} with {len(fold)} events")
        fold2foldfile(df.iloc[fold].copy(), out_file, fold_idx, cont_feats=cont_feats, cat_feats=cat_feats, targ_feats=targ_feats,
                      targ_type=targ_type, misc_feats=misc_feats, wgt_feat=wgt_feat, matrix_feats=matrix_feats)
                      