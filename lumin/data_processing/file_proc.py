import h5py
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any, Tuple, Dict
import os
from pathlib import Path
import json

from sklearn.model_selection import StratifiedKFold, KFold

__all__ = ['save_to_grp', 'fold2foldfile', 'df2foldfile', 'add_meta_data']


def save_to_grp(arr:np.ndarray, grp:h5py.Group, name:str) -> None:
    r'''
    Save Numpy array as a dataset in an h5py Group
    
    Arguments:
        arr: array to be saved
        grp: group in which to save arr
        name: name of dataset to create
    '''

    # TODO Option for string length

    ds = grp.create_dataset(name, shape=arr.shape, dtype=arr.dtype.name if arr.dtype.name not in ['object', 'str864'] else 'S64')
    ds[...] = arr if arr.dtype.name not in ['object', 'str864'] else arr.astype('S64')


def _build_matrix_lookups(feats:List[str], vecs:List[str], feats_per_vec:List[str], row_wise:bool) -> Tuple[List[str],np.ndarray,Tuple[int,int]]:
    shape = (len(vecs),len(feats_per_vec)) if row_wise else (len(feats_per_vec),len(vecs))
    lookup,missing = np.zeros(shape, dtype=np.array(feats).dtype),np.zeros(shape, dtype=np.bool)
    if row_wise:
        for i, v in enumerate(vecs):
            for j, c in enumerate(feats_per_vec):
                f = f'{v}_{c}'
                if f in feats:
                    lookup[i,j] = f
                else:
                    lookup[i,j] = feats[0]  # Temp value, to be set to null later using missing
                    missing[i,j] = True
    else:
        for j, v in enumerate(vecs):
            for i, c in enumerate(feats_per_vec):
                f = f'{v}_{c}'
                if f in feats:
                    lookup[i,j] = f
                else:
                    lookup[i,j] = feats[0]  # Temp value, to be set to null later using missing
                    missing[i,j] = True
    return list(lookup.flatten()),missing.flatten(),shape


def fold2foldfile(df:pd.DataFrame, out_file:h5py.File, fold_idx:int,
                  cont_feats:List[str], cat_feats:List[str], targ_feats:Union[str,List[str]], targ_type:Any,
                  misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None,
                  matrix_lookup:Optional[List[str]]=None, matrix_missing:Optional[np.ndarray]=None, matrix_shape:Optional[Tuple[int,int]]=None) -> None:
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
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes 
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
    '''

    # TODO infer target type automatically

    grp = out_file.create_group(f'fold_{fold_idx}')
    save_to_grp(np.hstack((df[cont_feats].values.astype('float32'), df[cat_feats].values.astype('float32'))), grp, 'inputs')
    save_to_grp(df[targ_feats].values.astype(targ_type), grp, 'targets')
    if wgt_feat is not None: 
        if wgt_feat in df.columns: save_to_grp(df[wgt_feat].values.astype('float32'), grp, 'weights')
        else:                      print(f'{wgt_feat} not found in file')
    if misc_feats is not None:
        for f in misc_feats:
            if f in df.columns: save_to_grp(df[f].values, grp, f)
            else:               print(f'{f} not found in file')


    if matrix_lookup is not None:
        mat = df[matrix_lookup].values
        mat[:,matrix_missing] = np.NaN
        mat = mat.reshape((len(df),*matrix_shape))
        save_to_grp(mat, grp, 'matrix_inputs')


def df2foldfile(df:pd.DataFrame, n_folds:int, cont_feats:List[str], cat_feats:List[str],
                targ_feats:Union[str,List[str]], savename:Union[Path,str], targ_type:str,
                strat_key:Optional[str]=None, misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None, cat_maps:Optional[Dict[str,Dict[int,Any]]]=None,
                matrix_vecs:Optional[List[str]]=None, matrix_feats_per_vec:Optional[List[str]]=None, matrix_row_wise:Optional[bool]=None) -> None:
    r'''
    Convert dataframe into h5py file by splitting data into sub-folds to be accessed by a :class:`~lumin.nn.data.fold_yielder.FoldYielder`
    
    Arguments:
        df: Dataframe from which to save data
        n_folds: number of folds to split df into
        cont_feats: list of columns in df to save as continuous variables
        cat_feats: list of columns in df to save as discreet variables
        targ_feats: (list of) column(s) in df to save as target feature(s)
        savename: name of h5py file to create (.h5py extension not required)
        targ_type: type of target feature, e.g. int,'float32'
        strat_key: column to use for stratified splitting
        misc_feats: any extra columns to save
        wgt_feat: column to save as data weights
        cat_maps: Dictionary mapping categorical features to dictionary mapping codes to categories
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes 
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
    '''

    savename = str(savename)
    os.system(f'rm {savename}.hdf5')
    os.makedirs(savename[:savename.rfind('/')], exist_ok=True)
    out_file = h5py.File(f'{savename}.hdf5', "w")
    lookup,missing,shape = None,None,None
    if matrix_vecs is not None:
        lookup,missing,shape = _build_matrix_lookups(df.columns, matrix_vecs, matrix_feats_per_vec, matrix_row_wise)
        mat_feats = list(np.array(lookup)[np.logical_not(missing)])  # Only features present in data
        dup = [f for f in cont_feats if f in mat_feats]
        if len(dup) > 1:
            print(f'{dup} present in both matrix features and continuous features; removing from continuous features')
            cont_feats = [f for f in cont_feats if f not in dup]

    if strat_key not in df.columns:
        print(f'{strat_key} not found in DataFrame')
        strat_key = None
    if strat_key is None:
        kf = KFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df, df[strat_key])
    for fold_idx, (_, fold) in enumerate(folds):
        print(f"Saving fold {fold_idx} with {len(fold)} events")
        fold2foldfile(df.iloc[fold].copy(), out_file, fold_idx, cont_feats=cont_feats, cat_feats=cat_feats, targ_feats=targ_feats,
                      targ_type=targ_type, misc_feats=misc_feats, wgt_feat=wgt_feat,
                      matrix_lookup=lookup, matrix_missing=missing, matrix_shape=shape)
    add_meta_data(out_file=out_file, feats=df.columns, cont_feats=cont_feats, cat_feats=cat_feats, cat_maps=cat_maps, targ_feats=targ_feats, wgt_feat=wgt_feat,
                  matrix_vecs=matrix_vecs, matrix_feats_per_vec=matrix_feats_per_vec, matrix_row_wise=matrix_row_wise)


def add_meta_data(out_file:h5py.File, feats:List[str], cont_feats:List[str], cat_feats:List[str], cat_maps:Optional[Dict[str,Dict[int,Any]]],
                  targ_feats:Union[str,List[str]], wgt_feat:Optional[str]=None,
                  matrix_vecs:Optional[List[str]]=None, matrix_feats_per_vec:Optional[List[str]]=None, matrix_row_wise:Optional[bool]=None) -> None:
    r'''
    Adds meta data to foldfile containing information about the data: feature names, matrix information, etc.
    :class:`~lumin.nn.data.fold_yielder.FoldYielder` objects will access this and automatically extract it to save the user from having to manually pass lists
    of features.

    Arguments:
        out_file: h5py file to save data in
        feats: list of all features in data
        cont_feats: list of continuous features
        cat_feats: list of categorical features
        cat_maps: Dictionary mapping categorical features to dictionary mapping codes to categories
        targ_feats: (list of) target feature(s)
        wgt_feat: name of weight feature
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes 
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
    '''

    grp = out_file.create_group('meta_data')
    grp.create_dataset('cont_feats',   data=json.dumps(cont_feats))
    grp.create_dataset('cat_feats',    data=json.dumps(cat_feats))
    grp.create_dataset('targ_feats',   data=json.dumps(targ_feats))
    if wgt_feat is not None: grp.create_dataset('wgt_feat', data=json.dumps(wgt_feat))
    if cat_maps is not None: grp.create_dataset('cat_maps', data=json.dumps(cat_maps))
    if matrix_vecs is not None:
        lookup,missing,shape = _build_matrix_lookups(feats, matrix_vecs, matrix_feats_per_vec, matrix_row_wise)
        use = list(np.array(lookup)[np.logical_not(missing)])  # Only features present in data
        grp.create_dataset('matrix_feats', data=json.dumps({'present_feats': use, 'vecs': matrix_vecs, 'missing': [int(m) for m in missing],
                                                            'feats_per_vec': matrix_feats_per_vec, 'row_wise': matrix_row_wise, 'shape': shape}))
