import h5py
from h5py import Group
import numpy as np
import pandas as pd
from typing import List, Union, Optional
import os
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold


def save_to_grp(arr:np.ndarray, grp:Group, name:str) -> None:
    '''Save array group in HDF5 file'''
    ds = grp.create_dataset(name, shape=arr.shape, dtype=arr.dtype.name if arr.dtype.name != 'object' else 'S16')
    ds[...] = arr if arr.dtype.name != 'object' else arr.astype('S16')


def fold2foldfile(df:pd.DataFrame, out_file:h5py.File, fold_idx:int,
                  cont_feats:List[str], cat_feats:List[str], targ_feats:Union[str,List[str]], targ_type:str,
                  misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None) -> None:
    '''Save fold data into foldfile group'''
    grp = out_file.create_group(f'fold_{fold_idx}')
    save_to_grp(np.hstack((df[cont_feats].values.astype('float32'), df[cat_feats].values.astype('float32'))), grp, 'inputs')
    save_to_grp(df[targ_feats].values.astype(targ_type), grp, 'targets')
    if wgt_feat is not None: save_to_grp(df[wgt_feat].values.astype('float32'), grp, 'weights')
    if misc_feats is not None:
        for f in misc_feats: save_to_grp(df[f].values, grp, f)  


def df2foldfile(df:pd.DataFrame, n_folds:int, cont_feats:List[str], cat_feats:List[str],
                targ_feats:Union[str,List[str]], savename:Union[Path,str], targ_type:str,
                strat_key:str=None, misc_feats:Optional[List[str]]=None, wgt_feat:Optional[str]=None):
    '''Convert dataframe into foldfile'''
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
                      targ_type=targ_type, misc_feats=misc_feats, wgt_feat=wgt_feat)
                      