import h5py
from h5py import Group
import numpy as np
import pandas as pd
from typing import List, Union, Any, Optional
import os

from sklearn.model_selection import StratifiedKFold, KFold


def save_to_grp(data:np.ndarray, grp:Group, name:str) -> None:
    d = grp.create_dataset(name, shape=data.shape, dtype=data.dtype.name if data.dtype.name != 'object' else 'S16')
    d[...] = data if data.dtype.name != 'object' else data.astype('S16')


def fold2foldfile(df:pd.DataFrame, out_file:h5py.File, fold_id:int,
                  cont_feats:List[str], cat_feats:List[str], targ_feats:Union[str,List[str]],
                  targ_type, misc_feats:List[str]=[], weight_feat:Optional[str]=None) -> None:
    grp = out_file.create_group(f'fold_{fold_id}')
    
    x = np.hstack((df[cont_feats].values.astype('float32'),
                   df[cat_feats].values.astype('float32')))
    save_to_grp(x, grp, 'inputs')
    save_to_grp(df[targ_feats].values.astype(targ_type), grp, 'targets')
    if weight_feat is not None: save_to_grp(df[weight_feat].values.astype('float32'), grp, 'weights')
    for m in misc_feats: save_to_grp(df[m].values, grp, m)  


def df2foldfile(df:pd.DataFrame, n_folds:int, cont_feats:List[str], cat_feats:List[str],
                targ_feats:Union[str,List[str]], savename:str, targ_type:Any,
                strat_key:str=None, misc_feats:List[str]=[], weight_feat:Optional[str]=None):
    os.system(f'rm {savename}.hdf5')
    os.makedirs(savename[:savename.rfind('/')], exist_ok=True)
    out_file = h5py.File(f'{savename}.hdf5', "w")

    if strat_key is None:
        kf = KFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(df, df[strat_key])
    
    for fold_id, (_, fold) in enumerate(folds):
        print(f"Saving fold: {fold_id} with {len(fold)} events")
        fold2foldfile(df.iloc[fold].copy(), out_file, fold_id, cont_feats=cont_feats, cat_feats=cat_feats, targ_feats=targ_feats,
                      targ_type=targ_type, misc_feats=misc_feats, weight_feat=weight_feat)
                      