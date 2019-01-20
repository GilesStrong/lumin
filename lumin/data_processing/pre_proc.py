import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
from six.moves import cPickle as pickle
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def get_pre_proc_pipes(norm_in=False, norm_out=False, pca=False, whiten=False, with_mean=True, with_std=True):
    steps_in = []
    if not norm_in and not pca:
        steps_in.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    else:
        if pca: steps_in.append(('pca', PCA(whiten=whiten)))
        if norm_in: steps_in.append(('norm_in', StandardScaler(with_mean=with_mean, with_std=with_std)))
    input_pipe = Pipeline(steps_in)

    steps_out = []
    if norm_out: steps_out.append(('norm_out', StandardScaler(with_mean=with_mean, with_std=with_std)))
    else:        steps_out.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    output_pipe = Pipeline(steps_out)
    return input_pipe, output_pipe


def fit_input_pipe(df:pd.DataFrame, cont_feats:List[str], save_path:Optional[Path]=None) -> Pipeline:
    input_pipe, _ = get_pre_proc_pipes(norm_in=True)
    input_pipe.fit(df[cont_feats].values.astype('float32'))
    if save_path is not None:
        with open(save_path/'input_pipe.pkl', 'wb') as fout: pickle.dump(input_pipe, fout)
    return input_pipe


def proc_cats(train_df:pd.DataFrame, cat_feats:List[str], val_df:Optional[pd.DataFrame]=None, test_df:Optional[pd.DataFrame]=None) -> Tuple[OrderedDict,OrderedDict]:
    cat_maps = OrderedDict()
    cat_szs = OrderedDict()

    for f in cat_feats:
        cat_maps[f] = {}
        vals = sorted(set(train_df[f]))
        cat_szs[f] = len(vals)
        if val_df is not None:
            if sorted(set(val_df[f])) != vals: raise Exception(f"Feature {f} declared categorical, but validation set contains categories different to the training set")
        if test_df is not None:
            if sorted(set(val_df[f])) != vals: raise Exception(f"Feature {f} declared categorical, but testing set contains categories different to the training set")
        
        for i, v in enumerate(vals):
            train_df.loc[train_df[f] == v, f] = i
            if val_df is not None: val_df.loc[val_df[f] == v, f] = i
            if test_df is not None: test_df.loc[test_df[f] == v, f] = i
            cat_maps[f][i] = v
    return cat_maps, cat_szs
