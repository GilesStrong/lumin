import pandas as pd
import numpy as np
from typing import List, Optional

from sklearn.ensemble.forest import ForestRegressor

from .hyper_param import get_opt_rf_params
from ..plotting.interpretation import plot_importance
from ..plotting.plot_settings import PlotSettings
from ..utils.mod_ver import check_rfpimp


def get_rf_feat_importance(rf:ForestRegressor, inputs:pd.DataFrame, targets:np.ndarray, weights:Optional[np.ndarray]=None) -> pd.DataFrame:
    '''Wrapper function for rfpimp which checks correct version is installed'''
    check_rfpimp(); from rfpimp import importances
    return importances(rf, inputs, targets, features=inputs.columns, sample_weights=weights).reset_index()


def rf_rank_features(train_df:pd.DataFrame, val_df:pd.DataFrame, objective:str,
                     train_feats:List[str], targ_name:str='gen_target', wgt_name:Optional[str]=None,
                     importance_cut:float=0.0, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> List[str]:
    '''Rank train_feats via permutation importance using random forests'''
    w_trn = None if wgt_name is None else train_df[wgt_name]
    w_val = None if wgt_name is None else val_df[wgt_name]
    print("Optimising RF")
    _, rf = get_opt_rf_params(train_df[train_feats], train_df[targ_name], val_df[train_feats], val_df[targ_name],
                              objective, w_trn=w_trn, w_val=w_val, verbose=False)

    fi = get_rf_feat_importance(rf, train_df[train_feats], train_df[targ_name], w_trn)
    print("Top ten most important features:\n", fi[:min(len(fi), 10)])
    plot_importance(fi[:min(len(fi), 30)], savename=savename, settings=settings)

    top_feats = list(fi[fi.Importance > importance_cut].Feature)
    print(f"\n{len(top_feats)} features found with importance greater than {importance_cut}:\n", top_feats, '\n')
    if len(top_feats) == 0:
        print('No features found to be important, returning all training features. Good luck.')
        return train_feats
    if len(top_feats) < len(train_feats): 
        print("\nOptimising new RF")
        _, rf_new = get_opt_rf_params(train_df[top_feats], train_df[targ_name], val_df[top_feats], val_df[targ_name],
                                      objective, w_trn=w_trn, w_val=w_val, verbose=False)  
        print("Comparing RF scores, higher = better")                           
        print(f"All features:\t{rf.score(val_df[train_feats], val_df[targ_name], w_val):.5f}")
        print(f"Top features:\t{rf_new.score(val_df[top_feats], val_df[targ_name], w_val):.5f}")
    else:
        print('All training features found to be important')
    return top_feats
