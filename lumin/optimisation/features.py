import pandas as pd
from typing import List, Optional
from rfpimp import importances

from sklearn.ensemble.forest import ForestRegressor

from .hyper_param import get_opt_rf_params
from ..plotting.interpretation import plot_fi


def get_rf_feat_importance(rf:ForestRegressor, x_val, y_val, w_val, feats:List[str]) -> pd.DataFrame:
    return importances(rf, x_val, y_val, features=feats, sample_weights=w_val).reset_index()


def rf_rank_features(df_trn:pd.DataFrame, df_val:pd.DataFrame, objective:str,
                     train_feats:List[str], targ_name:str='gen_target', weight_name:Optional[str]=None,
                     cut:float=0.0) -> List[str]:
    w_trn = None if weight_name is None else df_trn[weight_name]
    w_val = None if weight_name is None else df_val[weight_name]
    print("Optimising RF")
    _, rf = get_opt_rf_params(df_trn[train_feats], df_trn[targ_name], w_trn,
                              df_val[train_feats], df_val[targ_name], w_val, objective, verbose=False)

    fi = get_rf_feat_importance(rf, df_val[train_feats], df_val[targ_name], w_val, train_feats)
    print("Top ten most important features:\n", fi[:max(len(fi), 10)])
    plot_fi(fi[:max(len(fi), 30)])

    top_feats = list(fi[fi.Importance > cut].Feature)
    print(f"\n{len(top_feats)} features found with importance greater than {cut}:\n", top_feats)
    print("\nOptimising new RF")
    _, rf_new = get_opt_rf_params(df_trn[top_feats], df_trn[targ_name], w_trn,
                                  df_val[top_feats], df_val[targ_name], w_val, objective, verbose=False)  
    print("Comparing RF scores, higher = better")                           
    print(f"All features:\t{rf.score(df_val[train_feats], df_val[targ_name], w_val):.5f}")
    print(f"Top features:\t{rf_new.score(df_val[top_feats], df_val[targ_name], w_val):.5f}")
    return top_feats
