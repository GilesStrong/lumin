import pandas as pd
from typing import List, Optional

from sklearn.ensemble.forest import ForestRegressor

from .hyper_param import get_opt_rf_params
from ..plotting.interpretation import plot_fi


def get_rf_feat_importance(rf:ForestRegressor, df:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'feature':df.columns, 'importance':rf.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)


def rf_rank_features(df_trn:pd.DataFrame, df_val:pd.DataFrame, objective:str,
                     train_feats:List[str], targ_name:str='gen_target', weight_name:Optional[str]=None,
                     cut:float=0.005) -> List[str]:
    w_trn = None if weight_name is None else df_trn[weight_name]
    w_val = None if weight_name is None else df_val[weight_name]
    print("Optimising RF")
    _, rf = get_opt_rf_params(df_trn[train_feats], df_trn[targ_name], w_trn,
                              df_val[train_feats], df_val[targ_name], w_val, objective, verbose=False)

    fi = get_rf_feat_importance(rf, df_trn[train_feats])
    print("Top ten most important features:\n", fi[:10])
    plot_fi(fi[:30])

    top_feats = list(fi[fi.importance > cut].feature)
    print(f"\n{len(top_feats)} features found with importance greater than {cut}:\n", top_feats)
    print("\nOptimising new RF")
    _, rf_new = get_opt_rf_params(df_trn[top_feats], df_trn[targ_name], w_trn,
                                  df_val[top_feats], df_val[targ_name], w_val, objective, verbose=False)                         
    print(f"RF score with all features:\t{rf.score(df_val[train_feats], df_val[targ_name], w_val):.5f}")
    print(f"RF score with top features:\t{rf_new.score(df_val[top_feats], df_val[targ_name], w_val):.5f}")
    print("Higher is better")
    return top_feats
