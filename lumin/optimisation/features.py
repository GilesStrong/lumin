import pandas as pd
import numpy as np
from typing import List, Optional
from fastprogress import progress_bar
from rfpimp import importances

from sklearn.ensemble.forest import ForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .hyper_param import get_opt_rf_params
from ..plotting.interpretation import plot_importance
from ..plotting.plot_settings import PlotSettings


def get_rf_feat_importance(rf:ForestRegressor, inputs:pd.DataFrame, targets:np.ndarray, weights:Optional[np.ndarray]=None) -> pd.DataFrame:
    r'''
    Compute feature importance for a Random Forest model using rfpimp.

    Arguments:
        rf: trained Random Forest model
        inputs: input data as Pandas DataFrame
        targets: target data as Numpy array
        weights: Optional data weights as Numpy array
    '''
    return importances(rf, inputs, targets, features=inputs.columns, sample_weights=weights).reset_index()


def rf_rank_features(train_df:pd.DataFrame, val_df:pd.DataFrame, objective:str,
                     train_feats:List[str], targ_name:str='gen_target', wgt_name:Optional[str]=None,
                     importance_cut:float=0.0, n_estimators:int=40, n_rfs:int=1,
                     savename:Optional[str]=None, plot_settings:PlotSettings=PlotSettings()) -> List[str]:
    r'''
    Compute relative permutation importance of input features via using Random Forests.
    A reduced set of 'important features' is obtained by cutting on relative importance and a new model is trained and evaluated on this reduced set.
    RFs will have their hyper-parameters roughly optimised, both when training on all features and once when training on important features.
    Relative importances may be computed multiple times (via n_rfs) and averaged. In which case the standard error is also computed.

    Arguments:
        train_df: training data as Pandas DataFrame
        val_df: validation data as Pandas DataFrame
        objective: string representation of objective: either 'classification' or 'regression'
        train_feats: complete list of training features
        targ_name: name of column containing target data
        wgt_name: name of column containing weight data. If set, will use weights for training and evaluation, otherwise will not
        importance_cut: minimum importance required to be considered an 'important feature'
        n_estimators: number of trees to use in each forest
        n_rfs: number of trainings to perform on all training features in order to compute importances
        savename: Optional name of file to which to save the plot of feature importances
        plot_settings: :class:PlotSettings class to control figure appearance

    Returns:
        List of features passing importance_cut, ordered by importance
    '''

    w_trn = None if wgt_name is None else train_df[wgt_name]
    w_val = None if wgt_name is None else val_df[wgt_name]
    print("Optimising RF")
    opt_params, rf = get_opt_rf_params(train_df[train_feats], train_df[targ_name], val_df[train_feats], val_df[targ_name],
                                       objective, w_trn=w_trn, w_val=w_val, verbose=False)
    print("Evalualting importances")
    fi = get_rf_feat_importance(rf, train_df[train_feats], train_df[targ_name], w_trn)
    orig_score = rf.score(val_df[train_feats], val_df[targ_name], w_val)
    if n_rfs > 1:
        m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
        for _ in progress_bar(range(n_rfs-1)):
            rf = m(**opt_params)
            rf.fit(train_df[train_feats], train_df[targ_name], w_trn)
            fi = pd.merge(fi, get_rf_feat_importance(rf, train_df[train_feats], train_df[targ_name], w_trn), on='Feature', how='left')
            orig_score += rf.score(val_df[train_feats], val_df[targ_name], w_val)
        fi['Importance']  = np.mean(fi[[f for f in fi.columns if 'Importance' in f]].values, axis=1)
        fi['Uncertainty'] = np.std(fi[[f for f in fi.columns if 'Importance' in f]].values, ddof=1, axis=1)/np.sqrt(n_rfs)
        orig_score /= n_rfs
        fi.sort_values(by='Importance', ascending=False, inplace=True)
    print("Top ten most important features:\n", fi[['Feature', 'Importance']][:min(len(fi), 10)])
    plot_importance(fi[:min(len(fi), 30)], savename=savename, settings=plot_settings)

    top_feats = list(fi[fi.Importance > importance_cut].Feature)
    print(f"\n{len(top_feats)} features found with importance greater than {importance_cut}:\n", top_feats, '\n')
    if len(top_feats) == 0:
        print(f"Model score: :\t{orig_score:.5f}")
        print('No features found to be important, returning all training features. Good luck.')
        return train_feats
    if len(top_feats) < len(train_feats): 
        print("\nOptimising new RF")
        _, rf_new = get_opt_rf_params(train_df[top_feats], train_df[targ_name], val_df[top_feats], val_df[targ_name],
                                      objective, w_trn=w_trn, w_val=w_val, n_estimators=n_estimators, verbose=False)  
        print("Comparing RF scores, higher = better")                           
        print(f"All features:\t{orig_score:.5f}")
        print(f"Top features:\t{rf_new.score(val_df[top_feats], val_df[targ_name], w_val):.5f}")
    else:
        print('All training features found to be important')
    return top_feats
