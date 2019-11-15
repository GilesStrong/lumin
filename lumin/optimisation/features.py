import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from fastprogress import progress_bar
from rfpimp import importances
from prettytable import PrettyTable
import timeit
from collections import OrderedDict, defaultdict
from rfpimp import feature_dependence_matrix, plot_dependence_heatmap
import multiprocessing as mp

from sklearn.ensemble.forest import ForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .hyper_param import get_opt_rf_params
from ..plotting.interpretation import plot_importance
from ..plotting.plot_settings import PlotSettings
from ..plotting.data_viewing import plot_rank_order_dendrogram
from ..utils.statistics import uncert_round
from ..utils.misc import subsample_df
from ..utils.multiprocessing import mp_run

__all__ = ['get_rf_feat_importance', 'rf_rank_features', 'rf_check_feat_removal', 'repeated_rf_rank_features', 'auto_filter_on_linear_correlation',
           'auto_filter_on_mutual_dependence']


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
                     importance_cut:float=0.0, n_estimators:int=40, rf_params:Optional[Dict[str,Any]]=None, optimise_rf:bool=True,
                     n_rfs:int=1, n_max_display:int=30,
                     plot_results:bool=True, retrain_on_import_feats:bool=True, verbose:bool=True,
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
        rf_params: optional dictionary of keyword parameters for SK-Learn Random Forests
            Or ordered dictionary mapping parameters to optimise to list of values to consider
            If None and will optimise parameters using :meth:`lumin.optimisation.hyper_param.get_opt_rf_params`
        optimise_rf: if true will optimise RF params, passing `rf_params` to :meth:`~lumin.optimisation.hyper_param.get_opt_rf_params`
        n_rfs: number of trainings to perform on all training features in order to compute importances
        n_max_display: maximum number of features to display in importance plot
        plot_results: whether to plot the feature importances
        retrain_on_import_feats: whether to train a new model on important features to compare to full model
        verbose: whether to report results and progress
        savename: Optional name of file to which to save the plot of feature importances
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        List of features passing importance_cut, ordered by decreasing importance
    '''

    w_trn = None if wgt_name is None else train_df[wgt_name]
    w_val = None if wgt_name is None else val_df[wgt_name]

    if rf_params is None or optimise_rf is True:
        if verbose: print("Optimising RF parameters")
        rfp, rf = get_opt_rf_params(train_df[train_feats], train_df[targ_name], val_df[train_feats], val_df[targ_name],
                                    objective, w_trn=w_trn, w_val=w_val, n_estimators=n_estimators, params=rf_params, verbose=False)
    else:
        rfp = rf_params
        rfp['n_estimators'] = n_estimators
        m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
        rf = m(**rfp)
        rf.fit(train_df[train_feats], train_df[targ_name], w_trn)
    
    if verbose: print("Evalualting importances")
    fi = get_rf_feat_importance(rf, train_df[train_feats], train_df[targ_name], w_trn)
    orig_score = [rf.score(val_df[train_feats], val_df[targ_name], w_val)]
    if n_rfs > 1:
        m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
        for _ in progress_bar(range(n_rfs-1)):
            rf = m(**rfp)
            rf.fit(train_df[train_feats], train_df[targ_name], w_trn)
            fi = pd.merge(fi, get_rf_feat_importance(rf, train_df[train_feats], train_df[targ_name], w_trn), on='Feature', how='left')
            orig_score.append(rf.score(val_df[train_feats], val_df[targ_name], w_val))
        fi['Importance']  = np.mean(fi[[f for f in fi.columns if 'Importance' in f]].values, axis=1)
        fi['Uncertainty'] = np.std(fi[[f for f in fi.columns if 'Importance' in f]].values, ddof=1, axis=1)/np.sqrt(n_rfs)
        fi.sort_values(by='Importance', ascending=False, inplace=True)
    orig_score = uncert_round(np.mean(orig_score), np.std(orig_score, ddof=1))
    if verbose: print("Top ten most important features:\n", fi[['Feature', 'Importance']][:min(len(fi), 10)])
    if plot_results: plot_importance(fi[:min(len(fi), n_max_display)], threshold=importance_cut, savename=savename, settings=plot_settings)

    top_feats = list(fi[fi.Importance >= importance_cut].Feature)
    if verbose: print(f"\n{len(top_feats)} features found with importance greater than {importance_cut}:\n", top_feats, '\n')
    if len(top_feats) == 0:
        if verbose: print(f"Model score: :\t{orig_score[0]}±{orig_score[1]}")
        print('No features found to be important, returning all training features. Good luck.')
        return train_feats

    if retrain_on_import_feats:
        if len(top_feats) < len(train_feats):
            new_score = []
            if rf_params is None or optimise_rf is True:
                if verbose: print("Optimising new RF")
                rfp, rf_new = get_opt_rf_params(train_df[top_feats], train_df[targ_name], val_df[top_feats], val_df[targ_name],
                                                objective, w_trn=w_trn, w_val=w_val, n_estimators=n_estimators, params=rf_params, verbose=False)
                new_score.append(rf_new.score(val_df[top_feats], val_df[targ_name], w_val))
            else:
                rfp = rf_params
                rfp['n_estimators'] = n_estimators
            while len(new_score) < n_rfs:
                rf_new = m(**rfp)
                rf_new.fit(train_df[top_feats], train_df[targ_name], w_trn)
                new_score.append(rf_new.score(val_df[top_feats], val_df[targ_name], w_val))
            new_score = uncert_round(np.mean(new_score), np.std(new_score, ddof=1))
            print("Comparing RF scores, higher = better")                           
            print(f"All features:\t\t{orig_score[0]}±{orig_score[1]}")
            print(f"Important features:\t{new_score[0]}±{new_score[1]}")
        else:
            print('All training features found to be important')
    return top_feats


def rf_check_feat_removal(train_df:pd.DataFrame, objective:str,
                          train_feats:List[str], check_feats:List[str],
                          targ_name:str='gen_target', wgt_name:Optional[str]=None,
                          val_df:Optional[pd.DataFrame]=None,
                          subsample_rate:Optional[float]=None, strat_key:Optional[str]=None,
                          n_estimators:int=40, n_rfs:int=1, rf_params:Optional[Dict[str,Any]]=None) -> Dict[str,float]:
    r'''
    Checks whether features can be removed from the set of training features without degrading model performance using Random Forests
    Computes scores for model with all training features then for each feature listed in `check_feats` computes scores for a model trained on all training
    features except that feature
    E.g. if two features are highly correlated this function could be used to check whether one of them could be removed.
    
    Arguments:
        train_df: training data as Pandas DataFrame
        objective: string representation of objective: either 'classification' or 'regression'
        train_feats: complete list of training features
        check_feats: list of features to try removing
        targ_name: name of column containing target data
        wgt_name: name of column containing weight data. If set, will use weights for training and evaluation, otherwise will not
        val_df: optional validation data as Pandas DataFrame.
            If set will compute validation scores in addition to Out Of Bag scores
            And will optimise RF parameters if `rf_params` is None
        subsample_rate: if set, will subsample the training data to the provided fraction. Subsample is repeated per Random Forest training
        strat_key: column name to use for stratified subsampling, if desired
        n_estimators: number of trees to use in each forest
        n_rfs: number of trainings to perform on all training features in order to compute importances
        rf_params: optional dictionary of keyword parameters for SK-Learn Random Forests
            If None and val_df is None will use default parameters of 'min_samples_leaf':3, 'max_features':0.5
            Elif None and val_df is not None will optimise parameters using :meth:`lumin.optimisation.hyper_param.get_opt_rf_params`
            
    Returns:
        Dictionary of results
    '''

    for f in check_feats: assert f in train_feats, f"{f} not found in train_feats"
    
    if rf_params is None:
        if val_df is None:
            rf_params = {'min_samples_leaf':3, 'max_features':0.5, 'n_estimators':n_estimators}
            print('Using following default RF parameters:', rf_params)
        else:
            print('Optimising RF parameters')
            if subsample_rate is not None:
                tmp_trn = subsample_df(train_df, objective, targ_name, n_samples=int(subsample_rate*len(train_df)), strat_key=strat_key, wgt_name=wgt_name)
            else:
                tmp_trn = train_df
            rf_params, _ = get_opt_rf_params(tmp_trn[train_feats], tmp_trn[targ_name], val_df[train_feats], val_df[targ_name], objective,
                                             w_trn=None if wgt_name is None else tmp_trn[wgt_name],
                                             w_val=None if wgt_name is None else val_df[wgt_name], n_estimators=n_estimators, verbose=False)
    else:
        rf_params['n_estimators'] = n_estimators
    rf_params['n_jobs']    = -1
    rf_params['oob_score'] = True
            
    m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
    pt = PrettyTable(['Removed', 'OOB Score', 'Val Score'])
    oob,val = defaultdict(list),defaultdict(list)
    for _ in range(n_rfs):
        if subsample_rate is not None:
            tmp_trn = subsample_df(train_df, objective, targ_name, n_samples=int(subsample_rate*len(train_df)), strat_key=strat_key, wgt_name=wgt_name)
        else:
            tmp_trn = train_df
        for remove in ['None']+check_feats:
            feats = train_feats if remove == 'None' else [f for f in train_feats if f != remove]
            rf = m(**rf_params)
            rf.fit(tmp_trn[feats], tmp_trn[targ_name], sample_weight=None if wgt_name is None else tmp_trn[wgt_name])
            oob[remove].append(rf.oob_score_)
            if val_df is not None: val[remove].append(rf.score(val_df[feats], val_df[targ_name], None if wgt_name is None else val_df[wgt_name]))

    results = {}
    for remove in ['None']+check_feats:
        oob_score, oob_unc = np.mean(oob[remove]), np.std(oob[remove], ddof=1)/np.sqrt(n_rfs)
        results[f'{remove}_oob_score'] = oob_score
        results[f'{remove}_oob_unc']   = oob_unc
        oob_round = uncert_round(oob_score, oob_unc)
        if val_df is not None:
            val_score, val_unc = np.mean(val[remove]), np.std(val[remove], ddof=1)/np.sqrt(n_rfs)
            results[f'{remove}_val_score'] = val_score
            results[f'{remove}_val_unc']   = val_unc
            val_round = uncert_round(val_score, val_unc)
        else:
            val_round = ['-','-']
            
        pt.add_row([remove, f'{oob_round[0]}±{oob_round[1]}', f'{val_round[0]}±{val_round[1]}'])
        
    print(pt)
    return results


def repeated_rf_rank_features(train_df:pd.DataFrame, val_df:pd.DataFrame, n_reps:int, min_frac_import:float, objective:str, train_feats:List[str],
                              targ_name:str='gen_target', wgt_name:Optional[str]=None, strat_key:Optional[str]=None, subsample_rate:Optional[float]=None,
                              resample_val:bool=True, importance_cut:float=0.0, n_estimators:int=40, rf_params:Optional[Dict[str,Any]]=None,
                              optimise_rf:bool=True, n_rfs:int=1, n_max_display:int=30, n_threads:int=1,
                              savename:Optional[str]=None, plot_settings:PlotSettings=PlotSettings()) -> Tuple[List[str],pd.DataFrame]:
    r'''
    Runs :meth:`~lumin.optimisation.features.rf_rank_features` multiple times on bootstrap resamples of training data and computes the fraction of times each
    feature passes the importance cut.
    Then returns a list features which are have a fractional selection as important great than some number.
    I.e. in cases where :meth:`~lumin.optimisation.features.rf_rank_features` can be unstable (list of important features changes each run), this method can be 
    used to help stabailse the list of important features
    
    Arguments:
        train_df: training data as Pandas DataFrame
        val_df: validation data as Pandas DataFrame
        n_reps: number of times to resample and run :meth:`~lumin.optimisation.features.rf_rank_features`
        min_frac_import: minimum fraction of times feature must be selected as important by :meth:`~lumin.optimisation.features.rf_rank_features` in order to be
            considered generally important
        objective: string representation of objective: either 'classification' or 'regression'
        train_feats: complete list of training features
        targ_name: name of column containing target data
        wgt_name: name of column containing weight data. If set, will use weights for training and evaluation, otherwise will not
        strat_key: name of column to use to stratify data when resampling
        subsample_rate: if set, will subsample the training data to the provided fraction. Subsample is repeated per Random Forest training
        resample_val: whether to also resample the validation set, or use the original set for all evaluations
        importance_cut: minimum importance required to be considered an 'important feature'
        n_estimators: number of trees to use in each forest
        rf_params: optional dictionary of keyword parameters for SK-Learn Random Forests
            Or ordered dictionary mapping parameters to optimise to list of values to consider
            If None and will optimise parameters using :meth:`lumin.optimisation.hyper_param.get_opt_rf_params`
        optimise_rf: if true will optimise RF params, passing `rf_params` to :meth:`~lumin.optimisation.hyper_param.get_opt_rf_params`
        n_rfs: number of trainings to perform on all training features in order to compute importances
        n_max_display: maximum number of features to display in importance plot
        n_threads: number of rankings to run simultaneously
        savename: Optional name of file to which to save the plot of feature importances
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        - List of features with fractional selection greater than min_frac_import, ordered by decreasing fractional selection
        - DataFrame of number of selections and fractional selections for all features
    '''

    def _mp_rank(args:Dict[str,Any], out_q:mp.Queue) -> None:
        import_feats = rf_rank_features(args['tmp_trn'], args['tmp_val'], objective=objective, train_feats=train_feats,
                                        importance_cut=importance_cut, targ_name=targ_name, n_rfs=n_rfs, wgt_name=wgt_name,
                                        rf_params=rf_params, optimise_rf=optimise_rf, n_estimators=n_estimators, plot_results=False,
                                        retrain_on_import_feats=False, verbose=False)
        out_q.put({args['name']:import_feats})

    def _resample() -> Tuple[pd.DataFrame,pd.DataFrame]:
        tmp_trn = subsample_df(train_df, objective, targ_name, n_samples=int(subsample_rate*len(train_df)) if subsample_rate is not None else None,
                               replace=True, strat_key=strat_key, wgt_name=wgt_name)
        tmp_val = val_df if not resample_val else subsample_df(val_df, objective, targ_name, replace=True, strat_key=strat_key, wgt_name=wgt_name)
        return tmp_trn, tmp_val

    def _get_score(feats:List[str]) -> Tuple[float,float]:
        score = []
        if rf_params is None or optimise_rf is True:
            tmp_trn, tmp_val = _resample()
            w_trn = None if wgt_name is None else tmp_trn[wgt_name]
            w_val = None if wgt_name is None else tmp_val[wgt_name]
            rfp, rf = get_opt_rf_params(tmp_trn[feats], tmp_trn[targ_name], tmp_val[feats], tmp_val[targ_name],
                                        objective, w_trn=w_trn, w_val=w_val, n_estimators=n_estimators, params=rf_params, verbose=False)
            score.append(rf.score(tmp_val[feats], tmp_val[targ_name], w_val))
        else:
            rfp = rf_params
            rfp['n_estimators'] = n_estimators
        while len(score) < n_rfs:
            tmp_trn, tmp_val = _resample()
            w_trn = None if wgt_name is None else tmp_trn[wgt_name]
            w_val = None if wgt_name is None else tmp_val[wgt_name]
            m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
            rf = m(**rfp)
            rf.fit(tmp_trn[feats], tmp_trn[targ_name], w_trn)
            score.append(rf.score(tmp_val[feats], tmp_val[targ_name], w_val))
        return  uncert_round(np.mean(score), np.std(score, ddof=1))
    
    selections = pd.DataFrame({'Feature':train_feats, 'N_Selections':0})
    if n_threads == 1:
        for i in progress_bar(range(n_reps)):
            print(f'Repition {i}')
            tmp_trn, tmp_val = _resample()
            import_feats = rf_rank_features(tmp_trn, tmp_val, objective=objective, train_feats=train_feats,
                                            importance_cut=importance_cut, targ_name=targ_name, n_rfs=n_rfs, wgt_name=wgt_name,
                                            rf_params=rf_params, optimise_rf=optimise_rf, n_estimators=n_estimators, plot_results=False,
                                            retrain_on_import_feats=False, verbose=False)
            for f in import_feats: selections.loc[selections.Feature == f, 'N_Selections'] += 1
    
    else:
        for i in progress_bar(range(0,n_reps,n_threads)):
            args = []
            for j in range(n_threads):
                tmp_trn, tmp_val = _resample()
                args.append({'name':f'{i}_{j}', 'tmp_trn':tmp_trn, 'tmp_val': tmp_val})
            res = mp_run(args, _mp_rank)
            for r in res:
                for f in res[r]: selections.loc[selections.Feature == f, 'N_Selections'] += 1                               
            
    selections['Fractional_Selection'] = selections.N_Selections/n_reps
    selections.sort_values(by='Fractional_Selection', ascending=False, inplace=True)
    plot_importance(selections[:min(len(selections), n_max_display)], imp_name='Fractional_Selection',
                    x_lbl='Fraction of times important', threshold=min_frac_import, savename=savename, settings=plot_settings)
    top_feats = list(selections[selections.Fractional_Selection >= min_frac_import].Feature)
    print(f"\n{len(top_feats)} features found with fractional selection greater than {min_frac_import}:\n", top_feats, '\n')
    if len(top_feats) == 0:
        print('No features found to pass minimum fractional selection threshold, returning all training features. Good luck.')
        return train_feats, selections

    if len(top_feats) < len(train_feats):
        old_score, new_score = _get_score(train_feats), _get_score(top_feats)
        print("Comparing RF scores, higher = better")                           
        print(f"All features:\t\t{old_score[0]}±{old_score[1]}")
        print(f"Selected features:\t{new_score[0]}±{new_score[1]}")
    else:
        print('All training features found to be important')
    return top_feats, selections


def auto_filter_on_linear_correlation(train_df:pd.DataFrame, val_df:pd.DataFrame, check_feats:List[str], objective:str, targ_name:str,
                                      strat_key:Optional[str]=None, wgt_name:Optional[str]=None, corr_threshold:float=0.8,
                                      n_estimators:int=40, rf_params:Optional[Union[Dict,OrderedDict]]=None, optimise_rf:bool=True, n_rfs:int=5,
                                      subsample_rate:Optional[float]=None, savename:Optional[str]=None, plot_settings:PlotSettings=PlotSettings()) -> List[str]:

    r'''
    Filters a list of possible training features by identifying pairs of linearly correlated features and then attempting to remove either feature from each
    pair by checking whether doing so would not decrease the performance Random Forests trained to perform classification or regression.

    Linearly correlated features are identified by computing Spearman's rank-order correlation coefficients for every pair of features. Hierachical clustering
    is then used to group features. Pairs with a correlation coefficient greater than a set threshold are candidates for removal.
    Candidate pairs are tested, in order of decreasing correlation, by computing the mean performance of a Random Forests trained on: all remaining training
    features; all remaining training features except the first feature in the pair; and all remaining training features except the second feature in the pair.
    If the RF trained on all remaining features consistently outperforms the other two trainings, then neither feature from the pair is removed, otherwise the
    feature whose removal causes the largest mean increase in performance is removed.

    Since multiple features maybe correlated with one-another, but this function examines paris of features, it might be necessary/desirable to rerun it on the
    the previous results.

    Since this function involves training many models, it can be slow on large datasets. In such cases one can use the `subsample_rate` argument to sample
    randomly a fraction of the whole dataset (with optionaly stratification). Resampling is performed prior to each RF training for maximum genralisation, and
    any weights in the data are automatically renormalised to the original weight sum (within each class).

    .. Attention:: This function combines :meth:`~lumin.plotting.data_viewing.plot_rank_order_dendrogram` with
        :meth:`~lumin.optimisation.features.rf_check_feat_removal`. This is purely for convenience and should not be treated as a 'black box'. We encourage users to
        convince themselves that it is really is reasonable to remove the features which are identified as redundant.

    Arguments:
        train_df: training data as Pandas DataFrame
        val_df: validation data as Pandas DataFrame
        check_feats: complete list of features to consider for training and removal
        objective: string representation of objective: either 'classification' or 'regression'
        targ_name: name of column containing target data
        strat_key: name of column to use to stratify data when resampling
        wgt_name: name of column containing weight data. If set, will use weights for training and evaluation, otherwise will not
        corr_threshold: minimum threshold on Spearman's rank-order correlation coefficient for pairs to be considered 'correlated'
        n_estimators: number of trees to use in each forest
        rf_params: either: a dictionare of keyword hyper-parameters to use for the Random Forests, if optimse_rf is False;
            or an `OrderedDict` of a range of hyper-parameters to test during optimisation. See :meth:`~lumin.optimisation.hyper_param.get_opt_rf_params` for
            more details.
        optimise_rf: whether to optimise the Random Forest hyper-parameters for the (sub-sambled) dataset
        n_rfs: number of trainings to perform during each perfromance impact test
        subsample_rate: float between 0 and 1. If set will subsample the trainng data to the requested fraction
        savename: Optional name of file to which to save the first plot of feature clustering
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        Filtered list of training features
    '''

    tmr = timeit.default_timer()
    # Get pairs of linearly correlated features clustered by Spearman's rank-order correlation coefficient
    print("Computing Spearman's rank-order correlation coefficients")
    pairs = plot_rank_order_dendrogram(train_df[check_feats], threshold=corr_threshold, savename=savename, settings=plot_settings)
    if len(pairs) == 0:
        print(f'No pairs of features found to pass correlation threshold of {corr_threshold}')
        return check_feats
    else:
        print(f'{len(pairs)} pairs of features found to pass correlation threshold of {corr_threshold}:')
        print(pairs)
    
    if optimise_rf:  # Roughly optimise a Random Forest for the (subsampled) data
        print("\nOptimising RF")
        tmp_trn = subsample_df(train_df, objective=objective, targ_name=targ_name, strat_key=strat_key, wgt_name=wgt_name,
                               n_samples=int(subsample_rate*len(train_df)) if subsample_rate is not None else None)
        rf_params, _ = get_opt_rf_params(tmp_trn[check_feats], tmp_trn[targ_name], val_df[check_feats], val_df[targ_name],
                                         w_trn=tmp_trn[wgt_name] if wgt_name is not None else wgt_name,
                                         w_val=val_df[wgt_name] if wgt_name is not None else wgt_name,
                                         objective=objective, n_estimators=n_estimators, verbose=True, params=rf_params)
        
    remove = []
    for p in progress_bar(pairs):  # Loop through paris of features check whether features can be removed
        print(f'\nChecking pair: {p}')
        res = rf_check_feat_removal(check_feats=p, train_df=train_df, objective=objective, train_feats=[f for f in check_feats if f not in remove],
                                    targ_name=targ_name, wgt_name=wgt_name, val_df=val_df, n_rfs=n_rfs, subsample_rate=subsample_rate, strat_key=strat_key,
                                    rf_params=rf_params)
        arr = [['None', res['None_val_score']],
               [p[0], res[f'{p[0]}_val_score']],
               [p[1], res[f'{p[1]}_val_score']]]
        arr = sorted(arr, key=lambda x:x[1])
        drop = arr[-1][0]
        if drop != 'None':
            print(f'Dropping {drop}')
            remove.append(drop)
        elif arr[-2][1] == arr[-1][1]:
            drop = arr[-2][0]
            print(f'Dropping {drop}')
            remove.append(drop)
            
    filtered = [f for f in check_feats if f not in remove]
    print(f'\n{len(remove)} features removed from starting list of {len(check_feats)}, {len(filtered)} features remain')
    print("\nRecomputing Spearman's rank-order correlation coefficients on filtered features")
    pairs = plot_rank_order_dendrogram(train_df[filtered], threshold=corr_threshold, settings=plot_settings)
    if len(pairs) == 0:
        print(f'No pairs of features found to pass correlation threshold of {corr_threshold}')
    else:
        print(f'{len(pairs)} pairs of features still found to pass correlation threshold of {corr_threshold}.', 
              'You may wish to rerun this function on the filtered features.')

    def _get_score(feats:List[str]) -> Tuple[float,float]:
        score = []
        w_val = val_df[wgt_name] if wgt_name is not None else wgt_name
        m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
        while len(score) < n_rfs:
            tmp_trn = subsample_df(train_df, objective=objective, targ_name=targ_name, strat_key=strat_key, wgt_name=wgt_name,
                                   n_samples=int(subsample_rate*len(train_df)) if subsample_rate is not None else None)
            w_trn = None if wgt_name is None else tmp_trn[wgt_name]
            rf = m(**rf_params)
            rf.fit(tmp_trn[feats], tmp_trn[targ_name], w_trn)
            score.append(rf.score(val_df[feats], val_df[targ_name], w_val))
        return uncert_round(np.mean(score), np.std(score, ddof=1))
    
    if len(filtered) < len(check_feats):
        old_score, new_score = _get_score(check_feats), _get_score(filtered)
        print("\nComparing RF scores, higher = better")                           
        print(f"All features:\t\t{old_score[0]}±{old_score[1]}")
        print(f"Filtered features:\t{new_score[0]}±{new_score[1]}")
    
    print(f'\nFiltering took {timeit.default_timer()-tmr:.3f} seconds')
    return filtered


def auto_filter_on_mutual_dependence(train_df:pd.DataFrame, val_df:pd.DataFrame, check_feats:List[str], objective:str, targ_name:str,
                                     strat_key:Optional[str]=None, wgt_name:Optional[str]=None, md_threshold:float=0.8,
                                     n_estimators:int=40, rf_params:Optional[OrderedDict]=None, optimise_rf:bool=True, n_rfs:int=5,
                                     subsample_rate:Optional[float]=None, plot_settings:PlotSettings=PlotSettings()) -> List[str]:
    r'''
    Filters a list of possible training features via mutual dependence: By identifying features whose values can be accurately predicted using the other
    features. Features with a high 'dependence' are then checked to see whether removing them would not decrease the performance Random Forests trained to
    perform classification or regression. For best results, the features to check should be supplied in order to decreasing importance.

    Dependent features are identified by training Random Forest regressors on the other features. Features with a dependence greater than a set threshold are
    candidates for removal. Candidate features are tested, in order of increasing importance, by computing the mean performance of a Random Forests trained on:
    all remaining training features; and all remaining training features except the candidate feature.
    If the RF trained on all remaining features except the candidate feature consistently outperforms or matches the training which uses all remaining features,
    then the candidate feature is removed, otherwise the feature remains and is no longer tested.

    Since evaluating the mutual dependence via regression then allows the important features used by the regressor to be identified, it is possible to test
    multiple feature removals at once, provided a removal candidate is not important for predicting another removal candidate.

    Since this function involves training many models, it can be slow on large datasets. In such cases one can use the `subsample_rate` argument to sample
    randomly a fraction of the whole dataset (with optionaly stratification). Resampling is performed prior to each RF training for maximum genralisation, and
    any weights in the data are automatically renormalised to the original weight sum (within each class).

    .. Attention:: This function combines RFPImp's `feature_dependence_matrix` with :meth:`~lumin.optimisation.features.rf_check_feat_removal`.
        This is purely for convenience and should not be treated as a 'black box'. We encourage users to convince themselves that it is really is reasonable to
        remove the features which are identified as redundant.

    .. Note:: Technicalities related to RFPImp's use of SVG for plots mean that the mutual dependence plots can have low resolution when shown or saved.
        Therefore this function does not take a `savename` argument. Users wiching to save the plots as PNG or PDF should compute the dependence matrix themselves
        using `feature_dependence_matrix` and then plot using `plot_dependence_heatmap`, calling `.save([savename])` on the retunred object. The plotting backend
        might need to be set to SVG, using: `%config InlineBackend.figure_format = 'svg'`.

    Arguments:
        train_df: training data as Pandas DataFrame
        val_df: validation data as Pandas DataFrame
        check_feats: complete list of features to consider for training and removal
        objective: string representation of objective: either 'classification' or 'regression'
        targ_name: name of column containing target data
        strat_key: name of column to use to stratify data when resampling
        wgt_name: name of column containing weight data. If set, will use weights for training and evaluation, otherwise will not
        md_threshold: minimum threshold on the mutual dependence coefficient for a feature to be considered 'predictable'
        n_estimators: number of trees to use in each forest
        rf_params: either: a dictionare of keyword hyper-parameters to use for the Random Forests, if optimse_rf is False;
            or an `OrderedDict` of a range of hyper-parameters to test during optimisation. See :meth:`~lumin.optimisation.hyper_param.get_opt_rf_params` for
            more details.
        optimise_rf: whether to optimise the Random Forest hyper-parameters for the (sub-sambled) dataset
        n_rfs: number of trainings to perform during each perfromance impact test
        subsample_rate: float between 0 and 1. If set will subsample the trainng data to the requested fraction
        plot_settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        Filtered list of training features
    '''
    
    tmr = timeit.default_timer()
    remove,skip = [],[]    
    
    def _get_checks(remove:List[str], skip:List[str], plot:bool=False) -> List[str]:
        '''Compute dependencies and return list of features which can probably be safely tested simultaneously'''
        checks,predictors = [],[]
        dep = feature_dependence_matrix(train_df[[f for f in check_feats if f not in remove]], sort_by_dependence=False,
                                        rfmodel=RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, oob_score=True))
        if plot: plot_dependence_heatmap(dep, figsize=(plot_settings.h_large, plot_settings.h_large),
                                         label_fontsize=plot_settings.lbl_sz/2, value_fontsize=(plot_settings.lbl_sz-2)/2).view()
        print(f'\n{len(dep[dep.Dependence >= md_threshold])} predictable features found to pass mutual dependence threshold of {md_threshold}')
        for f,r in dep[dep.Dependence >= md_threshold][::-1].iterrows():
            if f in skip: continue
            if f in predictors: break  # Dependent feature already used to predict other features or lower importance
            checks.append(f)
            predictors += list(r[r >= 0.05].index.values[1:])  # Feature actuall important for regression
        return checks
    
    # Get initial dependencies
    print("Computing mutual dependencies")
    checks = _get_checks(remove, skip, True)
    if len(checks) == 0:
        print(f'No features found to pass mutual dependence threshold of {md_threshold}')
        return check_feats
    
    if optimise_rf:  # Roughly optimise a Random Forest for the (subsampled) data
        print("\nOptimising RF")
        tmp_trn = subsample_df(train_df, objective=objective, targ_name=targ_name,
                               n_samples=int(subsample_rate*len(train_df)) if subsample_rate is not None else None, strat_key=strat_key, wgt_name=wgt_name)
        rf_params, _ = get_opt_rf_params(tmp_trn[check_feats], tmp_trn[targ_name], val_df[check_feats], val_df[targ_name], objective=objective,
                                         w_trn=tmp_trn[wgt_name] if wgt_name is not None else wgt_name,
                                         w_val=val_df[wgt_name] if wgt_name is not None else wgt_name,
                                         n_estimators=n_estimators, verbose=True, params=rf_params)
        
    while len(checks) > 0:
        print(f'\nChecking {checks}')
        res = rf_check_feat_removal(check_feats=checks, train_df=train_df, objective=objective, train_feats=[f for f in check_feats if f not in remove],
                                    targ_name=targ_name, wgt_name=wgt_name, rf_params=rf_params, val_df=val_df, n_rfs=5, subsample_rate=subsample_rate,
                                    strat_key=strat_key)
        for c in checks:
            if res[f'{c}_val_score'] >= res['None_val_score']:
                print(f'Dropping {c}')
                remove.append(c)
            else:
                skip.append(c)
        checks = _get_checks(remove, skip)
    print('\nAll checks completed')
    
    filtered = [f for f in check_feats if f not in remove]
    print(f'{len(remove)} features removed from starting list of {len(check_feats)}, {len(filtered)} features remain')
    
    print("Recomputing mutual dependencies")
    _get_checks(remove, skip, True)

    def _get_score(feats:List[str]) -> Tuple[float,float]:
        score = []
        w_val = val_df[wgt_name] if wgt_name is not None else wgt_name
        m = RandomForestClassifier if 'class' in objective.lower() else RandomForestRegressor
        while len(score) < n_rfs:
            tmp_trn = subsample_df(train_df, objective=objective, targ_name=targ_name, strat_key=strat_key, wgt_name=wgt_name,
                                   n_samples=int(subsample_rate*len(train_df)) if subsample_rate is not None else None)
            w_trn = None if wgt_name is None else tmp_trn[wgt_name]
            rf = m(**rf_params)
            rf.fit(tmp_trn[feats], tmp_trn[targ_name], w_trn)
            score.append(rf.score(val_df[feats], val_df[targ_name], w_val))
        return uncert_round(np.mean(score), np.std(score, ddof=1))

    if len(filtered) < len(check_feats):
        old_score, new_score = _get_score(check_feats), _get_score(filtered)
        print("Comparing RF scores, higher = better")                           
        print(f"All features:\t\t{old_score[0]}±{old_score[1]}")
        print(f"Filtered features:\t{new_score[0]}±{new_score[1]}")
    
    print(f'\nFiltering took {timeit.default_timer()-tmr:.3f} seconds')
    return filtered
