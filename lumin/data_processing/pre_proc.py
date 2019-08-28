import pandas as pd
from typing import List, Optional, Tuple, Union
import pickle
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

__all__ = ['get_pre_proc_pipes', 'fit_input_pipe', 'fit_output_pipe', 'proc_cats']


def get_pre_proc_pipes(norm_in:bool=True, norm_out:bool=False, pca:bool=False, whiten:bool=False,
                       with_mean:bool=True, with_std:bool=True, n_components:Optional[int]=None) -> Tuple[Pipeline,Pipeline]:
    r'''
    Configure SKLearn Pipelines for processing inputs and targets with the requested transformations.

    Arguments:
        norm_in: whether to apply StandardScaler to inputs
        norm_out: whether to apply StandardScaler to outputs
        pca: whether to apply PCA to inputs. Perforemed prior to StandardScaler. No dimensionality reduction is applied, purely rotation.
        whiten: whether PCA should whiten inputs.
        with_mean: whether StandardScalers should shift means to 0
        with_std: whether StandardScalers should scale standard deviations to 1
        n_components: if set, causes PCA to reduce the dimensionality of the input data

    Returns:
        Pipeline for input data
        Pipeline for target data
    '''

    steps_in = []
    if not norm_in and not pca:
        steps_in.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    else:
        if pca: steps_in.append(('pca', PCA(n_components=n_components, whiten=whiten)))
        if norm_in: steps_in.append(('norm_in', StandardScaler(with_mean=with_mean, with_std=with_std)))
    input_pipe = Pipeline(steps_in)

    steps_out = []
    if norm_out: steps_out.append(('norm_out', StandardScaler(with_mean=with_mean, with_std=with_std)))
    else:        steps_out.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    output_pipe = Pipeline(steps_out)
    return input_pipe, output_pipe


def fit_input_pipe(df:pd.DataFrame, cont_feats:Union[str,List[str]], savename:Optional[str]=None, input_pipe:Optional[Pipeline]=None,
                   norm_in:bool=True, pca:bool=False, whiten:bool=False,
                   with_mean:bool=True, with_std:bool=True, n_components:Optional[int]=None) -> Pipeline:
    r'''
    Fit input pipeline to continuous features and optionally save.
    
    Arguments:
        df: DataFrame with data to fit pipeline
        cont_feats: (list of) column(s) to use as input data for fitting
        savename: if set will save the fitted Pipeline to with that name as Pickle (.pkl extension added automatically)
        input_pipe: if set will fit, otherwise will instantiate a new Pipeline
        norm_in: whether to apply StandardScaler to inputs. Only used if input_pipe is not set.
        pca: whether to apply PCA to inputs. Perforemed prior to StandardScaler.
             No dimensionality reduction is applied, purely rotation. Only used if input_pipe is not set.
        whiten: whether PCA should whiten inputs. Only used if input_pipe is not set.
        with_mean: whether StandardScalers should shift means to 0. Only used if input_pipe is not set.
        with_std: whether StandardScalers should scale standard deviations to 1. Only used if input_pipe is not set.
        n_components: if set, causes PCA to reduce the dimensionality of the input data. Only used if input_pipe is not set.

    Returns:
        Fitted Pipeline
    '''
    
    if input_pipe is None: input_pipe, _ = get_pre_proc_pipes(norm_in=norm_in, pca=pca, whiten=whiten,
                                                              with_mean=with_mean, with_std=with_std, n_components=n_components)
    input_pipe.fit(df[cont_feats].values.astype('float32'))
    if savename is not None:
        with open(f'{savename}.pkl', 'wb') as fout: pickle.dump(input_pipe, fout)
    return input_pipe


def fit_output_pipe(df:pd.DataFrame, targ_feats:Union[str,List[str]], savename:Optional[str]=None,
                    output_pipe:Optional[Pipeline]=None, norm_out:bool=True) -> Pipeline:
    r'''
    Fit output pipeline to target features and optionally save. Have you thought about using a y_range for regression instead?
    
    Arguments:
        df: DataFrame with data to fit pipeline
        targ_feats: (list of) column(s) to use as input data for fitting
        savename: if set will save the fitted Pipeline to with that name as Pickle (.pkl extension added automatically)
        output_pipe: if set will fit, otherwise will instantiate a new Pipeline
        norm_out: whether to apply StandardScaler to outputs . Only used if output_pipe is not set.

    Returns:
        Fitted Pipeline
    '''

    if output_pipe is None: _, output_pipe = get_pre_proc_pipes(norm_out=True)
    output_pipe.fit(df[targ_feats].values.astype('float32'))
    if savename is not None:
        with open(f'{savename}.pkl', 'wb') as fout: pickle.dump(output_pipe, fout)
    return output_pipe


def proc_cats(train_df:pd.DataFrame, cat_feats:List[str],
              val_df:Optional[pd.DataFrame]=None, test_df:Optional[pd.DataFrame]=None) -> Tuple[OrderedDict,OrderedDict]:
    r'''
    Process categorical features in train_df to be valued 0->cardinality-1. Applied inplace.
    Applies same transformation to validation and testing data is passed.
    Will complain if validation or testing sets contain categories which are not present in the training data.
    
    Arguments:
        train_df: DataFrame with the training data, which will also be used to specify all the categories to consider
        cat_feats: list of columns to use as categorical features
        val_df: if set will apply the same category to code mapping to the validation data as was performed on the training data
        test_df: if set will apply the same category to code mapping to the testing data as was performed on the training data
    
    Returns:
        ordered dictionary mapping categorical features to dictionaries mapping categories to codes
        ordered dictionary mapping categorical features to their cardinalities
    '''

    # TODO: check how this handles non-numerical categories

    cat_maps = OrderedDict()
    cat_szs = OrderedDict()
    for feat in cat_feats:
        cat_maps[feat] = {}
        vals = sorted(set(train_df[feat]))
        cat_szs[feat] = len(vals)
        if val_df is not None:
            if sorted(set(val_df[feat])) != vals:
                raise Exception(f"Feature {feat} declared categorical, but validation set contains categories different to the training set")
        if test_df is not None:
            if sorted(set(val_df[feat])) != vals:
                raise Exception(f"Feature {feat} declared categorical, but testing set contains categories different to the training set")
        
        for i, val in enumerate(vals):
            train_df.loc[train_df[feat] == val, feat] = i
            if val_df is not None: val_df.loc[val_df[feat] == val, feat] = i
            if test_df is not None: test_df.loc[test_df[feat] == val, feat] = i
            cat_maps[feat][i] = val
    return cat_maps, cat_szs
