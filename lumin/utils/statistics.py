import numpy as np
from typing import Tuple, Dict, Optional, Any, Union
import multiprocessing as mp
import math

from statsmodels.nonparametric.kde import KDEUnivariate

__all__ = ['bootstrap_stats', 'get_moments', 'uncert_round']


def bootstrap_stats(args:Dict[str,Any], out_q:Optional[mp.Queue]=None) -> Union[None,Dict[str,Any]]:
    r'''
    Computes statistics and KDEs of data via sampling with replacement

    Arguments:
        args: dictionary of arguments. Possible keys are:
            data - data to resample
            name - name prepended to returned keys in result dict
            weights - array of weights matching length of data to use for weighted resampling
            n - number of times to resample data
            x - points at which to compute the kde values of resample data
            kde - whether to compute the kde values at x-points for resampled data
            mean - whether to compute the means of the resampled data
            std - whether to compute standard deviation of resampled data
            c68 - whether to compute the width of the absolute central 68.2 percentile of the resampled data

        out_q: if using multiporcessing can place result dictionary in provided queue

    Returns:
        Result dictionary if `out_q` is `None` else `None`.
    '''

    out_dict, mean, std, c68, boot = {}, [], [], [], []
    name    = ''   if 'name'    not in args else args['name']
    weights = None if 'weights' not in args else args['weights']
    if 'n'    not in args: args['n']    = 100
    if 'kde'  not in args: args['kde']  = False
    if 'mean' not in args: args['mean'] = False
    if 'std'  not in args: args['std']  = False  
    if 'c68'  not in args: args['c68']  = False
    if args['kde'] and args['data'].dtype != 'float64': data = np.array(args['data'], dtype='float64')
    else:                                               data = args['data']
    len_d = len(data)

    np.random.seed()
    for i in range(args['n']):
        points = np.random.choice(data, len_d, replace=True, p=weights)
        if args['kde']:
            kde = KDEUnivariate(points)
            kde.fit()
            boot.append([kde.evaluate(x) for x in args['x']])
        if args['mean']: mean.append(np.mean(points))
        if args['std']:  std.append(np.std(points, ddof=1))
        if args['c68']:  c68.append(np.percentile(np.abs(points), 68.2))

    if args['kde']:  out_dict[f'{name}_kde']  = boot
    if args['mean']: out_dict[f'{name}_mean'] = mean
    if args['std']:  out_dict[f'{name}_std']  = std
    if args['c68']:  out_dict[f'{name}_c68']  = c68
    if out_q is not None: out_q.put(out_dict)
    else: return out_dict


def get_moments(arr:np.ndarray) -> Tuple[float,float,float,float]:
    r'''
    Computes mean and std of data, and their associated uncertainties

    Arguments:
        arr: univariate data

    Returns:
        - mean
        - statistical uncertainty of mean
        - standard deviation
        - statistical uncertainty of standard deviation
    '''

    n = len(arr)
    m = np.mean(arr)
    m_4 = np.mean((arr-m)**4)
    s = np.std(arr, ddof=1)
    s4 = s**4
    se_s2 = ((m_4-(s4*(n-3)/(n-1)))/n)**0.25
    se_s = se_s2/(2*s)
    return m, s/np.sqrt(n), s, se_s


def uncert_round(value:float, uncert:float) -> Tuple[float,float]:
    r'''
    Round value according to given uncertainty using one significant figure of the uncertainty

    Arguments:
        value: value to round
        uncert: uncertainty of value

    Returns:
        - rounded value
        - rounded uncertainty
    '''

    if uncert == math.inf: return value, uncert
    uncert = np.nan_to_num(uncert)
    if uncert == 0: return value, uncert
    
    factor = 1.0
    while uncert / factor > 1: factor *= 10.0
    value /= factor
    uncert /= factor   
    i = 0    
    while uncert * (10**i) <= 1: i += 1
    
    round_uncert = factor*round(uncert, i)
    round_value  = factor*round(value, i)
    if int(round_uncert) == round_uncert:
        round_uncert = int(round_uncert)
        round_value  = int(round_value)
    return round_value, round_uncert
