import numpy as np
from typing import Tuple, Dict, Optional, Any
import multiprocessing as mp
import math

from statsmodels.nonparametric.kde import KDEUnivariate


def bootstrap_stats(args:Dict[str,Any], out_q:Optional[mp.Queue]=None) -> [Dict[str,Any]]:
    '''Compute statistics and KDEs of data via sampling with replacement'''
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
    '''Compute mean and std of data, and their associated uncertainties'''
    n = len(arr)
    m = np.mean(arr)
    m_4 = np.mean((arr-m)**4)
    s = np.std(arr, ddof=1)
    s4 = s**4
    se_s2 = ((m_4-(s4*(n-3)/(n-1)))/n)**0.25
    se_s = se_s2/(2*s)
    return m, s/np.sqrt(n), s, se_s


def uncert_round(value:float, uncert:float) -> Tuple[float,float]:
    '''Round value according tp given uncertainty'''
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
