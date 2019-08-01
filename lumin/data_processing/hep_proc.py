import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Set
import warnings

'''
Todo:
- Add non inplace versions/options
'''


def to_cartesian(df:pd.DataFrame, vec:str, drop:bool=False) -> None:
    r'''
    Vectoriesed conversion of 3-momenta to Cartesian coordinates inplace, optionally dropping old pT,eta,phi features

    Arguments:
        df: DataFrame to alter
        vec: column prefix of vector components to alter, e.g. 'muon' for columns ['muon_pt', 'muon_phi', 'muon_eta']
        drop: Whether to remove original columns and just keep the new ones
    '''

    z = f'{vec}_eta' in df.columns
    try:
        pt = df[f'{vec}_pT']
        pt_name = f'{vec}_pT'
    except KeyError:
        pt = df[f'{vec}_pt']
        pt_name = f'{vec}_pt'

    if z: eta = df[f'{vec}_eta']  
    phi = df[f'{vec}_phi']
    df[f'{vec}_px'] = pt*np.cos(phi)
    df[f'{vec}_py'] = pt*np.sin(phi)
    if z: df[f'{vec}_pz'] = pt*np.sinh(eta)
    if drop:
        df.drop(columns=[pt_name, f"{vec}_phi"], inplace=True)
        if z: df.drop(columns=[f"{vec}_eta"], inplace=True)


def to_pt_eta_phi(df:pd.DataFrame, vec:str, eta:Optional[bool]=None, drop:bool=False) -> None:
    r'''
    Vectoriesed conversion of 3-momenta to pT,eta,phi coordinates inplace, optionally dropping old px,py,pz features

    Arguments:
            df: DataFrame to alter
            vec: column prefix of vector components to alter, e.g. 'muon' for columns ['muon_px', 'muon_py', 'muon_pz']
            drop: Whether to remove original columns and just keep the new ones
    '''

    # XXX Remove in v0.4
    if eta is not None: warnings.warn('''Passing eta is now no longer required, it is infered from DF.
                                         The eta argument is now depreciated and will be removed in v0.4''')

    eta = f'{vec}_pz' in df.columns
    px = df[f"{vec}_px"]
    py = df[f"{vec}_py"]
    if eta: pz = df[f"{vec}_pz"]  
    df[f'{vec}_pT'] = np.sqrt(np.square(px) + np.square(py))
    if eta: df[f'{vec}_eta'] = np.arcsinh(pz/df[f'{vec}_pT'])

    df[f'{vec}_phi'] = np.arcsin(py/df[f'{vec}_pT'])
    df.loc[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] > 0), f'{vec}_phi'] =   np.pi-df.loc[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] > 0), f'{vec}_phi']
    df.loc[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] < 0), f'{vec}_phi'] = -(np.pi+df.loc[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] < 0), f'{vec}_phi'])         
    df.loc[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] == 0), f'{vec}_phi'] = \
        np.random.choice((-np.pi, np.pi), df[(df[f"{vec}_px"] < 0) & (df[f"{vec}_py"] == 0)].shape[0])

    if drop:
        df.drop(columns=[f"{vec}_px", f"{vec}_py"], inplace=True)
        if eta: df.drop(columns=[f"{vec}_pz"], inplace=True)


def delta_phi(arr_a:Union[float,np.ndarray], arr_b:Union[float,np.ndarray]) -> Union[float,np.ndarray]:
    r'''
    Vectorised compututation of modulo 2pi angular seperation of array of angles b from array of angles a, in range [-pi,pi]

    Arguments:
        arr_a: reference angles
        arr_b: final angles

    Returns:
        angular separation as float or np.array
    '''

    df = pd.DataFrame()  # Better way to do this without df?
    df['dphi'] = arr_b-arr_a
    while len(df[df.dphi > np.pi]) > 0:  df.loc[df.dphi > np.pi, 'dphi']  -= 2*np.pi
    while len(df[df.dphi < -np.pi]) > 0: df.loc[df.dphi < -np.pi, 'dphi'] += 2*np.pi
    return df.dphi.values


def twist(dphi:Union[float,np.ndarray], deta:Union[float,np.ndarray]) -> Union[float,np.ndarray]:
    r'''
    Vectorised computation of twist between vectors (https://arxiv.org/abs/1010.3698)

    Arguments:
        dphi: delta phi separations
        deta: delta eta separations

    Returns:
        angular separation as float or np.array
    '''
    
    return np.arctan(np.abs(dphi/deta))


def add_abs_mom(df:pd.DataFrame, vec:str, z:bool=True) -> None:
    r'''
    Vectorised computation 3-momenta magnitude, adding new column in place. Currently only works for Cartesian vectors

    Arguments:
        df: DataFrame to alter
        vec: column prefix of vector components, e.g. 'muon' for columns ['muon_px', 'muon_py', 'muon_pz']
        z: whether to consider the z-component of the momenta
    '''

    # TODO extend to work on pT, eta, phi vectors

    if z: df[f'{vec}_absp'] = np.sqrt(np.square(df[f'{vec}_px'])+np.square(df[f'{vec}_py'])+np.square(df[f'{vec}_pz']))
    else: df[f'{vec}_absp'] = np.sqrt(np.square(df[f'{vec}_px'])+np.square(df[f'{vec}_py']))


def add_mass(df:pd.DataFrame, vec:str) -> None:
    r'''
    Vectorised computation of mass of 4-vector, adding new column in place.

    Arguments:
        df: DataFrame to alter
        vec: column prefix of vector components, e.g. 'muon' for columns ['muon_px', 'muon_py', 'muon_pz']
    '''

    if f'{vec}_absp' not in df.columns: add_abs_mom(df, vec)
    df[f'{vec}_mass'] = np.sqrt(np.square(df[f'{vec}_E'])-np.square(df[f'{vec}_absp']))


def add_energy(df:pd.DataFrame, vec:str) -> None:
    r'''
    Vectorised computation of energy of 4-vector, adding new column in place.

    Arguments:
        df: DataFrame to alter
        vec: column prefix of vector components, e.g. 'muon' for columns ['muon_px', 'muon_py', 'muon_pz']
    '''

    if f'{vec}_absp' not in df.columns: add_abs_mom(df, vec)
    df[f'{vec}_E'] = np.sqrt(np.square(df[f'{vec}_mass'])+np.square(df[f'{vec}_absp']))


def add_mt(df:pd.DataFrame, vec:str, mpt_name:str='mpt'):
    r'''
    Vectorised computation of transverse mass of 4-vector with respect to missing transverse momenta, adding new column in place.
    Currently only works for pT, eta, phi vectors

    Arguments:
        df: DataFrame to alter
        vec: column prefix of vector components, e.g. 'muon' for columns ['muon_px', 'muon_py', 'muon_pz']
        mpt_name: column prefix of vector of missing transverse momenta components, e.g. 'mpt' for columns ['mpt_pT', 'mpt_phi'] 
    '''

    # TODO: extend to work on Cartesian coordinates

    try:             df[f'{vec}_mT'] = np.sqrt(2*df[f'{vec}_pT']*df[f'{mpt_name}_pT']*(1-np.cos(delta_phi(df[f'{vec}_phi'], df[f'{mpt_name}_phi']))))
    except KeyError: df[f'{vec}_mt'] = np.sqrt(2*df[f'{vec}_pt']*df[f'{mpt_name}_pt']*(1-np.cos(delta_phi(df[f'{vec}_phi'], df[f'{mpt_name}_phi']))))


def get_vecs(feats:List[str], strict:bool=True) -> Set[str]:
    r'''
    Filter list of features to get list of 3-momenta defined in the list. Works for both pT, eta, phi and Cartesian coordinates.
    If strict, return only vectors with all coordinates present in feature list.

    Arguments:
        feats: list of features to filter
        strict: whether to require all 3-momenta components to be present in the list
    
    Returns:
        set of unique 3-momneta prefixes
    '''

    low = [f.lower() for f in feats]
    all_vecs = [f for f in feats if (f.lower().endswith('_pt') or f.lower().endswith('_phi') or f.lower().endswith('_eta')) or 
                                    (f.lower().endswith('_px') or f.lower().endswith('_py')  or f.lower().endswith('_pz'))]
    if not strict: return set([v[:v.rfind('_')] for v in all_vecs])
    vecs = [v[:v.rfind('_')] for v in all_vecs if (f'{v[:v.rfind("_")]}_pt'.lower() in low and f'{v[:v.rfind("_")]}_phi'.lower() in low) or 
                                                  (f'{v[:v.rfind("_")]}_px'.lower() in low and f'{v[:v.rfind("_")]}_py'.lower()  in low)]
    return set(vecs)


def fix_event_phi(df:pd.DataFrame, ref_vec:str) -> None:
    r'''
    Rotate event in phi such that ref_vec is at phi == 0. Performed inplace. Currently only works on vectors defined in pT, eta, phi

    Arguments:
        df: DataFrame to alter
        ref_vec: column prefix of vector components to use as reference, e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
    '''

    # TODO: extend to work on Cartesian coordinates

    for v in get_vecs(df.columns):
        if v != ref_vec: df[f'{v}_phi'] = delta_phi(df[f'{ref_vec}_phi'], df[f'{v}_phi'])
    df[f'{ref_vec}_phi'] = 0


def fix_event_z(df:pd.DataFrame, ref_vec:str) -> None:
    r'''
    Flip event in z-axis such that ref_vec is in positive z-direction. Performed inplace. Works for both pT, eta, phi and Cartesian coordinates.
    
    Arguments:
        df: DataFrame to alter
        ref_vec: column prefix of vector components to use as reference, e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
    '''

    if f'{ref_vec}_eta' in df.columns:
        cut = (df[f'{ref_vec}_eta'] < 0)
        for v in get_vecs(df.columns):
            try: df.loc[cut, f'{v}_eta'] = -df.loc[cut, f'{v}_eta'] 
            except KeyError: print(f'eta component of {v} not found')
    else:
        cut = cut = (df[f'{ref_vec}_pz'] < 0)
        for v in get_vecs(df.columns):
            try: df.loc[cut, f'{v}_pz'] = -df.loc[cut, f'{v}_pz']
            except KeyError: print(f'pz component of {v} not found')


def fix_event_y(df:pd.DataFrame, ref_vec_0:str, ref_vec_1:str) -> None:
    r'''
    Flip event in y-axis such that ref_vec_1 has a higher py than ref_vec_0. Performed in place. Works for both pT, eta, phi and Cartesian coordinates.
    
    Arguments:
        df: DataFrame to alter
        ref_vec_0: column prefix of vector components to use as reference 0, e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
        ref_vec_1: column prefix of vector components to use as reference 1, e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
    '''
    
    if f'{ref_vec_1}_phi' in df.columns:
        cut = (df[f'{ref_vec_1}_phi'] < 0)
        for v in get_vecs(df.columns):
            if v != ref_vec_0: df.loc[cut, f'{v}_phi'] = -df.loc[cut, f'{v}_phi'] 
    else:
        cut = (df[f'{ref_vec_1}_py'] < 0)
        for v in get_vecs(df.columns):
            if v != ref_vec_0: df.loc[cut, f'{v}_py'] = -df.loc[cut, f'{v}_py']


def event_to_cartesian(df:pd.DataFrame, drop:bool=False, ignore:Optional[List[str]]=None) -> None:
    r'''
    Convert entire event to Cartesian coordinates, except vectors listed in ignore. Optionally, drop old pT,eta,phi features. Perfomed inplace.
    
    Arguments:
        df: DataFrame to alter
        drop: whether to drop old coordinates
        ignore: vectors to ignore when converting
    '''
    
    for v in get_vecs(df.columns):
        if ignore is None or v not in ignore: to_cartesian(df, v, drop=drop)


def proc_event(df:pd.DataFrame, fix_phi:bool=False, fix_y=False, fix_z=False, use_cartesian=False,
               ref_vec_0:str=None, ref_vec_1:str=None, keep_feats:Optional[List[str]]=None, default_vals:Optional[List[str]]=None) -> None:
    r'''
    Process event: Pass data through inplace various conversions and drop uneeded columns. Data expected to consist of vectors defined in pT, eta, phi.
    
    Arguments:
        df: DataFrame to alter
        fix_phi: whether to rotate events using :meth:fix_event_phi
        fix_y: whether to flip events using :meth:fix_event_y
        fix_z: whether to flip events using :meth:fix_event_z
        use_cartesian: wether to convert vectors to Cartesian coordinates
        ref_vec_0: column prefix of vector components to use as reference (0) for :meth:fix_event_phi, :meth:fix_event_y, and :meth:fix_event_z
                   e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
        ref_vec_1: column prefix of vector components to use as reference 1 for :meth:fix_event_z, e.g. 'muon' for columns ['muon_pT', 'muon_eta', 'muon_phi']
        keep_feats: columns to keep which would otherwise be dropped
        default_vals:  list of default values which might be used to represent missing vector components. These will be replaced with np.nan.
    '''
    
    df.replace([np.inf, -np.inf]+default_vals if default_vals is not None else [np.inf, -np.inf], np.nan, inplace=True)
    if keep_feats is not None:
        for f in keep_feats: df[f'{f}keep'] = df[f'{f}']
    
    if fix_phi:
        print(f'Setting {ref_vec_0} to phi = 0')
        fix_event_phi(df, ref_vec_0)
        if fix_y:
            print(f'Setting {ref_vec_1} to positve phi')
            fix_event_y(df, ref_vec_0, ref_vec_1)
    if fix_z:
        print(f'Setting {ref_vec_0} to positive eta')
        fix_event_z(df, ref_vec_0) 
    if use_cartesian:
        print("Converting to use_cartesian coordinates")
        event_to_cartesian(df, drop=True)
    if   fix_phi and not use_cartesian: df.drop(columns=[f"{ref_vec_0}_phi"], inplace=True)
    elif fix_phi and     use_cartesian: df.drop(columns=[f"{ref_vec_0}_py"], inplace=True)
    
    if keep_feats is not None:
        for f in keep_feats:
            df[f'{f}'] = df[f'{f}keep']
            df.drop(columns=[f'{f}keep'], inplace=True)


def calc_pair_mass(df:pd.DataFrame, masses:Union[Tuple[float,float],Tuple[np.ndarray,np.ndarray]], feat_map:Dict[str,str]) -> np.ndarray:
    r'''
    Vectorised computation of invarient mass of pair of particles with given masses, using 4-momenta. Only works for vectors defined in Cartesian coordinates.

    Arguments:
        df: DataFrame vector components
        masses: tuple of masses of particles (either constant or different pair of masses per pair of particles)
        feat_map: dictionary mapping of requested momentum components to the features in df

    Returns:
        np.array of invarient masses
    '''

    # TODO: rewrite to not use a DataFrame for holding parent vector
    # TODO: add inplace option
    # TODO: extend to work on pT, eta, phi coordinates

    tmp = pd.DataFrame()
    tmp['0_E'] = np.sqrt((masses[0]**2)+np.square(df.loc[:, feat_map['0_px']])+np.square(df.loc[:, feat_map['0_py']])+np.square(df.loc[:, feat_map['0_pz']]))
    tmp['1_E'] = np.sqrt((masses[1]**2)+np.square(df.loc[:, feat_map['1_px']])+np.square(df.loc[:, feat_map['1_py']])+np.square(df.loc[:, feat_map['1_pz']]))
    tmp['p_px'] = df.loc[:, feat_map['0_px']]+df.loc[:, feat_map['1_px']]
    tmp['p_py'] = df.loc[:, feat_map['0_py']]+df.loc[:, feat_map['1_py']]
    tmp['p_pz'] = df.loc[:, feat_map['0_pz']]+df.loc[:, feat_map['1_pz']]
    tmp['p_E'] = tmp.loc[:, '0_E']+tmp.loc[:, '1_E']
    tmp['p_p2'] = np.square(tmp.loc[:, 'p_px'])+np.square(tmp.loc[:, 'p_py'])+np.square(tmp.loc[:, 'p_pz'])
    tmp['p_mass'] = np.sqrt(np.square(tmp.loc[:, 'p_E'])-tmp.loc[:, 'p_p2'])
    return tmp.p_mass.values
