import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

'''
Todo:
- Add non inplace versions/options
'''


def to_cartesian(in_data:pd.DataFrame, vec:str, drop:bool=False) -> None:
    z = f'{vec}_eta' in in_data.columns
    try:
        pt = in_data[f'{vec}_pT']
        pt_name = f'{vec}_pT'
    except KeyError:
        pt = in_data[f'{vec}_pt']
        pt_name = f'{vec}_pt'

    if z: eta = in_data[f'{vec}_eta']  
    phi = in_data[f'{vec}_phi']
    in_data[f'{vec}_px'] = pt*np.cos(phi)
    in_data[f'{vec}_py'] = pt*np.sin(phi)
    if z: in_data[f'{vec}_pz'] = pt*np.sinh(eta)
    if drop:
        in_data.drop(columns=[pt_name, f"{vec}_phi"], inplace=True)
        if z: in_data.drop(columns=[f"{vec}_eta"], inplace=True)


def to_pt_eta_phi(in_data:pd.DataFrame, vec:str, eta:bool=True, drop:bool=False) -> None:
    px = in_data[f"{vec}_px"]
    py = in_data[f"{vec}_py"]
    if eta: pz = in_data[f"{vec}_pz"]  
    in_data[f'{vec}_pT'] = np.sqrt(np.square(px) + np.square(py))
    if eta: in_data[f'{vec}_eta'] = np.arcsinh(pz/in_data[f'{vec}_pT'])

    in_data[f'{vec}_phi'] = np.arcsin(py/in_data[f'{vec}_pT'])
    in_data.loc[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] > 0), f'{vec}_phi'] = \
        np.pi-in_data.loc[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] > 0), f'{vec}_phi']
    in_data.loc[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] < 0), f'{vec}_phi'] = \
        -(np.pi + in_data.loc[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] < 0), f'{vec}_phi'])         
    in_data.loc[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] == 0), f'{vec}_phi'] = \
        np.random.choice([-np.pi, np.pi], in_data[(in_data[f"{vec}_px"] < 0) & (in_data[f"{vec}_py"] == 0)].shape[0])

    if drop:
        in_data.drop(columns=[f"{vec}_px", f"{vec}_py"], inplace=True)
        if eta: in_data.drop(columns=[f"{vec}_pz"], inplace=True)


def delta_phi(a:np.ndarray, b:np.ndarray) ->float:
    tmp = pd.DataFrame()
    tmp['dphi'] = b-a
    while len(tmp[tmp.dphi > np.pi]) > 0: tmp.loc[tmp.dphi > np.pi, 'dphi'] -= 2*np.pi
    while len(tmp[tmp.dphi < -np.pi]) > 0: tmp.loc[tmp.dphi < -np.pi, 'dphi'] += 2*np.pi
    return tmp.dphi.values


def twist(dphi:float, deta:float) -> float:
    return np.arctan(np.abs(dphi/deta))


def add_abs_mom(in_data:pd.DataFrame, vec:str, z:bool=True) -> None:
    if z: in_data[f'{vec}_|p|'] = np.sqrt(np.square(in_data[f'{vec}_px'])+np.square(in_data[f'{vec}_py'])+np.square(in_data[f'{vec}_pz']))
    else: in_data[f'{vec}_|p|'] = np.sqrt(np.square(in_data[f'{vec}_px'])+np.square(in_data[f'{vec}_py']))


def add_energy(in_data:pd.DataFrame, vec:str) -> None:
    if f'{vec}_|p|' not in in_data.columns: add_abs_mom(in_data, vec)
    in_data[f'{vec}_E'] = np.sqrt(np.square(in_data[f'{vec}_mass'])+np.square(in_data[f'{vec}_|p|']))


def add_mt(in_data:pd.DataFrame, vec:str, mpt_name:str='mpt'):
    try:             in_data[f'{vec}_mT'] = np.sqrt(2*in_data[f'{vec}_pT']*in_data[f'{mpt_name}_pT']*(1-np.cos(delta_phi(in_data[f'{vec}_phi'], in_data[f'{mpt_name}_phi']))))
    except KeyError: in_data[f'{vec}_mt'] = np.sqrt(2*in_data[f'{vec}_pt']*in_data[f'{mpt_name}_pt']*(1-np.cos(delta_phi(in_data[f'{vec}_phi'], in_data[f'{mpt_name}_phi']))))


def get_vecs(feats:List[str], strict:bool=True) -> List[str]:
    low = [f.lower() for f in feats]
    all_vecs = [f for f in feats if f.lower().endswith('_pt') or f.lower().endswith('_phi') or f.lower().endswith('_eta') or f.lower().endswith('_px') or f.lower().endswith('_py') or f.lower().endswith('_pz')]
    if not strict: return set([v[:v.rfind('_')] for v in all_vecs])
    vecs = [v[:v.rfind('_')] for v in all_vecs if (f'{v[:v.rfind("_")]}_pt'.lower() in low and f'{v[:v.rfind("_")]}_phi'.lower() in low) or 
                                                  (f'{v[:v.rfind("_")]}_px'.lower() in low and f'{v[:v.rfind("_")]}_py'.lower()  in low)]
    return set(vecs)


def fix_event_phi(df:pd.DataFrame, ref_vec:str) -> None:
    '''Rotate event in phi such that ref_vec is at phi == 0'''
    for v in get_vecs(df.columns):
        if v != ref_vec: 
            df[f'{v}_phi'] = delta_phi(df[f'{ref_vec}_phi'], df[f'{v}_phi'])
    df[f'{ref_vec}_phi'] = 0


def fix_event_z(df:pd.DataFrame, ref_vec:str) -> None:
    '''Flip event in z-axis such that ref_vec is in positive z-direction'''
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
    '''Flip event in y-axis such that ref_vec_1 has a higher py than ref_vec_0'''
    if f'{ref_vec_1}_phi' in df.columns:
        cut = (df[f'{ref_vec_1}_phi'] < 0)
        for v in get_vecs(df.columns):
            if v != ref_vec_0: df.loc[cut, f'{v}_phi'] = -df.loc[cut, f'{v}_phi'] 
    else:
        cut = (df[f'{ref_vec_1}_py'] < 0)
        for v in get_vecs(df.columns):
            if v != ref_vec_0: df.loc[cut, f'{v}_py'] = -df.loc[cut, f'{v}_py']


def event_to_cartesian(df:pd.DataFrame, drop:bool=False, ignore:List[str]=[]) -> None:
    for v in get_vecs(df.columns):
        if v not in ignore: to_cartesian(df, v, drop=drop)


def proc_event(df:pd.DataFrame, fix_phi:bool=False, fix_y=False, fix_z=False, cartesian=False,
               ref_vec_0:str=None, ref_vec_1:str=None, keep_feats=[], default_values=[]) -> None:
    '''Pass data through conversions and drop uneeded columns'''
    df.replace([np.inf, -np.inf]+default_values, np.nan, inplace=True)
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
            
    if cartesian:
        print("Converting to Cartesian coordinates")
        event_to_cartesian(df, drop=True)
        
    if   fix_phi and not cartesian: df.drop(columns=[f"{ref_vec_0}_phi"], inplace=True)
    elif fix_phi and cartesian:     df.drop(columns=[f"{ref_vec_0}_py"], inplace=True)
    
    for f in keep_feats:
        df[f'{f}'] = df[f'{f}keep']
        df.drop(columns=[f'{f}keep'], inplace=True)


def calc_pair_mass(df: pd.DataFrame, masses:Tuple[float,float], feat_map:Dict[str,str]) -> np.ndarray:
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
