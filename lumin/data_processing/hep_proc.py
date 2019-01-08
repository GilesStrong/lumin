import numpy as np
import pandas as pd

'''
Todo:
- Add non inplace versions/options
- Imporve delta_phi
'''


def to_cartesian(in_data:pd.DataFrame, vec:str, z:bool=True, drop:bool=False) -> None:
    try:
        pt = in_data[f'{vec}_pT']
        pt_name = f'{vec}_pT'
    except KeyError:
        pt = in_data[f'{vec}_pt']
        pt_name = f'{vec}_pt'

    if z: eta = in_data[f'{vec}_eta']  
    phi = in_data[f'{vec}_phi']
    in_data[f'{vec}_px'] = pt * np.cos(phi)
    in_data[f'{vec}_py'] = pt * np.sin(phi)
    if z: in_data[f'{vec}_pz'] = pt * np.sinh(eta)
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


def delta_phi(a:float, b:float) ->float:
    dphi = b-a
    while dphi > np.pi:  dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    return dphi


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
