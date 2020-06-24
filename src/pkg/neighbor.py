from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_ralim(RA_target: float, DEC_target: float) -> (float, float):
    """
    入力された赤経赤緯の近傍の赤経の範囲を返す。
    
    Parameters
    ----------
    RA_target : float
        近傍の赤経の範囲を知りたい天体の赤経の値。
    DEC_target : float
        近傍の赤経の範囲を知りたい天体の赤緯の値。
    
    Returns
    -------
    ra_min : float
        入力された赤経赤緯の近傍の赤経の下限。
    ra_max : float
        入力された赤経赤緯の近傍の赤経の上限。
    """
    
    return (RA_target
            - 128.0 * 0.17 / 60.0 / 60.0 / np.cos(np.deg2rad(DEC_target)),
            RA_target
            + 128.0 * 0.17 / 60.0 / 60.0 / np.cos(np.deg2rad(DEC_target)))


def get_declim(DEC_target: float) -> (float, float):
    """
    入力された赤緯の近傍の赤緯の範囲を返す。
    
    Parameters
    ----------
    DEC_target : float
        近傍の赤緯の範囲を知りたい赤緯の値。
    
    Returns
    -------
    dec_min : float
        入力された赤緯の近傍の赤緯の下限。
    dec_max : float
        入力された赤緯の近傍の赤緯の上限。
    """
    
    return (DEC_target - 128.0 * 0.17 / 60.0 / 60.0,
            DEC_target + 128.0 * 0.17 / 60.0 / 60.0)


def get_neighbor_idx(
        df: pd.DataFrame, df_coord: pd.DataFrame) -> List[List[int]]:
    """
    それぞれの object の空間的に近傍の object の index の list を返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        m_ap30 の表。
    df_coord : pandas.DataFrame of shape (n_objects, n_features)
        columns に `coord_ra` と `coord_dec` をもつ表。
        df_coord.columns[1] が `coord_ra` で、
        df_coord.columns[2] が `coord_dec`。
    
    Returns
    -------
    neighbor_idx : list of list of int
        それぞれの object の空間的に近傍の object の index の list。
    """
    
    df_coord_source = df_coord.loc[df.index]
    neighbor_idx = [[] for _ in range(len(df))]
    print('\nget_neighbor_idx\n')
    for i in tqdm(range(len(df))):
        ra_min, ra_max = get_ralim(df_coord_source.iat[i, 1],
                                   df_coord_source.iat[i, 2])
        dec_min, dec_max = get_declim(df_coord_source.iat[i, 2])
        is_neighbor = (ra_min <= df_coord_source['coord_ra']) \
                      & (df_coord_source['coord_ra'] <= ra_max) \
                      & (dec_min <= df_coord_source['coord_dec']) \
                      & (df_coord_source['coord_dec'] <= dec_max)
        if np.count_nonzero(is_neighbor) > 1:
            neighbor_idx_tmp = is_neighbor.index[is_neighbor].values
            neighbor_idx[i] \
                = neighbor_idx_tmp[neighbor_idx_tmp != df.index[i]].tolist()
    return neighbor_idx
