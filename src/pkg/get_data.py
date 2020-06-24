import os
import random
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def fix_random_seed(seed: int = 42) -> None:
    """
    乱数のシードを固定する。
    
    Parameters
    ----------
    seed : int
        乱数のシード。
    """
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_input_csv(input_filepath: str, usecols: List[str]) -> pd.DataFrame:
    """
    入力 csv ファイルを、必要な columns を選択して読み込む。
    
    Parameters
    ----------
    input_filepath : str
        入力 csv ファイルへのパス。
    usecols : list of str
        必要なカラム名。
    
    Returns
    -------
    df_orig : pandas.DataFrame
        入力 csv ファイルの、選択した columns の表。
    """
    
    df_orig = pd.read_csv(input_filepath, sep='\s+|\s+', usecols=usecols,
                          engine='python', skipinitialspace=True,
                          skiprows=[1], skipfooter=1)
    df_orig.dropna(inplace=True)
    df_orig = df_orig.astype({'objectid': np.int64})
    df_orig.reset_index(inplace=True, drop=True)
    return df_orig


def get_unique_list(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    指定した pandas.DataFrame のカラムの unique なリストを、
    昇順にソートしたものを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame
        目的の表。
    col : str
        目的のカラム名。
    
    Returns
    -------
    unique_list : numpy.ndarray
        指定した pandas.DataFrame のカラムの unique なリストを、
        昇順にソートしたもの。
    """
    
    unique_list = df[col].unique()
    unique_list.sort()
    return unique_list


def get_ididx_mjdcols_dataframe(
        df: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
    """
    df_source の unique な objectid を index、mjd を columns とした
    pandas.DataFrame を作成し、df の m_ap30 の値を埋めたものを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame
        埋める m_ap30 の値を保持した表。
    df_source : pandas.DataFrame
        返り値の index となる objectid と、columns となる mjd を保持した表。
    
    Returns
    -------
    df_return : pandas.DataFrame
        df_source の unique な objectid を index、mjd を columns とした
        pandas.DataFrame で、df の m_ap30 の値を埋めたもの。
    """
    
    list_id = get_unique_list(df_source, 'objectid')
    list_mjd = get_unique_list(df_source, 'mjd')
    df_return = pd.DataFrame(index=list_id, columns=list_mjd)
    df_reset_idx = df.reset_index(drop=True)
    print('\nget_ididx_mjdcols_dataframe\n')
    for i in tqdm(range(len(df_reset_idx))):
        idx = df_reset_idx.at[i, 'objectid']
        col = df_reset_idx.at[i, 'mjd']
        df_return.at[idx, col] = df_reset_idx.at[i, 'm_ap30'].round(3)
    return df_return
