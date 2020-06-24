import datetime

import julian
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def convert_to_DatetimeIndex(df: pd.DataFrame) -> None:
    """
    入力された pandas.DataFrame の index を mjd から DatetimeIndex に変換する。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_mjds, n_objects)
        m_ap30 の表。
    """
    
    idx_datetime = np.empty_like(df.index, dtype=datetime.datetime)
    for i in range(len(idx_datetime)):
        idx_datetime[i] = \
            julian.from_jd(df.index[i], fmt='mjd').replace(second=0,
                                                           microsecond=0)
    df.index = idx_datetime


def transform_dataframe_to_centering(df: pd.DataFrame) -> pd.DataFrame:
    """
    object ごとに m_ap30 の値を中央揃え（平均値を 0）にする。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        m_ap30 の表。
    
    Returns
    -------
    df_center : pandas.DataFrame of shape (n_objects, n_mjds)
        object ごとに m_ap30 の値を中央揃え（平均値を 0）にした m_ap30 の表。
    """
    
    scaler = StandardScaler(with_std=False)
    df_center = pd.DataFrame(scaler.fit_transform(df.T).T,
                             index=df.index, columns=df.columns)
    return df_center


def convert_date_into_mjd(date_st: str, date_en: str) -> (int, int):
    """
    '%Y-%m-%d' 形式の str を mjd に変換する。
    
    Parameters
    ----------
    date_st : str
        '%Y-%m-%d' 形式の文字列。
    date_en : str
        '%Y-%m-%d' 形式の文字列。
    
    Returns
    -------
    mjd_st : int
        date_st を mjd に変換したもの。
    mjd_en : int
        date_en を mjd に変換したもの。
    """
    
    datetime_st = datetime.datetime.strptime(date_st, '%Y-%m-%d')
    datetime_en = datetime.datetime.strptime(date_en, '%Y-%m-%d')
    mjd_st = int(julian.to_jd(datetime_st, fmt='mjd'))
    mjd_en = int(julian.to_jd(datetime_en, fmt='mjd'))
    return mjd_st, mjd_en
