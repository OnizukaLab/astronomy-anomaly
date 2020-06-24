from typing import List

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from pkg import transform as tfr


def get_anomaly_period(
        df: pd.DataFrame,
        width_detect: str,
        width_step: str) -> (np.ndarray, np.ndarray):
    """
    異常検知を行なう期間の開始日と最終日を返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_datetimes, n_objects)
        m_ap30 の表。
    width_detect : str, tuple, datetime.timedelta, DateOffset or None
        異常検知を行なう時間幅。
    width_step : str or DateOffset
        異常検知を行なうステップ幅。
    
    Returns
    -------
    anomaly_st : numpy.ndarray of str
        異常検知を行なう期間の開始日。
    anomaly_en : numpy.ndarray of str
        異常検知を行なう期間の最終日。
    """
    
    anomaly_st = pd.date_range(start=df.index[0].date(),
                               end=df.index[-1].date(), freq=width_step).date
    anomaly_en = anomaly_st \
                 + pd.to_timedelta(to_offset(width_detect) - to_offset('D'))
    anomaly_en[anomaly_en > df.index[-1].date()] = df.index[-1].date()
    anomaly_en = np.unique(anomaly_en)
    anomaly_st = anomaly_st[:len(anomaly_en)]
    anomaly_st[-1] = anomaly_en[-1] \
                     - pd.to_timedelta(to_offset(width_detect) - to_offset('D'))
    anomaly_st = anomaly_st.astype('str')
    anomaly_en = anomaly_en.astype('str')
    anomaly_st = anomaly_st.astype(object)
    anomaly_en = anomaly_en.astype(object)
    return anomaly_st, anomaly_en


def get_anomaly_data_period(
        df: pd.DataFrame,
        width_detect: str,
        width_step: str) -> (List[str], List[str]):
    """
    異常検知を行なう期間の、データが存在する開始日と最終日を返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_datetimes, n_objects)
        m_ap30 の表。
    width_detect : str, tuple, datetime.timedelta, DateOffset or None
        異常検知を行なう時間幅。
    width_step : str or DateOffset
        異常検知を行なうステップ幅。
    
    Returns
    -------
    anomaly_data_st : list of str
        異常検知を行なう期間の、データが存在する開始日。
    anomaly_data_en : list of str
        異常検知を行なう期間の、データが存在する最終日。
    """
    
    anomaly_st, anomaly_en = get_anomaly_period(df, width_detect=width_detect,
                                                width_step=width_step)
    anomaly_data_st = []
    anomaly_data_en = []
    for i in range(len(anomaly_st)):
        df_slice = df[anomaly_st[i]:anomaly_en[i]]
        if len(df_slice) > 0:
            anomaly_data_st.append(str(df_slice.index[0].date()))
            anomaly_data_en.append(str(df_slice.index[-1].date()))
    anomaly_data_period = np.empty(len(anomaly_data_st), dtype=object)
    for i in range(len(anomaly_data_period)):
        anomaly_data_period[i] = anomaly_data_st[i] + 'to' + anomaly_data_en[i]
    anomaly_data_period = np.unique(anomaly_data_period)
    anomaly_data_st = []
    anomaly_data_en = []
    for str_period in anomaly_data_period:
        str_date = str_period.split('to')
        anomaly_data_st.append(str_date[0])
        anomaly_data_en.append(str_date[1])
    return anomaly_data_st, anomaly_data_en


def get_y_pred(df, **kwargs):
    """
    異常検知アルゴリズムを適用し、異常かどうかのラベルを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_features)
        m_ap30 の表。
    **kwargs
        Arbitrary keyword arguments.
    """
    
    return None


def get_outlier_idx(df, **kwargs):
    """
    異常検知アルゴリズムを適用し、異常と判定された行ラベルを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_features)
        m_ap30 の表。
    **kwargs
        Arbitrary keyword arguments.
    
    Returns
    -------
    outlier_idx : list
        異常と判定された行ラベル。
    """
    
    y_pred = get_y_pred(df, **kwargs)
    outlier_idx = [df.index[i] for i in range(len(y_pred)) if y_pred[i] == -1]
    return outlier_idx


def detect_anomaly_per_period(df, width_detect, width_step, **kwargs):
    """
    入力された pandas.DataFrame に対して切り出した期間ごとに異常検知を行ない、
    異常と判定された idx、mjd_st、mjd_en を columns にもつ
    pandas.DataFrame を返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_datetimes, n_objects)
        m_ap30 の表。
    width_detect : str, tuple, datetime.timedelta, DateOffset or None
        異常検知を行なう時間幅。
    width_step : str or DateOffset
        異常検知を行なうステップ幅。
    **kwargs
        Arbitrary keyword arguments.
    
    Returns
    -------
    df_outlier : pandas.DataFrame
        異常と判定された object の idx とその期間をリストアップしたもの。
    """
    
    date_st, date_en = get_anomaly_data_period(df, width_detect=width_detect,
                                               width_step=width_step)
    
    df_outlier = pd.DataFrame(columns=['objectid', 'mjd_st', 'mjd_en'])
    row_label = 0
    for i in range(len(date_st)):
        outlier_idx = get_outlier_idx(df[date_st[i]:date_en[i]].T, **kwargs)
        mjd_st, mjd_en = tfr.convert_date_into_mjd(date_st[i], date_en[i])
        for idx in outlier_idx:
            df_outlier.loc[row_label] = [idx, mjd_st, mjd_en]
            row_label += 1
    
    return df_outlier
