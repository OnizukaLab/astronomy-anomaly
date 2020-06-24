import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pkg import neighbor as nbr


def impute_by_time_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    時間平均による欠損値補完を行なう。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行なう m_ap30 の表。
    
    Returns
    -------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        時間平均により欠損値補完を行なった m_ap30 の表。
    """
    
    df_mean = df.mean(axis=1)
    df = df.T
    df.fillna(df_mean, inplace=True)
    df = df.T
    return df


def get_moving_average_num(
        df: pd.DataFrame, width_moving_average: float) -> np.ndarray:
    """
    時間幅 width_moving_average の移動平均による欠損値補完を行なう際に、
    指定された pandas.DataFrame のそれぞれの mjd の列に対して、
    直前の何個の要素の平均を取ればよいのかを格納した numpy.ndarray を返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行いたい m_ap30 の表。
    width_moving_average : float
        移動平均の時間幅。単位は mjd。
    
    Returns
    -------
    ma_num : numpy.ndarray
        時間幅 width_moving_average の移動平均による欠損値補完を行なう際に、
        指定された pandas.DataFrame のそれぞれの mjd の列に対して、
        直前の何個の要素の平均を取ればよいのかのリスト。
    """
    
    mjd_list = df.columns.values
    mjd_diff = mjd_list[1:] - mjd_list[:-1]
    ma_num = np.empty_like(mjd_diff, dtype=np.int)
    for i in range(len(ma_num)):
        cnt = 0
        w_tmp = mjd_diff[i]
        while w_tmp <= width_moving_average:
            cnt += 1
            if i < cnt:
                break
            w_tmp += mjd_diff[i - cnt]
        ma_num[i] = cnt
    return ma_num


def impute_by_moving_average(
        df: pd.DataFrame, width_moving_average: float) -> None:
    """
    移動平均による欠損値補完を行なう。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行なう m_ap30 の表。
    width_moving_average : float
        移動平均の時間幅。単位は mjd。
    """
    
    ma_num = get_moving_average_num(df,
                                    width_moving_average=width_moving_average)
    # 先頭の mjd を除く mjd の NaN を、その object の直前の ma_num 個の平均値で補完
    for col in range(len(df.columns) - 1, 0, -1):
        if ma_num[col - 1] > 0:
            df.iloc[:, col].fillna(df.iloc[:, col \
                                   - ma_num[col - 1]:col].mean(axis=1).round(4),
            inplace=True)


def get_spatial_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    object ごとに標準化した計測値の、mjd ごとのすべての object に対する平均値を、
    それぞれの object の元の表現にスケールバックしたものを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        m_ap30 の表。
    
    Returns
    -------
    df_std_mean_inv : pandas.DataFrame of shape (n_objects, n_mjds)
        object ごとに標準化した計測値の、mjd ごとのすべての object に対する平均値を、
        それぞれの object の元の表現にスケールバックしたもの。
    """
    
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df.T).T,
                          index=df.index, columns=df.columns)
    s_std_mean = df_std.mean(axis=0)
    df_std_mean \
        = pd.DataFrame(np.array([s_std_mean.values for _ in range(len(df))]),
                       index=df.index, columns=df.columns)
    df_std_mean_inv = pd.DataFrame(scaler.inverse_transform(df_std_mean.T).T,
                                   index=df.index, columns=df.columns).round(4)
    return df_std_mean_inv


def impute_by_spatial_mean(df: pd.DataFrame) -> None:
    """
    空間平均による欠損値補完を行なう。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行なう m_ap30 の表。これが補完される。
    """
    
    df_std_mean_inv = get_spatial_mean(df)
    df.fillna(df_std_mean_inv, inplace=True)


def impute_by_nbr(df: pd.DataFrame, df_coord: pd.DataFrame) -> None:
    """
    空間的に近傍に存在する object の、object ごとに標準化した計測値の平均値を、
    それぞれの object の元の表現にスケールバックした値による欠損値補完を行なう。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行なう m_ap30 の表。これが補完される。
    df_coord : pandas.DataFrame of shape (n_objects, n_features)
        columns に `coord_ra` と `coord_dec` をもつ表。
        df_coord.columns[1] が `coord_ra` で、
        df_coord.columns[2] が `coord_dec`。
    """
    
    # object ごとに計測値を標準化
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df.T).T,
                          index=df.index, columns=df.columns)
    # それぞれの object の空間的に近傍の object の標準化された計測値の平均を、
    # mjd ごとに算出
    neighbor_idx = nbr.get_neighbor_idx(df, df_coord)
    df_std_mean_nbr = pd.DataFrame(index=df.index, columns=df.columns)
    print('\nimpute_by_nbr\n')
    for i in tqdm(range(len(df_std_mean_nbr))):
        df_std_mean_nbr.iloc[i] = df_std.loc[neighbor_idx[i]].mean(axis=0)
    # 得られた値を、それぞれの object の元の表現にスケールバックする
    scaler = StandardScaler()
    scaler.fit(df.T)
    df_std_mean_nbr_inv \
        = pd.DataFrame(scaler.inverse_transform(df_std_mean_nbr.T).T,
                       index=df_std_mean_nbr.index,
                       columns=df_std_mean_nbr.columns).round(4)
    # これを補完する
    df.fillna(df_std_mean_nbr_inv, inplace=True)


def impute_by_nbr_and_spatial_mean(
        df: pd.DataFrame, df_coord: pd.DataFrame) -> None:
    """
    まず空間的近傍平均により欠損値補完を行ない、
    補完されなかった部分を空間平均により補完する。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_mjds)
        欠損値補完を行なう m_ap30 の表。これが補完される。
    df_coord : pandas.DataFrame of shape (n_objects, n_features)
        columns に `coord_ra` と `coord_dec` をもつ表。
        df_coord.columns[1] が `coord_ra` で、
        df_coord.columns[2] が `coord_dec`。
    """
    
    df_std_mean_inv = get_spatial_mean(df)
    impute_by_nbr(df, df_coord)
    df.fillna(df_std_mean_inv, inplace=True)
