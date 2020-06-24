import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import _kmeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from pkg import detect_anomaly as da
from pkg import get_data as gd
from pkg import impute as imp
from pkg import my_kmeans as mk
from pkg import transform as tfr
from pkg import visualize as vis


def execute_iforest(df, **kwargs):
    """
    Isolation Forest を実行し、異常かどうかのラベルを返す。
    
    Parameters
    ----------
    df : pandas.DataFrame of shape (n_objects, n_features)
        m_ap30 の表。
    
    Other Parameters
    ----------------
    以下、詳細は
    https://scikit-learn.org/stable/modules/generated
        /sklearn.ensemble.IsolationForest.html
    を参照。
    n_estimators : int, default 100
    max_samples : 'auto', int or float, default 'auto'
    contamination : 'auto' or float, default 0.01
    random_state : int or RandomState instance, default 42
    
    Returns
    -------
    y_pred : numpy.ndarray
        異常かどうかのラベル。
    """
    
    n_estimators = kwargs.get('n_estimators', 100)
    max_samples = kwargs.get('max_samples', 'auto')
    contamination = kwargs.get('contamination', 0.01)
    random_state = kwargs.get('random_state', 42)
    
    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                          contamination=contamination,
                          random_state=random_state)
    y_pred = clf.fit_predict(df)
    return y_pred


def main() -> None:
    SEED_VALUE = 42
    FILEPATH_INPUT = '../data/raw/sample_10k_ver2.csv'
    FILEPATH_OUTPUT = '../results/2020-06-24/iforest.csv'
    DIRPATH_SAVEFIG = '../results/2020-06-24/figures/'
    
    N_CLUSTERS = 5
    WIDTH_DETECT = '180D'  # Time width for anomaly detection
    WIDTH_STEP = '30D'  # Step width for anomaly detection
    PARAMS_IFOREST = {
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 0.01,
        'random_state': SEED_VALUE,
    }
    
    # 乱数のシードを固定して再現性を担保
    gd.fix_random_seed(SEED_VALUE)
    
    #
    # 入力 csv ファイルからデータを整形
    #
    df_orig = gd.load_input_csv(FILEPATH_INPUT,
                                usecols=['objectid', 'mjd', 'm_ap30'])
    # objectid をそのまま使うとメモリ消費が激しいので、0 からの連番に置き換える
    list_id = gd.get_unique_list(df_orig, 'objectid')
    df_orig['objectid'].replace(list_id, np.arange(len(list_id)), inplace=True)
    df_ididx_mjdcols = gd.get_ididx_mjdcols_dataframe(df_orig, df_orig)
    
    df_coord = gd.load_input_csv(FILEPATH_INPUT,
                                 usecols=['objectid', 'coord_ra', 'coord_dec'])
    df_coord.sort_values('objectid', inplace=True)
    df_coord.drop_duplicates(inplace=True)
    df_coord.reset_index(inplace=True, drop=True)
    
    #
    # k-means への入力を作成
    #
    scaler = StandardScaler()
    X_kmeans = np.array([df_coord['coord_ra'].values,
                         df_coord['coord_dec'].values,
                         df_ididx_mjdcols.mean(axis=1).round(4).values]).T
    X_kmeans_std = scaler.fit_transform(X_kmeans)
    
    #
    # エルボー法によりクラスタ数を決定
    #
    # vis.set_rcParams()
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # mk.elbow(X=X_kmeans_std, ax=ax, random_state=SEED_VALUE)
    # ax.set_title('Elbow method')
    # fig.tight_layout()
    # fig.savefig(DIRPATH_SAVEFIG + 'elbow-method.pdf')
    # return
    
    #
    # k-means によりクラスタラベルを割り当てる
    #
    # monkey-patch により k-means の距離関数を自作関数に変更
    _kmeans.euclidean_distances = mk.new_euclidean_distances
    km = _kmeans.KMeans(n_clusters=N_CLUSTERS, random_state=SEED_VALUE)
    y_kmeans = km.fit_predict(X_kmeans_std)
    # monkey-patch による変更を元に戻す
    _kmeans.euclidean_distances = euclidean_distances
    
    # monkey-patch により適用させるアルゴリズムを設定
    da.get_y_pred = execute_iforest
    
    #
    # クラスタ別に異常検知
    #
    df_outlier = pd.DataFrame(columns=['objectid', 'mjd_st', 'mjd_en'])
    for cl in np.unique(y_kmeans):
        df_cl = df_ididx_mjdcols.iloc[y_kmeans == cl]
        # 計測値が 1 つも存在しない時刻は切り落とす
        df_cl_drop = df_cl.dropna(axis=1, how='all')
        # 空間的近傍平均＋空間平均により欠損値を補完
        imp.impute_by_nbr_and_spatial_mean(df_cl_drop, df_coord)
        # object ごとにデータを中央に揃える（平均 0 に正規化）
        df_cl_center = tfr.transform_dataframe_to_centering(df_cl_drop)
        # pandas で時系列データを扱う際は時間軸を index にしたほうが都合が良いので、
        # pandas.DataFrame を転置する（異常検知する際には元に戻す）
        df_cl_center = df_cl_center.T
        tfr.convert_to_DatetimeIndex(df_cl_center)
        # 異常検知
        df_outlier_cl = da.detect_anomaly_per_period(df_cl_center,
                                                     width_detect=WIDTH_DETECT,
                                                     width_step=WIDTH_STEP,
                                                     **PARAMS_IFOREST)
        df_outlier = pd.concat([df_outlier, df_outlier_cl])
    
    df_outlier['objectid'].replace(np.arange(len(list_id)), list_id,
                                   inplace=True)
    df_outlier.sort_values(['objectid', 'mjd_st', 'mjd_en'], inplace=True)
    df_outlier.to_csv(FILEPATH_OUTPUT, index=False)


if __name__ == '__main__':
    main()
