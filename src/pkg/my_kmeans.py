import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import _kmeans
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    """
    k-means で用いる独自の距離関数。
    
    n_features が 3 以上で、X[:, :2] が coord_ra と coord_dec であるような入力 X
    を想定しており（Y についても同様）、coord_ra と coord_dec の 2 次元空間の
    ユークリッド距離（ルートを取ったもの）と、それ以外の特徴量の空間のユークリッド距離
    （ルートを取ったもの）の和の距離行列を返す。
    `squared == True` ならば、その 2 乗を取った距離行列を返す。
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)
        n_features は 3 以上で、X[:, :2] が coord_ra と coord_dec。
    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features),
        default None
        n_features は 3 以上で、Y[:, :2] が coord_ra と coord_dec。
    Y_norm_squared : array-like, shape (n_samples_2, ), default None
        元のユークリッド距離関数（sklearn.metrics.pairwise.euclidean_distances）の
        入力を保証するためのパラメータ。本関数では使用できないため、以下のコードには
        現れていない。
    squared : boolean, default False
        Return squared distances.
    
    Returns
    -------
    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)
        X と Y の距離行列。`Y == None` ならば、X と X 自身の距離行列。
    
    Examples
    --------
    >>> from sklearn.cluster import _kmeans
    >>> from pkg import my_kmeans as mk
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> import numpy as np
    >>> X = np.array([[0, 0, 0], [1, 1, 0], [2, 1, 0],
                      [0, 0, 1], [1, 1, 1], [1, 2, 1]])
    >>> # monkey-patch による上書き
    >>> _kmeans.euclidean_distances = mk.new_euclidean_distances
    >>> km = _kmeans.KMeans(n_clusters=2, random_state=42)
    >>> km.fit(X)
    >>> # monkey-patch による変更を元に戻す
    >>> _kmeans.euclidean_distances = euclidean_distances
    >>> km = _kmeans.KMeans(n_clusters=2, random_state=42)
    >>> km.fit(X)
    """
    
    X, Y = check_pairwise_arrays(X, Y)
    X_coord = X[:, :2]
    Y_coord = Y[:, :2]
    X_others = X[:, 2:]
    Y_others = Y[:, 2:]
    distances_coord = euclidean_distances(X=X_coord, Y=Y_coord, squared=False)
    distances_others = euclidean_distances(X=X_others, Y=Y_others,
                                           squared=False)
    distances = distances_coord + distances_others
    return distances if not squared else np.square(distances, out=distances)


def elbow(X, ax, k_min=1, k_max=10, random_state=42):
    """
    エルボー法の結果を可視化する。
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster.
    ax : matplotlib.axes.Axes
        プロットを行なう Axes オブジェクト。これを更新する。
    k_min : int, default 1
        エルボー法を行なう、最小のクラスタ数。
    k_max : int, default 10
        エルボー法を行なう、最大のクラスタ数。
    random_state : int, RandomState instance, default 42
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.
    """
    
    _kmeans.euclidean_distances = new_euclidean_distances
    
    sse = []
    for k in range(k_min, k_max + 1):
        km = _kmeans.KMeans(n_clusters=k, random_state=random_state)
        km.fit(X)
        sse.append(km.inertia_)
    ax.plot(range(k_min, k_max + 1), sse, marker='o')
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('SSE')
