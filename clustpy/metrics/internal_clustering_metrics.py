from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import numpy as np


def cvnn_score(X: np.ndarray, labels: np.ndarray | int | tuple, n_neighbors: int = 5, metric: str = "euclidean") -> float | np.ndarray:
    """
    Evaluate the quality of predicted labels by computing the clustering validation index based on nearest neighbors (CVNN).
    The score is calculated by adding a nearest-neighbor-based cluster separation value with a cluster compactness vale based on inner-cluster distances.
    Usually, it is used with a list of label arrays, i.e., labels is of type list or tuple.
    In this case, the score will be normalized to a value within [0, 2].
    If labels is a single array (of type np.ndarray) a single score is returned that is not normalized.
    In both cases, a lower value indicates a better clustering result (less neighbors in separate clusters and lower inner-cluster distances).

    Parameters
    ----------
    X : np.ndarray
        The data set
    labels : np.ndarray | list | tuple
        The labels as predicted by a clustering algorithm. If labels is a list/tuple it should contain multiple labels arrays of type np.ndarray
    n_neighbors : int
        The amount of neighbors to consider when calculating the cluster separation score. An object is not considered its own neighbor (default: 5)
    metric : str
        The metric used to identify the neighbors and to calculate the inner-cluster distance.
        See scipy.spatial.distance.pdist for more information (default: 'euclidean')

    Returns
    -------
    cvnn : float | np.ndarray
        The cvnn score of type float if labels contains a single labels array, i.e., is of type np.ndarray.
        Alternatively, a np.ndarray containing the normalized cvnn scores.

    References
    -------
    Liu, Yanchi, et al. "Understanding and enhancement of internal clustering validation measures."
    IEEE transactions on cybernetics 43.3 (2013): 982-994.
    """
    def _internlal_cvnn_score(X: np.ndarray, labels: np.ndarray, n_neighbors: int, metric: str):
        """
        The real calculation method of the CVNN score. 
        """
        assert isinstance(labels, np.ndarray), "labels must be of type np.nddary. Your input has type {0}".format(type(labels))
        unique_clusters = np.unique(labels)
        cluster_separation_scores = np.zeros(unique_clusters.shape[0])
        cluster_compactness_scores = np.zeros(unique_clusters.shape[0])
        # Calculate neighbor weights
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
        _, indices = nbrs.kneighbors()
        n_neighbors_not_in_cluster = np.zeros(X.shape[0])
        for k in range(n_neighbors):
            n_neighbors_not_in_cluster += (labels != labels[indices[:, k]])
        n_neighbors_not_in_cluster /= n_neighbors
        # Do per-cluster calculations
        for c in unique_clusters:
            in_cluster = (labels == c)
            # Calculate separation (mean of neighbor weights in cluster)
            cluster_separation_scores[c] = n_neighbors_not_in_cluster[in_cluster].mean()
            # Calculate compartness (mean of pair-wise distances in cluster)
            X_in_cluster = X[in_cluster]
            cluster_distances = pdist(X_in_cluster, metric=metric)
            in_cluster_pairs = (X_in_cluster.shape[0] * (X_in_cluster.shape[0] - 1)) / 2
            cluster_compactness_scores[c] = cluster_distances.sum() / max(1, in_cluster_pairs)
        # Calculate final CVNN
        cluster_separation_final = cluster_separation_scores.max()
        cluster_compactness_final = cluster_compactness_scores.sum()
        return cluster_separation_final, cluster_compactness_final
    
    if isinstance(labels, list) or isinstance(labels, tuple):
        # Calculate cluster separation and cluster compactness for each labels array in the list
        n_labels = len(labels)
        cluster_separations = np.zeros(n_labels)
        cluster_compactnesses = np.zeros(n_labels)
        for i, l in enumerate(labels):
            cluster_separation_l, cluster_compactness_l = _internlal_cvnn_score(X, l, n_neighbors, metric)
            cluster_separations[i] = cluster_separation_l
            cluster_compactnesses[i] = cluster_compactness_l
        # Normalize scores
        cvnn = cluster_separations / cluster_separations.max() + cluster_compactnesses / cluster_compactnesses.max()
    elif isinstance(labels, np.ndarray):
        # Do not normalize scores
        cluster_separation, cluster_compactness = _internlal_cvnn_score(X, labels, n_neighbors, metric)
        cvnn = cluster_separation + cluster_compactness
    else:
        raise ValueError("The labels must be of type list/tuple (indicating a list of different labels) or np.ndarray (indicating a single labels array). Your input is {0}".format(type(labels)))
    return cvnn
