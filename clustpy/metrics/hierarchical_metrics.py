from clustpy.hierarchical._cluster_tree import BinaryClusterTree
from clustpy.metrics import purity
import numpy as np


def leaf_purity(
    labels_true: np.ndarray, labels_pred: np.ndarray, tree: BinaryClusterTree
) -> float:
    """
    Calculates the leaf purity of the tree.
    Uses labels fromm leafs in the tree to calculate the purity (see clustpy.metrics.purity).

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    tree : BinaryClusterTree
        The clustering tree

    Returns
    -------
    leaf_purity : float
        The leaf purity

    References
    -------
    Mautz, Dominik, Claudia Plant, and Christian Böhm. "Deepect: The deep embedded cluster tree."
    Data Science and Engineering 5 (2020): 419-432.
    """
    leaf_nodes, _ = tree.get_leaf_and_split_nodes()
    labels_pred_adj = -np.ones(labels_pred.shape[0])
    for i, leaf_node in enumerate(leaf_nodes):
        labels_pred_adj[np.isin(labels_pred, leaf_node.labels)] = i
    leaf_purity = purity(labels_true, labels_pred_adj)
    return leaf_purity


def dendrogram_purity_ClusterTree(
    labels_true: np.ndarray, labels_pred: np.ndarray, tree: BinaryClusterTree
) -> float:
    # tiny wrapper for compatibility
    dendrogram = tree.export_sklearn_dendrogram()
    return dendrogram_purity(dendrogram, labels_true, labels_pred)


def dendrogram_purity(
    dendrogram: np.ndarray, labels_true: np.ndarray, labels_pred: np.ndarray = None
):
    """
    Calculates the dendrogram purity of the tree.

    Parameters
    ----------
    dendrogram: sklearn/scipy-style dendrogram or the first two columns of it (e.g. AgglomerativeClustering->children_)
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    dendrogram_purity : float
        The dendrogram purity

    References
    -------
    Heller, Katherine A., and Zoubin Ghahramani. "Bayesian hierarchical clustering."
    Proceedings of the 22nd international conference on Machine learning. 2005.

    or

    Kobren, Ari, et al. "A hierarchical algorithm for extreme clustering."
    Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017.
    """
    n = len(labels_true)
    classes, y_idx = np.unique(labels_true, return_inverse=True)
    num_true_classes = len(classes)

    if labels_pred is None:
        labels_pred = np.arange(n)
    elif len(labels_pred) != n:
        raise ValueError("labels_pred must have the same length as labels_true")

    base_clusters, b_idx = np.unique(labels_pred, return_inverse=True)
    num_base_clusters = len(base_clusters)
    parent_matrix = _get_parent_matrix(dendrogram=dendrogram, y_pred=labels_pred)

    # counts_per_class contains for every true class label the number of datapoints
    # with this label for each base class. For a full hierarchy this is either 0 or 1.
    # For an overclustering, larger values are possible.
    counts_per_class = np.zeros((num_true_classes, num_base_clusters), dtype=int)
    np.add.at(counts_per_class, (y_idx, b_idx), 1)

    # parent_matrix[i,:] contains a 1 at position j if j is a descendant of i.
    # the matrix multiplication thus sums up how many original nodes per GT class are at each
    # node in the tree.
    node_counts = counts_per_class @ parent_matrix.T

    purity = 0.0
    total_pairs = 0
    # within-leaf purities (always 0 for complete hierarchies)
    for true_label_index in range(num_true_classes):
        for base_cluster_index in range(num_base_clusters):
            class_count = counts_per_class[true_label_index, base_cluster_index]
            leaf_purity = class_count / np.sum(counts_per_class[:, base_cluster_index])
            weight = class_count * (class_count - 1) / 2
            purity += leaf_purity * weight
            total_pairs += weight

    # purity where inner nodes of the dendrogram serve as LCA
    for true_label_index in range(num_true_classes):
        for index, (child1, child2) in enumerate(dendrogram[:, :2]):
            counts_child1 = node_counts[true_label_index, int(child1)]
            counts_child2 = node_counts[true_label_index, int(child2)]
            weight = counts_child1 * counts_child2

            class_count = node_counts[true_label_index, index + num_base_clusters]
            node_purity = class_count / np.sum(
                node_counts[:, index + num_base_clusters]
            )

            purity += node_purity * weight
            total_pairs += weight

    purity /= total_pairs
    return purity


def _get_parent_matrix(dendrogram: np.ndarray, y_pred: np.ndarray):
    "The parent matrix stores for each cluster which of the basic elements/clusters form it."
    base_clusters = np.unique(y_pred)
    num_base_clusters = len(base_clusters)
    # each line of the dendrogram corresponds to a cluster, plus we need the initial clusters
    parent_matrix = np.zeros(
        (dendrogram.shape[0] + num_base_clusters, num_base_clusters), dtype=np.int8
    )
    # each initial cluster consists just of itself
    for i in range(num_base_clusters):
        parent_matrix[i, i] = 1

    for i, (child1, child2) in enumerate(dendrogram[:, :2]):
        parent_matrix[i + num_base_clusters] = (
            parent_matrix[int(child1)] + parent_matrix[int(child2)]
        )

    assert np.max(parent_matrix) == 1, "input dendrogram does not correspond to a tree"

    return parent_matrix
