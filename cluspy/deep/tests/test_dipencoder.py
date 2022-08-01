from cluspy.data import load_optdigits
from cluspy.deep import DipEncoder
from cluspy.deep.dipencoder import plot_dipencoder_embedding
import numpy as np

def test_simple_dipencoder_with_optdigits():
    X, labels = load_optdigits()
    dipencoder = DipEncoder(10, pretrain_epochs=10, clustering_epochs=10)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X)
    assert dipencoder.labels_.shape == labels.shape

def test_plot_dipencoder_embedding():
    embedded_data = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
         [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17],
         [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21],
         [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37],
         [38, 38, 38], [39, 39, 39]])
    n_clusters = 3
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    projection_axes = np.array([[11, 11, 11], [30,30,30], [19,19,19]])
    index_dict = {(0,1):0, (0,2):1, (1,2):2}
    plot_dipencoder_embedding(embedded_data, n_clusters, cluster_labels, projection_axes, index_dict, show_plot=False)
    # Only check if error is thrown
    assert True