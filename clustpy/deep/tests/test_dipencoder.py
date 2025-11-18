from clustpy.deep import DipEncoder, get_dataloader, detect_device
from clustpy.deep.dipencoder import plot_dipencoder_embedding, _get_ssl_loss_of_first_batch
from clustpy.deep.neural_networks import FeedforwardAutoencoder, ConvolutionalAutoencoder
import numpy as np
import torch
from unittest.mock import patch
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _get_dc_test_data, _get_dc_test_neuralnetwork, _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_dipencoder_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DipEncoder(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_dipencoder():
    dipencoder = DipEncoder(3)
    dipencoder, dipencoder2 = _test_dc_algorithm_simple(dipencoder)
    assert np.allclose(dipencoder.projection_axes_, dipencoder2.projection_axes_, atol=1e-1)
    assert dipencoder.index_dict_ == dipencoder2.index_dict_


def test_supervised_dipencoder():
    X, labels = _get_dc_test_data()
    nn = _get_dc_test_neuralnetwork(X.shape[1])
    dipencoder = DipEncoder(3, batch_size=30, pretrain_epochs=3, clustering_epochs=3, random_state=1, neural_network=nn, embedding_size=4)
    assert not hasattr(dipencoder, "labels_")
    dipencoder.fit(X, labels)
    assert dipencoder.labels_.dtype == np.int32
    assert np.array_equal(labels, dipencoder.labels_)


def test_dipencoder_augmentation():
    dipencoder = DipEncoder(10)
    _test_dc_algorithm_with_augmentation(dipencoder)


def test_plot_dipencoder_embedding():
    embedded_data = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
         [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17],
         [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21],
         [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37],
         [38, 38, 38], [39, 39, 39]])
    n_clusters = 3
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    projection_axes = np.array([[11, 11, 11], [30, 30, 30], [19, 19, 19]])
    index_dict = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    assert None == plot_dipencoder_embedding(embedded_data, n_clusters, cluster_labels, projection_axes, index_dict,
                                             show_plot=False)


def test_get_rec_loss_of_first_batch():
    torch.use_deterministic_algorithms(True)
    X = torch.rand((512, 3, 32, 32))
    device = detect_device()
    # Test with FeedforwardAutoencoder
    X_flat = X.reshape(512, -1)
    ff_trainloader = get_dataloader(X_flat, 32, shuffle=True)
    ff_autoencoder = FeedforwardAutoencoder([X_flat.shape[1], 32, 10])
    ae_loss = _get_ssl_loss_of_first_batch(ff_trainloader, ff_autoencoder, torch.nn.MSELoss(), device)
    assert ae_loss > 0
    # Test with ConvolutionalAutoencoder
    conv_trainloader = get_dataloader(X, 32, shuffle=True)
    conv_autoencoder = ConvolutionalAutoencoder(X.shape[-1], [512, 10])
    ae_loss = _get_ssl_loss_of_first_batch(conv_trainloader, conv_autoencoder, torch.nn.MSELoss(), device)
    assert ae_loss > 0


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_dipencoder_obj(mock_fig):
    X, _ = _get_dc_test_data()
    nn = _get_dc_test_neuralnetwork(X.shape[1])
    dipencoder = DipEncoder(3, batch_size=30, pretrain_epochs=1, clustering_epochs=1, random_state=1, neural_network=nn, embedding_size=4)
    dipencoder.fit(X)
    assert None == dipencoder.plot(X)
