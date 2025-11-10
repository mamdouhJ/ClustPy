from clustpy.deep import get_dataloader, DEN
import numpy as np
from scipy.spatial.distance import pdist, squareform
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _get_dc_test_data
from clustpy.deep.neural_networks import FeedforwardAutoencoder


def test_den_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DEN(n_clusters=3, pretrain_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_den():
    den = DEN(3, group_size=1)
    den, den2 = _test_dc_algorithm_simple(den)
    assert np.array_equal(den.cluster_centers_, den2.cluster_centers_)
    # Use group_size
    X, labels = _get_dc_test_data()
    nn = (FeedforwardAutoencoder, {"layers": [X.shape[1], 10, 6], "random_state": 42})
    den = DEN(3, group_size=[2, 2, 2], batch_size=30, pretrain_epochs=3, random_state=42, neural_network=nn, embedding_size=6)
    assert not hasattr(den, "labels_")
    den.fit(X)
    assert den.labels_.dtype == np.int32
    assert den.labels_.shape == labels.shape
    X_embed = den.transform(X)
    assert X_embed.shape == (X.shape[0], den.embedding_size)


def test_den_with_predefined_neighbors():
    n_neighbors = 5
    # Get dataloader with neighbors
    X, labels = _get_dc_test_data()
    dist_matrix = squareform(pdist(X))
    neighbor_ids = np.argsort(dist_matrix, axis=1)
    neighbors = [X[neighbor_ids[:, 1 + i]] for i in range(n_neighbors)]
    trainloader = get_dataloader(X, 30, True, additional_inputs=neighbors)
    testloader = get_dataloader(X, 30, False)
    custom_dataloaders = (trainloader, testloader)
    # Start test
    nn = (FeedforwardAutoencoder, {"layers": [X.shape[1], 10, 6], "random_state": 42})
    den = DEN(3, n_neighbors=n_neighbors, group_size=[2, 2, 2], batch_size = 30, pretrain_epochs=3, random_state=42, 
              neural_network=nn, embedding_size=6,
              weight_locality_constraint=2, weight_sparsity_constraint=2, heat_kernel_t_parameter=2, 
              group_lasso_lambda_parameter=2, custom_dataloaders=custom_dataloaders)
    assert not hasattr(den, "labels_")
    den.fit(X)
    assert den.labels_.dtype == np.int32
    assert den.labels_.shape == labels.shape
    X_embed = den.transform(X)
    assert X_embed.shape == (X.shape[0], den.embedding_size)
