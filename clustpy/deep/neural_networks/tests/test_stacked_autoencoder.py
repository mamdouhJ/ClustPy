from clustpy.deep.neural_networks import StackedAutoencoder
from clustpy.deep.tests._helpers_for_tests import _get_dc_test_data


def test_stacked_autoencoder():
    data, _ = _get_dc_test_data()
    embedding_dim = 3
    autoencoder = StackedAutoencoder(layers=[data.shape[1], 10, embedding_dim])
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs_per_layer=3, n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True
