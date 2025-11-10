from clustpy.deep.neural_networks import VariationalAutoencoder
from clustpy.deep.neural_networks.variational_autoencoder import _vae_sampling
import torch
from clustpy.deep.tests._helpers_for_tests import _get_dc_test_data


def test_variational_autoencoder():
    data, _ = _get_dc_test_data()
    batch_size = 30
    data_batch = torch.Tensor(data[:batch_size])
    embedding_dim = 3
    autoencoder = VariationalAutoencoder(layers=[data.shape[1], 10, embedding_dim])
    # Test encoding
    embedded_mean, embedded_var = autoencoder.encode(data_batch)
    assert embedded_mean.shape == (batch_size, embedding_dim)
    assert embedded_var.shape == (batch_size, embedding_dim)
    # Test decoding
    torch.manual_seed(0)
    embedded_sample = _vae_sampling(embedded_mean, embedded_var)
    decoded = autoencoder.decode(embedded_sample)
    assert decoded.shape == (batch_size, data.shape[1])
    # Test forwarding (needs seed, since sampling is random)
    torch.manual_seed(0)
    forward_sample, forward_mean, forward_var, forwarded_reconstruct = autoencoder.forward(data_batch)
    assert torch.equal(forward_sample, embedded_sample)
    assert torch.equal(forward_mean, embedded_mean)
    assert torch.equal(forward_var, embedded_var)
    assert torch.equal(forwarded_reconstruct, decoded)
    # Test fitting
    assert autoencoder.fitted is False
    autoencoder.fit(n_epochs=3, optimizer_params={"lr": 1e-3}, data=data)
    assert autoencoder.fitted is True
