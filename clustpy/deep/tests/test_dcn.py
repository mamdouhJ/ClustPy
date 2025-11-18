from clustpy.deep import DCN
from clustpy.deep.dcn import _compute_centroids
import torch
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_dcn_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DCN(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_dcn():
    dcn = DCN(3)
    dcn, dcn2 = _test_dc_algorithm_simple(dcn)
    assert np.allclose(dcn.cluster_centers_, dcn2.cluster_centers_, atol=1e-1)
    assert np.array_equal(dcn.dcn_labels_, dcn2.dcn_labels_)
    assert np.allclose(dcn.dcn_cluster_centers_, dcn2.dcn_cluster_centers_, atol=1e-1)


def test_compute_centroids():
    embedded = torch.tensor([[0., 1., 1.], [1., 0., 1.], [2., 2., 1.], [1., 2., 2.], [3., 4., 5.]])
    centers = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    count = torch.tensor([1, 3, 1])
    labels = torch.tensor([0, 0, 1, 1, 2])
    new_centers, new_count = _compute_centroids(centers, embedded, count, labels)
    assert torch.equal(new_count, torch.tensor([3, 5, 2]))
    desired_centers = torch.tensor([[2 / 3 * 0.5 + 1 / 3 * 1., 2 / 3 * 1. + 1 / 3 * 0., 2 / 3 * 1. + 1 / 3 * 1.],
                                    [4 / 5 * 2. + 1 / 5 * 1., 4 / 5 * 2. + 1 / 5 * 2., 4 / 5 * 1.75 + 1 / 5 * 2.],
                                    [0.5 * 3. + 0.5 * 3., 0.5 * 3. + 0.5 * 4., 0.5 * 3. + 0.5 * 5.]])
    assert torch.all(torch.isclose(new_centers, desired_centers))  # torch.equal is not working due to numerical issues


def test_dcn_augmentation():
    dcn = DCN(10)
    _test_dc_algorithm_with_augmentation(dcn)
