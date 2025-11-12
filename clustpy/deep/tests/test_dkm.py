from clustpy.deep import DKM
from clustpy.deep.dkm import _get_default_alphas
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_dkm_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DKM(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_dkm():
    dkm = DKM(3, alphas=(None, 0.1, 1))
    dkm, dkm2 = _test_dc_algorithm_simple(dkm)
    assert np.array_equal(dkm.cluster_centers_, dkm2.cluster_centers_)
    assert np.array_equal(dkm.dkm_labels_, dkm2.dkm_labels_)
    assert np.array_equal(dkm.dkm_cluster_centers_, dkm2.dkm_cluster_centers_)
    # Test different alpha
    dkm = DKM(3, alphas=0.1)
    dkm, dkm2 = _test_dc_algorithm_simple(dkm)
    assert np.array_equal(dkm.cluster_centers_, dkm2.cluster_centers_)
    assert np.array_equal(dkm.dkm_labels_, dkm2.dkm_labels_)
    assert np.array_equal(dkm.dkm_cluster_centers_, dkm2.dkm_cluster_centers_)


def test_get_default_alphas():
    obtained_alphas = _get_default_alphas(init_alpha=0.1, n_alphas=5)
    expected_alphas = [0.1, 0.42320861065570825, 0.7515684111296623, 1.077971160195895, 1.4087110115785935]
    assert np.allclose(obtained_alphas, expected_alphas)


def test_dkm_augmentation():
    dkm = DKM(10, alphas=(None, 0.1, 2))
    _test_dc_algorithm_with_augmentation(dkm)
