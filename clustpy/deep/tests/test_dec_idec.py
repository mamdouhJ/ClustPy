from clustpy.deep import DEC, IDEC
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_dec_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DEC(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_idec_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(IDEC(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_dec():
    dec = DEC(3)
    dec, dec2 = _test_dc_algorithm_simple(dec)
    assert np.allclose(dec.cluster_centers_, dec2.cluster_centers_, atol=1e-1)
    assert np.array_equal(dec.dec_labels_, dec2.dec_labels_)
    assert np.allclose(dec.dec_cluster_centers_, dec2.dec_cluster_centers_, atol=1e-1)


def test_simple_idec():
    idec = IDEC(3)
    idec, idec2 = _test_dc_algorithm_simple(idec)
    assert np.allclose(idec.cluster_centers_, idec2.cluster_centers_, atol=1e-1)
    assert np.array_equal(idec.dec_labels_, idec2.dec_labels_)
    assert np.allclose(idec.dec_cluster_centers_, idec2.dec_cluster_centers_, atol=1e-1)


def test_dec_augmentation():
    dec = DEC(10)
    _test_dc_algorithm_with_augmentation(dec)


def test_idec_augmentation():
    idec = IDEC(10)
    _test_dc_algorithm_with_augmentation(idec)
