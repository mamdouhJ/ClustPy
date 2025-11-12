from clustpy.deep import AEC
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


def test_aec_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(AEC(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_aec():
    aec = AEC(3)
    aec, aec2 = _test_dc_algorithm_simple(aec)
    assert np.allclose(aec.cluster_centers_, aec2.cluster_centers_, atol=1e-1)


def test_aec_augmentation():
    aec = AEC(10)
    _test_dc_algorithm_with_augmentation(aec)
