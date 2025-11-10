from clustpy.deep import ENRC, ACeDeC
import numpy as np
from clustpy.utils.checks import check_clustpy_estimator
import pytest
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple, _test_dc_algorithm_with_augmentation


@pytest.mark.skip(reason="Checks are not completly implemented yet")
def test_acedec_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(ACeDeC(3, pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


@pytest.mark.skip(reason="Checks are not completly implemented yet")
def test_enrc_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(ENRC([3, 3], pretrain_epochs=3, clustering_epochs=3),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_simple_enrc():
    enrc = ENRC([3, 3], debug=True)
    enrc, _ = _test_dc_algorithm_simple(enrc, check_random_state=False, use_nr_data=True)
    # Standard mechanism for checking random state is not working as the orthogonal matrix V is only approximated
    enrc2 = ENRC([3, 3])
    enrc2, _ = _test_dc_algorithm_simple(enrc2, check_random_state=False, use_nr_data=True)
    assert np.array_equal(enrc.labels_, enrc.labels_)
    for i in range(len(enrc.cluster_centers_)):
        assert np.array_equal(enrc.cluster_centers_[i], enrc2.cluster_centers_[i])
    # Test if sgd as init is working
    enrc = ENRC([3, 3], init="sgd")
    enrc, _ = _test_dc_algorithm_simple(enrc, check_random_state=False, use_nr_data=True)


def test_simple_acedec():
    acedec = ACeDeC(3, debug=True)
    acedec, _ = _test_dc_algorithm_simple(acedec, check_random_state=False)
    # Standard mechanism for checking random state is not working as the orthogonal matrix V is only approximated
    acedec2 = ACeDeC(3)
    acedec2, _ = _test_dc_algorithm_simple(acedec2, check_random_state=False)
    assert np.array_equal(acedec.labels_, acedec2.labels_)
    assert np.array_equal(acedec.cluster_centers_[0], acedec2.cluster_centers_[0])


def test_acedec_augmentation():
    acedec = ACeDeC(3)
    _test_dc_algorithm_with_augmentation(acedec)


def test_enrc_augmentation():
    enrc = ENRC([5, 2])
    _test_dc_algorithm_with_augmentation(enrc, use_nr_data=True)
