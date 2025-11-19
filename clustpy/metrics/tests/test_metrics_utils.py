from clustpy.metrics.external_clustering_metrics import _check_number_of_points
import pytest
import numpy as np

def test_check_number_of_points():
    l1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    l2 = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4])
    assert _check_number_of_points(l1, l2) == True
    with pytest.raises(Exception):
        _check_number_of_points(l1, l2[1:])