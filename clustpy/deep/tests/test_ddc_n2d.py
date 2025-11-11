from clustpy.deep import DDC, N2D
from clustpy.deep.ddc_n2d import DDC_density_peak_clustering
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.manifold import Isomap
from clustpy.utils.checks import check_clustpy_estimator
from clustpy.deep.tests._helpers_for_tests import _test_dc_algorithm_simple


def test_ddc_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(DDC(pretrain_epochs=3, tsne_params={"perplexity": 5}),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_n2d_estimator():
    # Ignore check_methods_subset_invariance due to numerical issues
    check_clustpy_estimator(N2D(3, pretrain_epochs=3, manifold_params={"perplexity": 5}, initial_clustering_params={"covariance_type": "spherical"}),
                            ("check_complex_data", "check_methods_subset_invariance"))


def test_ddc_density_peak_clustering():
    X, labels = make_blobs(50, 2, centers=3, random_state=1)
    # With small and large ratio
    for ratio in [0.1, 999]:
        ddc_dpc = DDC_density_peak_clustering(ratio=ratio)
        assert not hasattr(ddc_dpc, "labels_")
        ddc_dpc.fit(X)
        assert ddc_dpc.labels_.dtype == np.int32
        assert ddc_dpc.labels_.shape == labels.shape
        assert len(np.unique(ddc_dpc.labels_)) == ddc_dpc.n_clusters_
        assert np.array_equal(np.unique(ddc_dpc.labels_), np.arange(ddc_dpc.n_clusters_))


def test_simple_ddc():
    ddc = DDC(ratio=1.1)
    _test_dc_algorithm_simple(ddc, True, False)


def test_simple_n2d():
    n2d = N2D(3)
    _test_dc_algorithm_simple(n2d, True, False)
    n2d_isomap = N2D(3, manifold_class=Isomap, manifold_params={"n_components": 2, "n_neighbors": 10})
    _test_dc_algorithm_simple(n2d_isomap, True, False)
