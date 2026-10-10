"""The batched numba GMM must reproduce the NumPy per-cell fit.

The kernel repeats NumPy's operations and summation orders, so results are
bit-identical where NumPy's exp/log are the C library's; NumPy's AVX-512
kernels can differ from those in the last bit, hence the tight tolerance.
"""

import numpy as np
import pytest

from copykat_py import baseline


def _profiles(n_genes: int, n_cells: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    states = rng.choice([-0.3, 0.0, 0.3], size=(n_genes, n_cells), p=[0.1, 0.8, 0.1])
    values = states + rng.normal(0.0, 0.1, size=(n_genes, n_cells))
    values[:, ::3] = rng.normal(0.0, 0.02, size=(n_genes, len(range(0, n_cells, 3))))  # near-diploid cells
    return values.astype(np.float32)


@pytest.mark.parametrize("n_genes", [5, 130, 1999])
def test_batched_gmm_matches_numpy_fit(n_genes: int) -> None:
    values = _profiles(n_genes, 12).astype(np.float64)
    sigma0 = [max(0.05, 0.5 * float(np.std(values[:, j]))) for j in range(values.shape[1])]
    means, weights, sigma = baseline._fit_gmm_3component_many(values, sigma0, max_iter=500, n_cores=4)
    for j in range(values.shape[1]):
        ref_means, ref_weights, ref_sigma = baseline._fit_gmm_3component(
            values[:, j], sigma_init=sigma0[j], max_iter=500
        )
        np.testing.assert_allclose(means[j], ref_means, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(weights[j], ref_weights, rtol=1e-12, atol=1e-15)
        assert sigma[j] == pytest.approx(ref_sigma, rel=1e-12)


def test_baseline_gmm_keeps_first_diploid_cells_in_order() -> None:
    values = _profiles(800, 60)
    names = [f"cell{i}" for i in range(values.shape[1])]
    serial = baseline.baseline_gmm(values, names, max_normal=5, n_cores=1, cluster=False)
    parallel = baseline.baseline_gmm(values, names, max_normal=5, n_cores=8, cluster=False)
    assert serial.normal_cells == parallel.normal_cells == names[0:15:3]
    np.testing.assert_array_equal(serial.baseline, parallel.baseline)
