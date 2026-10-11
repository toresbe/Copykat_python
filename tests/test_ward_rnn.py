"""The parallel reciprocal-nearest-neighbour Ward engine must give fastcluster's tree."""

import fastcluster
import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster

from copykat_py import baseline
from copykat_py.ward_rnn import ward_linkage_rnn


def _clustered(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(7, d))
    return centers[rng.integers(0, 7, n)] + rng.normal(size=(n, d))


@pytest.mark.parametrize(("n", "d", "seed"), [(2, 3, 0), (3, 1, 1), (60, 5, 2), (700, 40, 3), (1500, 128, 4)])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_rnn_matches_fastcluster(n: int, d: int, seed: int, n_threads: int) -> None:
    X = _clustered(n, d, seed)
    expected = fastcluster.linkage(X, method="ward", metric="euclidean")
    got = ward_linkage_rnn(X, n_threads=n_threads)
    np.testing.assert_array_equal(got[:, [0, 1, 3]], expected[:, [0, 1, 3]])
    np.testing.assert_allclose(got[:, 2], expected[:, 2], rtol=1e-12)


def test_rnn_handles_duplicate_rows() -> None:
    X = np.repeat(_clustered(40, 6, 5), 3, axis=0)
    Z = ward_linkage_rnn(X, n_threads=2)
    assert Z.shape == (119, 4)
    assert np.all(np.diff(Z[:, 2]) >= 0)
    expected = fastcluster.linkage(X, method="ward", metric="euclidean")
    for k in range(2, 8):
        assert np.array_equal(fcluster(Z, k, "maxclust"), fcluster(expected, k, "maxclust"))


def test_ward_linkage_uses_rnn_above_pdist_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPYKAT_WARD_PDIST_MAX_GB", "0")
    X = _clustered(300, 10, 6)
    Z, engine = baseline._ward_linkage(X, n_cores=2)
    assert engine == "ward_rnn"
    np.testing.assert_array_equal(Z[:, [0, 1, 3]], fastcluster.linkage(X, method="ward")[:, [0, 1, 3]])
