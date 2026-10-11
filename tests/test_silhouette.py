"""Blocked-GEMM silhouette must agree with scikit-learn's silhouette_score."""

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from copykat_py import silhouette


@pytest.mark.parametrize(("n", "d", "k"), [(40, 3, 2), (500, 60, 3), (900, 8, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_sklearn(monkeypatch: pytest.MonkeyPatch, n: int, d: int, k: int, dtype: type) -> None:
    rng = np.random.default_rng(n + k)
    X = (rng.normal(size=(k, d))[rng.integers(0, k, n)] + rng.normal(size=(n, d))).astype(dtype)
    labels = rng.integers(0, k, n)
    labels[0] = k + 1  # a singleton cluster scores 0
    monkeypatch.setattr(silhouette, "_BLOCK_BYTES", 8 * n * 7)  # several row blocks
    expected = silhouette_score(X, labels, metric="euclidean")
    assert silhouette.silhouette_mean(X, labels, n_threads=2) == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_rejects_a_single_cluster() -> None:
    with pytest.raises(ValueError):
        silhouette.silhouette_mean(np.zeros((5, 2)), np.ones(5))
