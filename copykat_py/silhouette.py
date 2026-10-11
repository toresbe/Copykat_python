"""Mean silhouette width (Euclidean) with blocked GEMM distances, as on the GPU backend."""

import numpy as np
from threadpoolctl import threadpool_limits

from copykat_py._types import CellByFeature, ClusterLabels

# Bytes of float64 distances per row block.
_BLOCK_BYTES = 256 << 20


def silhouette_mean(data: CellByFeature, labels: ClusterLabels, n_threads: int = 1) -> float:
    """Mean silhouette width of the rows of ``data`` under ``labels`` (Euclidean).

    Same definition as ``sklearn.metrics.silhouette_score``: a point's
    silhouette is 0 in a singleton cluster, and points in other clusters are
    compared by mean distance. Distances come from float32 GEMMs
    (||x||^2 + ||y||^2 - 2 x.y) on the centred data over row blocks, like
    sklearn's float32 distances, and are summed per cluster in float64; the
    result agrees with sklearn to about 1e-9.
    """
    X64 = np.asarray(data, dtype=np.float64)
    # translation-invariant; centring keeps the norms, hence cancellation, small
    X = (X64 - X64.mean(axis=0)).astype(np.float32)
    del X64
    n = X.shape[0]
    uniq, inv = np.unique(np.asarray(labels), return_inverse=True)
    k = len(uniq)
    if not 2 <= k <= n - 1:
        raise ValueError(f"Number of labels is {k}. Valid values are 2 to n_samples - 1 (inclusive)")
    onehot = np.zeros((n, k))
    onehot[np.arange(n), inv] = 1.0
    counts = onehot.sum(axis=0)
    sq = np.einsum("ij,ij->i", X, X, dtype=np.float64).astype(np.float32)
    sums = np.empty((n, k))
    step = max(1, _BLOCK_BYTES // (12 * n))  # float32 block plus its float64 copy
    with threadpool_limits(max(1, int(n_threads)), user_api="blas"):
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            D = X[lo:hi] @ X.T
            D *= -2.0
            D += sq[lo:hi, None]
            D += sq[None, :]
            np.maximum(D, 0.0, out=D)
            np.sqrt(D, out=D)
            D[np.arange(hi - lo), np.arange(lo, hi)] = 0.0
            sums[lo:hi] = D.astype(np.float64) @ onehot
    own = counts[inv]
    a = sums[np.arange(n), inv] / np.maximum(own - 1, 1)
    mean_other = sums / counts[None, :]
    mean_other[np.arange(n), inv] = np.inf
    b = mean_other.min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (b - a) / np.maximum(a, b)
    s = np.where(own > 1, np.nan_to_num(s), 0.0)
    return float(s.mean())
