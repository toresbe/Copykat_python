"""Baseline estimation: find diploid normal cells and compute copy-number baseline.

Mirrors baseline.norm.cl.R, baseline.GMM.R, and baseline.synthetic.R from the R package.
"""

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import fastcluster
import numba
import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from copykat_py import backend
from copykat_py._types import (
    AnchorPath,
    BaselineWarning,
    CellByFeature,
    ClusterLabels,
    FloatArray,
    GeneByCell,
    GeneProfile,
    Genome,
    IntArray,
    LinkageMatrix,
    ParallelInfo,
    PredictionLabel,
)
from copykat_py.ward_rnn import ward_linkage_rnn

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Normal-cell reference and baseline profile estimated from the input.

    ``normal_cells`` contains cell names when available, or zero-based column
    indices otherwise. Cluster labels are absent only when baseline clustering
    was explicitly skipped.
    """

    baseline: GeneProfile
    warning: BaselineWarning
    normal_cells: list[str] | IntArray
    cluster_labels: ClusterLabels | None
    anchor_path: AnchorPath = AnchorPath.SIGMA


@dataclass(frozen=True, slots=True)
class SyntheticBaselineResult:
    """Synthetic-normal-adjusted expression and its cluster labels."""

    relative_expression: GeneByCell
    cluster_labels: ClusterLabels


_LAST_CLUSTER_INFO: ParallelInfo = {
    "step": "hierarchical_cluster",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "approximate": False,
    "engine": "scipy.linkage",
}

FULL_CLUSTER_MAX_CELLS = 2000
# Ward linkage runs on a precomputed condensed distance matrix (n*(n-1)/2
# doubles) when the memory it needs fits in this many GB; above it,
# ward_rnn.ward_linkage_rnn avoids the quadratic memory.
# fastcluster.linkage copies the matrix even with preserve_input=False
# (fastcluster 1.3.0), so the peak is two matrices, n*(n-1)*8 bytes: 8 GB
# covers ~31,600 cells (20,000 cells: 3.2 GB; 30,000: 7.2 GB). Override with
# the COPYKAT_WARD_PDIST_MAX_GB environment variable.
WARD_PDIST_MAX_GB = 8.0


def _ward_pdist_fits(n_samples: int) -> bool:
    budget_gb = float(os.getenv("COPYKAT_WARD_PDIST_MAX_GB", WARD_PDIST_MAX_GB))
    # the condensed matrix plus fastcluster.linkage's working copy of it
    return n_samples * (n_samples - 1) * 8 <= budget_gb * 1e9


AUTO_PCA_CELL_COUNT_CUTOFF = 50000
AUTO_PCA_SMALL_SAMPLE = 256
AUTO_PCA_LARGE_SAMPLE = 128
MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF = 20000
MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF = 40000
MOUSE_AUTO_PCA_SMALL_SAMPLE = 512
MOUSE_AUTO_PCA_MEDIUM_SAMPLE = 256
MOUSE_AUTO_PCA_LARGE_SAMPLE = 128
ADAPTIVE_PCA_COMPONENTS = AUTO_PCA_LARGE_SAMPLE


def get_last_cluster_info() -> ParallelInfo:
    return _LAST_CLUSTER_INFO.copy()


def resolve_adaptive_pca_components(
    n_cells: int,
    pca_components: int | None = None,
    genome: Genome = Genome.HG20,
) -> int:
    """Choose the PCA component cap for large-cell clustering.

    When ``pca_components`` is provided, that explicit value is used.
    Otherwise, use the project default policy:
    - human (`hg20`): fewer than 50,000 input cells -> 256 PCs, otherwise 128
    - mouse (`mm10`): fewer than 20,000 input cells -> 512 PCs,
      fewer than 40,000 input cells -> 256 PCs, otherwise 128
    """
    if pca_components is not None:
        return int(pca_components)

    n_cells = int(n_cells)
    genome = Genome(str(genome).strip().lower())

    if genome is Genome.MM10:
        if n_cells < MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF:
            return MOUSE_AUTO_PCA_SMALL_SAMPLE
        if n_cells < MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF:
            return MOUSE_AUTO_PCA_MEDIUM_SAMPLE
        return MOUSE_AUTO_PCA_LARGE_SAMPLE

    if n_cells < AUTO_PCA_CELL_COUNT_CUTOFF:
        return AUTO_PCA_SMALL_SAMPLE
    return AUTO_PCA_LARGE_SAMPLE


def _cluster_sizes(labels: ClusterLabels) -> IntArray:
    _, counts = np.unique(labels, return_counts=True)
    return counts


def _reduce_for_clustering(data: CellByFeature, max_components: int = 64) -> tuple[CellByFeature, int | None]:
    n_samples, n_features = data.shape
    if n_samples <= FULL_CLUSTER_MAX_CELLS or n_features <= 256:
        return data, None

    n_components = min(max_components, n_samples - 1, n_features)
    if n_components < 8:
        return data, None

    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=1234)
    return reducer.fit_transform(data), n_components


def _effective_threads(n_cores: int) -> int:
    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    return max(1, min(int(n_cores), max_cores))


def _pdist_euclidean(data: CellByFeature, n_cores: int = 1, block_bytes: int = 64 << 20) -> FloatArray:
    """Condensed Euclidean distances, bit-identical to ``pdist(data, "euclidean")``.

    With several cores, row blocks are computed with ``cdist`` in threads
    (scipy releases the GIL); each distance is still computed by the same
    kernel from the same two rows, so the result is unchanged.
    """
    n_samples = data.shape[0]
    n_threads = _effective_threads(n_cores)
    if n_threads == 1 or n_samples < 1000:
        return pdist(data, metric="euclidean")

    data = np.ascontiguousarray(data, dtype=np.float64)
    dist = np.empty(n_samples * (n_samples - 1) // 2)
    rows_per_block = max(1, block_bytes // (8 * n_samples))
    # Condensed offset of row i (its distances to j > i)
    row_start = np.concatenate([[0], np.cumsum(np.arange(n_samples - 1, 0, -1))])

    def _block(lo: int) -> None:
        hi = min(lo + rows_per_block, n_samples - 1)
        block = cdist(data[lo:hi], data[lo:], metric="euclidean")
        for i in range(lo, hi):
            dist[row_start[i] : row_start[i] + n_samples - 1 - i] = block[i - lo, i - lo + 1 :]

    with ThreadPoolExecutor(n_threads) as executor:
        list(executor.map(_block, range(0, n_samples - 1, rows_per_block)))
    return dist


def _collapse_repeated_features(data: CellByFeature, block_rows: int = 4096) -> CellByFeature | None:
    """Merge runs of identical adjacent feature columns into one column each.

    Each kept column is scaled by sqrt(run length), so Euclidean distances
    between rows are mathematically unchanged:
    sum_b (x_b - y_b)^2 == sum_runs len * (x_run - y_run)^2.
    Segmented CNA profiles repeat each value across all bins of a segment, so
    this typically shrinks ~12k bins to roughly the number of segments.

    Returns the collapsed matrix, or None when no columns repeat.
    """
    n_samples, n_features = data.shape
    if n_features < 2:
        return None
    changes = np.zeros(n_features - 1, dtype=bool)
    for lo in range(0, n_samples, block_rows):
        block = data[lo : lo + block_rows]
        changes |= np.any(block[:, 1:] != block[:, :-1], axis=0)
    starts = np.concatenate([[0], np.flatnonzero(changes) + 1])
    if len(starts) == n_features:
        return None
    run_lengths = np.diff(np.concatenate([starts, [n_features]]))
    # float64 like pdist's internal arithmetic, so float32 inputs lose nothing extra
    return data[:, starts].astype(np.float64) * np.sqrt(run_lengths)


def _ward_linkage(data: CellByFeature, n_cores: int = 1) -> tuple[LinkageMatrix, str]:
    """Exact Ward linkage of the rows of ``data`` (Euclidean), fastest engine that fits.

    Returns the linkage matrix and the engine name. Both engines produce the
    same merge tree; merge heights can differ in the last bits.
    """
    n_samples = data.shape[0]
    if _ward_pdist_fits(n_samples):
        dist = _pdist_euclidean(data, n_cores=n_cores)
        return fastcluster.linkage(dist, method="ward", preserve_input=False), "pdist+fastcluster.linkage"
    # O(n * d) memory and multithreaded; 16x faster than fastcluster.linkage_vector
    # on 40,000 x 128 with 32 threads.
    return ward_linkage_rnn(data, n_threads=_effective_threads(n_cores)), "ward_rnn"


def _hierarchical_cluster(
    data: CellByFeature,
    n_clusters: int,
    method: str = "ward",
    metric: str = "euclidean",
    max_cells: int = 65536,
    n_cores: int = 1,
    reduce: bool = True,
    pca_components: int = 64,
) -> tuple[ClusterLabels, LinkageMatrix]:
    """Hierarchical clustering with fastcluster-first execution.

    For Ward + Euclidean clustering, use exact Ward linkage for
    both small and large inputs (see ``_ward_linkage``: a precomputed distance
    matrix within ``WARD_PDIST_MAX_GB``, ``ward_rnn.ward_linkage_rnn`` above).
    Runs of identical adjacent feature columns (bins inside one CNA segment)
    are collapsed first, which leaves distances unchanged; when the collapsed
    width fits within the PCA component cap, that PCA would be lossless and is
    skipped. The ``max_cells`` argument is kept for compatibility but is not used.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix (cells × genomic bins).
    n_clusters : int
        Number of clusters to cut.
    method : str
        Linkage method.
    metric : str
        Distance metric.
    max_cells : int
        Retained for backward compatibility; no longer used as a fastcluster
        cutoff.
    reduce : bool
        Whether to apply PCA reduction for large Ward/Euclidean clustering.
        Set to False for R-compatibility paths that should cluster the full
        cells × features matrix through an explicit distance object.
    pca_components : int
        Maximum number of PCA components to retain when reduction is enabled.

    Returns
    -------
    labels : np.ndarray
        Cluster labels (1-indexed to match R convention).
    linkage_matrix : np.ndarray or None
        Linkage matrix if available.
    """
    n_samples, n_features = data.shape
    _LAST_CLUSTER_INFO.update(
        {
            "requested_cores": int(n_cores),
            "effective_cores": 1,
            "tasks": int(n_samples),
            "approximate": False,
            "engine": "scipy.linkage",
        }
    )

    if backend.use_gpu() and metric == "euclidean" and method.startswith("ward"):
        from copykat_py.gpu import ops

        reduce_to = None
        if reduce and not backend.exact_algorithms():
            pca_components_used = min(pca_components, n_samples - 1, n_features)
            if n_samples > FULL_CLUSTER_MAX_CELLS and n_features > 256 and pca_components_used >= 8:
                reduce_to = pca_components_used
        Z, engine = ops.ward_cluster(data, reduce_to=reduce_to)
        _LAST_CLUSTER_INFO["engine"] = engine
        _LAST_CLUSTER_INFO["approximate"] = "pca" in engine
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return labels, Z

    if not reduce:
        if metric == "euclidean" and method.startswith("ward"):
            collapsed = _collapse_repeated_features(data)
            if collapsed is not None:
                Z, engine = _ward_linkage(collapsed, n_cores=n_cores)
                _LAST_CLUSTER_INFO["engine"] = f"full_matrix+dedup{collapsed.shape[1]}+{engine}"
            else:
                Z, engine = _ward_linkage(data, n_cores=n_cores)
                _LAST_CLUSTER_INFO["engine"] = f"full_matrix+{engine}"
            _LAST_CLUSTER_INFO["effective_cores"] = _effective_threads(n_cores)
        else:
            dist = pdist(data, metric=cast(Any, metric))  # validated by scipy at runtime
            Z = fastcluster.linkage(dist, method=method, preserve_input=True)
            _LAST_CLUSTER_INFO["engine"] = "full_pdist+fastcluster.linkage"
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return labels, Z

    # Keep Ward + Euclidean on the vectorized full/PCA matrix path.
    if metric == "euclidean" and method.startswith("ward"):
        _LAST_CLUSTER_INFO["effective_cores"] = _effective_threads(n_cores)
        collapsed = _collapse_repeated_features(data)
        if collapsed is not None:
            # Mirror _reduce_for_clustering's decision: PCA applies only for
            # large, wide inputs and keeps at most this many components.
            pca_components_used = min(pca_components, n_samples - 1, n_features)
            pca_applies = n_samples > FULL_CLUSTER_MAX_CELLS and n_features > 256 and pca_components_used >= 8
            if not pca_applies or collapsed.shape[1] <= pca_components_used:
                Z, engine = _ward_linkage(collapsed, n_cores=n_cores)
                _LAST_CLUSTER_INFO["engine"] = f"dedup{collapsed.shape[1]}+{engine}"
                labels = fcluster(Z, t=n_clusters, criterion="maxclust")
                return labels, Z
        cluster_data, n_components = _reduce_for_clustering(data, max_components=pca_components)
        Z, engine = _ward_linkage(cluster_data, n_cores=n_cores)
        _LAST_CLUSTER_INFO["engine"] = engine
        if n_components is not None:
            _LAST_CLUSTER_INFO["approximate"] = True
            _LAST_CLUSTER_INFO["engine"] = f"pca{n_components}+{engine}"
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return labels, Z

    if metric == "euclidean":
        dist = pdist(data, metric="euclidean")
    else:
        dist = pdist(data, metric=cast(Any, metric))  # validated by scipy at runtime

    Z = fastcluster.linkage(dist, method=method, preserve_input=True)
    _LAST_CLUSTER_INFO["engine"] = "fastcluster.linkage"

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, Z


def _fit_gmm_3component(
    data: FloatArray,
    mu_init: Sequence[float] | None = None,
    sigma_init: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[FloatArray, FloatArray, float]:
    """Fit a 3-component Gaussian Mixture Model (gain, neutral, loss).

    Parameters
    ----------
    data : np.ndarray, shape (n,)
        Expression values.
    mu_init : array-like or None
        Initial means for 3 components.
    sigma_init : float or None
        Initial shared standard deviation.

    Returns
    -------
    means : np.ndarray, shape (3,)
        Fitted means.
    weights : np.ndarray, shape (3,)
        Fitted weights (lambdas).
    sigma : float
        Fitted shared sigma.
    """
    if sigma_init is None:
        sigma_init = max(0.05, 0.5 * float(np.std(data)))
    if mu_init is None:
        mu_init = [-0.2, 0.0, 0.2]

    if backend.use_gpu():
        from copykat_py.gpu.gmm import fit_gmm_3component_batch

        means, weights, sigma = fit_gmm_3component_batch(
            np.asarray(data, dtype=np.float64).ravel()[None, :],
            [sigma_init],
            mu_init=mu_init,
            max_iter=max_iter,
            tol=tol,
        )
        return means[0], weights[0], float(sigma[0])

    x = np.asarray(data, dtype=np.float64).ravel()
    means = np.asarray(mu_init, dtype=np.float64).copy()
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    sigma = float(max(sigma_init, 1e-8))
    prev_loglik = -np.inf

    for _ in range(max_iter):
        z = (x[:, None] - means[None, :]) / sigma
        density = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
        weighted = density * weights[None, :]
        row_sums = weighted.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = np.finfo(np.float64).tiny
        resp = weighted / row_sums

        nk = resp.sum(axis=0)
        nk[nk <= 0] = np.finfo(np.float64).tiny
        weights = nk / len(x)
        means = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - means[None, :]) ** 2).sum() / len(x)
        sigma = float(np.sqrt(max(var, 1e-12)))

        loglik = float(np.log(row_sums.ravel()).sum())
        if abs(loglik - prev_loglik) < tol * (abs(prev_loglik) + tol):
            break
        prev_loglik = loglik

    return means, weights, sigma


@njit
def _pairwise_sum(a: FloatArray, lo: int, n: int) -> float:
    """``a[lo:lo + n].sum()`` with NumPy's pairwise summation order (bit-identical)."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        m = n - n % 8
        for i in range(lo + 8, lo + m, 8):
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for i in range(lo + m, lo + n):
            res += a[i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@njit(parallel=True)
def _fit_gmm_3component_rows(
    X: FloatArray,
    sigma_init: FloatArray,
    mu_init: FloatArray,
    max_iter: int,
    tol: float,
    out_means: FloatArray,
    out_weights: FloatArray,
    out_sigma: FloatArray,
) -> None:
    """``_fit_gmm_3component`` for every row of ``X`` (cells x genes), rows in parallel threads.

    Performs the same floating-point operations in the same order as the NumPy
    version, including NumPy's summation orders (pairwise for full sums,
    sequential down columns, left to right across a row of three), so each row
    gets the same result as fitting it alone with NumPy wherever NumPy's
    ``exp``/``log`` are the C library's (no AVX-512 kernels).
    """
    n_rows, n = X.shape
    tiny = np.finfo(np.float64).tiny
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    for r in prange(n_rows):
        x = X[r]
        resp = np.empty(3 * n)
        terms = np.empty(3 * n)
        log_rs = np.empty(n)
        m0, m1, m2 = mu_init[0], mu_init[1], mu_init[2]
        w0 = w1 = w2 = 1.0 / 3.0
        sigma = max(sigma_init[r], 1e-8)
        prev_loglik = -np.inf
        for _ in range(max_iter):
            norm = sigma * sqrt_2pi
            nk0 = nk1 = nk2 = 0.0
            sx0 = sx1 = sx2 = 0.0
            for i in range(n):
                xi = x[i]
                z0 = (xi - m0) / sigma
                z1 = (xi - m1) / sigma
                z2 = (xi - m2) / sigma
                d0 = np.exp(-0.5 * z0 * z0) / norm * w0
                d1 = np.exp(-0.5 * z1 * z1) / norm * w1
                d2 = np.exp(-0.5 * z2 * z2) / norm * w2
                rs = d0 + d1 + d2
                if rs <= 0:
                    rs = tiny
                log_rs[i] = np.log(rs)
                q0 = d0 / rs
                q1 = d1 / rs
                q2 = d2 / rs
                resp[3 * i] = q0
                resp[3 * i + 1] = q1
                resp[3 * i + 2] = q2
                nk0 += q0
                nk1 += q1
                nk2 += q2
                sx0 += q0 * xi
                sx1 += q1 * xi
                sx2 += q2 * xi
            if nk0 <= 0:
                nk0 = tiny
            if nk1 <= 0:
                nk1 = tiny
            if nk2 <= 0:
                nk2 = tiny
            w0, w1, w2 = nk0 / n, nk1 / n, nk2 / n
            m0, m1, m2 = sx0 / nk0, sx1 / nk1, sx2 / nk2
            for i in range(n):
                xi = x[i]
                e0 = xi - m0
                e1 = xi - m1
                e2 = xi - m2
                terms[3 * i] = resp[3 * i] * (e0 * e0)
                terms[3 * i + 1] = resp[3 * i + 1] * (e1 * e1)
                terms[3 * i + 2] = resp[3 * i + 2] * (e2 * e2)
            var = (0.0 + _pairwise_sum(terms, 0, 3 * n)) / n
            sigma = np.sqrt(max(var, 1e-12))
            loglik = 0.0 + _pairwise_sum(log_rs, 0, n)
            if abs(loglik - prev_loglik) < tol * (abs(prev_loglik) + tol):
                break
            prev_loglik = loglik
        out_means[r, 0], out_means[r, 1], out_means[r, 2] = m0, m1, m2
        out_weights[r, 0], out_weights[r, 1], out_weights[r, 2] = w0, w1, w2
        out_sigma[r] = sigma


def _gmm_threads(n_cores: int, n_tasks: int) -> int:
    return max(1, min(_effective_threads(n_cores), n_tasks, numba.config.NUMBA_NUM_THREADS))


def _fit_gmm_3component_many(
    columns: GeneByCell,
    sigma_init: Sequence[float],
    max_iter: int,
    n_cores: int = 1,
    tol: float = 1e-8,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit ``_fit_gmm_3component`` (default ``mu_init``) to every column, in parallel threads.

    Returns means (n x 3), weights (n x 3) and sigmas (n,).
    """
    X = np.ascontiguousarray(np.asarray(columns, dtype=np.float64).T)
    n_rows = X.shape[0]
    means = np.empty((n_rows, 3))
    weights = np.empty((n_rows, 3))
    sigma = np.empty(n_rows)
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(_gmm_threads(n_cores, n_rows))
    try:
        _fit_gmm_3component_rows(
            X,
            np.asarray(sigma_init, dtype=np.float64),
            np.array([-0.2, 0.0, 0.2]),
            int(max_iter),
            float(tol),
            means,
            weights,
            sigma,
        )
    finally:
        numba.set_num_threads(previous_threads)
    return means, weights, sigma


def baseline_norm_cl(
    norm_mat_smooth: GeneByCell,
    min_cells: int = 5,
    n_cores: int = 1,
    cell_names: Sequence[str] | None = None,
    pca_components: int | None = None,
    genome: Genome = Genome.HG20,
    anchor_selector: Callable[[ClusterLabels, int], tuple[int, AnchorPath]] | None = None,
) -> BaselineResult:
    """Find a cluster of diploid cells using integrative clustering + GMM variance test.

    Mirrors baseline.norm.cl.R:
    1. Hierarchical clustering into 6 groups
    2. GMM on each cluster consensus to estimate variance
    3. Cluster with minimum variance is 'confident normal'
    4. Silhouette + F-test to validate

    Parameters
    ----------
    norm_mat_smooth : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    min_cells : int
        Minimum cells per cluster.
    n_cores : int
        Number of cores.

    Returns
    -------
    A ``BaselineResult`` containing the estimated baseline profile, warning,
    normal-cell reference, cluster labels, and anchor path.
    """
    n_genes, n_cells = norm_mat_smooth.shape
    selected_pca_components = resolve_adaptive_pca_components(
        n_cells,
        pca_components=pca_components,
        genome=genome,
    )

    # Hierarchical clustering
    data_t = norm_mat_smooth.T  # cells × genes
    step4_reduce = n_cells > FULL_CLUSTER_MAX_CELLS
    km = 6
    labels, Z = _hierarchical_cluster(
        data_t,
        km,
        method="ward",
        metric="euclidean",
        n_cores=n_cores,
        reduce=step4_reduce,
        pca_components=selected_pca_components,
    )

    # Reduce clusters until all have > min_cells
    while not np.all(_cluster_sizes(labels) > min_cells):
        km -= 1
        if Z is not None:
            labels = fcluster(Z, t=km, criterion="maxclust")
        else:
            labels, Z = _hierarchical_cluster(
                data_t,
                km,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step4_reduce,
                pca_components=selected_pca_components,
            )
        if km == 2:
            break

    # GMM on each cluster consensus - parallelized for speed
    unique_clusters = sorted(set(labels))

    def fit_gmm_sigma(cluster_consensus: GeneProfile) -> float:
        """Fit GMM for a single cluster consensus and return its sigma (for parallel execution)."""
        sx = max(0.05, 0.5 * float(np.std(cluster_consensus)))
        return _fit_gmm_3component(cluster_consensus, sigma_init=sx, max_iter=5000)[2]

    # Parallel GMM fitting. Consensus profiles are computed up front and fitted
    # in threads so workers never receive a copy of the full matrix.
    consensus_profiles = [np.median(norm_mat_smooth[:, labels == cl_id], axis=1) for cl_id in unique_clusters]
    SDM = np.array(
        Parallel(n_jobs=n_cores, prefer="threads")(
            delayed(fit_gmm_sigma)(consensus) for consensus in consensus_profiles
        )
    )

    # Silhouette width for 2-cluster separation
    if Z is not None:
        labels_2 = fcluster(Z, t=2, criterion="maxclust")
    else:
        labels_2, _ = _hierarchical_cluster(
            data_t,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step4_reduce,
            pca_components=selected_pca_components,
        )

    # Compute silhouette on a stratified subsample for medium/large datasets.
    # Stratify by labels_2 so each cluster (diploid/aneuploid) is proportionally
    # represented; floor at 200 per cluster to protect rare populations.
    if n_cells > 3000:
        rng = np.random.RandomState(1234)
        target = max(3000, min(int(0.20 * n_cells), 20000))
        idx_parts = []
        for cl in np.unique(labels_2):
            cl_idx = np.where(labels_2 == cl)[0]
            n_take = max(200, int(target * len(cl_idx) / n_cells))
            n_take = min(n_take, len(cl_idx))
            idx_parts.append(rng.choice(cl_idx, size=n_take, replace=False))
        idx = np.concatenate(idx_parts)
        wn = silhouette_score(data_t[idx], labels_2[idx], metric="euclidean")
    else:
        wn = silhouette_score(data_t, labels_2, metric="euclidean")

    # F-test: compare max variance to min variance
    from scipy.stats import f as f_dist

    f_stat = max(SDM) ** 2 / min(SDM) ** 2
    PDt = f_dist.sf(f_stat, n_genes, n_genes)

    if wn <= 0.15 or not np.all(_cluster_sizes(labels) > min_cells) or PDt > 0.05:
        WNS: BaselineWarning = BaselineWarning.UNCLASSIFIED
        logger.warning("  low confidence in classification")
    else:
        WNS = BaselineWarning.NONE

    # Cluster with minimum sigma is the 'confident normal' cluster
    min_sigma_idx = np.argmin(SDM)
    normal_cluster_id = unique_clusters[min_sigma_idx]
    anchor_path = AnchorPath.SIGMA
    if anchor_selector is not None:
        normal_cluster_id, anchor_path = anchor_selector(labels, int(normal_cluster_id))

    normal_mask = labels == normal_cluster_id
    basel = np.median(norm_mat_smooth[:, normal_mask], axis=1)
    preN_indices = np.where(normal_mask)[0]
    if cell_names is not None:
        names = np.asarray(cell_names, dtype=object)
        preN = names[preN_indices].tolist()
    else:
        preN = preN_indices

    return BaselineResult(
        baseline=basel,
        warning=WNS,
        normal_cells=preN,
        cluster_labels=labels,
        anchor_path=anchor_path,
    )


# Cells per wave of parallel per-cell GMM fits in baseline_gmm, as a multiple
# of the thread count: waves start at one cell per thread and double up to this.
_GMM_MAX_WAVE_PER_THREAD = 16


def _gmm_fits_in_order(CNA_mat: GeneByCell, n_cores: int) -> Iterator[tuple[int, FloatArray, FloatArray]]:
    """Yield ``(cell index, means, weights)`` of each cell's GMM fit, in cell order.

    Cells are fitted in waves of parallel threads, so a caller that stops early
    wastes at most one wave; each fit is identical to a serial
    ``_fit_gmm_3component`` call.
    """
    n_cells = CNA_mat.shape[1]
    if backend.use_gpu():
        for m in range(n_cells):
            sam = CNA_mat[:, m]
            sg = max(0.05, 0.5 * float(np.std(sam)))
            means, weights, _sigma = _fit_gmm_3component(sam, sigma_init=sg, max_iter=500)
            yield m, means, weights
        return
    n_threads = _gmm_threads(n_cores, n_cells)
    wave = n_threads
    lo = 0
    while lo < n_cells:
        hi = min(n_cells, lo + wave)
        block = CNA_mat[:, lo:hi]
        sigma0 = [max(0.05, 0.5 * float(np.std(block[:, j]))) for j in range(hi - lo)]
        wave_means, wave_weights, _ = _fit_gmm_3component_many(block, sigma0, max_iter=500, n_cores=n_threads)
        for j in range(hi - lo):
            yield lo + j, wave_means[j], wave_weights[j]
        lo = hi
        wave = min(2 * wave, _GMM_MAX_WAVE_PER_THREAD * n_threads)


def baseline_gmm(
    CNA_mat: GeneByCell,
    cell_names: Sequence[str],
    max_normal: int = 5,
    mu_cut: float = 0.05,
    Nfraq_cut: float = 0.99,
    RE_before: BaselineResult | None = None,
    n_cores: int = 1,
    pca_components: int | None = None,
    genome: Genome = Genome.HG20,
    cluster: bool = True,
) -> BaselineResult:
    """Identify diploid cells one-by-one using GMM (fallback when clustering is uncertain).

    Mirrors baseline.GMM.R.

    Parameters
    ----------
    CNA_mat : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    cell_names : list
        Cell names corresponding to columns.
    max_normal : int
        Max number of normal cells to find before stopping.
    mu_cut : float
        Threshold for neutral mean.
    Nfraq_cut : float
        Min fraction of genes in neutral state.
    RE_before : BaselineResult or None
        Previous baseline result to fall back on.
    n_cores : int
        Number of cores.
    cluster : bool
        Whether to hierarchically cluster all cells for the returned
        ``cluster_labels``. Callers that only need the baseline and normal-cell
        reference can pass False to skip it; ``cluster_labels`` is then None
        (unless ``RE_before`` is returned).

    Returns
    -------
    A ``BaselineResult`` containing the estimated baseline profile, warning,
    normal-cell reference, and optional cluster labels.
    """
    n_cells = CNA_mat.shape[1]
    N_normal = []
    N_normal_labels = []

    for m, means, weights in _gmm_fits_in_order(CNA_mat, n_cores):
        # Check if any component mean is near zero (neutral)
        neutral_mask = np.abs(means) <= mu_cut
        has_neutral_component = np.any(neutral_mask)
        neutral_fraction = np.sum(weights[neutral_mask])
        pred = (
            PredictionLabel.DIPLOID
            if has_neutral_component and neutral_fraction > Nfraq_cut
            else PredictionLabel.ANEUPLOID
        )

        N_normal_labels.append(pred)

        if pred is PredictionLabel.DIPLOID:
            N_normal.append(cell_names[m])

        if len(N_normal) >= max_normal:
            break

    # Hierarchical clustering for the full dataset
    labels = None
    if cluster and (len(N_normal) > 2 or RE_before is None):
        selected_pca_components = resolve_adaptive_pca_components(
            n_cells,
            pca_components=pca_components,
            genome=genome,
        )
        labels, _ = _hierarchical_cluster(
            CNA_mat.T,
            6,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=n_cells > FULL_CLUSTER_MAX_CELLS,
            pca_components=selected_pca_components,
        )

    if len(N_normal) > 2:
        WNS: BaselineWarning = BaselineWarning.NONE
        preN = N_normal
        normal_mask = np.isin(np.asarray(cell_names, dtype=object), np.asarray(preN, dtype=object))
        basel = np.mean(CNA_mat[:, normal_mask], axis=1)
        return BaselineResult(baseline=basel, warning=WNS, normal_cells=preN, cluster_labels=labels)
    else:
        if RE_before is not None:
            return RE_before
        else:
            # Fallback: use the full dataset median as baseline
            WNS = BaselineWarning.UNCLASSIFIED
            return BaselineResult(
                baseline=np.median(CNA_mat, axis=1),
                warning=WNS,
                normal_cells=N_normal,
                cluster_labels=labels,
            )


def baseline_synthetic(
    norm_mat: GeneByCell,
    min_cells: int = 10,
    n_cores: int = 1,
    pca_components: int | None = None,
    genome: Genome = Genome.HG20,
) -> SyntheticBaselineResult:
    """Estimate baseline using synthetic normal profiles (for cell line data).

    Mirrors baseline.synthetic.R.

    Parameters
    ----------
    norm_mat : np.ndarray, shape (n_genes, n_cells)
        Smoothed expression matrix.
    min_cells : int
        Min cells per cluster.
    n_cores : int
        Number of cores.

    Returns
    -------
    A ``SyntheticBaselineResult`` containing relative expression values and
    their cluster labels.
    """
    n_cells = norm_mat.shape[1]
    selected_pca_components = resolve_adaptive_pca_components(
        n_cells,
        pca_components=pca_components,
        genome=genome,
    )
    data_t = norm_mat.T

    km = 6
    labels, Z = _hierarchical_cluster(
        data_t,
        km,
        method="ward",
        metric="euclidean",
        pca_components=selected_pca_components,
    )

    while not all(np.bincount(labels)[np.bincount(labels) > 0] > min_cells):
        km -= 1
        if Z is not None:
            labels = fcluster(Z, t=km, criterion="maxclust")
        else:
            labels, Z = _hierarchical_cluster(
                data_t,
                km,
                method="ward",
                metric="euclidean",
                pca_components=selected_pca_components,
            )
        if km == 2:
            break

    rng = np.random.RandomState(123)
    expr_relat_parts = []
    unique_clusters = sorted(set(labels))

    for cl_id in unique_clusters:
        mask = labels == cl_id
        data_c = norm_mat[:, mask]
        sd_per_gene = np.std(data_c, axis=1)
        syn_norm = rng.normal(0, sd_per_gene)
        relat = data_c - syn_norm[:, np.newaxis]
        expr_relat_parts.append(relat)

    expr_relat = np.hstack(expr_relat_parts)

    # Reorder to match original cell order
    cluster_order = np.concatenate([np.where(labels == cl_id)[0] for cl_id in unique_clusters])

    # Create inverse permutation
    inv_perm = np.argsort(cluster_order)
    expr_relat = expr_relat[:, inv_perm]

    return SyntheticBaselineResult(relative_expression=expr_relat, cluster_labels=labels)
