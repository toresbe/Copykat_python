"""MCMC segmentation: Poisson-Gamma posterior estimation with KS-test breakpoint detection.

Mirrors CNA.MCMC.R from the R package.

Algorithm:
1. Compute cluster consensus profiles (median per cluster)
2. For each cluster consensus, find breakpoints by:
   a. Divide genome into windows of `bins` genes
   b. Sample Poisson-Gamma posteriors for adjacent windows (MCpoissongamma)
   c. KS-test between posteriors; if D > cut_cor, call breakpoint
3. Union all breakpoints across clusters
4. For each cell, compute posterior means per segment
"""

import itertools

import numpy as np
from numba import jit
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv, gammaln
from scipy.stats import ks_2samp

from copykat_py._types import ClusterLabels, FloatArray, GeneByCell, ParallelInfo, SegmentationResult

_LAST_PAR_INFO: ParallelInfo = {
    "step": "cna_mcmc",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
    "mc_samples": 1000,
    "engine": "posterior_sampling",
}


def get_last_cna_mcmc_info() -> ParallelInfo:
    return _LAST_PAR_INFO.copy()


@jit(nopython=True)
def _mc_poisson_gamma_numba(
    data: FloatArray, alpha: float, beta: float = 1.0, mc: int = 1000, seed: int = 42
) -> FloatArray:
    """Sample from the posterior of a Poisson-Gamma model using Numba.

    Prior: lambda ~ Gamma(alpha, beta)
    Likelihood: X_i ~ Poisson(lambda)
    Posterior: lambda | X ~ Gamma(alpha + sum(X), beta + n)

    Parameters
    ----------
    data : np.ndarray
        Observed counts (back-transformed expression values).
    alpha : float
        Prior shape parameter (set to the mean of data).
    beta : float
        Prior rate parameter.
    mc : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (mc,)
        Posterior samples.
    """
    # numba only supports the legacy np.random API.
    np.random.seed(seed)  # noqa: NPY002
    n = len(data)
    post_shape = alpha + np.sum(data)
    post_rate = beta + n
    # Gamma distribution: shape and scale=1/rate
    samples = np.random.gamma(post_shape, 1.0 / post_rate, mc)  # noqa: NPY002
    return samples


def _find_breakpoints_for_cluster(
    consensus: FloatArray, bins: int, cut_cor: float, rng_seed: int = 42, mc_samples: int = 1000
) -> list[int]:
    """Find breakpoints in a cluster consensus profile.

    Parameters
    ----------
    consensus : np.ndarray, shape (n_genes,)
        Exponentiated consensus profile.
    bins : int
        Window size.
    cut_cor : float
        KS test cutoff.
    rng_seed : int
        Random seed.
    mc_samples : int
        Number of MCMC samples.

    Returns
    -------
    list
        List of breakpoint indices.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(consensus)

    # Create bin boundaries
    breks = [*range(0, (n // bins - 1) * bins, bins), n - 1]

    bre = []
    for i in range(len(breks) - 2):
        seg1 = consensus[breks[i] : breks[i + 1] + 1]
        a1 = max(float(np.mean(seg1)), 0.001)
        posterior1 = _mc_poisson_gamma_numba(seg1, a1, 1.0, mc=mc_samples, seed=rng.randint(0, 2**31))

        seg2 = consensus[breks[i + 1] + 1 : breks[i + 2] + 1]
        a2 = max(float(np.mean(seg2)), 0.001)
        posterior2 = _mc_poisson_gamma_numba(seg2, a2, 1.0, mc=mc_samples, seed=rng.randint(0, 2**31))

        ks_stat, _ = ks_2samp(posterior1, posterior2)
        if ks_stat > cut_cor:
            bre.append(breks[i + 1])

    return bre


def _gamma_ks_distance(a1: float, r1: float, a2: float, r2: float, tail: float = 1e-12) -> float:
    """Supremum CDF distance between Gamma(shape, rate) posteriors.

    The CDF difference is extremal where the densities cross. In log-x space
    the log-density ratio has at most two roots; tail quantiles bound omitted
    probability mass by ``tail``.
    """
    if a1 == a2 and r1 == r2:
        return 0.0
    c = a1 * np.log(r1) - gammaln(a1) - a2 * np.log(r2) + gammaln(a2)

    def log_density_ratio(t: float) -> float:
        return (a1 - a2) * t - (r1 - r2) * np.exp(t) + c

    lo = np.log(min(gammaincinv(a1, tail) / r1, gammaincinv(a2, tail) / r2))
    hi = np.log(max(gammaincinv(a1, 1 - tail) / r1, gammaincinv(a2, 1 - tail) / r2))
    edges = [lo, hi]
    if r1 != r2 and (a1 - a2) / (r1 - r2) > 0:
        turning_point = np.log((a1 - a2) / (r1 - r2))
        if lo < turning_point < hi:
            edges = [lo, turning_point, hi]

    distance = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        f_left, f_right = log_density_ratio(left), log_density_ratio(right)
        if f_left == 0:
            root = left
        elif f_right == 0:
            root = right
        elif f_left * f_right < 0:
            root = brentq(log_density_ratio, left, right, xtol=1e-14, rtol=1e-14)
        else:
            continue
        x = np.exp(root)
        distance = max(distance, abs(gammainc(a1, r1 * x) - gammainc(a2, r2 * x)))
    return float(distance)


def _find_breakpoints_exact(consensus: FloatArray, bins: int, cut_cor: float) -> list[int]:
    """Find breakpoints with the exact posterior-Gamma KS statistic."""
    n = len(consensus)
    boundaries = [*range(0, (n // bins - 1) * bins, bins), n - 1]
    breaks = []
    for i in range(len(boundaries) - 2):
        left = consensus[boundaries[i] : boundaries[i + 1] + 1]
        right = consensus[boundaries[i + 1] + 1 : boundaries[i + 2] + 1]
        shape_left = max(float(np.mean(left)), 0.001) + float(np.sum(left))
        shape_right = max(float(np.mean(right)), 0.001) + float(np.sum(right))
        distance = _gamma_ks_distance(shape_left, 1.0 + len(left), shape_right, 1.0 + len(right))
        if distance > cut_cor:
            breaks.append(boundaries[i + 1])
    return breaks


def cna_mcmc(
    clu: ClusterLabels,
    fttmat: GeneByCell,
    bins: int = 25,
    cut_cor: float = 0.1,
    n_cores: int = 1,
    mc_samples: int | None = None,
    ks_method: str = "mc",
) -> SegmentationResult:
    """MCMC segmentation of copy number data.

    Parameters
    ----------
    clu : np.ndarray, shape (n_cells,)
        Cluster assignments for cells (1-indexed).
    fttmat : np.ndarray, shape (n_genes, n_cells)
        Relative expression matrix (log-scale).
    bins : int
        Window size for segmentation.
    cut_cor : float
        KS test cutoff for breakpoint calling.
    n_cores : int
        Number of parallel workers.
    mc_samples : int or None
        Number of MCMC samples (default: 1000); reduce for speed on small datasets.

    Returns
    -------
    dict with keys:
        'logCNA': np.ndarray, shape (n_genes, n_cells) - segmented CNA values
        'breaks': list - breakpoint positions
    """
    n_genes, n_cells = fttmat.shape
    if ks_method not in {"mc", "exact"}:
        raise ValueError("ks_method must be 'mc' or 'exact'")

    # Adaptive MC sample size: fewer samples for small datasets (speed optimization)
    if mc_samples is None:
        mc_samples = 1000 if n_cells > 500 else min(1000, max(500, n_cells * 2))

    # Step 1: Compute cluster consensus profiles (median per cluster)
    unique_clusters = sorted(set(clu))
    CON = np.column_stack([np.median(fttmat[:, clu == cl_id], axis=1) for cl_id in unique_clusters])

    # Back-transform: exp()
    norm_mat_sm = np.exp(CON)

    # Step 2: Find breakpoints for each cluster consensus
    breakpoints: set[int] = set()
    for c in range(norm_mat_sm.shape[1]):
        if ks_method == "exact":
            bre = _find_breakpoints_exact(norm_mat_sm[:, c], bins, cut_cor)
        else:
            bre = _find_breakpoints_for_cluster(
                norm_mat_sm[:, c], bins, cut_cor, rng_seed=42 + c, mc_samples=mc_samples
            )
        breakpoints.update({0, *bre, n_genes - 1})

    BR = sorted(breakpoints)

    # Step 3: For each cell, compute segment posterior means.
    # With alpha initialized to the segment mean and beta fixed to 1,
    # the Poisson-Gamma posterior mean reduces exactly to the segment mean.
    norm_mat_all = np.exp(fttmat)
    cumsum = np.vstack(
        [
            np.zeros((1, n_cells), dtype=norm_mat_all.dtype),
            np.cumsum(norm_mat_all, axis=0),
        ]
    )

    _LAST_PAR_INFO.update(
        {
            "parallel": False,
            "requested_cores": int(n_cores),
            "effective_cores": 1,
            "tasks": int(len(BR) - 1),
            "chunk_size": 0,
            "mc_samples": int(mc_samples),
            "engine": f"{ks_method}_ks+closed_form_segment_mean",
        }
    )

    logCNA = np.empty((n_genes, n_cells), dtype=np.float32)
    for left, right in itertools.pairwise(BR):
        seg_sum = cumsum[right + 1] - cumsum[left]
        seg_len = max(1, right - left + 1)
        seg_mean = np.maximum(seg_sum / seg_len, 1e-300)
        logCNA[left : right + 1, :] = np.log(seg_mean).astype(np.float32, copy=False)

    return {"logCNA": logCNA, "breaks": BR}
