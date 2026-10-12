"""Step 4 as CopyKAT-R runs it: ``baseline.norm.cl``, ``baseline.GMM`` and known normals.

Same decisions as R, on this package's fast engines: Ward linkage on every gene
(no PCA), cluster numbers in ``cutree``'s order, mixtools' EM (``mixtools.py``),
``sd()`` with n - 1, the silhouette over every cell, every cluster tied for the
lowest sigma taken as normal, and R's ``baseline.GMM`` fallback.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.stats import f as f_dist

from copykat_py._medians import group_medians
from copykat_py._types import BaselineWarning, ClusterLabels, GeneByCell, GeneProfile, LinkageMatrix
from copykat_py.baseline import _effective_threads, _ward_linkage
from copykat_py.rcompat.mixtools import normalmix_em
from copykat_py.silhouette import silhouette_mean

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RBaseline:
    baseline: GeneProfile
    warning: BaselineWarning
    normal_cells: list[str]
    cluster_labels: ClusterLabels


def cutree(Z: LinkageMatrix, k: int) -> ClusterLabels:
    """R's ``cutree(fit, k)``: clusters numbered 1, 2, ... in order of their first cell."""
    labels = fcluster(Z, t=k, criterion="maxclust")
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(np.argsort(first))  # rank of each fcluster label by first appearance
    return (order[np.searchsorted(np.unique(labels), labels)] + 1).astype(np.int64)


def ward_d2_tree(norm_mat_smooth: GeneByCell, n_cores: int) -> LinkageMatrix:
    """``hclust(parDist(t(x)), method = "ward.D2")`` on every gene."""
    Z, _engine = _ward_linkage(np.ascontiguousarray(norm_mat_smooth.T, dtype=np.float64), n_cores=n_cores)
    return Z


def cut_with_min_size(Z: LinkageMatrix, min_cells: int) -> ClusterLabels:
    """k = 6, then fewer until every cluster has more than ``min_cells`` cells (k >= 2)."""
    km = 6
    labels = cutree(Z, km)
    while not np.all(np.bincount(labels)[1:] > min_cells):
        km -= 1
        labels = cutree(Z, km)
        if km == 2:
            break
    return labels


def r_sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1))


def baseline_norm_cl(norm_mat_smooth: GeneByCell, cell_names: Sequence[str], min_cells: int, n_cores: int) -> RBaseline:
    """``copykat::baseline.norm.cl``."""
    n_genes = norm_mat_smooth.shape[0]
    Z = ward_d2_tree(norm_mat_smooth, n_cores)
    ct = cut_with_min_size(Z, min_cells)
    clusters = np.arange(1, int(np.max(ct)) + 1)
    consensus = group_medians(norm_mat_smooth, [ct == i for i in clusters], n_cores)
    sx = np.array([max(0.05, 0.5 * r_sd(c)) for c in consensus])
    _mu, _lam, SDM, restarted = normalmix_em(np.column_stack(consensus), sx, maxit=5000)
    if restarted.any():
        logger.warning("  mixtools would restart from random values here; R's results may differ")

    wn = silhouette_mean(norm_mat_smooth.T, cutree(Z, 2), n_threads=_effective_threads(n_cores))
    PDt = f_dist.sf(SDM.max() ** 2 / SDM.min() ** 2, n_genes, n_genes)
    sizes_ok = bool(np.all(np.bincount(ct)[1:] > min_cells))
    if wn <= 0.15 or not sizes_ok or PDt > 0.05:
        warning = BaselineWarning.UNCLASSIFIED
        logger.warning("  low confidence in classification")
    else:
        warning = BaselineWarning.NONE

    normal = np.isin(ct, clusters[np.equal(SDM, SDM.min())])
    (basel,) = group_medians(norm_mat_smooth, [normal], n_cores)
    names = np.asarray(cell_names, dtype=object)
    return RBaseline(basel, warning, names[normal].tolist(), ct)


def baseline_gmm(
    CNA_mat: GeneByCell,
    cell_names: Sequence[str],
    before: RBaseline,
    *,
    max_normal: int = 5,
    mu_cut: float = 0.05,
    Nfraq_cut: float = 0.99,
    n_cores: int = 1,
) -> RBaseline:
    """``copykat:::baseline.GMM``: scan cells in order until ``max_normal`` look diploid.

    R's version also re-clusters every cell, but ``copykat()`` discards those
    clusters (it keeps ``CL`` from ``baseline.norm.cl``), so that is skipped.
    """
    n_cells = CNA_mat.shape[1]
    found: list[int] = []
    wave = max(1, _effective_threads(n_cores))
    lo = 0
    while lo < n_cells and len(found) < max_normal:
        hi = min(n_cells, lo + wave)
        block = np.asarray(CNA_mat[:, lo:hi], dtype=np.float64)
        sg = np.array([max(0.05, 0.5 * r_sd(block[:, j])) for j in range(hi - lo)])
        mu, lam, _sigma, _restarted = normalmix_em(block, sg, maxit=500)
        for j in range(hi - lo):
            neutral = np.abs(mu[j]) <= mu_cut
            if neutral.any() and lam[j][neutral].sum() > Nfraq_cut:
                found.append(lo + j)
                if len(found) >= max_normal:
                    break
        lo = hi
        wave = min(2 * wave, 16 * max(1, _effective_threads(n_cores)))
    if len(found) > 2:
        mask = np.zeros(n_cells, dtype=bool)
        mask[found] = True
        basel = np.asarray(CNA_mat[:, mask], dtype=np.float64).mean(axis=1)
        names = np.asarray(cell_names, dtype=object)
        return RBaseline(basel, BaselineWarning.NONE, names[mask].tolist(), before.cluster_labels)
    return before


def known_normal_clusters(norm_mat_smooth: GeneByCell, min_cells: int, n_cores: int) -> ClusterLabels:
    """Clusters for the known-normal path of ``copykat()``."""
    return cut_with_min_size(ward_d2_tree(norm_mat_smooth, n_cores), min_cells)
