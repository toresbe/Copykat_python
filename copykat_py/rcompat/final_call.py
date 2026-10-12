"""Steps 7-8 as CopyKAT-R runs them (hg20, not cell-line mode), for releases 1.2.5 and 1.1.0.

Step 7 clusters the bins matrix with ``hclust(method = "ward.D")``, calls the
cluster(s) with the best normal-cell score diploid, and re-centres every cell on
those cells (``adjust``). Step 8 clusters the adjusted matrix again and makes the
final call.

R 1.2.5 cuts both trees at the largest k in 4, 3, 2 whose clusters all have more
than 10 cells, scores a cluster by its number of normal cells, and in step 8
compares the per-gene median profiles of the best- and worst-scoring clusters: if
they correlate at 0.6 or more every cell is diploid; otherwise each cluster goes to
the nearer of the two by Wasserstein distance. R 1.1.0 cuts at k = 2, scores a
cluster by its fraction of normal cells, and calls the better cluster diploid.

``ward.D`` is Ward's update applied to unsquared distances. fastcluster's ``ward``
squares the distances it is given, so it computes ``ward.D`` when given their
square roots; its merge heights are then the square roots of R's.
"""

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import fastcluster
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

from copykat_py._types import ClusterLabels, DistanceMetric, FloatArray, LinkageMatrix, PredictionLabel
from copykat_py.baseline import _collapse_repeated_features
from copykat_py.rcompat.baseline import cutree

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RFinalCall:
    step7_labels: ClusterLabels
    labels: ClusterLabels
    linkage: LinkageMatrix
    predictions: np.ndarray
    adjusted: FloatArray
    consensus_correlation: float | None


def ward_d(mat: FloatArray, distance: DistanceMetric) -> LinkageMatrix:
    """``hclust(d, method = "ward.D")`` of the columns of ``mat`` (features x cells).

    ``d`` is ``parDist(t(mat), method = "euclidean")`` or ``as.dist(1 - cor(mat, method))``.
    Heights are returned on R's scale.
    """
    if distance is DistanceMetric.EUCLIDEAN:
        cells = mat.T
        collapsed = _collapse_repeated_features(cells)  # same distances, fewer columns
        d = pdist(cells if collapsed is None else collapsed, metric="euclidean")
    else:
        x = mat if distance is DistanceMetric.PEARSON else np.apply_along_axis(rankdata, 0, mat)
        corr = np.corrcoef(x, rowvar=False)
        iu = np.triu_indices(corr.shape[0], k=1)
        d = 1.0 - corr[iu]
    Z = fastcluster.linkage(np.sqrt(np.maximum(d, 0.0)), method="ward", preserve_input=False)
    Z[:, 2] = Z[:, 2] ** 2
    return Z


def _choose_k(Z: LinkageMatrix, version: str) -> ClusterLabels:
    if version == "1.1.0":
        return cutree(Z, 2)
    for k in (4, 3, 2):
        labels = cutree(Z, k)
        if np.min(np.bincount(labels)[1:]) > 10:
            return labels
    logger.warning("  no cut into 2-4 clusters has more than 10 cells in each; CopyKAT-R stops here. Using k = 2")
    return cutree(Z, 2)


def _scores(labels: ClusterLabels, cell_names: Sequence[str], preN: Collection[str], version: str) -> np.ndarray:
    """``cl.ID``: per cluster, ``length(intersect(cli, preN))`` over ``length(preN)`` (1.2.5) or ``length(cli)``."""
    names = np.asarray(cell_names, dtype=object)
    normal = set(preN)
    scores = []
    for i in range(1, int(np.max(labels)) + 1):
        cli = names[labels == i]
        shared = len(set(cli) & normal)
        scores.append(shared / (len(normal) if version == "1.2.5" else len(cli)))
    return np.asarray(scores)


def _recycled_equal(labels: ClusterLabels, ids: np.ndarray) -> np.ndarray:
    """R's ``labels == ids``: ``ids`` is recycled along ``labels`` when it has several values."""
    return labels == ids[np.arange(len(labels)) % len(ids)]


def _step7_diploid(labels: ClusterLabels, scores: np.ndarray, version: str) -> np.ndarray:
    best = np.flatnonzero(scores == scores.max()) + 1
    if version == "1.2.5":
        return np.isin(labels, best)  # hc.short %in% which(cl.ID %in% max(cl.ID))
    # 1.1.0: com.pred[which(hc.umap == which(cl.ID == max(cl.ID)))] <- "diploid", then the
    # minimum is written over it the same way, so only cells left "diploid" count.
    worst = np.flatnonzero(scores == scores.min()) + 1
    return _recycled_equal(labels, best) & ~_recycled_equal(labels, worst)


def adjust(uber: FloatArray, diploid: np.ndarray) -> FloatArray:
    """Step 7's re-centring (``results.com.rat`` ... ``mat.adj``), with ``sd()`` using n - 1."""
    rat = uber - uber[:, diploid].mean(axis=1, keepdims=True)
    rat -= rat.mean(axis=0, keepdims=True)
    normal = rat[:, diploid]
    cf_h = normal.std(axis=1, ddof=1)
    base = normal.mean(axis=1)
    # a[abs(a - base) <= 0.25 * cf.h] <- mean(a), with mean(a) taken before the change
    col_means = np.broadcast_to(rat.mean(axis=0, keepdims=True), rat.shape)
    np.copyto(rat, col_means, where=np.abs(rat - base[:, None]) <= (0.25 * cf_h)[:, None])
    rat -= rat.mean(axis=0, keepdims=True)
    return rat


def _wasserstein1d(a: np.ndarray, b: np.ndarray) -> float:
    """``transport::wasserstein1d(a, b)`` for equal lengths: ``mean(abs(sort(b) - sort(a)))``."""
    return float(np.mean(np.abs(np.sort(b) - np.sort(a))))


def call(
    uber: FloatArray,
    gene_level: FloatArray | None,
    cell_names: Sequence[str],
    preN: Collection[str],
    *,
    version: str,
    distance: DistanceMetric,
    low_confidence: bool,
) -> RFinalCall:
    """Steps 7 and 8. ``gene_level`` is ``results.com`` (genes x cells), used by 1.2.5's step 8."""
    uber = np.asarray(uber, dtype=np.float64)
    step7 = _choose_k(ward_d(uber, distance), version)
    diploid = _step7_diploid(step7, _scores(step7, cell_names, preN, version), version)
    mat_adj = adjust(uber, diploid)
    del uber

    Z = ward_d(mat_adj, distance)
    labels = _choose_k(Z, version)
    scores = _scores(labels, cell_names, preN, version)
    names = np.asarray(cell_names, dtype=object)
    consensus_correlation = None
    if version == "1.2.5":
        assert gene_level is not None
        best = np.flatnonzero(scores == scores.max()) + 1
        worst = np.flatnonzero(scores == scores.min()) + 1
        conses_diploid = np.median(gene_level[:, np.isin(labels, best)], axis=1)
        conses_aneuploid = np.median(gene_level[:, np.isin(labels, worst)], axis=1)
        consensus_correlation = float(np.corrcoef(conses_diploid, conses_aneuploid)[0, 1])
        predictions = np.empty(len(labels), dtype=object)
        if consensus_correlation >= 0.6:
            predictions[:] = PredictionLabel.DIPLOID
        else:
            for i in range(1, int(np.max(labels)) + 1):
                conses = np.median(gene_level[:, labels == i], axis=1)
                nearer_diploid = _wasserstein1d(conses, conses_diploid) < _wasserstein1d(conses, conses_aneuploid)
                predictions[labels == i] = PredictionLabel.DIPLOID if nearer_diploid else PredictionLabel.ANEUPLOID
        # R also has `if (0.4 <= cor < 0.6) WNS == "unclassified.prediction"`, a comparison
        # that changes nothing, so that band is not flagged.
    else:
        best = np.flatnonzero(scores == scores.max()) + 1
        worst = np.flatnonzero(scores == scores.min()) + 1
        predictions = names.copy()  # cells matched by neither rule keep their name, as in R
        predictions[_recycled_equal(labels, best)] = PredictionLabel.DIPLOID
        predictions[_recycled_equal(labels, worst)] = PredictionLabel.ANEUPLOID
    if low_confidence:
        predictions[predictions == PredictionLabel.DIPLOID] = PredictionLabel.DIPLOID_LOW_CONFIDENCE
        predictions[predictions == PredictionLabel.ANEUPLOID] = PredictionLabel.ANEUPLOID_LOW_CONFIDENCE
    return RFinalCall(step7, labels, Z, predictions, mat_adj, consensus_correlation)
