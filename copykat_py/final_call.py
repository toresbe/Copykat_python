"""Final cell clustering, copy-number calls, and baseline adjustment."""

from collections.abc import Sequence
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt

from copykat_py import backend
from copykat_py._types import (
    BoolArray,
    ClusterLabels,
    FloatArray,
    LinkageMatrix,
    PredictionLabel,
)
from copykat_py.baseline import FULL_CLUSTER_MAX_CELLS, _hierarchical_cluster


class FinalCallResult(TypedDict):
    """Ward clustering and the per-cell copy-number calls derived from it."""

    labels: ClusterLabels
    Z: LinkageMatrix
    predictions: npt.NDArray[np.object_]


def _normal_cells_to_names(
    normal_cells: str | bytes | Sequence[Any] | npt.NDArray[Any] | None,
    reference_cell_names: Sequence[str],
) -> set[str]:
    """Resolve named normal cells and integer indices against the reference-cell list."""
    if normal_cells is None:
        return set()

    if isinstance(normal_cells, str | bytes):
        return {str(normal_cells)}

    values = list(normal_cells)
    if not values:
        return set()

    names = []
    for value in values:
        if isinstance(value, int | np.integer):
            idx = int(value)
            if 0 <= idx < len(reference_cell_names):
                names.append(reference_cell_names[idx])
        else:
            names.append(str(value))
    return set(names)


def _assign_binary_labels(
    cluster_labels: ClusterLabels,
    scores: Sequence[float] | FloatArray,
    max_score_label: PredictionLabel,
    min_score_label: PredictionLabel,
) -> npt.NDArray[np.object_]:
    """Map the maximum and minimum cluster scores to their respective labels.

    Scores must be ordered to match the sorted unique values in ``cluster_labels``.
    Tied maximum or minimum scores assign that label to every tied cluster. If
    all scores tie, the minimum-score label wins because that assignment runs last.
    """
    labels = np.empty(len(cluster_labels), dtype=object)
    labels[:] = ""
    cluster_vals = np.asarray(sorted(set(cluster_labels)))
    score_arr = np.asarray(scores, dtype=float)
    max_score = np.max(score_arr)
    min_score = np.min(score_arr)

    for cluster_val in cluster_vals[score_arr == max_score]:
        labels[cluster_labels == cluster_val] = max_score_label
    for cluster_val in cluster_vals[score_arr == min_score]:
        labels[cluster_labels == cluster_val] = min_score_label
    return labels


def cluster_cells(
    values: FloatArray,
    *,
    n_cores: int,
    pca_components: int | None,
) -> tuple[ClusterLabels, LinkageMatrix]:
    """Ward-cluster the cells in a feature-by-cell matrix.

    Args:
        values: Matrix shaped ``(features, cells)``; cells are clustered on its transpose.
        n_cores: Maximum number of CPU workers requested for clustering.
        pca_components: Optional PCA cap used by the large-sample clustering path.

    Returns:
        A pair containing 1-based cluster labels and the Ward linkage matrix.
    """
    return _hierarchical_cluster(
        values.T,
        2,
        method="ward",
        metric="euclidean",
        n_cores=n_cores,
        reduce=values.shape[1] > FULL_CLUSTER_MAX_CELLS,
        pca_components=pca_components,
    )


def cluster_and_call(
    values: FloatArray,
    cell_names: Sequence[str],
    reference_cell_names: Sequence[str],
    normal_cells: str | bytes | Sequence[Any] | npt.NDArray[Any] | None,
    *,
    n_cores: int,
    pca_components: int | None,
    prediction_override: BoolArray | None = None,
    low_confidence: bool = False,
) -> FinalCallResult:
    """Cluster cells and assign diploid/aneuploid labels from reference overlap or CNA magnitude.

    Args:
        values: Feature-by-cell CNA matrix used both for clustering and fallback scoring.
        cell_names: Cell names in the same order as the matrix columns and returned labels.
        reference_cell_names: Ordered source for resolving integer entries in ``normal_cells``.
        normal_cells: Known normal-cell names, a single name, or integer indices into
            ``reference_cell_names``. When absent or empty, calls use CNA magnitude.
        n_cores: Maximum number of CPU workers requested for clustering.
        pca_components: Optional PCA cap used by the large-sample clustering path.
        prediction_override: Optional boolean call per cell. ``True`` means aneuploid;
            the Ward result is still returned for ordering.
        low_confidence: Replace ordinary diploid/aneuploid labels with their low-confidence forms.

    Returns:
        A ``FinalCallResult`` containing Ward ``labels``, linkage ``Z``, and per-cell
        ``predictions``. With known normals, clusters are scored by their normal-cell
        fraction. Otherwise, the cluster with the lower mean absolute CNA magnitude is
        called diploid. Overrides are applied after cluster scoring, then low-confidence
        labels are applied last.
    """
    if len(cell_names) != values.shape[1]:
        raise ValueError("cell_names must contain one name per matrix column")
    if prediction_override is not None and prediction_override.shape != (values.shape[1],):
        raise ValueError("prediction_override must contain one boolean value per cell")

    labels, linkage = cluster_cells(values, n_cores=n_cores, pca_components=pca_components)

    if normal_cells is not None and len(normal_cells) > 0:
        normal_names = _normal_cells_to_names(normal_cells, reference_cell_names)
        scores = []
        for cluster_value in sorted(set(labels)):
            cluster_names = [cell_names[index] for index in range(len(cell_names)) if labels[index] == cluster_value]
            scores.append(len(set(cluster_names) & normal_names) / max(len(cluster_names), 1))
    else:
        scores = [float(np.mean(np.abs(values[:, labels == cluster_value]))) for cluster_value in sorted(set(labels))]
        scores = [-score for score in scores]

    predictions = _assign_binary_labels(labels, scores, PredictionLabel.DIPLOID, PredictionLabel.ANEUPLOID)
    if prediction_override is not None:
        predictions = np.where(prediction_override, PredictionLabel.ANEUPLOID, PredictionLabel.DIPLOID)
    if low_confidence:
        predictions = np.where(
            predictions == PredictionLabel.DIPLOID,
            PredictionLabel.DIPLOID_LOW_CONFIDENCE,
            predictions,
        )
        predictions = np.where(
            predictions == PredictionLabel.ANEUPLOID,
            PredictionLabel.ANEUPLOID_LOW_CONFIDENCE,
            predictions,
        )
    return {"labels": labels, "Z": linkage, "predictions": predictions}


def adjust_baseline_inplace(
    mat: FloatArray,
    diploid_mask: BoolArray,
    chunk_elems: int = 1 << 24,
) -> FloatArray:
    """Subtract the diploid baseline and flatten noise around it, mutating ``mat``.

    ``diploid_mask`` selects the cell columns used to estimate the per-feature
    baseline and noise threshold. Callers should pass a mask containing at least
    one diploid cell. The CPU implementation processes rows in chunks; the GPU
    backend performs the equivalent operation on device.
    """
    if backend.use_gpu():
        from copykat_py.gpu import ops

        return cast(FloatArray, ops.adjust_baseline(mat, diploid_mask))
    mat -= mat[:, diploid_mask].mean(axis=1, keepdims=True)
    mat -= mat.mean(axis=0, keepdims=True)

    diploid = mat[:, diploid_mask]
    cf_h = np.std(diploid, axis=1)
    base = np.mean(diploid, axis=1)
    del diploid
    threshold = 0.25 * cf_h
    cell_means = mat.mean(axis=0, keepdims=True)

    step = max(1, chunk_elems // max(1, mat.shape[1]))
    for start in range(0, mat.shape[0], step):
        block = mat[start : start + step]
        noise_mask = (
            np.abs(block - base[start : start + step, np.newaxis]) <= threshold[start : start + step, np.newaxis]
        )
        np.copyto(block, np.broadcast_to(cell_means, block.shape), where=noise_mask)

    mat -= mat.mean(axis=0, keepdims=True)
    return mat
