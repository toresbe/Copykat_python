"""Final cell clustering, copy-number calls, and baseline adjustment."""

from collections.abc import Sequence
from typing import Any, cast

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


def _preN_to_names(
    preN: str | bytes | Sequence[Any] | npt.NDArray[Any] | None,
    cell_names: Sequence[str],
) -> set[str]:
    if preN is None:
        return set()

    if isinstance(preN, str | bytes):
        return {str(preN)}

    values = list(preN)
    if not values:
        return set()

    names = []
    for value in values:
        if isinstance(value, int | np.integer):
            idx = int(value)
            if 0 <= idx < len(cell_names):
                names.append(cell_names[idx])
        else:
            names.append(str(value))
    return set(names)


def _assign_binary_labels(
    cluster_labels: ClusterLabels,
    scores: Sequence[float] | FloatArray,
    high_label: PredictionLabel,
    low_label: PredictionLabel,
) -> npt.NDArray[np.object_]:
    labels = np.empty(len(cluster_labels), dtype=object)
    labels[:] = ""
    cluster_vals = np.asarray(sorted(set(cluster_labels)))
    score_arr = np.asarray(scores, dtype=float)
    max_score = np.max(score_arr)
    min_score = np.min(score_arr)

    for cluster_val in cluster_vals[score_arr == max_score]:
        labels[cluster_labels == cluster_val] = high_label
    for cluster_val in cluster_vals[score_arr == min_score]:
        labels[cluster_labels == cluster_val] = low_label
    return labels


def cluster_cells(
    values: FloatArray,
    *,
    n_cores: int,
    pca_components: int | None,
) -> tuple[ClusterLabels, LinkageMatrix | None]:
    """Cluster the cells represented by columns in a feature-by-cell matrix."""
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
    preN: str | bytes | Sequence[Any] | npt.NDArray[Any] | None,
    *,
    n_cores: int,
    pca_components: int | None,
    prediction_override: npt.NDArray[Any] | None = None,
    low_confidence: bool = False,
) -> tuple[ClusterLabels, LinkageMatrix | None, npt.NDArray[np.object_]]:
    """Cluster cells and assign diploid/aneuploid labels from reference overlap or CNA magnitude.

    ``prediction_override`` can supply an external per-cell call, such as
    arm-correlation results, while retaining Ward clustering for ordering.
    """
    labels, linkage = cluster_cells(values, n_cores=n_cores, pca_components=pca_components)
    if preN is not None and len(preN) > 0:
        normal_names = _preN_to_names(preN, reference_cell_names)
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
    return labels, linkage, predictions


def adjust_baseline_inplace(
    mat: FloatArray,
    diploid_mask: BoolArray,
    chunk_elems: int = 1 << 24,
) -> FloatArray:
    """Subtract the diploid baseline and flatten noise around it, overwriting ``mat``."""
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
