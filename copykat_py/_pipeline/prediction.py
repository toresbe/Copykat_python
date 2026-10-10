"""Baseline adjustment and final calls on gene or genomic-bin CNA values."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copykat_py import anchor as _anchor
from copykat_py._pipeline.baseline import _BaselineState
from copykat_py._pipeline.runtime import _format_seconds, _record_step
from copykat_py._types import (
    BaselineWarning,
    CellLineMode,
    FinalCallStrategy,
    FloatArray,
    Genome,
    PredictionLabel,
    RuntimeInfo,
)
from copykat_py.baseline import get_last_cluster_info
from copykat_py.final_call import (
    FinalCallResult,
    WardClusteringResult,
    cluster_and_call,
    cluster_cells,
)
from copykat_py.final_call import (
    adjust_baseline_inplace as _adjust_baseline_inplace,
)
from copykat_py.normal_cells import normal_cells_to_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionOptions:
    """Final-call strategy and clustering settings.

    Attributes
    ----------
    genome : Genome
        Reference genome of the CNA values: ``HG20`` uses genomic bins and
        ``MM10`` uses gene-level values.
    cell_line : CellLineMode
        Pure cell-line mode. For ``HG20``, ``YES`` skips diploid baseline
        adjustment and returns clustering without predictions. The existing
        ``MM10`` path still adjusts the baseline and produces predictions.
    final_call : FinalCallStrategy
        ``CLUSTERS`` assigns calls from cluster/reference overlap or CNA
        magnitude. ``ARM_CORRELATION`` uses chromosome-arm correlation calls
        when at least five normal anchor cells remain; otherwise it falls
        back to clusters. Arm correlation is supported only for ``HG20``.
    n_cores : int
        Maximum number of CPU workers requested for final clustering.
    pca_components : int
        Resolved PCA component cap used by large final clustering steps,
        selected earlier from the input cell count and genome or an override.
    """

    genome: Genome
    cell_line: CellLineMode
    final_call: FinalCallStrategy
    n_cores: int
    pca_components: int


@dataclass(frozen=True, slots=True)
class FinalPrediction:
    """The adjusted input matrix and its final clustering/calls.

    Attributes
    ----------
    values : FloatArray
        CNA values with shape ``(features, cells)`` after baseline adjustment.
        Features are genomic bins for ``HG20`` and genes for ``MM10``. The
        array shares the input's backing storage and remains mutable.
    clustering : WardClusteringResult or FinalCallResult
        Cluster labels and linkage tree aligned to the matrix columns, plus
        per-cell predictions when the selected genome and mode produce calls.
    """

    values: FloatArray
    clustering: WardClusteringResult | FinalCallResult


def adjust_and_call(
    uber_mat_adj: FloatArray,
    coordinates: pd.DataFrame,
    cell_cols_seg: list[str],
    cell_name_list: list[str],
    baseline_state: _BaselineState,
    *,
    options: PredictionOptions,
    runtime_info: RuntimeInfo,
) -> FinalPrediction:
    """Adjust values in place; retain the genome-specific cell-line behavior."""
    synthetic_mode = options.genome is Genome.HG20 and options.cell_line is CellLineMode.YES
    logger.info("step 7: adjust baseline ...")
    step_start = time.perf_counter()

    if synthetic_mode:
        mat_adj = uber_mat_adj
    else:
        arm_calls = None
        if (
            options.genome is Genome.HG20
            and options.final_call is FinalCallStrategy.ARM_CORRELATION
            and baseline_state.normal_cells is not None
            and len(baseline_state.normal_cells) > 0
        ):
            normal_cell_names = normal_cells_to_names(baseline_state.normal_cells, cell_name_list)
            anchor_mask = np.array([cell in normal_cell_names for cell in cell_cols_seg], dtype=bool)
            if anchor_mask.sum() >= 5:
                arm_calls = _anchor.arm_correlation_calls(
                    uber_mat_adj,
                    coordinates["chrom"].to_numpy(),
                    coordinates["chrompos"].to_numpy(),
                    anchor_mask,
                )
                runtime_info["final_call_path"] = FinalCallStrategy.ARM_CORRELATION
            else:
                runtime_info["final_call_path"] = "clusters_insufficient_anchor"
        initial_call = cluster_and_call(
            uber_mat_adj,
            cell_cols_seg,
            cell_name_list,
            baseline_state.normal_cells,
            n_cores=options.n_cores,
            pca_components=options.pca_components,
            prediction_override=arm_calls,
        )
        # Baseline adjustment: subtract diploid mean, then denoise
        diploid_mask = initial_call.predictions == PredictionLabel.DIPLOID
        if diploid_mask.sum() > 0:
            mat_adj = _adjust_baseline_inplace(uber_mat_adj, diploid_mask)
        else:
            mat_adj = uber_mat_adj
    del uber_mat_adj
    cluster_info = get_last_cluster_info()
    elapsed = _record_step(
        runtime_info,
        "baseline_adjustment",
        step_start,
        parallel_info=cluster_info,
        extra={"warning": baseline_state.warning},
    )
    logger.info(
        f"  step 7 runtime: {_format_seconds(elapsed)} "
        f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, "
        f"engine={cluster_info.get('engine', 'n/a')})"
    )

    # =========================================================================
    # Step 8: Final prediction
    # =========================================================================
    logger.info("step 8: final prediction ...")
    step_start = time.perf_counter()
    clustering_result: WardClusteringResult | FinalCallResult
    if not synthetic_mode:
        clustering_result = cluster_and_call(
            mat_adj,
            cell_cols_seg,
            cell_name_list,
            baseline_state.normal_cells,
            n_cores=options.n_cores,
            pca_components=options.pca_components,
            prediction_override=arm_calls,
            low_confidence=baseline_state.warning is BaselineWarning.UNCLASSIFIED,
        )
    else:
        clustering_result = cluster_cells(
            mat_adj,
            n_cores=options.n_cores,
            pca_components=options.pca_components,
        )
    cluster_info = get_last_cluster_info()
    elapsed = _record_step(
        runtime_info,
        "final_prediction",
        step_start,
        parallel_info=cluster_info,
        extra={"warning": baseline_state.warning},
    )
    logger.info(
        f"  step 8 runtime: {_format_seconds(elapsed)} "
        f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, "
        f"engine={cluster_info.get('engine', 'n/a')})"
    )

    return FinalPrediction(mat_adj, clustering_result)
