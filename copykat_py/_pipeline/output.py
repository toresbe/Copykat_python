"""Write and plot CopyKAT results without retaining earlier stage matrices."""

import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
import pandas as pd

from copykat_py._pipeline.runtime import _format_seconds, _record_step, finish_runtime_report
from copykat_py._types import (
    BaselineWarning,
    ClusteringResult,
    CopyKATResult,
    DataQualityStatus,
    DistanceMetric,
    FloatArray,
    Genome,
    PredictionLabel,
    RuntimeInfo,
)
from copykat_py.final_call import FinalCallResult, WardClusteringResult
from copykat_py.output import _frame_with_leading_columns, _write_cna_csv, _write_seg_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class _HeatmapOptions:
    """Small display and execution settings, separate from heatmap data.

    Attributes
    ----------
    sample_name : str
        Label shown in the heatmap title. Pipeline callers supply the output
        prefix ``{sam_name}_copykat_``.
    distance : DistanceMetric
        Distance metric used to order cells for the heatmap; supports
        ``EUCLIDEAN``, ``PEARSON``, and ``SPEARMAN``.
    n_cores : int
        Number of CPU workers requested for plotting-time clustering.
    genome : Genome
        Reference genome of the plotted values, used to decode chromosome
        labels, including genome-specific numeric X and Y chromosome codes.
    output_path : str
        PNG destination supplied explicitly by the pipeline.
    """

    sample_name: str
    distance: DistanceMetric
    n_cores: int
    genome: Genome
    output_path: str


def _meta_with_pred(meta_csv: str, pred_dict: dict[str, str] | None, sample_name: str) -> str:
    """Read *meta_csv*, append copykat-py predictions as the last column.

    Returns the path to a new CSV written alongside the original outputs.
    Cells absent from *pred_dict* receive ``PredictionLabel.NOT_DEFINED``.
    """
    import pandas as pd

    meta = pd.read_csv(meta_csv)
    cell_col = meta.columns[0]
    meta = meta.set_index(cell_col)
    if pred_dict is not None:
        meta["copykat_pred_py"] = meta.index.map(pred_dict).fillna(PredictionLabel.NOT_DEFINED)
    else:
        meta["copykat_pred_py"] = PredictionLabel.NOT_DEFINED
    out_path = f"{sample_name}meta_with_pred.csv"
    meta.reset_index().to_csv(out_path, index=False)
    return out_path


def _run_plot_heatmap(
    mat_adj: FloatArray,
    chrom_info: npt.NDArray[Any],
    predictions: dict[str, str] | None,
    WNS1: DataQualityStatus,
    WNS: BaselineWarning,
    *,
    options: _HeatmapOptions,
) -> None:
    from copykat_py.plotting import plot_heatmap

    plot_heatmap(
        mat_adj,
        chrom_info,
        predictions=predictions,
        sample_name=options.sample_name,
        distance=options.distance,
        n_cores=options.n_cores,
        WNS1=WNS1,
        WNS=WNS,
        output_path=options.output_path,
        genome=options.genome,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputOptions:
    """Small file and plotting settings, shared by both genome paths.

    Attributes
    ----------
    sample_name : str
        Complete prefix for output filenames, constructed by the public
        function as ``{sam_name}_copykat_``; also used in plot titles.
    genome : Genome
        Reference genome of the CNA results. ``HG20`` has bin coordinates;
        ``MM10`` has gene annotations. Also controls chromosome labels and
        whether SEG export is available.
    distance : DistanceMetric
        Distance metric used to order cells in standard and annotated heatmaps.
    n_cores : int
        Number of CPU workers requested for CNA output writing and
        plotting-time clustering.
    plot_genes : bool
        Whether to plot the final CNA heatmap. Despite the public argument's
        name, ``HG20`` plots genomic bins and ``MM10`` plots genes. Also enables
        the additional annotated heatmap when ``meta_csv`` is provided.
    output_seg : bool
        Whether to export an IGV SEG file. Applied only to ``HG20`` results;
        the existing ``MM10`` path does not export SEG files.
    meta_csv : str or None
        Optional per-cell annotation CSV path. Its first column contains cell
        names; remaining columns supply annotation sidebars. When plotting is
        enabled, predictions are appended to a companion CSV and an additional
        annotated heatmap is written. ``None`` disables that additional plot.
    row_split_col : str or None
        Metadata column used to split and label annotated-heatmap rows.
        ``None`` uses the first metadata column after the cell-name column;
        an empty string disables row splitting.
    """

    sample_name: str
    genome: Genome
    distance: DistanceMetric
    n_cores: int
    plot_genes: bool
    output_seg: bool
    meta_csv: str | None
    row_split_col: str | None


def write_results(
    mat_adj: FloatArray,
    coordinates: pd.DataFrame,
    chrom_info: npt.NDArray[Any],
    cell_cols_seg: list[str],
    original_cell_names: list[str],
    clustering_result: WardClusteringResult | FinalCallResult,
    quality: DataQualityStatus,
    warning: BaselineWarning,
    *,
    options: OutputOptions,
    runtime_info: RuntimeInfo,
    start_time: float,
) -> CopyKATResult:
    """Save final calls, CNA values, clustering, heatmaps, and the runtime report."""
    sample_name = options.sample_name
    pred_dict = None
    res = None
    if isinstance(clustering_result, FinalCallResult):
        pred_dict = {cell_cols_seg[i]: clustering_result.predictions[i] for i in range(len(cell_cols_seg))}
        for cell in original_cell_names:
            if cell not in pred_dict:
                pred_dict[cell] = PredictionLabel.NOT_DEFINED
        res = pd.DataFrame(
            {
                "cell.names": list(pred_dict.keys()),
                "copykat.pred": list(pred_dict.values()),
            }
        )

    logger.info("step 9: saving results ...")
    step_start = time.perf_counter()

    if res is not None:
        res.to_csv(f"{sample_name}prediction.txt", sep="\t", index=False)

    # Save CNA results
    cna_out = _frame_with_leading_columns(coordinates, mat_adj, cell_cols_seg)
    _write_cna_csv(f"{sample_name}CNA_results.txt", coordinates, mat_adj, cell_cols_seg, n_cores=options.n_cores)

    # Save clustering
    clustering_data: ClusteringResult = {
        "labels": clustering_result.labels,
        "Z": clustering_result.linkage,
    }
    with open(f"{sample_name}clustering_results.pkl", "wb") as f:
        pickle.dump(clustering_data, f)
    elapsed = _record_step(
        runtime_info,
        "write_final_outputs",
        step_start,
        extra={"bins": int(cna_out.shape[0]), "cells": len(cell_cols_seg)},
    )
    logger.info(f"  step 9 runtime: {_format_seconds(elapsed)}")

    # =========================================================================
    # Step 10: Plot heatmap
    # =========================================================================
    if options.plot_genes:
        logger.info("step 10: plotting heatmap ...")
        plot_step_start = time.perf_counter()
        predictions = pred_dict
        _run_plot_heatmap(
            mat_adj,
            chrom_info,
            predictions,
            quality,
            warning,
            options=_HeatmapOptions(
                sample_name=sample_name,
                distance=options.distance,
                n_cores=options.n_cores,
                genome=options.genome,
                output_path=f"{sample_name}heatmap.png",
            ),
        )
        elapsed = _record_step(runtime_info, "plot_heatmap", plot_step_start)
        logger.info(f"  step 10 runtime: {_format_seconds(elapsed)}")

    if options.plot_genes and options.meta_csv is not None:
        logger.info("step 10b: plotting annotated heatmap ...")
        step_ann = time.perf_counter()
        from copykat_py.plotting import AnnotatedHeatmapOptions, plot_heatmap_annotated

        meta_pred_path = _meta_with_pred(options.meta_csv, pred_dict, sample_name)
        plot_heatmap_annotated(
            mat=mat_adj,
            cell_names=cna_out.columns[len(coordinates.columns) :].tolist(),
            chrom_info=chrom_info,
            meta_csv=meta_pred_path,
            genome=options.genome,
            options=AnnotatedHeatmapOptions(
                row_split_col=options.row_split_col,
                sample_name=sample_name,
                distance=options.distance,
                n_cores=options.n_cores,
                output_path=f"{sample_name}annotated_heatmap.png",
            ),
        )
        elapsed = _record_step(runtime_info, "plot_annotated_heatmap", step_ann)
        logger.info(f"  step 10b runtime: {_format_seconds(elapsed)}")

    # =========================================================================
    # Output SEG file
    # =========================================================================
    if options.output_seg and options.genome is Genome.HG20:
        logger.info("  generating seg files for IGV viewer")
        _write_seg_file(coordinates, mat_adj, cell_cols_seg, sample_name)
    finish_runtime_report(runtime_info, sample_name, start_time)

    if res is None:  # cell-line mode makes no predictions
        return {
            "CNAmat": cna_out,
            "hclustering": clustering_data,
            "runtime": runtime_info,
        }
    return {
        "prediction": res,
        "CNAmat": cna_out,
        "hclustering": clustering_data,
        "runtime": runtime_info,
    }
