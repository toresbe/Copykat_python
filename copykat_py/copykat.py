"""Main CopyKAT function: end-to-end copy number inference from scRNA-seq data.

Faithfully reimplements the R copykat() function workflow:
1. Read and filter data
2. Annotate gene coordinates
3. Freeman-Tukey transformation + DLM smoothing
4. Baseline estimation (normal cell identification)
5. MCMC segmentation with KS breakpoint detection
6. Convert to genomic bins (hg20)
7. Baseline adjustment
8. Final prediction (aneuploid vs diploid)
9. Output files + heatmap
"""

import logging
import time
from dataclasses import asdict
from typing import Unpack, cast

import numpy as np
import pandas as pd
from scipy import sparse

from copykat_py import anchor as _anchor
from copykat_py import backend
from copykat_py._logging import with_default_progress_output
from copykat_py._pipeline.baseline import select_baseline
from copykat_py._pipeline.output import write_results
from copykat_py._pipeline.prediction import adjust_and_call
from copykat_py._pipeline.preprocessing import annotate_genes, transform_counts_inplace
from copykat_py._pipeline.runtime import _format_seconds, _record_step, new_runtime_info, select_pca_components
from copykat_py._pipeline.segmentation import _segment_with_retries
from copykat_py._types import (
    AnchorStrategy,
    CopyKATResult,
    DataQualityStatus,
    Genome,
    RawMatrix,
    SparseMatrix,
)
from copykat_py.baseline import (
    FULL_CLUSTER_MAX_CELLS,
    _hierarchical_cluster,
)
from copykat_py.convert_bins import convert_to_bins, get_last_convert_bins_info
from copykat_py.input import (
    _keep_cells_by_chr_coverage,
    _prepare_input_matrix,
)
from copykat_py.output import (
    _write_cna_csv,
)
from copykat_py.run_context import CopyKATArguments, RunContext
from copykat_py.segmentation import get_last_cna_mcmc_info
from copykat_py.smoothing import dlm_smooth, get_last_dlm_smooth_info

logger = logging.getLogger(__name__)


@with_default_progress_output
def copykat(rawmat: RawMatrix, **options: Unpack[CopyKATArguments]) -> CopyKATResult:
    """Infer copy-number profiles from a genes-by-cells UMI count matrix.

    Parameters
    ----------
    rawmat : pd.DataFrame, np.ndarray, scipy.sparse, dict, or str
        UMI counts with genes in rows and cells in columns. A string is a path
        to a .mtx, .csv, or .tsv file; a dict supplies matrix, genes, and barcodes.
    **options : Unpack[CopyKATArguments]
        Optional named settings used to construct :class:`RunContext`.
        Its Attributes section documents every keyword, default, and special
        value. Options after ``rawmat`` must be passed by keyword.

    Returns
    -------
    CopyKATResult
        CNA values, clustering, and runtime metadata; also includes predictions
        when the genome and cell-line mode produce calls.

    See Also
    --------
    RunContext : Documented defaults, normalized settings, and stage options.
    CopyKATArguments : Typed dictionary of the accepted optional keywords.
    """
    run_context = RunContext(**options)
    backend.set_backend(run_context.backend_name)
    start_time = time.perf_counter()
    # Keep the original random stream for reproducibility.
    np.random.seed(run_context.random_seed)  # noqa: NPY002
    runtime_info = new_runtime_info(run_context.sam_name, run_context.n_cores, run_context.runtime_parameters())

    logger.info("running copykat-py v1.0.0")

    # =========================================================================
    # Step 1: Read and filter data
    # =========================================================================
    logger.info("step 1: read and filter data ...")
    step_start = time.perf_counter()
    marker_counts = None
    if run_context.anchor is AnchorStrategy.MARKERS:
        marker_counts = (
            _anchor.count_markers(rawmat, _anchor.IMMUNE_MARKERS),
            _anchor.count_markers(rawmat, _anchor.ENDOTHELIAL_MARKERS),
        )
    filtering = run_context.filtering_options
    prepared_input = _prepare_input_matrix(rawmat, filtering.min_gene_per_cell, filtering.lower_detection_rate)
    del rawmat  # release the caller's unfiltered input once preparation is complete
    input_cell_count = len(prepared_input.original_cell_names)
    selected_pca_components = select_pca_components(
        input_cell_count, requested=run_context.pca_components, genome=run_context.genome, runtime_info=runtime_info
    )
    logger.info(f"  {prepared_input.matrix.shape[0]} genes, {prepared_input.matrix.shape[1]} cells in raw data")
    pca_selection = (
        "manual override"
        if run_context.pca_components is not None
        else f"auto from input cell count {input_cell_count}"
    )
    logger.info(f"  adaptive PCA components: {selected_pca_components} ({pca_selection})")
    if prepared_input.stats.filtered_cells > 0:
        logger.info(
            f"  filtered out {prepared_input.stats.filtered_cells} cells "
            f"with <= {run_context.min_gene_per_cell} genes; "
            f"remaining {prepared_input.matrix.shape[1]} cells"
        )
    logger.info(f"  {prepared_input.matrix.shape[0]} genes past LOW_DR filtering")

    effective_detection_rate = filtering.upper_detection_rate
    WNS1 = DataQualityStatus.OK
    if prepared_input.matrix.shape[0] < 7000:
        WNS1 = DataQualityStatus.LOW
        effective_detection_rate = filtering.lower_detection_rate
        logger.warning("  WARNING: low data quality; assigned LOW_DR to UP_DR...")
        runtime_info["warnings"].append("Low data quality; effective UP_DR was set to LOW_DR")
    runtime_info["parameters"]["UP_DR_effective"] = effective_detection_rate
    elapsed = _record_step(runtime_info, "read_and_filter", step_start, extra=asdict(prepared_input.stats))
    logger.info(f"  step 1 runtime: {_format_seconds(elapsed)}")

    # =========================================================================
    # Step 2: Annotate gene coordinates
    # =========================================================================
    anno_mat, anno_rows, anno_cols = annotate_genes(
        prepared_input.genes, id_type=run_context.id_type, genome=run_context.genome, runtime_info=runtime_info
    )

    # Secondary cell filtering: ensure each cell has genes across chromosomes.
    step_start = time.perf_counter()
    expr_values = prepared_input.matrix[anno_rows]
    keep_cells = _keep_cells_by_chr_coverage(expr_values, anno_mat["chromosome_name"].to_numpy(), filtering.ngene_chr)
    if keep_cells.sum() == 0:
        raise ValueError("All cells are filtered out")
    cell_cols = list(prepared_input.barcodes)
    original_cell_names = prepared_input.original_cell_names
    del prepared_input  # release the filtered input matrix while retaining original names for output
    if not np.all(keep_cells):
        cell_cols = [cell_cols[i] for i in np.where(keep_cells)[0]]
        expr_values = expr_values[:, keep_cells]
    # Column-major, as pandas' DataFrame.to_numpy() returned it before: the
    # per-cell centering below sums down each column, and numpy's summation
    # order (hence the last bits of every value) depends on the layout. Those
    # bits can flip near-tie merges in the step-4 clustering on low-confidence
    # samples, so keep the original layout to reproduce results exactly.
    if sparse.issparse(expr_values):
        rawmat3 = cast(SparseMatrix, expr_values).astype(np.float64).toarray(order="F")
    else:
        rawmat3 = np.asfortranarray(expr_values, dtype=np.float64)
    del expr_values
    _record_step(runtime_info, "cell_filter_pre_smoothing", step_start, extra={"cells_after_filter": len(cell_cols)})

    norm_mat, seg_mask, keep_cells2 = transform_counts_inplace(
        rawmat3,
        anno_mat["chromosome_name"].to_numpy(),
        detection_rate=effective_detection_rate,
        ngene_chr=filtering.ngene_chr,
        runtime_info=runtime_info,
    )
    del rawmat3  # transform_counts_inplace returned this same backing array

    logger.info(f"  {norm_mat.shape[0]} genes, {norm_mat.shape[1]} cells after preprocessing")

    # =========================================================================
    # Step 3: DLM smoothing
    # =========================================================================
    logger.info("step 3: smoothing data with DLM ...")
    step_start = time.perf_counter()
    norm_mat_smooth = dlm_smooth(norm_mat, n_cores=run_context.n_cores)
    del norm_mat
    dlm_info = get_last_dlm_smooth_info()
    elapsed = _record_step(runtime_info, "dlm_smoothing", step_start, parallel_info=dlm_info)
    logger.info(
        f"  smoothing runtime: {_format_seconds(elapsed)} "
        f"(parallel={dlm_info['parallel']}, cores={dlm_info['effective_cores']})"
    )

    # =========================================================================
    # Step 4: Measure baselines
    # =========================================================================
    cell_name_list = cell_cols
    baseline_selection = select_baseline(
        norm_mat_smooth,
        cell_name_list,
        run_context.norm_cell_names,
        marker_counts,
        WNS1,
        options=run_context.baseline_options(selected_pca_components),
        runtime_info=runtime_info,
    )
    norm_mat_relat = baseline_selection.relative_expression
    CL = baseline_selection.cluster_labels
    baseline_state = baseline_selection.reference
    del baseline_selection, norm_mat_smooth

    # =========================================================================
    # Apply stricter gene filtering for segmentation
    # =========================================================================
    step_start = time.perf_counter()
    norm_mat_relat = norm_mat_relat[seg_mask, :]
    anno_mat2 = anno_mat.iloc[seg_mask].reset_index(drop=True)

    # Filter cells again with the reduced gene set (keep_cells2 computed before the transform)
    if keep_cells2.sum() == 0:
        raise ValueError("All cells are filtered out after UP_DR filtering")
    if not np.all(keep_cells2):
        norm_mat_relat = norm_mat_relat[:, keep_cells2]
        cell_cols_seg = [cell_cols[i] for i in np.where(keep_cells2)[0]]
        CL_filtered = CL[keep_cells2] if len(CL) == len(cell_cols) else CL
    else:
        cell_cols_seg = cell_cols
        CL_filtered = CL
    _record_step(
        runtime_info,
        "cell_filter_pre_segmentation",
        step_start,
        extra={"cells_after_filter": len(cell_cols_seg), "genes_after_filter": int(norm_mat_relat.shape[0])},
    )

    # Ensure CL alignment
    if len(CL_filtered) != norm_mat_relat.shape[1]:
        # Recompute if shape mismatch
        data_t = norm_mat_relat.T
        step4_reduce = data_t.shape[0] > FULL_CLUSTER_MAX_CELLS
        CL_filtered, _ = _hierarchical_cluster(
            data_t,
            min(6, norm_mat_relat.shape[1]),
            method="ward",
            metric="euclidean",
            n_cores=run_context.n_cores,
            reduce=step4_reduce,
            pca_components=selected_pca_components,
        )

    # =========================================================================
    # Step 5: Segmentation
    # =========================================================================
    logger.info("step 5: segmentation ...")
    step_start = time.perf_counter()
    results = _segment_with_retries(
        CL_filtered,
        norm_mat_relat,
        options=run_context.segmentation_options,
    )
    seg_info = get_last_cna_mcmc_info()
    elapsed = _record_step(
        runtime_info,
        "segmentation",
        step_start,
        parallel_info=seg_info,
        extra={"breakpoints": len(results.breakpoints)},
    )
    logger.info(
        f"  segmentation runtime: {_format_seconds(elapsed)} "
        f"(parallel={seg_info['parallel']}, cores={seg_info['effective_cores']}, "
        f"engine={seg_info.get('engine', 'n/a')})"
    )

    results_com = results.log_cna
    del results, norm_mat_relat
    # Center each cell
    results_com -= results_com.mean(axis=0, keepdims=True)

    # Save gene-by-cell CNA results
    gene_anno = anno_mat2[anno_cols].reset_index(drop=True)

    step_start = time.perf_counter()
    _write_cna_csv(
        f"{run_context.sample_name}CNA_raw_results_gene_by_cell.txt",
        gene_anno,
        results_com,
        cell_cols_seg,
        round_floats=False,
        quote_strings=False,
        n_cores=run_context.n_cores,
    )
    _record_step(
        runtime_info,
        "write_gene_level_output",
        step_start,
        extra={"rows": int(results_com.shape[0]), "cols": int(len(anno_cols) + results_com.shape[1])},
    )

    # =========================================================================
    # Step 6: Convert to genomic bins (hg20 only)
    # =========================================================================
    if run_context.genome is Genome.HG20:
        logger.info("step 6: convert to genomic bins ...")
        step_start = time.perf_counter()
        Aj = convert_to_bins(
            gene_anno,
            genome=run_context.genome,
            n_cores=run_context.n_cores,
            values=results_com,
            cell_names=cell_cols_seg,
        )
        del results_com
        convert_info = get_last_convert_bins_info()
        elapsed = _record_step(runtime_info, "convert_to_bins", step_start, parallel_info=convert_info)
        logger.info(
            f"  bin conversion runtime: {_format_seconds(elapsed)} "
            f"(parallel={convert_info['parallel']}, cores={convert_info['effective_cores']})"
        )

        assert Aj is not None  # convert_to_bins only returns None for non-hg20 genomes
        # uber_mat_adj is adjusted in place below, so drop the DataFrame view of it
        uber_mat_adj = Aj.values
        bin_coords = Aj.rna_table[["chrom", "chrompos", "abspos"]].copy()
        chrom_info = Aj.dna_annotations["chrom"].to_numpy()
        del Aj

    else:
        # mm10 uses gene-level results and retains its existing final-call path.
        uber_mat_adj = results_com
        del results_com
        bin_coords = gene_anno
        chrom_info = pd.to_numeric(anno_mat2["chromosome_name"], errors="coerce").fillna(0).to_numpy()

    final_prediction = adjust_and_call(
        uber_mat_adj,
        bin_coords,
        cell_cols_seg,
        cell_name_list,
        baseline_state,
        options=run_context.prediction_options(selected_pca_components),
        runtime_info=runtime_info,
    )
    mat_adj = final_prediction.values
    clustering_result = final_prediction.clustering
    del final_prediction, uber_mat_adj

    return write_results(
        mat_adj,
        bin_coords,
        chrom_info,
        cell_cols_seg,
        original_cell_names,
        clustering_result,
        WNS1,
        baseline_state.warning,
        options=run_context.output_options,
        runtime_info=runtime_info,
        start_time=start_time,
    )
