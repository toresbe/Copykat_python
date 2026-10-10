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
from typing import cast

import numpy as np
import pandas as pd
from scipy import sparse

from copykat_py import anchor as _anchor
from copykat_py import backend
from copykat_py._logging import with_default_progress_output
from copykat_py._pipeline.baseline import BaselineOptions, select_baseline
from copykat_py._pipeline.output import OutputOptions, write_results
from copykat_py._pipeline.prediction import PredictionOptions, adjust_and_call
from copykat_py._pipeline.preprocessing import FilteringOptions, annotate_genes, transform_counts_inplace
from copykat_py._pipeline.runtime import _format_seconds, _record_step, new_runtime_info, select_pca_components
from copykat_py._pipeline.segmentation import _segment_with_retries, _SegmentationOptions
from copykat_py._types import (
    AnchorStrategy,
    CellLineMode,
    CopyKATResult,
    DataQualityStatus,
    DistanceMetric,
    ExecutionBackend,
    FinalCallStrategy,
    GeneIdType,
    Genome,
    KSMethod,
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
from copykat_py.segmentation import get_last_cna_mcmc_info
from copykat_py.smoothing import dlm_smooth, get_last_dlm_smooth_info

logger = logging.getLogger(__name__)


@with_default_progress_output
def copykat(
    rawmat: RawMatrix,
    id_type: GeneIdType = GeneIdType.SYMBOL,
    cell_line: CellLineMode = CellLineMode.NO,
    ngene_chr: int = 5,
    min_gene_per_cell: int = 200,
    LOW_DR: float = 0.05,
    UP_DR: float = 0.1,
    win_size: int = 25,
    norm_cell_names: str | list[str] = "",
    KS_cut: float = 0.1,
    sam_name: str = "",
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    output_seg: bool = False,
    plot_genes: bool = True,
    genome: Genome = Genome.HG20,
    n_cores: int = 1,
    pca_components: int | None = None,
    meta_csv: str | None = None,
    row_split_col: str | None = None,
    backend_name: ExecutionBackend = ExecutionBackend.CPU,
    ks_method: KSMethod = KSMethod.MONTE_CARLO,
    anchor: AnchorStrategy = AnchorStrategy.SIGMA,
    final_call: FinalCallStrategy = FinalCallStrategy.CLUSTERS,
) -> CopyKATResult:
    """Run CopyKAT analysis: infer copy number profiles from scRNA-seq data.

    Parameters
    ----------
    rawmat : pd.DataFrame, np.ndarray, scipy.sparse, or str
        UMI count matrix (genes in rows, cells in columns).
        If str, path to .mtx, .csv, or .tsv file.
    id_type : GeneIdType
        Gene ID type: ``GeneIdType.SYMBOL`` or ``GeneIdType.ENSEMBL``.
    cell_line : CellLineMode
        ``CellLineMode.YES`` for pure cell line data, ``CellLineMode.NO`` for tumor/normal mixture.
    ngene_chr : int
        Minimum number of genes per chromosome for cell filtering.
    min_gene_per_cell : int
        Minimum genes detected per cell.
    LOW_DR : float
        Minimum gene detection rate for smoothing.
    UP_DR : float
        Minimum gene detection rate for segmentation.
    win_size : int
        Window size for MCMC segmentation.
    norm_cell_names : str or list
        Known normal cell barcodes ("" for auto-detection).
    KS_cut : float
        KS test cutoff for breakpoint detection (0 to 1).
    sam_name : str
        Sample name prefix for output files.
    distance : DistanceMetric
        Cell-ordering distance metric.
    output_seg : bool
        Whether to output .seg file for IGV.
    plot_genes : bool
        Whether to plot gene-level heatmap.
    genome : Genome
        Genome.HG20 or Genome.MM10.
    n_cores : int
        Number of CPU cores for parallel computation.
    pca_components : int or None
        Adaptive PCA component cap for large clustering steps. When omitted,
        CopyKAT-Py uses the built-in rule:
        - `hg20`: 256 PCs for fewer than 50,000 input cells, otherwise 128
        - `mm10`: 512 PCs for fewer than 20,000 input cells, 256 PCs for
          fewer than 40,000 input cells, otherwise 128
    meta_csv : str or None
        Path to a per-cell annotation CSV for the annotated heatmap.
        First column = cell name; remaining columns become coloured annotation
        sidebars.  When provided (and ``plot_genes=True``), an annotated
        heatmap is saved as ``{sam_name}annotated_heatmap.png`` in addition to
        the standard heatmap.  Header row is auto-detected.
    row_split_col : str or None
        Column in ``meta_csv`` used to split and label heatmap rows.
        Defaults to the second column when ``None``.

    Returns
    -------
    dict with keys:
        'prediction': pd.DataFrame with columns [cell.names, copykat.pred]
        'CNAmat': pd.DataFrame with CNA results
        'hclustering': linkage matrix or cluster labels
    """
    distance = DistanceMetric(distance)
    backend_name = ExecutionBackend(backend_name)
    id_type = GeneIdType.normalize(id_type)
    cell_line = CellLineMode(cell_line)
    ks_method = KSMethod(ks_method)
    anchor = AnchorStrategy(anchor)
    final_call = FinalCallStrategy(final_call)
    backend.set_backend(backend_name)
    genome = Genome(genome)
    if final_call is FinalCallStrategy.ARM_CORRELATION and genome is Genome.MM10:
        raise ValueError("arm_correlation currently uses hg38 centromere coordinates and is only supported for hg20")
    start_time = time.perf_counter()
    # Global seed kept for reproducibility with earlier versions; moving to a
    # Generator would change any random stream that depends on it.
    np.random.seed(1234)  # noqa: NPY002
    sample_name = f"{sam_name}_copykat_"
    runtime_info = new_runtime_info(
        sam_name,
        n_cores,
        {
            "id_type": id_type,
            "cell_line": cell_line,
            "genome": genome,
            "ngene_chr": ngene_chr,
            "min_gene_per_cell": min_gene_per_cell,
            "LOW_DR": LOW_DR,
            "UP_DR": UP_DR,
            "win_size": win_size,
            "KS_cut": KS_cut,
            "distance": distance,
            "n_cores": n_cores,
            "pca_components_requested": pca_components,
            "output_seg": output_seg,
            "plot_genes": plot_genes,
            "random_seed": 1234,
            "meta_csv": meta_csv,
            "row_split_col": row_split_col,
            "backend": backend.get_backend(),
            "ks_method": ks_method,
            "anchor": anchor,
            "final_call": final_call,
        },
    )

    logger.info("running copykat-py v1.0.0")

    # =========================================================================
    # Step 1: Read and filter data
    # =========================================================================
    logger.info("step 1: read and filter data ...")
    step_start = time.perf_counter()
    marker_counts = None
    if anchor is AnchorStrategy.MARKERS:
        marker_counts = (
            _anchor.count_markers(rawmat, _anchor.IMMUNE_MARKERS),
            _anchor.count_markers(rawmat, _anchor.ENDOTHELIAL_MARKERS),
        )
    filtering = FilteringOptions(
        min_gene_per_cell=min_gene_per_cell,
        ngene_chr=ngene_chr,
        lower_detection_rate=LOW_DR,
        upper_detection_rate=UP_DR,
    )
    prepared_input = _prepare_input_matrix(rawmat, filtering.min_gene_per_cell, filtering.lower_detection_rate)
    del rawmat  # release the caller's unfiltered input once preparation is complete
    input_cell_count = len(prepared_input.original_cell_names)
    selected_pca_components = select_pca_components(
        input_cell_count, requested=pca_components, genome=genome, runtime_info=runtime_info
    )
    logger.info(f"  {prepared_input.matrix.shape[0]} genes, {prepared_input.matrix.shape[1]} cells in raw data")
    logger.info(
        f"  adaptive PCA components: {selected_pca_components} "
        f"({'manual override' if pca_components is not None else f'auto from input cell count {input_cell_count}'})"
    )
    if prepared_input.stats.filtered_cells > 0:
        logger.info(
            f"  filtered out {prepared_input.stats.filtered_cells} cells with <= {min_gene_per_cell} genes; "
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
        prepared_input.genes, id_type=id_type, genome=genome, runtime_info=runtime_info
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
    norm_mat_smooth = dlm_smooth(norm_mat, n_cores=n_cores)
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
        norm_cell_names,
        marker_counts,
        WNS1,
        options=BaselineOptions(
            cell_line=cell_line,
            anchor=anchor,
            genome=genome,
            n_cores=n_cores,
            pca_components=selected_pca_components,
        ),
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
            n_cores=n_cores,
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
        options=_SegmentationOptions(
            window_size=win_size,
            ks_cutoff=KS_cut,
            ks_method=ks_method,
            n_cores=n_cores,
        ),
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
        f"{sample_name}CNA_raw_results_gene_by_cell.txt",
        gene_anno,
        results_com,
        cell_cols_seg,
        round_floats=False,
        quote_strings=False,
        n_cores=n_cores,
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
    if genome is Genome.HG20:
        logger.info("step 6: convert to genomic bins ...")
        step_start = time.perf_counter()
        Aj = convert_to_bins(gene_anno, genome=genome, n_cores=n_cores, values=results_com, cell_names=cell_cols_seg)
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
        options=PredictionOptions(
            genome=genome,
            cell_line=cell_line,
            final_call=final_call,
            n_cores=n_cores,
            pca_components=selected_pca_components,
        ),
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
        options=OutputOptions(
            sample_name=sample_name,
            genome=genome,
            distance=distance,
            n_cores=n_cores,
            plot_genes=plot_genes,
            output_seg=output_seg,
            meta_csv=meta_csv,
            row_split_col=row_split_col,
        ),
        runtime_info=runtime_info,
        start_time=start_time,
    )
