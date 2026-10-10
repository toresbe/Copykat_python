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

import json
import logging
import os
import pickle
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster

from copykat_py import anchor as _anchor
from copykat_py import backend
from copykat_py._logging import with_default_progress_output
from copykat_py._types import (
    AnchorPath,
    AnchorStrategy,
    BaselineWarning,
    CellLineMode,
    ClusteringResult,
    ClusterLabels,
    CopyKATResult,
    DataQualityStatus,
    DistanceMetric,
    ExecutionBackend,
    FinalCallStrategy,
    FloatArray,
    GeneIdType,
    GeneProfile,
    Genome,
    KSMethod,
    ParallelInfo,
    PredictionLabel,
    RawMatrix,
    ReferenceMode,
    RuntimeInfo,
    SparseMatrix,
)
from copykat_py.annotation import annotate_gene_rows
from copykat_py.baseline import (
    AUTO_PCA_CELL_COUNT_CUTOFF,
    AUTO_PCA_LARGE_SAMPLE,
    AUTO_PCA_SMALL_SAMPLE,
    FULL_CLUSTER_MAX_CELLS,
    MOUSE_AUTO_PCA_LARGE_SAMPLE,
    MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF,
    MOUSE_AUTO_PCA_MEDIUM_SAMPLE,
    MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF,
    MOUSE_AUTO_PCA_SMALL_SAMPLE,
    _fit_gmm_3component,
    _hierarchical_cluster,
    baseline_gmm,
    baseline_norm_cl,
    baseline_synthetic,
    get_last_cluster_info,
    resolve_adaptive_pca_components,
)
from copykat_py.convert_bins import convert_to_bins, get_last_convert_bins_info
from copykat_py.data_loader import load_cyclegenes
from copykat_py.final_call import (
    FinalCallResult,
    _normal_cells_to_names,
    cluster_and_call,
    cluster_cells,
)
from copykat_py.final_call import (
    adjust_baseline_inplace as _adjust_baseline_inplace,
)
from copykat_py.input import (
    _keep_cells_by_chr_coverage,
    _prepare_input_matrix,
)
from copykat_py.output import (
    _frame_with_leading_columns,
    _write_cna_csv,
    _write_seg_file,
)
from copykat_py.segmentation import cna_mcmc, get_last_cna_mcmc_info
from copykat_py.smoothing import dlm_smooth, get_last_dlm_smooth_info

logger = logging.getLogger(__name__)


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
    sample_name: str,
    distance: DistanceMetric,
    n_cores: int,
    WNS1: DataQualityStatus,
    WNS: BaselineWarning,
    output_path: str,
) -> None:
    from copykat_py.plotting import plot_heatmap

    plot_heatmap(
        mat_adj,
        chrom_info,
        predictions=predictions,
        sample_name=sample_name,
        distance=distance,
        n_cores=n_cores,
        WNS1=WNS1,
        WNS=WNS,
        output_path=output_path,
    )


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"


def _record_step(
    runtime_info: RuntimeInfo,
    step: str,
    start_time: float,
    parallel_info: ParallelInfo | None = None,
    extra: Mapping[str, Any] | None = None,
) -> float:
    elapsed = time.perf_counter() - start_time
    entry = {"step": step, "seconds": round(float(elapsed), 4)}
    if parallel_info:
        entry["parallel"] = bool(parallel_info.get("parallel", False))
        entry["requested_cores"] = int(parallel_info.get("requested_cores", 1))
        entry["effective_cores"] = int(parallel_info.get("effective_cores", 1))
        for key in ("tasks", "chunk_size", "mc_samples", "engine", "approximate"):
            if key in parallel_info:
                entry[key] = parallel_info[key]
    if extra:
        entry.update(extra)
    runtime_info["steps"].append(entry)
    return elapsed


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
    runtime_info: RuntimeInfo = {
        "sample_name": sam_name,
        "requested_cores": int(n_cores),
        "available_cores": int(os.cpu_count() or 1),
        "steps": [],
        "parameters": {
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
        "versions": {},
        "warnings": [],
    }
    for package in ("copykat-py", "numpy", "scipy", "pandas", "scikit-learn", "fastcluster"):
        try:
            runtime_info["versions"][package] = version(package)
        except PackageNotFoundError:
            runtime_info["versions"][package] = "unavailable (source checkout or package not installed)"

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
    rawmat, gene_names, barcodes, original_cell_names, prep_stats = _prepare_input_matrix(
        rawmat, min_gene_per_cell, LOW_DR
    )
    input_cell_count = len(original_cell_names)
    selected_pca_components = resolve_adaptive_pca_components(
        input_cell_count,
        pca_components=pca_components,
        genome=genome,
    )
    runtime_info["pca_components"] = int(selected_pca_components)
    runtime_info["pca_selection_mode"] = "manual" if pca_components is not None else "auto_by_input_cell_count"
    runtime_info["pca_selection_genome"] = str(genome)
    runtime_info["pca_selection_input_cells"] = input_cell_count
    if genome is Genome.MM10:
        runtime_info["pca_selection_rule"] = (
            f"<{MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_SMALL_SAMPLE},"
            f"<{MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_MEDIUM_SAMPLE},"
            f">={MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_LARGE_SAMPLE}"
        )
    else:
        runtime_info["pca_selection_rule"] = (
            f"<{AUTO_PCA_CELL_COUNT_CUTOFF}->{AUTO_PCA_SMALL_SAMPLE},"
            f">={AUTO_PCA_CELL_COUNT_CUTOFF}->{AUTO_PCA_LARGE_SAMPLE}"
        )
    logger.info(f"  {rawmat.shape[0]} genes, {rawmat.shape[1]} cells in raw data")
    logger.info(
        f"  adaptive PCA components: {selected_pca_components} "
        f"({'manual override' if pca_components is not None else f'auto from input cell count {input_cell_count}'})"
    )
    if prep_stats["filtered_cells"] > 0:
        logger.info(
            f"  filtered out {prep_stats['filtered_cells']} cells with <= {min_gene_per_cell} genes; "
            f"remaining {rawmat.shape[1]} cells"
        )
    logger.info(f"  {rawmat.shape[0]} genes past LOW_DR filtering")

    WNS1 = DataQualityStatus.OK
    if rawmat.shape[0] < 7000:
        WNS1 = DataQualityStatus.LOW
        UP_DR = LOW_DR
        logger.warning("  WARNING: low data quality; assigned LOW_DR to UP_DR...")
        runtime_info["warnings"].append("Low data quality; effective UP_DR was set to LOW_DR")
    runtime_info["parameters"]["UP_DR_effective"] = UP_DR
    elapsed = _record_step(runtime_info, "read_and_filter", step_start, extra=prep_stats)
    logger.info(f"  step 1 runtime: {_format_seconds(elapsed)}")

    # =========================================================================
    # Step 2: Annotate gene coordinates
    # =========================================================================
    logger.info("step 2: annotating gene coordinates ...")
    step_start = time.perf_counter()
    anno_mat, anno_rows = annotate_gene_rows(gene_names, id_type=id_type, genome=genome)
    runtime_info["parameters"]["gene_order"] = "chromosome,start_position" if genome is Genome.MM10 else "abspos"

    # =========================================================================
    # Step 3: Remove cell cycle genes and HLA genes (hg20 only)
    # =========================================================================
    if genome is Genome.HG20:
        symbol_col = "hgnc_symbol"
        cyclegenes = load_cyclegenes()
        hla_genes = anno_mat[symbol_col][anno_mat[symbol_col].str.startswith("HLA-", na=False)].tolist()
        genes_to_remove = set(cyclegenes) | set(hla_genes)
        keep_genes = ~anno_mat[symbol_col].isin(genes_to_remove).to_numpy()
        anno_mat = anno_mat[keep_genes].reset_index(drop=True)
        anno_rows = anno_rows[keep_genes]
    else:
        symbol_col = "mgi_symbol"
    elapsed = _record_step(
        runtime_info, "annotate_genes", step_start, extra={"genes_after_annotation": int(anno_mat.shape[0])}
    )
    logger.info(f"  step 2 runtime: {_format_seconds(elapsed)}")

    # Secondary cell filtering: ensure each cell has genes across chromosomes
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position", "ensembl_gene_id", symbol_col, "band"]
    step_start = time.perf_counter()
    expr_values = rawmat[anno_rows]
    del rawmat
    keep_cells = _keep_cells_by_chr_coverage(expr_values, anno_mat["chromosome_name"].to_numpy(), ngene_chr)
    if keep_cells.sum() == 0:
        raise ValueError("All cells are filtered out")
    cell_cols = list(barcodes)
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

    # Gene detection rates and post-UP_DR cell coverage only need the raw
    # counts; compute them now so rawmat3 can be transformed in place.
    DR2 = (rawmat3 > 0).sum(axis=1) / rawmat3.shape[1]
    seg_mask = DR2 >= UP_DR
    keep_cells2 = _keep_cells_by_chr_coverage(
        (rawmat3 != 0)[seg_mask], anno_mat["chromosome_name"].to_numpy()[seg_mask], ngene_chr
    )

    # Freeman-Tukey transformation: log(sqrt(x) + sqrt(x+1)), in place
    step_start = time.perf_counter()
    sqrt_plus_one = rawmat3 + 1
    np.sqrt(sqrt_plus_one, out=sqrt_plus_one)
    norm_mat = np.sqrt(rawmat3, out=rawmat3)
    del rawmat3
    norm_mat += sqrt_plus_one
    del sqrt_plus_one
    np.log(norm_mat, out=norm_mat)
    # Center each cell
    norm_mat -= norm_mat.mean(axis=0, keepdims=True)
    _record_step(runtime_info, "freeman_tukey_transform", step_start, extra={"matrix_shape": list(norm_mat.shape)})

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
    logger.info("step 4: measuring baselines ...")
    step_start = time.perf_counter()

    cell_name_list = cell_cols

    if cell_line is CellLineMode.YES:
        logger.info("  running pure cell line mode")
        relt = baseline_synthetic(
            norm_mat_smooth,
            min_cells=10,
            n_cores=n_cores,
            pca_components=selected_pca_components,
            genome=genome,
        )
        norm_mat_relat = relt["expr_relat"]
        CL = relt["cl"]
        WNS = BaselineWarning.CELL_LINE
        preN = None
    elif isinstance(norm_cell_names, list) and len(norm_cell_names) > 1:
        # Known normal cells provided
        norm_cell_set = set(norm_cell_names)
        known_normal_mask = np.array([c in norm_cell_set for c in cell_name_list], dtype=bool)
        NNN = known_normal_mask.sum()
        logger.info(f"  {NNN} known normal cells found in dataset")

        if NNN == 0:
            raise ValueError("Known normal cells provided but none found in dataset")

        logger.info("  run with known normal...")
        basel = np.median(norm_mat_smooth[:, known_normal_mask], axis=1)

        # Cluster all cells
        data_t = norm_mat_smooth.T
        step4_reduce = data_t.shape[0] > FULL_CLUSTER_MAX_CELLS
        km = 6
        CL, Z = _hierarchical_cluster(
            data_t,
            km,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step4_reduce,
            pca_components=selected_pca_components,
        )

        while not all(np.bincount(CL)[np.bincount(CL) > 0] > 5):
            km -= 1
            if Z is not None:
                CL = fcluster(Z, t=km, criterion="maxclust")
            else:
                CL, Z = _hierarchical_cluster(
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

        WNS = BaselineWarning.KNOWN_NORMAL
        preN = np.asarray(cell_name_list, dtype=object)[known_normal_mask].tolist()
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    else:
        # Auto-detect normal cells
        anchor_selector = None
        if anchor is AnchorStrategy.MARKERS and marker_counts is not None:
            immune_counts = marker_counts[0].groupby(level=0).first().reindex(cell_name_list).fillna(0).to_numpy()
            endothelial_counts = marker_counts[1].groupby(level=0).first().reindex(cell_name_list).fillna(0).to_numpy()

            def anchor_selector(labels: ClusterLabels, sigma_cluster: int) -> tuple[int, AnchorPath]:
                selected, path = _anchor.choose_anchor_cluster(labels, immune_counts, endothelial_counts, sigma_cluster)
                return int(selected), path

        basa = baseline_norm_cl(
            norm_mat_smooth,
            min_cells=5,
            n_cores=n_cores,
            cell_names=cell_name_list,
            pca_components=selected_pca_components,
            genome=genome,
            anchor_selector=anchor_selector,
        )
        basel = basa["basel"]
        WNS = basa["WNS"]
        preN = basa["preN"]
        clustered = basa["cl"]
        assert clustered is not None  # baseline_norm_cl always clusters
        CL = clustered
        runtime_info["anchor_path"] = basa.get("anchor_path", AnchorPath.SIGMA)
        if anchor is AnchorStrategy.MARKERS:
            WNS = (
                BaselineWarning.NONE
                if runtime_info["anchor_path"] in {AnchorPath.IMMUNE, AnchorPath.ENDOTHELIAL}
                else BaselineWarning.UNCLASSIFIED
            )
            logger.info(f"  normal reference from markers: path={runtime_info['anchor_path']}, cells={len(preN)}")

        if WNS is BaselineWarning.UNCLASSIFIED and anchor is not AnchorStrategy.MARKERS:
            cluster_preN = list(preN) if preN is not None else []
            keep_cluster_anchor = WNS1 is DataQualityStatus.LOW and len(cluster_preN) >= max(
                50, int(0.05 * len(cell_name_list))
            )
            if keep_cluster_anchor:
                logger.info("  low-data-quality mode: keeping cluster-based normal anchor")
            else:
                basa_cluster = basa
                basa_gmm = baseline_gmm(
                    norm_mat_smooth,
                    cell_name_list,
                    max_normal=5,
                    mu_cut=0.05,
                    Nfraq_cut=0.99,
                    RE_before=basa_cluster,
                    n_cores=n_cores,
                    pca_components=selected_pca_components,
                    genome=genome,
                    cluster=False,  # only basel/preN are used; CL stays from clustering
                )

                # baseline_gmm anchors on a handful of individually-scanned
                # cells (it stops at the first `max_normal` hits in raw cell
                # order) and can be far noisier than the clustering candidate
                # it is meant to replace -- a contaminated anchor set here
                # silently inverts the final diploid/aneuploid call downstream,
                # since cluster identity is decided purely by preN overlap.
                # Only adopt the fallback when its baseline profile is a
                # tighter, more confidently-neutral fit than the candidate it
                # would discard; otherwise keep the clustering answer even
                # though confidence is flagged low.
                #
                # Compare the two candidates with the same 3-component GMM
                # sigma that baseline_norm_cl already uses to rank its own
                # six clusters against each other, rather than a raw
                # mean(|basel|) magnitude. Magnitude is fit over genes for
                # both candidates, so it isn't literally biased by the
                # cell-count each basel was averaged over -- but a bigger,
                # more heterogeneous candidate can still land on a smaller
                # mean(|basel|) via cross-subpopulation cancellation rather
                # than genuine uniform neutrality, without that cancellation
                # showing up as a tighter (lower-sigma) GMM fit. Sigma
                # measures how cleanly the profile separates into
                # loss/neutral/gain, which is what "confidently neutral"
                # actually means here, so it is the more consistent yardstick
                # to reuse for this cross-candidate comparison.
                def _basel_sigma(basel_vec: GeneProfile) -> float:
                    sigma_init = max(0.05, 0.5 * float(np.std(basel_vec)))
                    return _fit_gmm_3component(basel_vec, sigma_init=sigma_init, max_iter=5000)[2]

                clustering_sigma = float(_basel_sigma(basa_cluster["basel"]))
                gmm_sigma = float(_basel_sigma(basa_gmm["basel"]))
                if gmm_sigma < clustering_sigma:
                    basa = basa_gmm
                else:
                    logger.info(
                        f"  GMM fallback baseline (sigma={gmm_sigma:.4f}) is not tighter/more confidently "
                        f"neutral than the clustering candidate (sigma={clustering_sigma:.4f}); "
                        "keeping cluster-based normal anchor"
                    )
                basel = basa["basel"]
                preN = basa["preN"]
                WNS = BaselineWarning.UNCLASSIFIED

        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    del norm_mat_smooth
    baseline_cluster_info = get_last_cluster_info()
    if cell_line is CellLineMode.YES:
        reference_mode = ReferenceMode.SYNTHETIC
    elif WNS is BaselineWarning.KNOWN_NORMAL:
        reference_mode = ReferenceMode.KNOWN_NORMAL
    else:
        reference_mode = ReferenceMode.AUTOMATIC
    runtime_info["reference"] = {
        "mode": reference_mode,
        "supplied_count": len(set(norm_cell_names)) if isinstance(norm_cell_names, list) else 0,
        "matched_supplied_count": len(set(norm_cell_names).intersection(cell_name_list))
        if isinstance(norm_cell_names, list)
        else 0,
        "baseline_anchor_count": len(_normal_cells_to_names(preN, cell_name_list).intersection(cell_name_list)),
    }
    elapsed = _record_step(
        runtime_info, "baseline_estimation", step_start, parallel_info=baseline_cluster_info, extra={"warning": WNS}
    )
    logger.info(
        f"  baseline runtime: {_format_seconds(elapsed)} "
        f"(parallel={baseline_cluster_info['parallel']}, cores={baseline_cluster_info['effective_cores']}, "
        f"engine={baseline_cluster_info.get('engine', 'n/a')})"
    )

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
    results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=KS_cut, n_cores=n_cores, ks_method=ks_method)

    if len(results["breaks"]) < 25:
        logger.info("  too few breakpoints; decreased KS_cut to 50%")
        results = cna_mcmc(
            CL_filtered,
            norm_mat_relat,
            bins=win_size,
            cut_cor=0.5 * KS_cut,
            n_cores=n_cores,
            ks_method=ks_method,
        )

    if len(results["breaks"]) < 25:
        logger.info("  too few breakpoints; decreased KS_cut to 25%")
        results = cna_mcmc(
            CL_filtered,
            norm_mat_relat,
            bins=win_size,
            cut_cor=0.25 * KS_cut,
            n_cores=n_cores,
            ks_method=ks_method,
        )

    if len(results["breaks"]) < 25:
        raise ValueError("Too few segments; try decreasing KS_cut or improving data quality")
    seg_info = get_last_cna_mcmc_info()
    elapsed = _record_step(
        runtime_info,
        "segmentation",
        step_start,
        parallel_info=seg_info,
        extra={"breakpoints": len(results["breaks"])},
    )
    logger.info(
        f"  segmentation runtime: {_format_seconds(elapsed)} "
        f"(parallel={seg_info['parallel']}, cores={seg_info['effective_cores']}, "
        f"engine={seg_info.get('engine', 'n/a')})"
    )

    results_com = results["logCNA"]
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
        uber_mat_adj = Aj["RNA_adj_values"]
        bin_coords = Aj["RNA_adj"][["chrom", "chrompos", "abspos"]].copy()
        chrom_info = Aj["DNA_adj"]["chrom"].to_numpy()
        del Aj

        logger.info("step 7: adjust baseline ...")
        step_start = time.perf_counter()

        if cell_line is CellLineMode.YES:
            mat_adj = uber_mat_adj
        else:
            arm_calls = None
            if final_call is FinalCallStrategy.ARM_CORRELATION and preN is not None and len(preN) > 0:
                preN_names = _normal_cells_to_names(preN, cell_name_list)
                anchor_mask = np.array([cell in preN_names for cell in cell_cols_seg], dtype=bool)
                if anchor_mask.sum() >= 5:
                    arm_calls = _anchor.arm_correlation_calls(
                        uber_mat_adj,
                        bin_coords["chrom"].to_numpy(),
                        bin_coords["chrompos"].to_numpy(),
                        anchor_mask,
                    )
                    runtime_info["final_call_path"] = FinalCallStrategy.ARM_CORRELATION
                else:
                    runtime_info["final_call_path"] = "clusters_insufficient_anchor"
            initial_call = cluster_and_call(
                uber_mat_adj,
                cell_cols_seg,
                cell_name_list,
                preN,
                n_cores=n_cores,
                pca_components=selected_pca_components,
                prediction_override=arm_calls,
            )
            com_pred = initial_call["predictions"]

            # Baseline adjustment: subtract diploid mean, then denoise
            diploid_mask = com_pred == PredictionLabel.DIPLOID
            if diploid_mask.sum() > 0:
                mat_adj = _adjust_baseline_inplace(uber_mat_adj, diploid_mask)
            else:
                mat_adj = uber_mat_adj
        del uber_mat_adj
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(
            runtime_info, "baseline_adjustment", step_start, parallel_info=cluster_info, extra={"warning": WNS}
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
        if cell_line is not CellLineMode.YES:
            final_call_result: FinalCallResult = cluster_and_call(
                mat_adj,
                cell_cols_seg,
                cell_name_list,
                preN,
                n_cores=n_cores,
                pca_components=selected_pca_components,
                prediction_override=arm_calls,
                low_confidence=WNS is BaselineWarning.UNCLASSIFIED,
            )
            labels_final, Z_final = final_call_result["labels"], final_call_result["Z"]
            com_preN = final_call_result["predictions"]
        else:
            clustering = cluster_cells(
                mat_adj,
                n_cores=n_cores,
                pca_components=selected_pca_components,
            )
            labels_final = clustering["labels"]
            Z_final = clustering["Z"]
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(
            runtime_info, "final_prediction", step_start, parallel_info=cluster_info, extra={"warning": WNS}
        )
        logger.info(
            f"  step 8 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, "
            f"engine={cluster_info.get('engine', 'n/a')})"
        )

        # =========================================================================
        # Step 9: Save results
        # =========================================================================
        pred_dict = None
        res = None
        if cell_line is not CellLineMode.YES:
            pred_dict = {cell_cols_seg[i]: com_preN[i] for i in range(len(cell_cols_seg))}
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
        cna_out = _frame_with_leading_columns(bin_coords, mat_adj, cell_cols_seg)
        _write_cna_csv(f"{sample_name}CNA_results.txt", bin_coords, mat_adj, cell_cols_seg, n_cores=n_cores)

        # Save clustering
        clustering_data: ClusteringResult = {
            "labels": labels_final,
            "Z": Z_final,
        }
        with open(f"{sample_name}clustering_results.pkl", "wb") as f:
            pickle.dump(clustering_data, f)
        elapsed = _record_step(
            runtime_info,
            "write_final_outputs",
            step_start,
            extra={"bins": int(cna_out.shape[0]), "cells": int(cna_out.shape[1] - 3)},
        )
        logger.info(f"  step 9 runtime: {_format_seconds(elapsed)}")

        # =========================================================================
        # Step 10: Plot heatmap
        # =========================================================================
        if plot_genes:
            logger.info("step 10: plotting heatmap ...")
            plot_step_start = time.perf_counter()
            predictions = pred_dict if cell_line is not CellLineMode.YES else None
            _run_plot_heatmap(
                mat_adj,
                chrom_info,
                predictions,
                sample_name,
                distance,
                n_cores,
                WNS1,
                WNS,
                f"{sample_name}heatmap.png",
            )
            elapsed = _record_step(runtime_info, "plot_heatmap", plot_step_start)
            logger.info(f"  step 10 runtime: {_format_seconds(elapsed)}")

        if plot_genes and meta_csv is not None:
            logger.info("step 10b: plotting annotated heatmap ...")
            step_ann = time.perf_counter()
            from copykat_py.plotting import plot_heatmap_annotated

            meta_pred_path = _meta_with_pred(meta_csv, pred_dict, sample_name)
            plot_heatmap_annotated(
                mat=mat_adj,
                cell_names=cna_out.columns[3:].tolist(),
                chrom_info=chrom_info,
                meta_csv=meta_pred_path,
                row_split_col=row_split_col,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                output_path=f"{sample_name}annotated_heatmap.png",
            )
            elapsed = _record_step(runtime_info, "plot_annotated_heatmap", step_ann)
            logger.info(f"  step 10b runtime: {_format_seconds(elapsed)}")

        # =========================================================================
        # Output SEG file
        # =========================================================================
        if output_seg:
            logger.info("  generating seg files for IGV viewer")
            _write_seg_file(bin_coords, mat_adj, cell_cols_seg, sample_name)
        runtime_info["total_seconds"] = round(time.perf_counter() - start_time, 4)
        with open(f"{sample_name}runtime.json", "w", encoding="utf-8") as report:
            json.dump(runtime_info, report, indent=2)
        logger.info(f"Done. Elapsed time: {_format_seconds(runtime_info['total_seconds'])}")
        logger.info(f"Runtime report saved to: {sample_name}runtime.json")

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

    else:
        # mm10: no bin conversion, use gene-level results directly
        uber_mat_adj = results_com  # adjusted in place below; results_com is not used again
        del results_com
        chrom_info = anno_mat2["chromosome_name"].to_numpy()

        logger.info("step 7: adjust baseline ...")
        step_start = time.perf_counter()
        initial_call = cluster_and_call(
            uber_mat_adj,
            cell_cols_seg,
            cell_name_list,
            preN,
            n_cores=n_cores,
            pca_components=selected_pca_components,
        )
        com_pred = initial_call["predictions"]

        # Baseline adjustment
        diploid_mask = com_pred == PredictionLabel.DIPLOID
        if diploid_mask.sum() > 0:
            mat_adj = _adjust_baseline_inplace(uber_mat_adj, diploid_mask)
        else:
            mat_adj = uber_mat_adj
        del uber_mat_adj
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(
            runtime_info, "baseline_adjustment", step_start, parallel_info=cluster_info, extra={"warning": WNS}
        )
        logger.info(
            f"  step 7 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, "
            f"engine={cluster_info.get('engine', 'n/a')})"
        )

        # Final prediction
        logger.info("step 8: final prediction ...")
        step_start = time.perf_counter()
        final_call_result = cluster_and_call(
            mat_adj,
            cell_cols_seg,
            cell_name_list,
            preN,
            n_cores=n_cores,
            pca_components=selected_pca_components,
            low_confidence=WNS is BaselineWarning.UNCLASSIFIED,
        )
        labels_final, Z_final = final_call_result["labels"], final_call_result["Z"]
        com_preN = final_call_result["predictions"]
        cluster_info = get_last_cluster_info()
        elapsed = _record_step(
            runtime_info, "final_prediction", step_start, parallel_info=cluster_info, extra={"warning": WNS}
        )
        logger.info(
            f"  step 8 runtime: {_format_seconds(elapsed)} "
            f"(parallel={cluster_info['parallel']}, cores={cluster_info['effective_cores']}, "
            f"engine={cluster_info.get('engine', 'n/a')})"
        )

        # Save
        logger.info("step 9: saving results ...")
        step_start = time.perf_counter()
        pred_dict = {cell_cols_seg[i]: com_preN[i] for i in range(len(cell_cols_seg))}
        for cell in original_cell_names:
            if cell not in pred_dict:
                pred_dict[cell] = PredictionLabel.NOT_DEFINED

        res = pd.DataFrame(
            {
                "cell.names": list(pred_dict.keys()),
                "copykat.pred": list(pred_dict.values()),
            }
        )
        res.to_csv(f"{sample_name}prediction.txt", sep="\t", index=False)

        cna_out = _frame_with_leading_columns(gene_anno, mat_adj, cell_cols_seg)
        _write_cna_csv(f"{sample_name}CNA_results.txt", gene_anno, mat_adj, cell_cols_seg, n_cores=n_cores)

        clustering_data = {"labels": labels_final, "Z": Z_final}
        with open(f"{sample_name}clustering_results.pkl", "wb") as f:
            pickle.dump(clustering_data, f)
        elapsed = _record_step(
            runtime_info,
            "write_final_outputs",
            step_start,
            extra={"bins": int(cna_out.shape[0]), "cells": len(cell_cols_seg)},
        )
        logger.info(f"  step 9 runtime: {_format_seconds(elapsed)}")

        chrom_numeric = pd.to_numeric(anno_mat2["chromosome_name"], errors="coerce").fillna(0).values
        if plot_genes:
            logger.info("step 10: plotting heatmap ...")
            step_start = time.perf_counter()
            from copykat_py.plotting import plot_heatmap

            plot_heatmap(
                mat_adj,
                chrom_numeric,
                predictions=pred_dict,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                WNS1=WNS1,
                WNS=WNS,
                output_path=f"{sample_name}heatmap.png",
                genome=genome,
            )
            elapsed = _record_step(runtime_info, "plot_heatmap", step_start)
            logger.info(f"  step 10 runtime: {_format_seconds(elapsed)}")

        if plot_genes and meta_csv is not None:
            logger.info("step 10b: plotting annotated heatmap ...")
            step_ann = time.perf_counter()
            from copykat_py.plotting import plot_heatmap_annotated

            meta_pred_path = _meta_with_pred(meta_csv, pred_dict, sample_name)
            plot_heatmap_annotated(
                mat=mat_adj,
                cell_names=cna_out.columns[7:].tolist(),
                chrom_info=chrom_numeric,
                meta_csv=meta_pred_path,
                row_split_col=row_split_col,
                sample_name=sample_name,
                distance=distance,
                n_cores=n_cores,
                output_path=f"{sample_name}annotated_heatmap.png",
                genome=genome,
            )
            elapsed = _record_step(runtime_info, "plot_annotated_heatmap", step_ann)
            logger.info(f"  step 10b runtime: {_format_seconds(elapsed)}")
        runtime_info["total_seconds"] = round(time.perf_counter() - start_time, 4)
        with open(f"{sample_name}runtime.json", "w", encoding="utf-8") as report:
            json.dump(runtime_info, report, indent=2)
        logger.info(f"Done. Elapsed time: {_format_seconds(runtime_info['total_seconds'])}")
        logger.info(f"Runtime report saved to: {sample_name}runtime.json")

        return {
            "prediction": res,
            "CNAmat": cna_out,
            "hclustering": clustering_data,
            "runtime": runtime_info,
        }
