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

import io
import json
import logging
import os
import pickle
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from scipy import sparse
from scipy.cluster.hierarchy import fcluster
from scipy.io import mmread

from copykat_py._logging import with_default_progress_output
from copykat_py._types import (
    BoolArray,
    ClusteringResult,
    ClusterLabels,
    CopyKATResult,
    FloatArray,
    GeneProfile,
    InputStats,
    ParallelInfo,
    RawMatrix,
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
    _effective_threads,
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
from copykat_py.segmentation import cna_mcmc, get_last_cna_mcmc_info
from copykat_py.smoothing import dlm_smooth, get_last_dlm_smooth_info

logger = logging.getLogger(__name__)

_WRITE_CHUNK_BYTES = 512 << 20
# Parallel formatting keeps several chunks (plus their rounded copies and
# text) in flight, so it gets a smaller total budget. Below the minimum rows
# per chunk, per-column pyarrow overhead outweighs the gain from parallelism,
# so very wide matrices stay serial.
_PARALLEL_WRITE_BYTES = 128 << 20
_PARALLEL_WRITE_MIN_ROWS = 128


def _write_cna_csv(
    path: str,
    lead_df: pd.DataFrame,
    values: FloatArray,
    value_columns: Sequence[str],
    round_floats: bool = True,
    quote_strings: bool = True,
    n_cores: int = 1,
) -> None:
    """Write ``lead_df`` columns followed by ``values`` columns as TSV, in row chunks.

    Equivalent to writing ``pd.concat([lead_df, pd.DataFrame(values, columns=value_columns)], axis=1)``
    but never materializes that frame (or a rounded copy of it); only one chunk
    of rows is converted at a time. Uses pyarrow (fast) with 6 d.p. float
    precision. With ``round_floats=False`` floats keep their shortest
    round-trip repr, and with ``quote_strings=False`` strings and the header
    are left unquoted; both together match pandas ``.to_csv(sep="\\t", index=False)``
    output.

    With ``n_cores > 1``, up to ``n_cores`` chunks are formatted
    concurrently and written in order, producing the same bytes; the total
    in-flight chunk size stays within ``_PARALLEL_WRITE_BYTES``.
    """
    lead_df = lead_df.reset_index(drop=True)
    if round_floats:
        lead_float_cols = [c for c in lead_df.columns if pd.api.types.is_float_dtype(lead_df[c])]
        if lead_float_cols:
            lead_df = lead_df.copy()
            lead_df[lead_float_cols] = lead_df[lead_float_cols].round(6)
    value_columns = list(value_columns)
    n_rows = values.shape[0]
    row_bytes = max(1, len(value_columns) * values.itemsize)
    chunk_rows = max(1, _WRITE_CHUNK_BYTES // row_bytes)

    def _value_chunks() -> Iterator[tuple[int, int, FloatArray]]:
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            block = values[start:stop]
            if round_floats and np.issubdtype(block.dtype, np.floating):
                block = np.round(block, 6)
            yield start, stop, block

    write_kwargs = {"delimiter": "\t"}
    if not quote_strings:
        write_kwargs["quoting_style"] = "none"
        write_kwargs["quoting_header"] = "none"
    write_options = pa_csv.WriteOptions(**write_kwargs)

    lead_table = pa.Table.from_pandas(lead_df, preserve_index=False).replace_schema_metadata(None)
    value_type = pa.from_numpy_dtype(values.dtype)
    schema = pa.schema(list(lead_table.schema) + [pa.field(str(c), value_type) for c in value_columns])

    def _chunk_table(start: int, stop: int, block: FloatArray) -> Any:
        block = np.asfortranarray(block)  # contiguous columns
        arrays = [col.combine_chunks() for col in lead_table.slice(start, stop - start).columns]
        arrays += [pa.array(block[:, j], type=value_type, from_pandas=True) for j in range(block.shape[1])]
        return pa.Table.from_arrays(arrays, schema=schema)

    n_threads = _effective_threads(n_cores)
    parallel_rows = _PARALLEL_WRITE_BYTES // (n_threads * row_bytes)
    if n_threads > 1 and parallel_rows >= _PARALLEL_WRITE_MIN_ROWS and n_rows > parallel_rows:
        chunk_rows = parallel_rows

        def _format(start: int) -> bytes:
            stop = min(start + chunk_rows, n_rows)
            block = values[start:stop]
            if round_floats and np.issubdtype(block.dtype, np.floating):
                block = np.round(block, 6)
            buf = io.BytesIO()
            options = pa_csv.WriteOptions(include_header=(start == 0), **write_kwargs)
            pa_csv.write_csv(_chunk_table(start, stop, block), buf, write_options=options)
            return buf.getvalue()

        pending: deque[Future[bytes]] = deque()
        with open(path, "wb") as f, ThreadPoolExecutor(n_threads) as executor:
            for start in range(0, n_rows, chunk_rows):
                pending.append(executor.submit(_format, start))
                if len(pending) >= n_threads:
                    f.write(pending.popleft().result())
            while pending:
                f.write(pending.popleft().result())
    else:
        with open(path, "wb") as f, pa_csv.CSVWriter(f, schema, write_options=write_options) as writer:
            for start, stop, block in _value_chunks():
                writer.write_table(_chunk_table(start, stop, block))
    # Arrow's allocator keeps freed chunk buffers cached; hand them back
    # so they don't inflate the peak of the steps that follow.
    pa.default_memory_pool().release_unused()


def _frame_with_leading_columns(
    lead_df: pd.DataFrame, values: FloatArray, value_columns: Sequence[str]
) -> pd.DataFrame:
    """DataFrame of ``lead_df`` columns followed by ``values``, without copying ``values``."""
    frame = pd.DataFrame(values, columns=list(value_columns), copy=False)
    lead_df = lead_df.reset_index(drop=True)
    for pos, col in enumerate(lead_df.columns):
        frame.insert(pos, col, lead_df[col].to_numpy())
    return frame


def _adjust_baseline_inplace(mat: FloatArray, diploid_mask: BoolArray, chunk_elems: int = 1 << 24) -> FloatArray:
    """Subtract the diploid baseline and flatten noise around it, overwriting ``mat``.

    Same arithmetic as the original out-of-place version: center on the
    diploid mean, re-center cells, replace values within 0.25 SD of the
    diploid profile with the cell mean, and re-center cells again.
    """
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


def _meta_with_pred(meta_csv: str, pred_dict: dict[str, str] | None, sample_name: str) -> str:
    """Read *meta_csv*, append copykat-py predictions as the last column.

    Returns the path to a new CSV written alongside the original outputs.
    Cells absent from *pred_dict* receive ``"not.defined"``.
    """
    import pandas as pd

    meta = pd.read_csv(meta_csv)
    cell_col = meta.columns[0]
    meta = meta.set_index(cell_col)
    if pred_dict is not None:
        meta["copykat_pred_py"] = meta.index.map(pred_dict).fillna("not.defined")
    else:
        meta["copykat_pred_py"] = "not.defined"
    out_path = f"{sample_name}meta_with_pred.csv"
    meta.reset_index().to_csv(out_path, index=False)
    return out_path


def _run_plot_heatmap(
    mat_adj: FloatArray,
    chrom_info: npt.NDArray[Any],
    predictions: dict[str, str] | None,
    sample_name: str,
    distance: str,
    n_cores: int,
    WNS1: str,
    WNS: str,
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


def _load_matrix(rawmat: RawMatrix) -> pd.DataFrame:
    """Load raw matrix from various input formats.

    Supports: pd.DataFrame, scipy sparse, numpy array, dict (matrix/genes/barcodes), or file path (mtx/csv/tsv).
    """
    if isinstance(rawmat, dict):
        # Dict format from CLI with sparse matrix + gene/barcode names
        mat = rawmat.get("matrix")
        genes = rawmat.get("genes")
        barcodes = rawmat.get("barcodes")

        if mat is None:
            raise ValueError("Dict input must contain 'matrix' key")

        # Convert to dense if sparse
        if hasattr(mat, "toarray"):
            mat = mat.toarray()

        df = pd.DataFrame(
            mat,
            index=None if genes is None else pd.Index(genes),
            columns=None if barcodes is None else pd.Index(barcodes),
        )
        return df
    elif isinstance(rawmat, pd.DataFrame):
        return rawmat
    elif isinstance(rawmat, np.ndarray):
        return pd.DataFrame(rawmat)
    elif hasattr(rawmat, "toarray"):
        # scipy sparse
        return pd.DataFrame(rawmat.toarray())
    elif isinstance(rawmat, str):
        if rawmat.endswith(".mtx") or rawmat.endswith(".mtx.gz"):
            mat = mmread(rawmat)
            return pd.DataFrame(mat.toarray() if hasattr(mat, "toarray") else mat)
        elif rawmat.endswith(".csv"):
            return pd.read_csv(rawmat, index_col=0)
        elif rawmat.endswith(".tsv") or rawmat.endswith(".txt"):
            return pd.read_csv(rawmat, sep="\t", index_col=0)
        else:
            return pd.read_csv(rawmat, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported rawmat type: {type(rawmat)}")


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


def _preN_to_names(preN: str | bytes | Sequence[Any] | npt.NDArray[Any] | None, cell_names: Sequence[str]) -> set[str]:
    if preN is None:
        return set()

    if isinstance(preN, (str, bytes)):
        return {str(preN)}

    values = list(preN)
    if not values:
        return set()

    names = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if 0 <= idx < len(cell_names):
                names.append(cell_names[idx])
        else:
            names.append(str(value))
    return set(names)


def _assign_binary_labels(
    cluster_labels: ClusterLabels, scores: Sequence[float] | FloatArray, high_label: str, low_label: str
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


def _aggregate_duplicate_genes_sparse(
    mat: sparse.csc_matrix, genes: npt.NDArray[Any]
) -> tuple[sparse.csc_matrix, npt.NDArray[Any]]:
    genes = np.asarray(genes, dtype=object)
    unique_genes, inverse = np.unique(genes.astype(str), return_inverse=True)
    if len(unique_genes) == len(genes):
        return mat, genes

    row_map = sparse.csr_matrix(
        (np.ones(len(genes), dtype=np.float64), (inverse, np.arange(len(genes)))),
        shape=(len(unique_genes), len(genes)),
    )
    return (row_map @ mat).tocsc(), unique_genes.astype(object)


def _aggregate_duplicate_genes_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df

    # Match R rowsum(..., reorder=TRUE): sum duplicate symbols and sort groups.
    return df.groupby(level=0, sort=True).sum()


def _prepare_sparse_input(
    mat: SparseMatrix,
    genes: npt.NDArray[Any],
    barcodes: npt.NDArray[Any],
    min_gene_per_cell: int,
    low_dr: float,
) -> tuple[sparse.csr_matrix, npt.NDArray[Any], list[str], list[str], InputStats]:
    """Sparse branch of ``_prepare_input_matrix``: filter without densifying."""
    csc, genes = _aggregate_duplicate_genes_sparse(mat.tocsc(copy=False), genes)
    original_cell_names = barcodes.tolist()
    genes_per_cell = np.asarray(csc.getnnz(axis=0)).ravel()
    if (genes_per_cell > min_gene_per_cell).sum() == 0:
        raise ValueError("No cells have more than min_gene_per_cell genes")
    keep_cells = genes_per_cell >= min_gene_per_cell
    filtered_cells = int((~keep_cells).sum())
    csc = csc[:, keep_cells]
    barcodes = barcodes[keep_cells]
    detection_rate = np.asarray(csc.getnnz(axis=1)).ravel() / max(csc.shape[1], 1)
    keep_genes = detection_rate > low_dr
    filtered_gene_rows = int((~keep_genes).sum())
    return (
        csc[keep_genes, :].tocsr(),
        genes[keep_genes],
        barcodes.tolist(),
        original_cell_names,
        {
            "input_type": "sparse_dict",
            "filtered_cells": filtered_cells,
            "filtered_gene_rows": filtered_gene_rows,
        },
    )


def _prepare_input_matrix(
    rawmat: RawMatrix, min_gene_per_cell: int, low_dr: float
) -> tuple[sparse.csr_matrix | npt.NDArray[Any], pd.Index | npt.NDArray[Any], list[str], list[str], InputStats]:
    """Filter cells and genes; returns (matrix, genes, barcodes, original_cell_names, stats).

    The matrix is genes x cells, scipy CSR for sparse dict input (so it is
    only densified after annotation and cell filtering) or a numpy array.
    """
    if isinstance(rawmat, dict):
        mat = rawmat.get("matrix")
        if mat is None:
            raise ValueError("Dict input must contain 'matrix' key")
        if sparse.issparse(mat):
            return _prepare_sparse_input(
                mat,
                np.asarray(rawmat.get("genes")),
                np.asarray(rawmat.get("barcodes")),
                min_gene_per_cell,
                low_dr,
            )

    loaded = _load_matrix(rawmat)
    loaded = _aggregate_duplicate_genes_frame(loaded)
    original_cell_names = list(loaded.columns)
    genes_per_cell = (loaded > 0).sum(axis=0)
    if (genes_per_cell > min_gene_per_cell).sum() == 0:
        raise ValueError("No cells have more than min_gene_per_cell genes")
    low_gene_cells = genes_per_cell < min_gene_per_cell
    filtered_cells = int(low_gene_cells.sum())
    if filtered_cells > 0:
        loaded = loaded.loc[:, ~low_gene_cells]
    detection_rate = (loaded > 0).sum(axis=1) / loaded.shape[1]
    keep_genes = detection_rate > low_dr
    filtered_gene_rows = int((~keep_genes).sum())
    if keep_genes.sum() >= 1:
        loaded = loaded.loc[keep_genes]
    return (
        loaded.to_numpy(),
        loaded.index,
        list(loaded.columns),
        original_cell_names,
        {
            "input_type": "dense",
            "filtered_cells": filtered_cells,
            "filtered_gene_rows": filtered_gene_rows,
        },
    )


def _keep_cells_by_chr_coverage(
    values: SparseMatrix | npt.NDArray[Any], chroms: npt.NDArray[Any], ngene_chr: int
) -> BoolArray:
    chrom_codes, unique_chroms = pd.factorize(chroms, sort=False)
    if sparse.issparse(values):
        # Per-chromosome nonzero counts as (chromosome indicator) @ (nonzero pattern)
        indicator = sparse.csr_matrix(
            (np.ones(len(chrom_codes), dtype=np.int64), (chrom_codes, np.arange(len(chrom_codes)))),
            shape=(len(unique_chroms), len(chrom_codes)),
        )
        counts = (indicator @ values.astype(bool).astype(np.int64)).toarray()
    else:
        nonzero = values != 0
        counts = np.vstack([nonzero[chrom_codes == chrom_idx].sum(axis=0) for chrom_idx in range(len(unique_chroms))])
    keep = (counts.sum(axis=0) >= 5) & (counts > 0).all(axis=0) & (counts.min(axis=0) >= ngene_chr)
    return keep


@with_default_progress_output
def copykat(
    rawmat: RawMatrix,
    id_type: str = "S",
    cell_line: str = "no",
    ngene_chr: int = 5,
    min_gene_per_cell: int = 200,
    LOW_DR: float = 0.05,
    UP_DR: float = 0.1,
    win_size: int = 25,
    norm_cell_names: str | list[str] = "",
    KS_cut: float = 0.1,
    sam_name: str = "",
    distance: str = "euclidean",
    output_seg: bool = False,
    plot_genes: bool = True,
    genome: str = "hg20",
    n_cores: int = 1,
    pca_components: int | None = None,
    meta_csv: str | None = None,
    row_split_col: str | None = None,
) -> CopyKATResult:
    """Run CopyKAT analysis: infer copy number profiles from scRNA-seq data.

    Parameters
    ----------
    rawmat : pd.DataFrame, np.ndarray, scipy.sparse, or str
        UMI count matrix (genes in rows, cells in columns).
        If str, path to .mtx, .csv, or .tsv file.
    id_type : str
        Gene ID type: "S" for Symbol, "E" for Ensembl.
    cell_line : str
        "yes" for pure cell line data, "no" for tumor/normal mixture.
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
    distance : str
        Distance metric: "euclidean", "pearson", or "spearman".
    output_seg : bool
        Whether to output .seg file for IGV.
    plot_genes : bool
        Whether to plot gene-level heatmap.
    genome : str
        "hg20" or "mm10".
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
            "id_type": id_type, "cell_line": cell_line, "genome": genome,
            "ngene_chr": ngene_chr, "min_gene_per_cell": min_gene_per_cell,
            "LOW_DR": LOW_DR, "UP_DR": UP_DR, "win_size": win_size, "KS_cut": KS_cut,
            "distance": distance, "n_cores": n_cores, "pca_components_requested": pca_components,
            "output_seg": output_seg, "plot_genes": plot_genes, "random_seed": 1234,
            "meta_csv": meta_csv, "row_split_col": row_split_col,
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
    if str(genome).strip().lower() == "mm10":
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

    WNS1 = "data quality is ok"
    if rawmat.shape[0] < 7000:
        WNS1 = "low data quality"
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
    runtime_info["parameters"]["gene_order"] = "chromosome,start_position" if genome == "mm10" else "abspos"

    # =========================================================================
    # Step 3: Remove cell cycle genes and HLA genes (hg20 only)
    # =========================================================================
    if genome == "hg20":
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

    if cell_line == "yes":
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
        WNS = "run with cell line mode"
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

        WNS = "run with known normal"
        preN = np.asarray(cell_name_list, dtype=object)[known_normal_mask].tolist()
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    else:
        # Auto-detect normal cells
        basa = baseline_norm_cl(
            norm_mat_smooth,
            min_cells=5,
            n_cores=n_cores,
            cell_names=cell_name_list,
            pca_components=selected_pca_components,
            genome=genome,
        )
        basel = basa["basel"]
        WNS = basa["WNS"]
        preN = basa["preN"]
        clustered = basa["cl"]
        assert clustered is not None  # baseline_norm_cl always clusters
        CL = clustered

        if WNS == "unclassified.prediction":
            cluster_preN = list(preN) if preN is not None else []
            keep_cluster_anchor = WNS1 == "low data quality" and len(cluster_preN) >= max(
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
                WNS = "unclassified.prediction"

        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    del norm_mat_smooth
    baseline_cluster_info = get_last_cluster_info()
    runtime_info["reference"] = {
        "mode": (
            "synthetic" if cell_line == "yes" else "known_normal" if WNS == "run with known normal" else "automatic"
        ),
        "supplied_count": len(set(norm_cell_names)) if isinstance(norm_cell_names, list) else 0,
        "matched_supplied_count": len(set(norm_cell_names).intersection(cell_name_list))
        if isinstance(norm_cell_names, list) else 0,
        "baseline_anchor_count": len(_preN_to_names(preN, cell_name_list).intersection(cell_name_list)),
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
    results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=KS_cut, n_cores=n_cores)

    if len(results["breaks"]) < 25:
        logger.info("  too few breakpoints; decreased KS_cut to 50%")
        results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=0.5 * KS_cut, n_cores=n_cores)

    if len(results["breaks"]) < 25:
        logger.info("  too few breakpoints; decreased KS_cut to 25%")
        results = cna_mcmc(CL_filtered, norm_mat_relat, bins=win_size, cut_cor=0.25 * KS_cut, n_cores=n_cores)

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
    if genome == "hg20":
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
        step7_reduce = uber_mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS

        if cell_line == "yes":
            mat_adj = uber_mat_adj
        else:
            # First hierarchical clustering for initial prediction
            labels, Z = _hierarchical_cluster(
                uber_mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step7_reduce,
                pca_components=selected_pca_components,
            )
            hc_umap = labels

            # Determine which cluster is normal based on preN enrichment
            if preN is not None and len(preN) > 0:
                preN_names = _preN_to_names(preN, cell_name_list)
                cl_ID = []
                for cl_val in sorted(set(hc_umap)):
                    cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_umap[j] == cl_val]
                    pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                    cl_ID.append(pid)
                com_pred = _assign_binary_labels(hc_umap, cl_ID, "diploid", "aneuploid")
            else:
                # If no preN, assign based on total CNA magnitude
                cl_mag = []
                for cl_val in sorted(set(hc_umap)):
                    mask = hc_umap == cl_val
                    cl_mag.append(np.mean(np.abs(uber_mat_adj[:, mask])))
                com_pred = _assign_binary_labels(hc_umap, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")

            # Baseline adjustment: subtract diploid mean, then denoise
            diploid_mask = com_pred == "diploid"
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
        step8_reduce = mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        if cell_line != "yes":
            labels_final, Z_final = _hierarchical_cluster(
                mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step8_reduce,
                pca_components=selected_pca_components,
            )
            hc_final = labels_final

            if preN is not None and len(preN) > 0:
                preN_names = _preN_to_names(preN, cell_name_list)
                cl_ID_final = []
                for cl_val in sorted(set(hc_final)):
                    cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_final[j] == cl_val]
                    pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                    cl_ID_final.append(pid)
                com_preN = _assign_binary_labels(hc_final, cl_ID_final, "diploid", "aneuploid")
            else:
                cl_mag = []
                for cl_val in sorted(set(hc_final)):
                    mask = hc_final == cl_val
                    cl_mag.append(np.mean(np.abs(mat_adj[:, mask])))
                com_preN = _assign_binary_labels(hc_final, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")

            if WNS == "unclassified.prediction":
                com_preN = np.where(com_preN == "diploid", "c1:diploid:low.conf", com_preN)
                com_preN = np.where(com_preN == "aneuploid", "c2:aneuploid:low.conf", com_preN)
        else:
            labels, Z = _hierarchical_cluster(
                mat_adj.T,
                2,
                method="ward",
                metric="euclidean",
                n_cores=n_cores,
                reduce=step8_reduce,
                pca_components=selected_pca_components,
            )
            labels_final, Z_final = labels, Z
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
        if cell_line != "yes":
            pred_dict = {cell_cols_seg[i]: com_preN[i] for i in range(len(cell_cols_seg))}
            for cell in original_cell_names:
                if cell not in pred_dict:
                    pred_dict[cell] = "not.defined"
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
            "labels": labels_final if cell_line != "yes" else labels,
            "Z": Z_final if cell_line != "yes" else Z,
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
            predictions = pred_dict if cell_line != "yes" else None
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
        step7_reduce = uber_mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        # Same prediction logic as hg20 (mirroring the R code mm10 section)
        labels, Z = _hierarchical_cluster(
            uber_mat_adj.T,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step7_reduce,
            pca_components=selected_pca_components,
        )
        hc_umap = labels

        if preN is not None and len(preN) > 0:
            preN_names = _preN_to_names(preN, cell_name_list)

            cl_ID = []
            for cl_val in sorted(set(hc_umap)):
                cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_umap[j] == cl_val]
                pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                cl_ID.append(pid)
        else:
            cl_ID = []
            for cl_val in sorted(set(hc_umap)):
                mask = hc_umap == cl_val
                cl_ID.append(float(np.mean(np.abs(uber_mat_adj[:, mask]))))

        if preN is not None and len(preN) > 0:
            com_pred = _assign_binary_labels(hc_umap, cl_ID, "diploid", "aneuploid")
        else:
            com_pred = _assign_binary_labels(hc_umap, -np.asarray(cl_ID, dtype=float), "diploid", "aneuploid")

        # Baseline adjustment
        diploid_mask = com_pred == "diploid"
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
        step8_reduce = mat_adj.shape[1] > FULL_CLUSTER_MAX_CELLS
        labels_final, Z_final = _hierarchical_cluster(
            mat_adj.T,
            2,
            method="ward",
            metric="euclidean",
            n_cores=n_cores,
            reduce=step8_reduce,
            pca_components=selected_pca_components,
        )
        hc_final = labels_final

        if preN is not None and len(preN) > 0:
            preN_names = _preN_to_names(preN, cell_name_list)
            cl_ID_final = []
            for cl_val in sorted(set(hc_final)):
                cli_names = [cell_cols_seg[j] for j in range(len(cell_cols_seg)) if hc_final[j] == cl_val]
                pid = len(set(cli_names) & preN_names) / max(len(cli_names), 1)
                cl_ID_final.append(pid)
            com_preN = _assign_binary_labels(hc_final, cl_ID_final, "diploid", "aneuploid")
        else:
            cl_mag = []
            for cl_val in sorted(set(hc_final)):
                mask = hc_final == cl_val
                cl_mag.append(np.mean(np.abs(mat_adj[:, mask])))
            com_preN = _assign_binary_labels(hc_final, -np.asarray(cl_mag, dtype=float), "diploid", "aneuploid")

        if WNS == "unclassified.prediction":
            com_preN = np.where(com_preN == "diploid", "c1:diploid:low.conf", com_preN)
            com_preN = np.where(com_preN == "aneuploid", "c2:aneuploid:low.conf", com_preN)
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
                pred_dict[cell] = "not.defined"

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


def _write_seg_file(RNA_adj_df: pd.DataFrame, mat_adj: FloatArray, cell_cols: Sequence[str], sample_name: str) -> None:
    """Write .seg file for IGV visualization."""
    rows = []
    chroms = RNA_adj_df["chrom"].to_numpy()
    chrompos = RNA_adj_df["chrompos"].values
    unique_chroms = np.unique(chroms)

    for ci, cell in enumerate(cell_cols):
        vals = mat_adj[:, ci]
        for chrom_val in unique_chroms:
            mask = chroms == chrom_val
            sub_vals = vals[mask]
            sub_pos = chrompos[mask]

            # RLE encoding
            if len(sub_vals) == 0:
                continue

            changes = np.where(np.diff(sub_vals) != 0)[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(sub_vals)]])

            for s, e in zip(starts, ends, strict=True):
                rows.append(
                    {
                        "ID": cell,
                        "chrom": chrom_val,
                        "loc.start": sub_pos[s],
                        "loc.end": sub_pos[e - 1],
                        "num.mark": e - s,
                        "seg.mean": sub_vals[s],
                    }
                )

    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(f"{sample_name}CNA_results.seg", sep="\t", index=False)
