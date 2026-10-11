"""Gene annotation and in-place count transformation for CopyKAT."""

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from copykat_py._pipeline.runtime import _format_seconds, _record_step
from copykat_py._types import BoolArray, FloatArray, GeneIdType, Genome, IntArray, RuntimeInfo, SparseMatrix
from copykat_py.annotation import annotate_gene_rows
from copykat_py.data_loader import load_cyclegenes
from copykat_py.input import _keep_cells_by_chr_coverage
from copykat_py.smoothing import dlm_smooth

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class FilteringOptions:
    """Requested cell coverage and gene detection thresholds.

    Attributes
    ----------
    min_gene_per_cell : int
        Minimum number of detected genes needed to retain an input cell.
    ngene_chr : int
        Minimum number of detected genes per represented chromosome for each
        retained cell, checked before smoothing and again before segmentation.
    lower_detection_rate : float
        Requested ``LOW_DR`` threshold, expressed as a fraction of cells.
        Input gene filtering uses detection rates strictly above this value.
    upper_detection_rate : float
        Requested ``UP_DR`` threshold for segmentation, expressed as a fraction
        of cells. Segmentation retains genes at or above the effective value;
        when fewer than 7,000 genes survive input filtering, the effective
        value is replaced with ``lower_detection_rate``. This field retains
        the original request for reporting.
    """

    min_gene_per_cell: int
    ngene_chr: int
    lower_detection_rate: float
    upper_detection_rate: float


def annotate_genes(
    genes: pd.Index | npt.NDArray[Any],
    *,
    id_type: GeneIdType,
    genome: Genome,
    runtime_info: RuntimeInfo,
) -> tuple[pd.DataFrame, IntArray, list[str]]:
    """Order annotated genes and remove human cell-cycle and HLA genes."""
    logger.info("step 2: annotating gene coordinates ...")
    step_start = time.perf_counter()
    anno_mat, anno_rows = annotate_gene_rows(genes, id_type=id_type, genome=genome)
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

    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position", "ensembl_gene_id", symbol_col, "band"]
    return anno_mat, anno_rows, anno_cols


def transform_counts_inplace(
    rawmat3: FloatArray,
    chromosomes: npt.NDArray[Any],
    *,
    detection_rate: float,
    ngene_chr: int,
    runtime_info: RuntimeInfo,
) -> tuple[FloatArray, BoolArray, BoolArray]:
    """Compute segmentation masks, then transform the same backing count array."""
    # Gene detection rates and post-UP_DR cell coverage only need the raw
    # counts; compute them now so rawmat3 can be transformed in place.
    DR2 = (rawmat3 > 0).sum(axis=1) / rawmat3.shape[1]
    seg_mask = detection_rate <= DR2
    keep_cells2 = _keep_cells_by_chr_coverage((rawmat3 != 0)[seg_mask], chromosomes[seg_mask], ngene_chr)

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

    return norm_mat, seg_mask, keep_cells2


# Cells per block in transform_and_smooth: two float64 genes x cells blocks
# (counts and their sqrt(x + 1)) stay around 0.5 GB for 8,000 genes.
_STREAM_BLOCK_CELLS = 4096


def transform_and_smooth(
    counts: SparseMatrix | FloatArray,
    chromosomes: npt.NDArray[Any],
    *,
    detection_rate: float,
    ngene_chr: int,
    n_cores: int,
    runtime_info: RuntimeInfo,
) -> tuple[FloatArray, BoolArray, BoolArray, float]:
    """Freeman-Tukey transform, centre and DLM-smooth ``counts`` (genes x cells) block by block.

    Same values as densifying ``counts`` to a column-major float64 matrix,
    ``transform_counts_inplace`` and ``dlm_smooth``: every step works on
    whole cells, and each block is column-major like the full matrix, so
    NumPy sums each cell in the same order. Only one block of cells is ever
    dense in float64; the result is the float32 smoothed matrix.

    Returns the smoothed matrix, the segmentation gene mask, the
    post-UP_DR cell mask and the seconds spent smoothing (the transform is
    recorded as the ``freeman_tukey_transform`` step).
    """
    n_genes, n_cells = counts.shape
    if sparse.issparse(counts):
        counts = cast(SparseMatrix, counts).tocsc()
        positive = cast(SparseMatrix, counts > 0)
        DR2 = np.asarray(positive.sum(axis=1)).ravel() / n_cells
    else:
        DR2 = (counts > 0).sum(axis=1) / n_cells
    seg_mask = detection_rate <= DR2
    keep_cells2 = _keep_cells_by_chr_coverage(counts[seg_mask], chromosomes[seg_mask], ngene_chr)

    smoothed = np.empty((n_genes, n_cells), dtype=np.float32)
    transform_seconds = 0.0
    smooth_seconds = 0.0
    for lo in range(0, n_cells, _STREAM_BLOCK_CELLS):
        hi = min(n_cells, lo + _STREAM_BLOCK_CELLS)
        start = time.perf_counter()
        if sparse.issparse(counts):
            block = cast(SparseMatrix, counts[:, lo:hi]).astype(np.float64).toarray(order="F")
        else:
            block = np.asfortranarray(counts[:, lo:hi], dtype=np.float64)
        # Freeman-Tukey transformation: log(sqrt(x) + sqrt(x+1)), then centre each cell
        sqrt_plus_one = block + 1
        np.sqrt(sqrt_plus_one, out=sqrt_plus_one)
        np.sqrt(block, out=block)
        block += sqrt_plus_one
        del sqrt_plus_one
        np.log(block, out=block)
        block -= block.mean(axis=0, keepdims=True)
        transform_seconds += time.perf_counter() - start
        start = time.perf_counter()
        dlm_smooth(block, n_cores=n_cores, out=smoothed[:, lo:hi])
        smooth_seconds += time.perf_counter() - start
    _record_step(
        runtime_info,
        "freeman_tukey_transform",
        time.perf_counter() - transform_seconds,
        extra={"matrix_shape": [int(n_genes), int(n_cells)]},
    )
    return smoothed, seg_mask, keep_cells2, smooth_seconds
