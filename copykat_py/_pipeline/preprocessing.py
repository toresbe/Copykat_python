"""Gene annotation and in-place count transformation for CopyKAT."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from copykat_py._pipeline.runtime import _format_seconds, _record_step
from copykat_py._types import BoolArray, FloatArray, GeneIdType, Genome, IntArray, RuntimeInfo
from copykat_py.annotation import annotate_gene_rows
from copykat_py.data_loader import load_cyclegenes
from copykat_py.input import _keep_cells_by_chr_coverage

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
