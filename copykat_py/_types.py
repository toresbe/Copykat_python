"""Shared type definitions: names for the data shapes passed between pipeline steps.

Array orientation conventions (mypy cannot check shapes, so the aliases below
document them rather than enforce them):

- Expression and copy-number matrices passed between steps are features x
  cells: ``GeneByCell`` (genes x cells) and ``BinByCell`` (genomic bins x cells).
- Clustering works on the transpose, ``CellByFeature`` (cells x features).
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

GeneByCell: TypeAlias = FloatArray
"""(n_genes, n_cells) expression or copy-number values."""

BinByCell: TypeAlias = FloatArray
"""(n_bins, n_cells) copy-number values on the genomic bins (hg20 only)."""

CellByFeature: TypeAlias = FloatArray
"""(n_cells, n_features) clustering input: one row per cell."""

GeneProfile: TypeAlias = FloatArray
"""(n_genes,) one value per gene, e.g. a baseline copy-number profile."""

ClusterLabels: TypeAlias = IntArray
"""(n_cells,) cluster ids, 1-based like scipy's fcluster and R's cutree."""

LinkageMatrix: TypeAlias = npt.NDArray[np.float64]
"""(n_cells - 1, 4) hierarchical clustering linkage, as from scipy.cluster.hierarchy.linkage."""

# Sparse count matrices as they reach the pipeline: COO from scipy.io.mmread,
# CSC from the AnnData wrapper, CSR from callers and after row selection.
SparseMatrix: TypeAlias = sparse.coo_matrix | sparse.csr_matrix | sparse.csc_matrix


class DistanceMetric(StrEnum):
    """Supported distance metrics for ordering cells in CNA heatmaps."""

    EUCLIDEAN = "euclidean"
    PEARSON = "pearson"
    SPEARMAN = "spearman"


class Genome(StrEnum):
    """Supported reference genome assemblies."""

    HG20 = "hg20"
    MM10 = "mm10"


class RawInput(TypedDict):
    """Count matrix with its gene and cell names (genes x cells)."""

    matrix: SparseMatrix | npt.NDArray[Any]
    genes: Sequence[str] | npt.NDArray[Any]
    barcodes: Sequence[str] | npt.NDArray[Any]


RawMatrix: TypeAlias = pd.DataFrame | npt.NDArray[Any] | SparseMatrix | RawInput | str
"""Inputs accepted by ``copykat()``: genes x cells, or a path to a .mtx/.csv/.tsv file."""

BaselineWarning: TypeAlias = Literal["", "unclassified.prediction", "run with cell line mode", "run with known normal"]
"""Classification note carried through the pipeline (``WNS`` in the R code)."""


class BaselineResult(TypedDict):
    """Reference normal cells and the baseline profile derived from them."""

    basel: GeneProfile
    WNS: BaselineWarning
    preN: list[str] | IntArray
    """Reference normal cells: names when cell names are known, otherwise column indices."""
    cl: ClusterLabels | None
    anchor_path: NotRequired[str]


class SyntheticBaselineResult(TypedDict):
    """Cell-line mode: expression relative to a synthetic normal per cluster."""

    expr_relat: GeneByCell
    cl: ClusterLabels


class SegmentationResult(TypedDict):
    logCNA: GeneByCell
    breaks: list[int]
    """Segment boundaries as gene indices, including the first and last gene."""


class BinConversion(TypedDict):
    DNA_adj: pd.DataFrame
    """Genomic bins (chrom, chrompos, abspos), chromosome Y removed."""
    RNA_adj: pd.DataFrame
    """chrom, chrompos, abspos, then one column per cell."""
    RNA_adj_values: BinByCell
    """The values backing ``RNA_adj``'s cell columns (column-major)."""


class ClusteringResult(TypedDict):
    labels: ClusterLabels
    Z: LinkageMatrix | None


class ParallelInfo(TypedDict, total=False):
    """How a step ran, for the runtime report (``get_last_*_info()``)."""

    step: str
    parallel: bool
    requested_cores: int
    effective_cores: int
    tasks: int
    chunk_size: int
    mc_samples: int
    engine: str
    approximate: bool


class InputStats(TypedDict):
    input_type: str
    filtered_cells: int
    filtered_gene_rows: int


class RuntimeInfo(TypedDict):
    """Contents of ``<sample>_copykat_runtime.json``."""

    sample_name: str
    requested_cores: int
    available_cores: int
    steps: list[dict[str, Any]]
    pca_components: NotRequired[int]
    pca_selection_mode: NotRequired[str]
    pca_selection_genome: NotRequired[str]
    pca_selection_input_cells: NotRequired[int]
    pca_selection_rule: NotRequired[str]
    total_seconds: NotRequired[float]
    parameters: NotRequired[dict[str, Any]]
    versions: NotRequired[dict[str, str]]
    reference: NotRequired[dict[str, Any]]
    warnings: NotRequired[list[str]]


class CopyKATResult(TypedDict):
    """Return value of ``copykat()``."""

    prediction: NotRequired[pd.DataFrame]
    """cell.names, copykat.pred; absent in cell-line mode."""
    CNAmat: pd.DataFrame
    """hg20: chrom, chrompos, abspos, then one column per cell (genomic bins).
    mm10: the seven gene annotation columns, then one column per cell (genes)."""
    hclustering: ClusteringResult
    runtime: RuntimeInfo
    allele_orientation: NotRequired[dict[str, Any]]
