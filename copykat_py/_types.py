"""Shared type definitions: names for the data shapes passed between pipeline steps.

Array orientation conventions (mypy cannot check shapes, so the aliases below
document them rather than enforce them):

- Expression and copy-number matrices passed between steps are features x
  cells: ``GeneByCell`` (genes x cells) and ``BinByCell`` (genomic bins x cells).
- Clustering works on the transpose, ``CellByFeature`` (cells x features).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, NotRequired, TypeAlias, TypedDict

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


class ExecutionBackend(StrEnum):
    """Supported execution backends."""

    CPU = "cpu"
    GPU = "gpu"
    GPU_COMPAT = "gpu-compat"


class AnchorStrategy(StrEnum):
    """How CopyKAT chooses the normal-cell reference."""

    SIGMA = "sigma"
    MARKERS = "markers"


class FinalCallStrategy(StrEnum):
    """Strategy used to label final copy-number calls."""

    CLUSTERS = "clusters"
    ARM_CORRELATION = "arm_correlation"


class KSMethod(StrEnum):
    """Breakpoint statistic used during segmentation."""

    MONTE_CARLO = "mc"
    EXACT = "exact"


class GeneIdType(StrEnum):
    """Gene identifier representation accepted by annotation."""

    SYMBOL = "S"
    ENSEMBL = "E"

    @classmethod
    def normalize(cls, value: GeneIdType | str) -> GeneIdType:
        """Normalize the ID-type argument: E-prefixed values select Ensembl; others select symbols."""
        if isinstance(value, cls):
            return value
        return cls.ENSEMBL if str(value).upper().startswith(cls.ENSEMBL.value) else cls.SYMBOL


class CellLineMode(StrEnum):
    """Whether input contains a pure cell line sample."""

    YES = "yes"
    NO = "no"


class ReportFormat(StrEnum):
    """Supported output formats for run reports."""

    TEXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class ReferenceMode(StrEnum):
    """Source used to establish the normal reference population."""

    SYNTHETIC = "synthetic"
    KNOWN_NORMAL = "known_normal"
    AUTOMATIC = "automatic"


class AnchorPath(StrEnum):
    """Evidence used to choose the normal-cell reference."""

    SIGMA = "sigma"
    IMMUNE = "immune"
    ENDOTHELIAL = "endothelial"
    ENRICHMENT = "enrichment"


class PredictionLabel(StrEnum):
    """Copy-number classification labels emitted by CopyKAT."""

    DIPLOID = "diploid"
    ANEUPLOID = "aneuploid"
    DIPLOID_LOW_CONFIDENCE = "c1:diploid:low.conf"
    ANEUPLOID_LOW_CONFIDENCE = "c2:aneuploid:low.conf"
    NOT_DEFINED = "not.defined"
    UNKNOWN = "unknown"


class BaselineWarning(StrEnum):
    """Baseline classification status reported by the pipeline."""

    NONE = ""
    UNCLASSIFIED = "unclassified.prediction"
    CELL_LINE = "run with cell line mode"
    KNOWN_NORMAL = "run with known normal"


class DataQualityStatus(StrEnum):
    """Data-quality status used to choose baseline fallback behavior."""

    OK = "data quality is ok"
    LOW = "low data quality"


class RawInput(TypedDict):
    """Count matrix with its gene and cell names (genes x cells)."""

    matrix: SparseMatrix | npt.NDArray[Any]
    genes: Sequence[str] | npt.NDArray[Any]
    barcodes: Sequence[str] | npt.NDArray[Any]


RawMatrix: TypeAlias = pd.DataFrame | npt.NDArray[Any] | SparseMatrix | RawInput | str
"""Inputs accepted by ``copykat()``: genes x cells, or a path to a .mtx/.csv/.tsv file."""


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
