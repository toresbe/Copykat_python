"""Documented keyword arguments and immutable configuration for a CopyKAT run."""

from dataclasses import dataclass, field, fields
from typing import Any, TypedDict

from copykat_py._pipeline.baseline import BaselineOptions
from copykat_py._pipeline.output import OutputOptions
from copykat_py._pipeline.prediction import PredictionOptions
from copykat_py._pipeline.preprocessing import FilteringOptions
from copykat_py._pipeline.segmentation import _SegmentationOptions
from copykat_py._types import (
    AnchorStrategy,
    CellLineMode,
    DistanceMetric,
    ExecutionBackend,
    FinalCallStrategy,
    GeneIdType,
    Genome,
    KSMethod,
)


class CopyKATArguments(TypedDict, total=False):
    """Optional keywords accepted by ``copykat(rawmat, **options)``.

    Names and types are available to type checkers through ``Unpack``.
    Defaults, normalization, and field descriptions are documented on
    :class:`RunContext`, which constructs the effective configuration.
    """

    id_type: GeneIdType
    cell_line: CellLineMode
    ngene_chr: int
    min_gene_per_cell: int
    LOW_DR: float
    UP_DR: float
    win_size: int
    norm_cell_names: str | list[str]
    KS_cut: float
    sam_name: str
    distance: DistanceMetric
    output_seg: bool
    plot_genes: bool
    genome: Genome
    n_cores: int
    pca_components: int | None
    meta_csv: str | None
    row_split_col: str | None
    backend_name: ExecutionBackend
    ks_method: KSMethod
    anchor: AnchorStrategy
    final_call: FinalCallStrategy


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext:
    """Requested settings for one run, normalized before any analysis starts.

    Defaults match the public CopyKAT interface. Enum-valued fields accept
    their string values at runtime and are normalized during construction.
    Construction does not select a backend, allocate matrices, or write files.
    The input matrix stays outside this context, and stages receive only their
    own options. ``norm_cell_names`` retains the supplied list by reference;
    freezing this context does not freeze the contents of that list.

    Attributes
    ----------
    id_type : GeneIdType, default GeneIdType.SYMBOL
        Gene identifiers: ``SYMBOL`` or ``ENSEMBL``. Legacy aliases accepted
        by ``GeneIdType.normalize`` are normalized during construction.
    cell_line : CellLineMode, default CellLineMode.NO
        ``YES`` selects a synthetic reference for pure cell-line data;
        ``NO`` uses supplied normals or automatically detects a reference.
    ngene_chr : int, default 5
        Minimum detected genes per represented chromosome for cell filtering,
        checked before smoothing and again before segmentation.
    min_gene_per_cell : int, default 200
        Minimum number of detected genes required to retain an input cell.
    LOW_DR : float, default 0.05
        Lower gene-detection threshold, expressed as a fraction of cells.
        Input gene filtering uses detection rates strictly above this value.
    UP_DR : float, default 0.1
        Requested segmentation gene-detection threshold. Genes at or above
        the effective threshold are retained. For fewer than 7,000 genes
        after input filtering, the effective value becomes ``LOW_DR``;
        this context retains the requested value for reporting.
    win_size : int, default 25
        Genes per window for MCMC segmentation.
    norm_cell_names : str or list of str, default ""
        Known normal-cell barcodes. A list with more than one entry selects
        the supplied-normal path; an empty string selects automatic reference
        detection. Synthetic cell-line mode takes precedence.
    KS_cut : float, default 0.1
        Requested KS breakpoint threshold from 0 to 1. Sparse breakpoint
        results are retried at 50% and then 25% of this value.
    sam_name : str, default ""
        User's sample name. ``sample_name`` derives the output prefix
        ``{sam_name}_copykat_``, also used by pipeline plot titles.
    distance : DistanceMetric, default DistanceMetric.EUCLIDEAN
        Cell-ordering distance metric: ``EUCLIDEAN``, ``PEARSON``, or
        ``SPEARMAN``. Normalized to ``DistanceMetric`` during construction.
    output_seg : bool, default False
        Whether to export an IGV SEG file for ``HG20``. The existing
        ``MM10`` path does not export SEG files.
    plot_genes : bool, default True
        Whether to plot final CNA values: genomic bins for ``HG20`` and
        genes for ``MM10``. Also enables annotated plotting with ``meta_csv``.
    genome : Genome, default Genome.HG20
        Reference genome, ``HG20`` or ``MM10``. Controls annotation, adaptive
        PCA selection, genomic-bin conversion, and chromosome labels.
    n_cores : int, default 1
        Requested CPU workers for parallel analysis, output, and plotting.
    pca_components : int or None, default None
        Requested component cap for large clustering steps. ``None`` uses
        adaptive selection from the original input cell count: ``HG20``
        selects 256 below 50,000 cells, otherwise 128; ``MM10`` selects 512
        below 20,000 cells, 256 below 40,000 cells, otherwise 128. Stage
        options receive the resolved cap after input preparation.
    meta_csv : str or None, default None
        Optional per-cell annotation CSV. The first column contains cell
        names; remaining columns supply sidebars for an additional annotated
        heatmap when ``plot_genes`` is enabled. ``None`` disables that plot.
    row_split_col : str or None, default None
        Metadata column used to split and label annotated-heatmap rows.
        ``None`` selects the first metadata column after the cell-name column;
        an empty string disables row splitting.
    backend_name : ExecutionBackend, default ExecutionBackend.CPU
        Requested execution backend. Normalized during construction; the
        orchestrator activates it after the context has been validated.
    ks_method : KSMethod, default KSMethod.MONTE_CARLO
        ``MONTE_CARLO`` compares sampled posterior distributions; ``EXACT``
        computes the posterior-Gamma KS statistic directly.
    anchor : AnchorStrategy, default AnchorStrategy.SIGMA
        Strategy for automatic reference selection: ``SIGMA`` uses fitted
        neutral-profile spread; ``MARKERS`` also considers immune and
        endothelial marker evidence. Supplied normals take precedence.
    final_call : FinalCallStrategy, default FinalCallStrategy.CLUSTERS
        ``CLUSTERS`` assigns calls from clusters; ``ARM_CORRELATION`` uses
        chromosome-arm correlations when at least five normal anchor cells
        remain, falling back to clusters otherwise. Arm correlation is
        supported only for ``HG20``; construction rejects it for ``MM10``.
    random_seed : int, fixed at 1234
        Reproducibility seed retained from the existing implementation.
        Not accepted as a constructor or public ``copykat`` keyword.
    """

    id_type: GeneIdType = GeneIdType.SYMBOL
    cell_line: CellLineMode = CellLineMode.NO
    ngene_chr: int = 5
    min_gene_per_cell: int = 200
    LOW_DR: float = 0.05
    UP_DR: float = 0.1
    win_size: int = 25
    norm_cell_names: str | list[str] = ""
    KS_cut: float = 0.1
    sam_name: str = ""
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN
    output_seg: bool = False
    plot_genes: bool = True
    genome: Genome = Genome.HG20
    n_cores: int = 1
    pca_components: int | None = None
    meta_csv: str | None = None
    row_split_col: str | None = None
    backend_name: ExecutionBackend = ExecutionBackend.CPU
    ks_method: KSMethod = KSMethod.MONTE_CARLO
    anchor: AnchorStrategy = AnchorStrategy.SIGMA
    final_call: FinalCallStrategy = FinalCallStrategy.CLUSTERS
    random_seed: int = field(default=1234, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "distance", DistanceMetric(self.distance))
        object.__setattr__(self, "backend_name", ExecutionBackend(self.backend_name))
        object.__setattr__(self, "id_type", GeneIdType.normalize(self.id_type))
        object.__setattr__(self, "cell_line", CellLineMode(self.cell_line))
        object.__setattr__(self, "ks_method", KSMethod(self.ks_method))
        object.__setattr__(self, "anchor", AnchorStrategy(self.anchor))
        object.__setattr__(self, "final_call", FinalCallStrategy(self.final_call))
        object.__setattr__(self, "genome", Genome(self.genome))
        if self.final_call is FinalCallStrategy.ARM_CORRELATION and self.genome is Genome.MM10:
            raise ValueError(
                "arm_correlation currently uses hg38 centromere coordinates and is only supported for hg20"
            )

    @property
    def sample_name(self) -> str:
        """Complete output prefix, ``{sam_name}_copykat_``."""
        return f"{self.sam_name}_copykat_"

    @property
    def filtering_options(self) -> FilteringOptions:
        """Requested input and chromosome-coverage thresholds."""
        return FilteringOptions(
            min_gene_per_cell=self.min_gene_per_cell,
            ngene_chr=self.ngene_chr,
            lower_detection_rate=self.LOW_DR,
            upper_detection_rate=self.UP_DR,
        )

    @property
    def segmentation_options(self) -> _SegmentationOptions:
        """Requested segmentation settings, before retry cutoffs are applied."""
        return _SegmentationOptions(
            window_size=self.win_size, ks_cutoff=self.KS_cut, ks_method=self.ks_method, n_cores=self.n_cores
        )

    @property
    def output_options(self) -> OutputOptions:
        """File and plotting settings without reference names or matrix data."""
        return OutputOptions(
            sample_name=self.sample_name,
            genome=self.genome,
            distance=self.distance,
            n_cores=self.n_cores,
            plot_genes=self.plot_genes,
            output_seg=self.output_seg,
            meta_csv=self.meta_csv,
            row_split_col=self.row_split_col,
        )

    def baseline_options(self, pca_components: int) -> BaselineOptions:
        """Baseline settings with the component cap resolved after input preparation."""
        return BaselineOptions(
            cell_line=self.cell_line,
            anchor=self.anchor,
            genome=self.genome,
            n_cores=self.n_cores,
            pca_components=pca_components,
        )

    def prediction_options(self, pca_components: int) -> PredictionOptions:
        """Final-call settings with the resolved clustering component cap."""
        return PredictionOptions(
            genome=self.genome,
            cell_line=self.cell_line,
            final_call=self.final_call,
            n_cores=self.n_cores,
            pca_components=pca_components,
        )

    def runtime_parameters(self) -> dict[str, Any]:
        """Requested settings in the existing report schema, without copying lists."""
        parameters = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name not in {"sam_name", "norm_cell_names"}
        }
        parameters["backend"] = parameters.pop("backend_name")
        parameters["pca_components_requested"] = parameters.pop("pca_components")
        return parameters
