"""Command line interfaces for CopyKAT-Py wrappers."""

import argparse
import logging
import os
import sys
import time
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.io import mmread

from copykat_py._logging import default_progress_output, log_progress_to, with_default_progress_output
from copykat_py._types import (
    AnchorStrategy,
    CellLineMode,
    CopyKATResult,
    DistanceMetric,
    ExecutionBackend,
    FinalCallStrategy,
    GeneIdType,
    Genome,
    KSMethod,
    RawInput,
    RawMatrix,
    SparseMatrix,
)
from copykat_py.genomic_coordinates import split_cna_table

logger = logging.getLogger(__name__)


class TeeStream:
    """Write stream output to both terminal and file."""

    def __init__(self, stream: TextIO, logfile_handle: TextIO) -> None:
        self.stream = stream
        self.logfile_handle = logfile_handle

    def write(self, data: str) -> None:
        self.stream.write(data)
        self.logfile_handle.write(data)
        if "\n" in data or "\r" in data:
            self.flush()

    def flush(self) -> None:
        self.stream.flush()
        self.logfile_handle.flush()


def _add_common_copykat_args(parser: argparse.ArgumentParser) -> None:
    """Attach CopyKAT runtime arguments shared by matrix and Python wrappers."""
    parser.add_argument(
        "--id-type",
        type=GeneIdType,
        default=GeneIdType.SYMBOL,
        choices=list(GeneIdType),
        help="[optional] Gene ID type: S=Symbol, E=Ensembl (default: S)",
    )
    parser.add_argument(
        "--cell-line",
        type=CellLineMode,
        default=CellLineMode.NO,
        choices=list(CellLineMode),
        help="[optional] Pure cell line mode (default: no)",
    )
    parser.add_argument(
        "--ngene-chr",
        type=int,
        default=5,
        help="[optional] Min genes per chromosome (default: 5)",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="[optional] Min genes per cell (default: 200)",
    )
    parser.add_argument(
        "--low-dr",
        type=float,
        default=0.05,
        help="[optional] Min detection rate for smoothing (default: 0.05)",
    )
    parser.add_argument(
        "--up-dr",
        type=float,
        default=0.1,
        help="[optional] Min detection rate for segmentation (default: 0.1)",
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=25,
        help="[optional] Window size for segmentation (default: 25)",
    )
    parser.add_argument(
        "--norm-cells",
        default="",
        help="[optional] File with known normal cell barcodes (one per line)",
    )
    parser.add_argument(
        "--ks-cut",
        type=float,
        default=0.1,
        help="[optional] KS test cutoff for breakpoints (default: 0.1)",
    )
    parser.add_argument(
        "--sample-name",
        default="",
        help="[optional] Sample name prefix for output files",
    )
    parser.add_argument(
        "--distance",
        type=DistanceMetric,
        default=DistanceMetric.EUCLIDEAN,
        choices=list(DistanceMetric),
        help="[optional] Distance metric for clustering (default: euclidean)",
    )
    parser.add_argument(
        "--output-seg",
        action="store_true",
        help="[optional] Output .seg file for IGV",
    )
    parser.add_argument(
        "--plot-genes",
        dest="plot_genes",
        action="store_true",
        default=True,
        help="[optional] Generate the heatmap plot (default: enabled)",
    )
    parser.add_argument(
        "--no-plot-genes",
        dest="plot_genes",
        action="store_false",
        help="[optional] Skip heatmap plotting for faster runs",
    )
    parser.add_argument(
        "--genome",
        type=Genome,
        default=Genome.HG20,
        choices=list(Genome),
        help="[optional] Genome assembly (default: hg20)",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="[optional] Number of CPU cores (default: 1)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="[optional] Adaptive PCA component cap for large clustering steps (default: automatic by cell count)",
    )
    parser.add_argument(
        "--backend",
        type=ExecutionBackend,
        default=ExecutionBackend.CPU,
        choices=list(ExecutionBackend),
        help="[experimental] Execution backend; GPU modes require CUDA-enabled torch and CuPy.",
    )
    parser.add_argument(
        "--anchor",
        type=AnchorStrategy,
        default=AnchorStrategy.SIGMA,
        choices=list(AnchorStrategy),
        help=(
            "[experimental] Normal reference: smallest GMM sigma (default) or marker-defined immune/endothelial cells."
        ),
    )
    parser.add_argument(
        "--final-call",
        type=FinalCallStrategy,
        default=FinalCallStrategy.CLUSTERS,
        choices=list(FinalCallStrategy),
        help="[experimental] Final calls from the Ward split (default) or arm-level correlation.",
    )
    parser.add_argument(
        "--ks-method",
        type=KSMethod,
        default=KSMethod.MONTE_CARLO,
        choices=list(KSMethod),
        help="Breakpoint statistic: Monte Carlo posterior KS (default) or exact posterior-Gamma KS.",
    )
    parser.add_argument(
        "--allele-counts",
        default=None,
        help="[optional] cellsnp-lite allele-count directory; requires --allele-phase.",
    )
    parser.add_argument(
        "--allele-phase",
        default=None,
        help="[optional] phased SNP CSV produced by copykat-py-allele phase.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="[optional] Output directory (default: current)",
    )


def _add_matrix_metadata_args(parser: argparse.ArgumentParser) -> None:
    """Attach metadata CSV arguments used by the matrix-style wrappers."""
    parser.add_argument(
        "--meta",
        default=None,
        metavar="CSV",
        help="[optional] Per-cell annotation CSV for the annotated heatmap. "
        "First column = cell name; remaining columns are drawn as "
        "coloured sidebars.",
    )
    parser.add_argument(
        "--row-split",
        default=None,
        metavar="COLUMN",
        help="[optional] Column in --meta used to split and label heatmap rows. "
        "Defaults to the second column of the CSV when not given.",
    )


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copykat-py",
        description="CopyKAT-Py: Inference of genomic copy number from single cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="[required] Input UMI count matrix (.csv, .tsv, .txt, or .mtx)",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="[optional] Gene names file (required for .mtx input)",
    )
    parser.add_argument(
        "--barcodes",
        default=None,
        help="[optional] Barcode names file (required for .mtx input)",
    )
    _add_common_copykat_args(parser)
    _add_matrix_metadata_args(parser)
    return parser


def _build_matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copykat_matrix",
        description=(
            "Matrix-focused CopyKAT-Py wrapper for raw count matrices plus an "
            "optional metadata CSV, designed for R and 10X-style outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="[required] Input UMI count matrix (.csv, .tsv, .txt, or .mtx)",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="[optional] Gene names file (required for .mtx input)",
    )
    parser.add_argument(
        "--barcodes",
        default=None,
        help="[optional] Barcode names file (required for .mtx input)",
    )
    _add_common_copykat_args(parser)
    _add_matrix_metadata_args(parser)
    return parser


def _normalize_selected_meta(values: str | Iterable[object] | None) -> list[str]:
    """Return selected AnnData obs columns, handling CSV-style input too."""
    if not values:
        return []

    columns = []
    seen = set()
    for raw_value in values:
        for value in str(raw_value).split(","):
            column = value.strip()
            if not column or column in seen:
                continue
            seen.add(column)
            columns.append(column)
    return columns


def _load_matrix_input(
    input_path: str | os.PathLike[str], genes_path: str | None = None, barcodes_path: str | None = None
) -> RawInput | str:
    """Load raw matrix input from file paths used by CLI wrappers."""
    input_path = str(input_path)
    if input_path.endswith(".mtx") or input_path.endswith(".mtx.gz"):
        mat = mmread(input_path)

        if genes_path is None or barcodes_path is None:
            input_dir = os.path.dirname(input_path)
            if genes_path is None:
                for gf in ["genes.tsv", "genes.txt", "features.tsv", "features.tsv.gz"]:
                    candidate = os.path.join(input_dir, gf)
                    if os.path.exists(candidate):
                        genes_path = candidate
                        break
            if barcodes_path is None:
                for bf in ["barcodes.tsv", "barcodes.txt"]:
                    candidate = os.path.join(input_dir, bf)
                    if os.path.exists(candidate):
                        barcodes_path = candidate
                        break

        if genes_path:
            genes = pd.read_csv(genes_path, sep="\t", header=None)
            gene_names = genes.iloc[:, -1].to_numpy() if genes.shape[1] > 1 else genes.iloc[:, 0].to_numpy()
        else:
            gene_names = np.array([f"gene_{i}" for i in range(mat.shape[0])])

        if barcodes_path:
            barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()
        else:
            barcodes = np.array([f"cell_{i}" for i in range(mat.shape[1])], dtype=object)

        raw_input: RawInput = {
            "matrix": mat,
            "genes": gene_names,
            "barcodes": barcodes,
        }
        return raw_input

    return input_path


def _load_normal_cells(norm_cells_path: str) -> str | list[str]:
    """Read known-normal barcode names from a text file when provided."""
    if not norm_cells_path or not os.path.exists(norm_cells_path):
        return ""

    with open(norm_cells_path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prepare_output_dir(output_dir: str | os.PathLike[str]) -> Path:
    """Create and activate the output directory used by a run."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    os.chdir(output_path)
    mpl_dir = output_path / ".mplconfig"
    cache_dir = output_path / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return output_path


def _sample_stub(sample_name: str, default_name: str) -> str:
    """Build a readable file stem for wrapper-created helper files."""
    cleaned = str(sample_name).strip()
    return cleaned if cleaned else default_name


def _anndata_to_rawmat(adata: Any, layer: str | None = None, use_raw: bool = False) -> tuple[Any, RawInput, str]:
    """Convert an AnnData object to CopyKAT's rawmat structure."""
    if layer and use_raw:
        raise ValueError("--layer and --use-raw cannot be used together")

    if use_raw:
        if adata.raw is None:
            raise ValueError("The supplied AnnData object does not contain adata.raw")
        matrix = adata.raw.X
        gene_names = adata.raw.var_names.astype(str).to_numpy()
        matrix_label = "adata.raw.X"
    elif layer:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}"
            )
        matrix = adata.layers[layer]
        gene_names = adata.var_names.astype(str).to_numpy()
        matrix_label = f"adata.layers['{layer}']"
    else:
        matrix = adata.X
        gene_names = adata.var_names.astype(str).to_numpy()
        matrix_label = "adata.X"

    if sp.issparse(matrix):
        matrix_t = cast(SparseMatrix, matrix).T.tocsc(copy=True).astype(np.float32)
    else:
        matrix_t = sp.csc_matrix(np.asarray(matrix, dtype=np.float32).T)

    rawmat: RawInput = {
        "matrix": matrix_t,
        "genes": gene_names,
        "barcodes": adata.obs_names.astype(str).to_numpy(),
    }
    return adata, rawmat, matrix_label


def _write_selected_obs_meta_csv(
    adata: Any, selecting_meta: str | Iterable[object] | None, output_dir: str | os.PathLike[str], sample_name: str
) -> tuple[str | None, list[str]]:
    """Persist selected AnnData obs columns as the metadata CSV expected by copykat()."""
    columns = _normalize_selected_meta(selecting_meta)
    if not columns:
        return None, []

    obs_columns = [str(col) for col in adata.obs.columns.tolist()]
    missing = [column for column in columns if column not in obs_columns]
    if missing:
        raise ValueError(f"Requested obs columns not found: {missing}. Available obs columns: {obs_columns}")

    meta_df = adata.obs.loc[:, columns].copy()
    meta_df.index = meta_df.index.astype(str)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    meta_path = output_path / f"{_sample_stub(sample_name, 'copykat_anndata')}_selected_obs_meta.csv"
    meta_df.to_csv(meta_path, index=True, index_label="cell_name")
    return str(meta_path), columns


def _run_copykat_analysis(
    args: argparse.Namespace,
    rawmat: RawMatrix,
    *,
    meta_csv: str | None = None,
    row_split_col: str | None = None,
    input_label: str | None = None,
    post_plot_meta: str | None = None,
) -> CopyKATResult:
    """Run copykat() with consistent logging and output-directory setup."""
    norm_cells_path = os.path.abspath(args.norm_cells) if args.norm_cells else ""
    allele_counts_path = os.path.abspath(args.allele_counts) if args.allele_counts else None
    allele_phase_path = os.path.abspath(args.allele_phase) if args.allele_phase else None
    if bool(allele_counts_path) != bool(allele_phase_path):
        raise ValueError("--allele-counts and --allele-phase must be provided together")
    meta_csv = os.path.abspath(meta_csv) if meta_csv is not None else None
    post_plot_meta = os.path.abspath(post_plot_meta) if post_plot_meta is not None else None

    norm_cell_names = _load_normal_cells(norm_cells_path)
    output_dir = _prepare_output_dir(args.output_dir)

    from copykat_py.copykat import copykat

    log_path = output_dir / "copykat_run.log"
    log_handle = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in the `finally` below
    # Progress goes to the console (unless the caller configured logging) and
    # to the run log. Anything else written to stderr, such as third-party
    # warnings, is copied to the run log too.
    progress_output = ExitStack()
    progress_output.enter_context(default_progress_output())
    progress_output.enter_context(log_progress_to(log_handle))
    old_stderr = sys.stderr
    sys.stderr = TeeStream(sys.stderr, log_handle)

    logger.info("=" * 80)
    logger.info(f"CopyKAT-Py run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {output_dir}")
    logger.info(f"Input: {input_label or getattr(args, 'input', '<in-memory>')}")
    logger.info(f"Requested cores: {args.n_cores}")
    if args.pca_components is not None:
        logger.info(f"Requested adaptive PCA components: {args.pca_components}")
    if meta_csv is not None:
        logger.info(f"Metadata source: {meta_csv}")
    if row_split_col is not None:
        logger.info(f"Row split column: {row_split_col}")

    try:
        result = copykat(
            rawmat=rawmat,
            id_type=args.id_type,
            cell_line=args.cell_line,
            ngene_chr=args.ngene_chr,
            min_gene_per_cell=args.min_genes,
            LOW_DR=args.low_dr,
            UP_DR=args.up_dr,
            win_size=args.win_size,
            norm_cell_names=norm_cell_names,
            KS_cut=args.ks_cut,
            sam_name=args.sample_name,
            distance=args.distance,
            output_seg=args.output_seg,
            plot_genes=args.plot_genes,
            genome=args.genome,
            n_cores=args.n_cores,
            pca_components=args.pca_components,
            backend_name=args.backend,
            ks_method=args.ks_method,
            anchor=args.anchor,
            final_call=args.final_call,
            meta_csv=meta_csv,
            row_split_col=row_split_col,
        )

        if allele_counts_path and "prediction" in result:
            import json

            from copykat_py.allele.orient import orient_prediction

            oriented, allele_report = orient_prediction(result["prediction"], allele_counts_path, allele_phase_path)
            prefix = f"{args.sample_name}_copykat_"
            with open(f"{prefix}allele_orientation.json", "w", encoding="utf-8") as report_file:
                json.dump(allele_report, report_file, indent=2)
            if allele_report["flipped"]:
                result["prediction"].to_csv(f"{prefix}prediction.before_allele.txt", sep="\t", index=False)
                oriented.to_csv(f"{prefix}prediction.txt", sep="\t", index=False)
                result["prediction"] = oriented
            result["allele_orientation"] = allele_report

        logger.info("CopyKAT-Py analysis complete.")
        if "prediction" in result:
            pred = result["prediction"]["copykat.pred"].value_counts()
            for key, value in pred.items():
                logger.info(f"  {key}: {value} cells")

        if post_plot_meta and args.plot_genes and "CNAmat" in result:
            logger.info("\nGenerating annotated heatmap...")
            from copykat_py.plotting import plot_heatmap_annotated

            cna_df = result["CNAmat"]
            ann_mat, ann_cell_names, ann_chrom_info, ann_genome = split_cna_table(cna_df)
            ann_output = f"{args.sample_name}_copykat_annotated_heatmap.png"
            plot_heatmap_annotated(
                mat=ann_mat,
                cell_names=ann_cell_names,
                chrom_info=ann_chrom_info,
                meta_csv=post_plot_meta,
                row_split_col=args.row_split,
                sample_name=args.sample_name,
                distance=args.distance,
                n_cores=args.n_cores,
                output_path=ann_output,
                genome=ann_genome,
            )
        return result
    finally:
        logger.info(f"CopyKAT-Py run finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Detailed log saved to: {log_path}")
        sys.stderr.flush()
        sys.stderr = old_stderr
        progress_output.close()
        log_handle.close()


def main() -> CopyKATResult:
    """Entry point for the legacy matrix-focused ``copykat-py`` CLI."""
    parser = _build_main_parser()
    args = parser.parse_args()
    rawmat = _load_matrix_input(args.input, args.genes, args.barcodes)
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=None,
        row_split_col=None,
        input_label=args.input,
        post_plot_meta=args.meta,
    )


def matrix_main() -> CopyKATResult:
    """Entry point for ``copykat_matrix``."""
    parser = _build_matrix_parser()
    args = parser.parse_args()
    rawmat = _load_matrix_input(args.input, args.genes, args.barcodes)
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=args.meta,
        row_split_col=args.row_split,
        input_label=args.input,
        post_plot_meta=None,
    )


@with_default_progress_output
def copykat_anndata(
    adata: Any,
    *,
    selecting_meta: str | Iterable[str] | None = None,
    row_split: str | None = None,
    sample_name: str = "",
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    genome: Genome = Genome.HG20,
    n_cores: int = 1,
    output_dir: str | os.PathLike[str] = ".",
    layer: str | None = None,
    use_raw: bool = False,
    id_type: GeneIdType = GeneIdType.SYMBOL,
    cell_line: CellLineMode = CellLineMode.NO,
    ngene_chr: int = 5,
    min_genes: int = 200,
    low_dr: float = 0.05,
    up_dr: float = 0.1,
    win_size: int = 25,
    norm_cells: str | os.PathLike[str] = "",
    ks_cut: float = 0.1,
    output_seg: bool = False,
    plot_genes: bool = True,
    pca_components: int | None = None,
) -> CopyKATResult:
    """Python-friendly AnnData wrapper that accepts an in-memory AnnData object."""
    genome = Genome(genome)
    distance = DistanceMetric(distance)
    id_type = GeneIdType.from_legacy(id_type)
    cell_line = CellLineMode(cell_line)
    _, rawmat, matrix_label = _anndata_to_rawmat(
        adata,
        layer=layer,
        use_raw=use_raw,
    )
    meta_csv, selected_meta = _write_selected_obs_meta_csv(
        adata,
        selecting_meta,
        output_dir,
        sample_name,
    )

    if row_split and meta_csv is None:
        raise ValueError("row_split requires selecting_meta")

    row_split_col = row_split
    if meta_csv is not None and row_split_col is None:
        row_split_col = selected_meta[0]

    if selected_meta:
        logger.info(f"Selected obs columns: {selected_meta}")

    args = argparse.Namespace(
        input="<AnnData object>",
        layer=layer,
        use_raw=use_raw,
        selecting_meta=selecting_meta,
        row_split=row_split,
        id_type=id_type,
        cell_line=cell_line,
        ngene_chr=ngene_chr,
        min_genes=min_genes,
        low_dr=low_dr,
        up_dr=up_dr,
        win_size=win_size,
        norm_cells=str(norm_cells) if norm_cells else "",
        ks_cut=ks_cut,
        sample_name=sample_name,
        distance=distance,
        output_seg=output_seg,
        plot_genes=plot_genes,
        genome=genome,
        n_cores=n_cores,
        pca_components=pca_components,
        output_dir=str(output_dir),
    )
    return _run_copykat_analysis(
        args,
        rawmat,
        meta_csv=meta_csv,
        row_split_col=row_split_col,
        input_label=f"<AnnData object> ({matrix_label})",
        post_plot_meta=None,
    )


@with_default_progress_output
def plot_main() -> None:
    """Entry point for ``copykat-py-plot``: annotated heatmap from CNA results."""
    parser = argparse.ArgumentParser(
        prog="copykat-py-plot",
        description=(
            "Draw an annotated CNA heatmap from a copykat-py CNA results file "
            "and a per-cell metadata CSV. Rows are split and colour-labelled by "
            "a chosen metadata column; cells within each group are ordered by "
            "hierarchical or K-means clustering so intra-group CNA structure is "
            "preserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use second column (leiden_cluster) as row-split (default):
  copykat-py-plot \\
      --cna  sample_copykat_CNA_results.txt \\
      --meta meta.csv

  # Explicitly split rows by inferred_CellType and also annotate leiden_cluster:
  copykat-py-plot \\
      --cna  sample_copykat_CNA_results.txt \\
      --meta xenium_ft_full_meta_celltype_leiden.csv \\
      --row-split inferred_CellType \\
      --sample-name xenium_all_cells \\
      --n-cores 40 \\
      --output xenium_annotated_heatmap.png

Meta CSV format
---------------
  First column : cell name (must match column names in the CNA results file)
  Remaining    : any metadata (cell type, cluster, condition, ...)

  With header:    cell_name,leiden_cluster,inferred_CellType
  Without header: aaaajgij-1,5,lumhr   (first row treated as data)
""",
    )

    parser.add_argument(
        "--cna",
        "-c",
        required=True,
        help="[required] CNA results file produced by copykat-py "
        "(*_copykat_CNA_results.txt, tab-separated). "
        "Accepts hg20 bin tables or mm10 gene tables; annotation columns are detected automatically.",
    )
    parser.add_argument(
        "--meta",
        "-m",
        required=True,
        help="[required] Annotation CSV. First column = cell name; remaining columns are "
        "drawn as coloured sidebars. Header row is auto-detected.",
    )
    parser.add_argument(
        "--row-split",
        default=None,
        metavar="COLUMN",
        help="[optional] Metadata column used to split and label rows. "
        "Defaults to the second column of the CSV when not supplied.",
    )
    parser.add_argument(
        "--sample-name",
        default="",
        help="[optional] Label shown in the figure title and used as the output filename "
        "prefix when --output is not given.",
    )
    parser.add_argument(
        "--distance",
        type=DistanceMetric,
        default=DistanceMetric.EUCLIDEAN,
        choices=list(DistanceMetric),
        help="[optional] Distance metric for within-group cell clustering (default: euclidean).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="[optional] Parallel threads for clustering (default: 1).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="[optional] Output PNG path. Defaults to {sample_name}_copykat_annotated_heatmap.png.",
    )
    parser.add_argument(
        "--continuous-meta",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help="Force selected numeric metadata columns to use continuous colors (e.g. n_umi).",
    )
    parser.add_argument(
        "--no-row-split",
        action="store_true",
        help="Cluster all cells together without categorical row groups.",
    )

    args = parser.parse_args()

    logger.info(f"Loading CNA results: {args.cna}")
    cna_df = pd.read_csv(args.cna, sep="\t", index_col=False)
    mat, cell_names, chrom_info, genome = split_cna_table(cna_df)
    logger.info(f"  {mat.shape[1]} cells x {mat.shape[0]} bins")

    from copykat_py.plotting import plot_heatmap_annotated

    plot_heatmap_annotated(
        mat=mat,
        cell_names=cell_names,
        chrom_info=chrom_info,
        meta_csv=args.meta,
        row_split_col="" if args.no_row_split else args.row_split,
        sample_name=args.sample_name,
        distance=args.distance,
        n_cores=args.n_cores,
        output_path=args.output,
        continuous_meta=args.continuous_meta,
        genome=genome,
    )


if __name__ == "__main__":
    main()
