"""Convert gene-by-cell CNA matrix to genomic-bin-by-cell matrix.

Mirrors convert.all.bins.hg20.R from the R package.
Maps gene-level copy number values into 220KB variable genomic bins.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from copykat_py._types import BinByCell, GeneByCell, Genome, ParallelInfo
from copykat_py.data_loader import load_dna_bins, load_full_anno

_LAST_PAR_INFO: ParallelInfo = {
    "step": "convert_to_bins",
    "parallel": False,
    "requested_cores": 1,
    "effective_cores": 1,
    "tasks": 0,
    "chunk_size": 0,
}


@dataclass(frozen=True, slots=True)
class BinConversionResult:
    """Genomic-bin annotations and the backing copy-number matrix.

    ``values`` is the matrix backing the cell columns of ``rna_table``.
    """

    dna_annotations: pd.DataFrame
    rna_table: pd.DataFrame
    values: BinByCell


def get_last_convert_bins_info() -> ParallelInfo:
    return _LAST_PAR_INFO.copy()


# Cells per task when computing bin medians.
_BIN_BLOCK_CELLS = 2048


def convert_to_bins(
    RNA_mat: pd.DataFrame,
    genome: Genome = Genome.HG20,
    n_cores: int = 1,
    values: GeneByCell | None = None,
    cell_names: Sequence[str] | None = None,
) -> BinConversionResult | None:
    """Convert gene-by-cell CNA results to 220KB genomic bins.

    Parameters
    ----------
    RNA_mat : pd.DataFrame
        CNA results with columns: abspos, chromosome_name, start_position,
        end_position, ensembl_gene_id, hgnc_symbol (or mgi_symbol), band, cell1, cell2, ...
    genome : Genome
        "hg20" (bins conversion only supported for hg20).
    n_cores : int
        Number of parallel workers.
    values : np.ndarray or None
        Optional (n_genes, n_cells) CNA values. When given, ``RNA_mat`` only
        needs the annotation columns and ``cell_names`` names the columns of
        ``values``; this avoids copying the cell columns out of a DataFrame.
    cell_names : list or None
        Cell names for ``values``.

    Returns
    -------
    A ``BinConversionResult`` containing DNA bin coordinates, the output table,
    and the matrix backing the table's cell columns.
    """
    if genome != Genome.HG20:
        # For mm10, return gene-level results (no bin conversion, same as R)
        return None

    full_anno = load_full_anno(Genome.HG20)
    DNA_mat = load_dna_bins(Genome.HG20)

    # Remove chromosome 24 (Y)
    DNA = DNA_mat[DNA_mat["chrom"] != 24].copy().reset_index(drop=True)

    end = DNA["chrompos"].to_numpy()
    start = np.concatenate([[0], end[:-1]])

    # Determine gene symbol column
    if "hgnc_symbol" in RNA_mat.columns:
        symbol_col = "hgnc_symbol"
    elif "mgi_symbol" in RNA_mat.columns:
        symbol_col = "mgi_symbol"
    else:
        raise ValueError("Cannot find gene symbol column in RNA_mat")

    # Cell data columns (after the 7 annotation columns)
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position", "ensembl_gene_id", symbol_col, "band"]
    if values is not None:
        if cell_names is None:
            raise ValueError("cell_names is required when values is given")
        cell_cols = list(cell_names)
        RNA_values = np.asarray(values, dtype=np.float32)  # shape: (n_genes, n_cells)
    else:
        cell_cols = [c for c in RNA_mat.columns if c not in anno_cols]
        RNA_values = RNA_mat[cell_cols].values.astype(np.float32, copy=False)  # shape: (n_genes, n_cells)
    gene_symbols = RNA_mat[symbol_col].values

    # Map genes in RNA_mat to indices for fast lookup.
    gene_to_idx: dict[str, list[int]] = {}
    for i, g in enumerate(gene_symbols):
        if g not in gene_to_idx:
            gene_to_idx[g] = []
        gene_to_idx[g].append(i)

    # Pre-compute which RNA rows belong to each DNA bin while preserving
    # the original R logic of using the full annotation table for bin membership.
    bin_gene_indices: list[list[int]] = [[] for _ in range(len(DNA))]
    chrom_values = DNA["chrom"].to_numpy()

    for chrom_id in np.unique(chrom_values):
        dna_idx = np.where(chrom_values == chrom_id)[0]
        if len(dna_idx) == 0:
            continue

        sub = full_anno[full_anno["chromosome_name"] == chrom_id].copy()
        if sub.empty:
            continue

        sub["center"] = 0.5 * (sub["start_position"] + sub["end_position"])
        centers = sub["center"].to_numpy()
        symbols = sub["hgnc_symbol"].to_numpy()
        order = np.argsort(centers, kind="mergesort")
        centers = centers[order]
        symbols = symbols[order]

        left_edges = start[dna_idx]
        right_edges = end[dna_idx]
        left_idx = np.searchsorted(centers, left_edges, side="left")
        right_idx = np.searchsorted(centers, right_edges, side="right")

        for local_i, bin_i in enumerate(dna_idx):
            if right_idx[local_i] <= left_idx[local_i]:
                continue
            row_indices: list[int] = []
            for symbol in symbols[left_idx[local_i] : right_idx[local_i]]:
                row_indices.extend(gene_to_idx.get(symbol, ()))
            if row_indices:
                bin_gene_indices[bin_i] = row_indices

    # Column-major so each cell's profile is contiguous, matching the layout
    # the downstream baseline adjustment has always operated on. float32 like
    # the gene-level values: a median of float32 values is one of them, so
    # nothing is lost, and the bins x cells matrix takes half the memory.
    n_cells = len(cell_cols)
    n_bins = len(DNA)
    RNA_adj = np.zeros((n_bins, n_cells), order="F", dtype=np.float32)

    # Bins with the same number of genes are computed together (one gather and
    # one median per group, as on the GPU backend). A median selects values,
    # so grouping cannot change them.
    by_len: dict[int, list[int]] = {}
    for b, rows in enumerate(bin_gene_indices):
        if rows:
            by_len.setdefault(len(rows), []).append(b)
    groups = [
        (np.asarray(bins), np.asarray([bin_gene_indices[b] for b in bins], dtype=np.intp)) for bins in by_len.values()
    ]
    # Empty bins copy their nearest valid bin (the lower one on ties).
    valid_indices = np.flatnonzero([bool(rows) for rows in bin_gene_indices])
    fill_to = fill_from = np.empty(0, dtype=np.intp)
    if 0 < len(valid_indices) < n_bins:
        fill_to = np.setdiff1d(np.arange(n_bins), valid_indices)
        pos = np.searchsorted(valid_indices, fill_to)
        left = valid_indices[np.maximum(pos - 1, 0)]
        right = valid_indices[np.minimum(pos, len(valid_indices) - 1)]
        fill_from = np.where((pos > 0) & ((fill_to - left) <= (right - fill_to)), left, right)

    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    n_jobs = max(1, min(int(n_cores), max_cores))
    block_cells = max(1, min(_BIN_BLOCK_CELLS, -(-n_cells // n_jobs)))
    blocks = [(lo, min(n_cells, lo + block_cells)) for lo in range(0, n_cells, block_cells)]
    _LAST_PAR_INFO.update(
        {
            "parallel": n_jobs > 1 and len(blocks) > 1,
            "requested_cores": int(n_cores),
            "effective_cores": int(min(n_jobs, len(blocks))),
            "tasks": len(blocks),
            "chunk_size": int(block_cells),
        }
    )

    def _process_cells(span: tuple[int, int]) -> None:
        lo, hi = span
        block_values = RNA_values[:, lo:hi]
        out = RNA_adj[:, lo:hi]
        for bins, rows in groups:
            gathered = block_values[rows]  # (bins, genes per bin, cells)
            out[bins] = gathered[:, 0] if rows.shape[1] == 1 else np.median(gathered, axis=1)
        if len(fill_to):
            out[fill_to] = out[fill_from]

    # Threads share RNA_values and write disjoint cell columns of RNA_adj.
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_process_cells)(span) for span in blocks)

    # Build output DataFrame on top of RNA_adj without copying it
    RNA_adj_df = pd.DataFrame(RNA_adj, columns=cell_cols, copy=False)
    for pos, col in enumerate(["chrom", "chrompos", "abspos"]):
        RNA_adj_df.insert(pos, col, DNA[col].to_numpy())

    return BinConversionResult(dna_annotations=DNA, rna_table=RNA_adj_df, values=RNA_adj)
