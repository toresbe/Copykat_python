"""Input loading and filtering for CopyKAT's genes-by-cells matrices."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from copykat_py._types import BoolArray, RawMatrix, SparseMatrix


@dataclass(frozen=True, slots=True)
class InputStats:
    """Counts and input representation recorded during input filtering."""

    input_type: str
    filtered_cells: int
    filtered_gene_rows: int


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """Filtered genes-by-cells matrix and its aligned input metadata.

    ``barcodes`` aligns to the filtered matrix columns; ``original_cell_names``
    retains all input names before cell filtering.
    """

    matrix: sparse.csr_matrix | npt.NDArray[Any]
    genes: pd.Index | npt.NDArray[Any]
    barcodes: list[str]
    original_cell_names: list[str]
    stats: InputStats


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
) -> PreparedInput:
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
    return PreparedInput(
        matrix=csc[keep_genes, :].tocsr(),
        genes=genes[keep_genes],
        barcodes=barcodes.tolist(),
        original_cell_names=original_cell_names,
        stats=InputStats(
            input_type="sparse_dict",
            filtered_cells=filtered_cells,
            filtered_gene_rows=filtered_gene_rows,
        ),
    )


def _prepare_input_matrix(rawmat: RawMatrix, min_gene_per_cell: int, low_dr: float) -> PreparedInput:
    """Filter cells and genes, returning a ``PreparedInput`` result.

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
    return PreparedInput(
        matrix=loaded.to_numpy(),
        genes=loaded.index,
        barcodes=list(loaded.columns),
        original_cell_names=original_cell_names,
        stats=InputStats(
            input_type="dense",
            filtered_cells=filtered_cells,
            filtered_gene_rows=filtered_gene_rows,
        ),
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
