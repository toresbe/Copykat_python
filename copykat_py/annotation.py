"""Gene annotation: map genes to genomic coordinates, mirroring annotateGenes.hg20.R / annotateGenes.mm10.R."""

import logging
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from copykat_py._types import GeneIdType, Genome
from copykat_py.data_loader import load_full_anno
from copykat_py.genomic_coordinates import annotation_order

logger = logging.getLogger(__name__)


def annotate_gene_rows(
    genes: Sequence[Hashable] | pd.Index | npt.NDArray[Any],
    id_type: GeneIdType = GeneIdType.SYMBOL,
    genome: Genome = Genome.HG20,
) -> tuple[pd.DataFrame, npt.NDArray[np.intp]]:
    """Annotate gene identifiers with genomic coordinates, without touching expression values.

    Parameters
    ----------
    genes : array-like
        Gene identifiers, one per expression-matrix row.
    id_type : str
        "S" for gene Symbol, "E" for Ensembl ID.
    genome : Genome
        Genome.HG20 or Genome.MM10.

    Returns
    -------
    anno : pd.DataFrame
        Annotation sorted in genomic order. Columns: <id column>, then the remaining
        annotation columns (abspos, chromosome_name, start_position, ...).
    rows : np.ndarray
        For each annotation row, the index of the matching expression-matrix row.
    """
    genome = Genome(genome)
    id_type = GeneIdType.normalize(id_type)
    logger.info("  start annotation ...")
    full_anno = load_full_anno(genome)

    if genome is Genome.MM10:
        symbol_col = "mgi_symbol"
    else:
        symbol_col = "hgnc_symbol"

    if id_type is GeneIdType.ENSEMBL:
        id_col = "ensembl_gene_id"
    else:
        id_col = symbol_col

    # Intersect genes, keeping matrix row order
    genes = pd.Index(genes)
    rows = np.flatnonzero(genes.isin(full_anno[id_col]))
    if len(rows) == 0:
        raise ValueError("No shared genes found between input matrix and annotation.")

    kept = genes[rows]
    anno = full_anno[full_anno[id_col].isin(kept)].copy()
    anno = anno.drop_duplicates(subset=id_col)
    # Align annotation to matrix order
    anno = anno.set_index(id_col).reindex(kept)
    anno.index.name = id_col
    anno = anno.reset_index()

    order = annotation_order(anno, genome)
    anno = anno.iloc[order].reset_index(drop=True)
    rows = rows[order]

    # Re-derive 'chrom' as integer for downstream compatibility
    # hg20: X=23, Y=24; mm10: X=20, Y=21
    anno["chromosome_name"] = anno["chromosome_name"].astype(str)

    logger.info(f"  {len(anno)} genes annotated")
    return anno, rows


def annotate_genes(
    mat: pd.DataFrame, id_type: GeneIdType = GeneIdType.SYMBOL, genome: Genome = Genome.HG20
) -> pd.DataFrame:
    """Annotate gene expression matrix with genomic coordinates.

    Parameters
    ----------
    mat : pd.DataFrame
        Gene expression matrix, genes in rows, cells in columns.
    id_type : str
        "S" for gene Symbol, "E" for Ensembl ID.
    genome : Genome
        Genome.HG20 or Genome.MM10.

    Returns
    -------
    pd.DataFrame
        Combined annotation + expression, sorted in genomic order.
        Columns: abspos, chromosome_name, start_position, end_position,
                 ensembl_gene_id, hgnc_symbol (or mgi_symbol), band, <cell1>, <cell2>, ...
    """
    anno, rows = annotate_gene_rows(mat.index, id_type=id_type, genome=genome)
    expr = mat.iloc[rows].reset_index(drop=True)
    return pd.concat([anno, expr], axis=1)
