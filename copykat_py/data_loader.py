"""Load reference data (gene annotations, DNA bins, cycle genes) and example data."""

import os

import pandas as pd
from scipy import sparse
from scipy.io import mmread

from copykat_py._types import Genome

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(_DATA_DIR, filename))


def load_full_anno(genome: Genome = Genome.HG20) -> pd.DataFrame:
    """Load the gene annotation table.

    Columns: abspos, chromosome_name, start_position, end_position,
    ensembl_gene_id, hgnc_symbol (or mgi_symbol for mm10), band.
    """
    genome = Genome(genome)
    if genome is Genome.HG20:
        return _load_csv("full_anno_hg20.csv")
    elif genome is Genome.MM10:
        return _load_csv("full_anno_mm10.csv")
    else:
        raise ValueError(f"Unsupported genome: {genome}")


def load_dna_bins(genome: Genome = Genome.HG20) -> pd.DataFrame:
    """Load 220KB variable genomic bins (chrom, chrompos, abspos)."""
    genome = Genome(genome)
    if genome is Genome.HG20:
        return _load_csv("DNA_hg20.csv")
    else:
        raise ValueError(f"DNA bins only available for hg20, got {genome}")


def load_cyclegenes() -> list[str]:
    """Load cell-cycle gene list."""
    df = _load_csv("cyclegenes.csv")
    return df["gene"].tolist()


def load_example_data() -> pd.DataFrame:
    """Load the built-in breast tumor example dataset (302 cells, 33694 genes).

    Returns
    -------
    pd.DataFrame
        UMI count matrix with genes as rows and cells as columns.
    """
    mtx = sparse.coo_matrix(mmread(os.path.join(_DATA_DIR, "exp_rawdata_sparse.mtx")))
    with open(os.path.join(_DATA_DIR, "exp_rawdata_genes.txt")) as f:
        genes = f.read().strip().split("\n")
    with open(os.path.join(_DATA_DIR, "exp_rawdata_barcodes.txt")) as f:
        barcodes = f.read().strip().split("\n")

    dense = mtx.toarray()
    return pd.DataFrame(dense, index=genes, columns=barcodes)
