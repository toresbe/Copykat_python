"""Genomic ordering and chromosome labels shared by inference and plotting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from copykat_py._types import Genome


@dataclass(frozen=True, slots=True)
class CnaTableData:
    """Cell matrix, names, chromosome labels, and genome from a saved CNA table."""

    matrix: npt.NDArray[np.float32]
    cell_names: list[str]
    chromosome_info: npt.NDArray[Any]
    genome: Genome


def annotation_order(annotation: pd.DataFrame, genome: Genome) -> npt.NDArray[np.intp]:
    """Order mm10 by chromosome and gene start; retain hg20 absolute-position ordering.

    Mouse annotation abspos values are chromosome offsets, not individual gene
    positions. Stable sorting keeps equal-position genes in their original order.
    """
    if genome != Genome.MM10:
        return np.argsort(annotation["abspos"].to_numpy(), kind="mergesort")
    chrom = pd.to_numeric(annotation["chromosome_name"], errors="raise").to_numpy(dtype=float)
    start = pd.to_numeric(annotation["start_position"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(chrom).all() or not np.isfinite(start).all():
        raise ValueError("mm10 annotation requires finite chromosome and gene-start coordinates")
    return np.lexsort((start, chrom)).astype(np.intp)


def chromosome_label(chromosome: Any, genome: Genome) -> str:
    """Decode numeric sex chromosomes according to the supplied genome."""
    try:
        numeric = float(str(chromosome))
        if not numeric.is_integer():
            return str(chromosome)
        code = int(numeric)
        sex = {20: "X", 21: "Y"} if genome == Genome.MM10 else {23: "X", 24: "Y"}
        return sex.get(code, str(code))
    except (ValueError, OverflowError):
        return str(chromosome)


def split_cna_table(frame: pd.DataFrame) -> CnaTableData:
    """Split saved human-bin or mouse-gene CNA results without treating annotation as cells."""
    if "mgi_symbol" in frame.columns and "chromosome_name" in frame.columns:
        leading, genome, chromosome = 7, Genome.MM10, "chromosome_name"
    elif list(frame.columns[:3]) == ["chrom", "chrompos", "abspos"]:
        leading, genome, chromosome = 3, Genome.HG20, "chrom"
    else:
        raise ValueError("Unrecognized CNA annotation columns; expected hg20 bins or mm10 gene results")
    return CnaTableData(
        matrix=frame.iloc[:, leading:].to_numpy(dtype=np.float32),
        cell_names=frame.columns[leading:].tolist(),
        chromosome_info=pd.to_numeric(frame[chromosome], errors="raise").to_numpy(),
        genome=genome,
    )
