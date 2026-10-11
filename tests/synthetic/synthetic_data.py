"""Seeded scRNA-seq count simulator with planted copy-number truth.

Genes are real hg20 symbols, so the pipeline annotates and bins them as it would
real data. Counts are gamma-Poisson (negative binomial) draws whose means scale
with each cell's library size, each gene's base rate, the tumour copy number
over the gene (CN / 2) and, for normal cells, a cell-type program that changes
expression without changing copy number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from copykat_py.data_loader import load_full_anno

AUTOSOMES = [str(c) for c in range(1, 23)]


@dataclass(frozen=True)
class Event:
    """A clonal copy-number change over ``chrom:[start, end)`` (base pairs)."""

    chrom: str
    copies: float
    start: int = 0
    end: int = 10**10

    def label(self) -> str:
        kind = "gain" if self.copies > 2 else "loss"
        whole = self.start == 0 and self.end >= 10**9
        where = f"chr{self.chrom}" if whole else f"chr{self.chrom}:{self.start // 10**6}-{self.end // 10**6}Mb"
        return f"{kind} {where} (CN={self.copies:g})"


# Two arm-scale gains and two losses, the kind of karyotype CopyKAT is built for.
DEFAULT_EVENTS: tuple[Event, ...] = (
    Event("7", 3.0),
    Event("20", 4.0),
    Event("10", 1.0),
    Event("3", 1.0, end=90_000_000),  # 3p loss
)


@dataclass(frozen=True)
class SimulationConfig:
    n_tumor: int = 180
    n_normal: int = 120
    n_genes: int = 8000
    mean_library_size: float = 6000.0
    library_size_sdlog: float = 0.3
    gene_rate_sdlog: float = 1.0
    dispersion: float = 0.2  # NB overdispersion phi; variance = mu + phi * mu^2
    normal_cell_types: int = 2
    program_fraction: float = 0.03  # share of genes up-regulated in each normal cell type
    program_fold: float = 3.0
    events: tuple[Event, ...] = DEFAULT_EVENTS
    seed: int = 20261011


@dataclass
class SyntheticDataset:
    counts: pd.DataFrame  # genes x cells
    is_tumor: pd.Series  # bool, indexed by cell name
    genes: pd.DataFrame  # symbol, chrom, start; indexed by symbol, genomic order
    tumor_copies: pd.Series  # per-gene tumour copy number, indexed by symbol
    config: SimulationConfig
    mean: pd.DataFrame  # genes x cells expected counts the draws were made from

    @property
    def tumor_cells(self) -> list[str]:
        return self.is_tumor.index[self.is_tumor].tolist()

    @property
    def normal_cells(self) -> list[str]:
        return self.is_tumor.index[~self.is_tumor].tolist()

    def subset_cells(self, cells: list[str]) -> SyntheticDataset:
        return replace(self, counts=self.counts[cells], is_tumor=self.is_tumor[cells], mean=self.mean[cells])


def _gene_panel(n_genes: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pick ``n_genes`` uniquely-named autosomal genes spread evenly over the genome."""
    anno = load_full_anno()
    anno = anno[anno["chromosome_name"].astype(str).isin(AUTOSOMES)]
    anno = anno.dropna(subset=["hgnc_symbol"])
    anno = anno[anno["hgnc_symbol"].astype(str).str.len() > 0]
    anno = anno.drop_duplicates(subset="hgnc_symbol", keep=False)
    # Genes sharing a position are ordered by input row (see test_invariance), so
    # leave them out of the panel; tests that need a tie add one on purpose.
    anno = anno.drop_duplicates(subset="abspos", keep=False)
    anno = anno.sort_values("abspos")
    picked = np.sort(rng.choice(len(anno), size=n_genes, replace=False))
    anno = anno.iloc[picked]
    return pd.DataFrame(
        {
            "chrom": anno["chromosome_name"].astype(str).to_numpy(),
            "start": anno["start_position"].astype(np.int64).to_numpy(),
        },
        index=pd.Index(anno["hgnc_symbol"].astype(str).to_numpy(), name="symbol"),
    )


def copies_for(genes: pd.DataFrame, events: tuple[Event, ...]) -> np.ndarray:
    copies = np.full(len(genes), 2.0)
    for event in events:
        hit = (genes["chrom"].to_numpy() == event.chrom) & (genes["start"].to_numpy() >= event.start)
        hit &= genes["start"].to_numpy() < event.end
        copies[hit] = event.copies
    return copies


def simulate(config: SimulationConfig | None = None) -> SyntheticDataset:
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    genes = _gene_panel(config.n_genes, rng)
    n_cells = config.n_tumor + config.n_normal

    base_rate = rng.lognormal(0.0, config.gene_rate_sdlog, size=len(genes))
    base_rate /= base_rate.sum()
    library = config.mean_library_size * rng.lognormal(
        -0.5 * config.library_size_sdlog**2, config.library_size_sdlog, size=n_cells
    )

    tumor_copies = copies_for(genes, config.events)
    multiplier = np.ones((len(genes), n_cells))
    multiplier[:, : config.n_tumor] = (tumor_copies / 2.0)[:, None]

    # Normal cell types get their own expression programs: real, non-CNV
    # expression differences the reference must not mistake for copy number.
    normal_type = rng.integers(0, config.normal_cell_types, size=config.n_normal)
    n_program = round(config.program_fraction * len(genes))
    for cell_type in range(config.normal_cell_types):
        program = rng.choice(len(genes), size=n_program, replace=False)
        cols = config.n_tumor + np.flatnonzero(normal_type == cell_type)
        multiplier[np.ix_(program, cols)] *= config.program_fold

    mu = base_rate[:, None] * library[None, :] * multiplier
    shape = 1.0 / config.dispersion
    counts = rng.poisson(rng.gamma(shape, mu / shape)).astype(np.int32)
    mean = pd.DataFrame(mu, index=genes.index.copy())

    names = [f"tumor_{i:04d}" for i in range(config.n_tumor)] + [f"normal_{i:04d}" for i in range(config.n_normal)]
    # Interleave so nothing downstream can lean on input order.
    order = rng.permutation(n_cells)
    names = [names[i] for i in order]
    counts = counts[:, order]
    mean = mean.iloc[:, order]
    mean.columns = names
    is_tumor = pd.Series([n.startswith("tumor") for n in names], index=names, name="is_tumor")
    frame = pd.DataFrame(counts, index=genes.index.copy(), columns=names)
    return SyntheticDataset(
        counts=frame,
        is_tumor=is_tumor,
        genes=genes,
        tumor_copies=pd.Series(tumor_copies, index=genes.index, name="copies"),
        config=config,
        mean=mean,
    )


def scale_region(
    data: SyntheticDataset, cells: list[str], chrom: str, factor: float, seed: int, start: int = 0, end: int = 10**10
) -> SyntheticDataset:
    """Change the copy number of a region in ``cells`` by ``factor``, leaving every other count untouched.

    A loss (``factor < 1``) binomially thins the observed counts, which turns a
    negative binomial draw into an exact draw at the lower mean. A gain adds an
    independent negative binomial draw at ``(factor - 1)`` times the simulated
    mean, so cells pick up reads at genes where they had none, as a real gain
    does; scaling the observed counts instead would leave every zero a zero.
    """
    rng = np.random.default_rng(seed)
    in_region = (data.genes["chrom"] == chrom) & (data.genes["start"] >= start) & (data.genes["start"] < end)
    rows = in_region.to_numpy()
    counts = data.counts.copy()
    mean = data.mean.copy()
    block = counts.loc[rows, cells].to_numpy()
    mu = mean.loc[rows, cells].to_numpy()
    if factor < 1:
        block = rng.binomial(block, factor)
    else:
        shape = 1.0 / data.config.dispersion
        block = block + rng.poisson(rng.gamma(shape, (factor - 1.0) * mu / shape))
    counts.loc[rows, cells] = block.astype(counts.dtypes.iloc[0])
    mean.loc[rows, cells] = mu * factor
    return replace(data, counts=counts, mean=mean)
