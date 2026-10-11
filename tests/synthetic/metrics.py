"""Run the pipeline on a synthetic dataset and score it against the planted truth."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from synthetic_data import Event, SyntheticDataset

from copykat_py._types import PredictionLabel
from copykat_py.copykat import copykat

ANEUPLOID_LABELS = {PredictionLabel.ANEUPLOID, PredictionLabel.ANEUPLOID_LOW_CONFIDENCE}
DIPLOID_LABELS = {PredictionLabel.DIPLOID, PredictionLabel.DIPLOID_LOW_CONFIDENCE}
COORDINATE_COLUMNS = ["chrom", "chrompos", "abspos"]


@dataclass
class Run:
    predictions: pd.Series  # copykat.pred, indexed by cell name
    cna: pd.DataFrame  # bins x cells
    bins: pd.DataFrame  # chrom (str), chrompos, abspos
    runtime: dict[str, Any]

    @property
    def called_aneuploid(self) -> pd.Series:
        """True/False per cell with a diploid or aneuploid call; undefined calls dropped."""
        defined = self.predictions[self.predictions.isin(ANEUPLOID_LABELS | DIPLOID_LABELS)]
        return defined.isin(ANEUPLOID_LABELS)


def run_copykat(data: SyntheticDataset, out_dir: Path, counts: Any = None, **kwargs: Any) -> Run:
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"plot_genes": False, "n_cores": int(os.environ.get("SYNTHETIC_N_CORES", "2"))}
    options.update(kwargs)
    # The pipeline logs every step at INFO; keep pytest output readable.
    with _quiet():
        result = copykat(data.counts if counts is None else counts, sam_name=str(out_dir / "synthetic"), **options)
    cna = result["CNAmat"]
    bins = cna[COORDINATE_COLUMNS].copy()
    bins["chrom"] = bins["chrom"].astype(str)
    return Run(
        predictions=result["prediction"].set_index("cell.names")["copykat.pred"],
        cna=cna.drop(columns=COORDINATE_COLUMNS),
        bins=bins,
        runtime=result["runtime"],
    )


@contextlib.contextmanager
def _quiet():
    package = logging.getLogger("copykat_py")
    previous = package.level
    package.setLevel(logging.WARNING)
    try:
        yield
    finally:
        package.setLevel(previous)


def call_accuracy(run: Run, data: SyntheticDataset, cells: list[str] | None = None) -> float:
    """Share of cells whose diploid/aneuploid call matches the truth (undefined calls count as wrong)."""
    cells = data.is_tumor.index.tolist() if cells is None else cells
    called = run.called_aneuploid.reindex(cells)
    truth = data.is_tumor.reindex(cells)
    return float((called == truth).mean())


def region_mask(bins: pd.DataFrame, chrom: str, start: int = 0, end: int = 10**10) -> np.ndarray:
    pos = bins["chrompos"].to_numpy()
    return ((bins["chrom"] == chrom).to_numpy()) & (pos >= start) & (pos < end)


def region_shift(
    run: Run, cells: list[str], ref_cells: list[str], chrom: str, start: int = 0, end: int = 10**10
) -> float:
    """Mean CNA over a region in ``cells`` minus the same in ``ref_cells``."""
    mask = region_mask(run.bins, chrom, start, end)
    cells = [c for c in cells if c in run.cna.columns]
    ref_cells = [c for c in ref_cells if c in run.cna.columns]
    region = run.cna.loc[mask]
    return float(region[cells].to_numpy().mean() - region[ref_cells].to_numpy().mean())


def event_shift(run: Run, data: SyntheticDataset, event: Event) -> float:
    return region_shift(run, data.tumor_cells, data.normal_cells, event.chrom, event.start, event.end)


def expected_log_ratio(bins: pd.DataFrame, events: tuple[Event, ...]) -> np.ndarray:
    expected = np.zeros(len(bins))
    for event in events:
        expected[region_mask(bins, event.chrom, event.start, event.end)] = np.log2(event.copies / 2.0)
    return expected


def profile_correlation(run: Run, data: SyntheticDataset) -> float:
    """Pearson r between the tumour pseudo-bulk profile and the planted log2(CN/2) per bin."""
    tumor = [c for c in data.tumor_cells if c in run.cna.columns]
    normal = [c for c in data.normal_cells if c in run.cna.columns]
    observed = run.cna[tumor].mean(axis=1).to_numpy() - run.cna[normal].mean(axis=1).to_numpy()
    expected = expected_log_ratio(run.bins, data.config.events)
    return float(np.corrcoef(observed, expected)[0, 1])


def call_agreement(a: Run, b: Run) -> tuple[float, int]:
    """Share of shared cells given the same diploid/aneuploid call, and the shared count."""
    a_calls, b_calls = a.called_aneuploid, b.called_aneuploid
    shared = a_calls.index.intersection(b_calls.index)
    return float((a_calls[shared] == b_calls[shared]).mean()), len(shared)


def per_cell_correlation(a: Run, b: Run) -> np.ndarray:
    """Pearson r between each shared cell's CNA profile in two runs (bins aligned on position)."""
    left = a.cna.set_index(pd.MultiIndex.from_frame(a.bins[["chrom", "chrompos"]]))
    right = b.cna.set_index(pd.MultiIndex.from_frame(b.bins[["chrom", "chrompos"]]))
    rows = left.index.intersection(right.index)
    cells = left.columns.intersection(right.columns)
    x = left.loc[rows, cells].to_numpy()
    y = right.loc[rows, cells].to_numpy()
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    return (x * y).sum(axis=0) / np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))


def pseudobulk_noise(run: Run) -> float:
    """95th percentile of |per-bin mean CNA| over all cells: how much spurious signal a population carries."""
    return float(np.quantile(np.abs(run.cna.mean(axis=1).to_numpy()), 0.95))
