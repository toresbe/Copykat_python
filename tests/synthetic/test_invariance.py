"""Results must not depend on the order of cells or genes in the input, or on the run."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from metrics import run_copykat


def _assert_same(a, b):
    pd.testing.assert_series_equal(a.predictions.sort_index(), b.predictions.sort_index())
    np.testing.assert_array_equal(a.cna[sorted(a.cna.columns)].to_numpy(), b.cna[sorted(b.cna.columns)].to_numpy())


def _max_difference(a, b):
    cells = sorted(a.cna.columns)
    return float(np.abs(a.cna[cells].to_numpy() - b.cna[cells].to_numpy()).max())


def test_repeat_run_is_identical(dataset, baseline_run, runs_dir, record):
    again = run_copykat(dataset, runs_dir / "repeat")
    record("invariance", "repeat_max_abs_difference", _max_difference(baseline_run, again))
    _assert_same(baseline_run, again)


def test_cell_order(dataset, baseline_run, runs_dir, record):
    order = np.random.default_rng(11).permutation(dataset.counts.shape[1])
    shuffled = run_copykat(replace(dataset, counts=dataset.counts.iloc[:, order]), runs_dir / "cell_order")
    record("invariance", "cell_order_max_abs_difference", _max_difference(baseline_run, shuffled))
    _assert_same(baseline_run, shuffled)


def test_gene_order(dataset, baseline_run, runs_dir, record):
    order = np.random.default_rng(12).permutation(dataset.counts.shape[0])
    shuffled = run_copykat(replace(dataset, counts=dataset.counts.iloc[order]), runs_dir / "gene_order")
    record("invariance", "gene_order_max_abs_difference", _max_difference(baseline_run, shuffled))
    _assert_same(baseline_run, shuffled)


@pytest.mark.xfail(
    strict=True,
    reason="annotation_order breaks ties in gene position by input row, so two genes at the same "
    "position swap places when the input is reordered and the smoothed profile changes",
)
def test_gene_order_with_tied_positions(dataset, runs_dir, record):
    # RP11-3P17.3 and RP11-3P17.5 share an hg20 position. Put them in the panel
    # in place of two genes that sit next to them.
    neighbours = dataset.genes.index[(dataset.genes["chrom"] == "22")][-2:]
    counts = dataset.counts.rename(index=dict(zip(neighbours, ["RP11-3P17.3", "RP11-3P17.5"], strict=True)))
    forward = run_copykat(replace(dataset, counts=counts), runs_dir / "tie_forward")
    backward = run_copykat(replace(dataset, counts=counts.iloc[::-1]), runs_dir / "tie_backward")
    record("invariance", "tied_gene_order_max_abs_difference", _max_difference(forward, backward))
    _assert_same(forward, backward)
