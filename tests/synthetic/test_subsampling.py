"""Dropping cells should leave the calls and profiles of the cells that remain about where they were."""

import numpy as np
import pytest
from metrics import call_agreement, per_cell_correlation, run_copykat

SEEDS = [100, 101, 102]


@pytest.mark.parametrize("keep", [0.8, 0.5], ids=["keep80", "keep50"])
def test_subsampling_stability(dataset, baseline_run, runs_dir, keep, record):
    agreements, medians = [], []
    seeds = SEEDS if keep == 0.8 else SEEDS[:1]
    for seed in seeds:
        n_keep = round(keep * dataset.counts.shape[1])
        cells = sorted(np.random.default_rng(seed).choice(dataset.counts.columns, size=n_keep, replace=False))
        run = run_copykat(dataset.subset_cells(list(cells)), runs_dir / f"subsample_{keep}_{seed}")
        agreement, shared = call_agreement(baseline_run, run)
        assert shared >= 0.95 * n_keep
        agreements.append(agreement)
        medians.append(float(np.median(per_cell_correlation(baseline_run, run))))
    record("subsampling", f"keep {keep:.0%} min call agreement", min(agreements))
    record("subsampling", f"keep {keep:.0%} min median per-cell r", min(medians))
    assert min(agreements) >= 0.97
    assert min(medians) >= 0.75
