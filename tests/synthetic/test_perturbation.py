"""Inject a known copy-number shift and check the output moves with it, and only where it was put."""

from itertools import pairwise

import numpy as np
import pytest
from metrics import call_accuracy, region_shift, run_copykat
from synthetic_data import scale_region

FACTORS = [0.5, 0.75, 1.5, 2.0, 3.0]
TARGET = "5"  # copy-neutral in the default karyotype
CONTROLS = ["1", "2", "12"]


@pytest.fixture(scope="module")
def dose_response(dataset, runs_dir):
    runs = {}
    for factor in FACTORS:
        perturbed = scale_region(dataset, dataset.tumor_cells, TARGET, factor, seed=5)
        runs[factor] = run_copykat(perturbed, runs_dir / f"perturb_{factor}")
    return runs


def _shift(run, dataset, chrom, start=0, end=10**10):
    return region_shift(run, dataset.tumor_cells, dataset.normal_cells, chrom, start, end)


def test_shift_follows_injected_copy_change(dataset, baseline_run, dose_response, record):
    before = _shift(baseline_run, dataset, TARGET)
    response = {factor: _shift(run, dataset, TARGET) - before for factor, run in dose_response.items()}
    for factor, delta in response.items():
        record("perturbation", f"chr{TARGET} x{factor:g} shift", delta)
    slope = np.polyfit(np.log2(FACTORS), [response[f] for f in FACTORS], 1)[0]
    record("perturbation", "slope per log2 fold", slope)

    for factor, delta in response.items():
        assert np.sign(delta) == np.sign(np.log2(factor)), f"x{factor}: {delta:+.3f}"
        assert abs(delta) >= 0.04, f"x{factor}: {delta:+.3f}"
    ordered = [response[f] for f in FACTORS]
    assert all(a < b for a, b in pairwise(ordered)), ordered


def test_injection_leaves_other_chromosomes_and_calls_alone(dataset, baseline_run, dose_response, record):
    before = _shift(baseline_run, dataset, TARGET)
    worst = 0.0
    for factor, run in dose_response.items():
        # Each cell is centred on its own mean, so a large change on the target
        # nudges every other chromosome the opposite way by roughly the target's
        # share of the genome times the change. Allow for that, and no more.
        target = abs(_shift(run, dataset, TARGET) - before)
        change = max(abs(_shift(run, dataset, c) - _shift(baseline_run, dataset, c)) for c in CONTROLS)
        worst = max(worst, change)
        assert change < 0.01 + 0.15 * target, f"x{factor}: controls moved {change:.3f}"
        assert call_accuracy(run, dataset) >= 0.97
    record("perturbation", "max_control_chromosome_change", worst)


def test_focal_gain_stays_focal(dataset, runs_dir, record):
    start, end = 60_000_000, 100_000_000
    run = run_copykat(
        scale_region(dataset, dataset.tumor_cells, "12", 2.0, seed=6, start=start, end=end), runs_dir / "focal"
    )
    inside = _shift(run, dataset, "12", start, end)
    flank = _shift(run, dataset, "12", 0, 50_000_000)
    record("perturbation", "focal chr12:60-100Mb x2 inside", inside)
    record("perturbation", "focal chr12:0-50Mb flank", flank)
    assert inside >= 0.1
    assert abs(flank) < 0.03
