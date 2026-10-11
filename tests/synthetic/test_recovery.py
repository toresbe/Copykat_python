"""Planted-CNV recovery, a pure-diploid null, and robustness to depth and purity."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from metrics import (
    ANEUPLOID_LABELS,
    call_accuracy,
    event_shift,
    profile_correlation,
    region_mask,
    run_copykat,
)
from synthetic_data import AUTOSOMES, SimulationConfig, simulate

from copykat_py._types import PredictionLabel

# Smallest whole-population CNA shift the default karyotypes produce is about
# 0.12 (one-copy gain); a null population should stay well under that.
NULL_CHROMOSOME_LIMIT = 0.08


def test_planted_events_are_recovered(dataset, baseline_run, record):
    accuracy = call_accuracy(baseline_run, dataset)
    correlation = profile_correlation(baseline_run, dataset)
    shifts = {event.label(): event_shift(baseline_run, dataset, event) for event in dataset.config.karyotype}
    planted = {event.chrom for event in dataset.config.karyotype}
    neutral = [c for c in AUTOSOMES[dataset.config.genome] if c not in planted]
    tumor, normal = dataset.tumor_cells, dataset.normal_cells
    off_target = max(
        abs(
            baseline_run.cna.loc[region_mask(baseline_run.bins, c), tumor].to_numpy().mean()
            - baseline_run.cna.loc[region_mask(baseline_run.bins, c), normal].to_numpy().mean()
        )
        for c in neutral
    )
    record("recovery", "call_accuracy", accuracy)
    record("recovery", "profile_correlation", correlation)
    record("recovery", "max_neutral_chromosome_shift", off_target)
    for label, shift in shifts.items():
        record("recovery", f"shift {label}", shift)

    assert accuracy >= 0.97
    assert correlation >= 0.85
    for event in dataset.config.karyotype:
        shift = shifts[event.label()]
        assert np.sign(shift) == np.sign(event.copies - 2), event.label()
        assert abs(shift) >= 0.06, event.label()
    by_copies = {event.copies: shifts[event.label()] for event in dataset.config.karyotype if event.copies > 2}
    assert by_copies[4.0] > by_copies[3.0], "a two-copy gain should read stronger than a one-copy gain"
    assert off_target < 0.03


def test_pure_diploid_population_has_no_confident_calls(genome, runs_dir, record):
    data = simulate(SimulationConfig(genome=genome, n_tumor=0, n_normal=300, events=()))
    run = run_copykat(data, runs_dir / "null")
    labels = run.predictions.value_counts()
    confident_aneuploid = int(run.predictions.eq(PredictionLabel.ANEUPLOID).sum())
    chromosome_means = run.cna.groupby(run.bins["chrom"].to_numpy()).mean().mean(axis=1).abs()
    record("null", "confident_aneuploid_calls", confident_aneuploid)
    record("null", "low_confidence_calls", int(run.predictions.isin(ANEUPLOID_LABELS).sum()) - confident_aneuploid)
    record("null", "max_chromosome_pseudobulk", float(chromosome_means.max()))
    record("null", "labels", labels.to_dict())

    assert confident_aneuploid == 0, labels.to_dict()
    assert chromosome_means.max() < NULL_CHROMOSOME_LIMIT


def test_half_sequencing_depth(dataset, runs_dir, record):
    rng = np.random.default_rng(3)
    thinned = pd.DataFrame(
        rng.binomial(dataset.counts.to_numpy(), 0.5), index=dataset.counts.index, columns=dataset.counts.columns
    )
    run = run_copykat(replace(dataset, counts=thinned), runs_dir / "half_depth")
    accuracy, correlation = call_accuracy(run, dataset), profile_correlation(run, dataset)
    record("depth", "half_depth_call_accuracy", accuracy)
    record("depth", "half_depth_profile_correlation", correlation)
    assert accuracy >= 0.95
    assert correlation >= 0.85


@pytest.mark.parametrize("n_tumor", [30, 270], ids=["10pct_tumor", "90pct_tumor"])
def test_tumor_purity(n_tumor, genome, runs_dir, record):
    data = simulate(SimulationConfig(genome=genome, n_tumor=n_tumor, n_normal=300 - n_tumor))
    run = run_copykat(data, runs_dir / f"purity_{n_tumor}")
    accuracy, correlation = call_accuracy(run, data), profile_correlation(run, data)
    record("purity", f"{n_tumor / 3:.0f}pct_tumor_call_accuracy", accuracy)
    record("purity", f"{n_tumor / 3:.0f}pct_tumor_profile_correlation", correlation)
    assert accuracy >= 0.95
    assert correlation >= 0.85
