"""Mix tumour cells into a supplied normal reference and measure how far results drift."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from metrics import call_accuracy, profile_correlation, region_mask, run_copykat

# Share of the supplied reference that is actually tumour. 0.75 uses every tumour cell.
FRACTIONS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75]
POOL_SIZE = 60


def _raw_event_shift(run_dir, dataset):
    """Mean tumour-minus-normal shift over planted events in the reference-subtracted, pre-adjustment output."""
    raw = pd.read_csv(run_dir / "synthetic_copykat_CNA_raw_results_gene_by_cell.txt", sep="\t")
    chrom, start = raw["chromosome_name"].astype(str).to_numpy(), raw["start_position"].to_numpy()
    tumor = [c for c in dataset.tumor_cells if c in raw.columns]
    shifts = []
    for event in dataset.config.karyotype:
        rows = (chrom == event.chrom) & (start >= event.start) & (start < event.end)
        shifts.append(abs(raw.loc[rows, tumor].to_numpy().mean()))
    return float(np.mean(shifts))


def _normal_offset(run, dataset):
    """Mean |final CNA| of truly normal cells over planted regions: zero when the reference is clean."""
    normal = [c for c in dataset.normal_cells if c in run.cna.columns]
    offsets = [
        abs(run.cna.loc[region_mask(run.bins, e.chrom, e.start, e.end), normal].to_numpy().mean())
        for e in dataset.config.karyotype
    ]
    return float(np.mean(offsets))


@pytest.fixture(scope="module")
def drift_curve(dataset, runs_dir):
    pool = dataset.normal_cells[:POOL_SIZE]
    curve = {}
    for fraction in FRACTIONS:
        n_bad = round(fraction * POOL_SIZE / (1 - fraction))
        reference = pool + dataset.tumor_cells[:n_bad]
        out = runs_dir / f"contamination_{fraction}"
        run = run_copykat(dataset, out, norm_cell_names=reference)
        curve[fraction] = {
            "tumor_cells_in_reference": n_bad,
            "call_accuracy": call_accuracy(run, dataset),
            "profile_correlation": profile_correlation(run, dataset),
            "raw_event_shift": _raw_event_shift(out, dataset),
            "normal_offset": _normal_offset(run, dataset),
        }
    clean = curve[0.0]["raw_event_shift"]
    for point in curve.values():
        point["raw_signal_retained"] = point["raw_event_shift"] / clean
    return curve


def test_reference_contamination_drift(drift_curve, record):
    for fraction, point in drift_curve.items():
        for key, value in point.items():
            record("contamination", f"{fraction:.2f} {key}", value)
    failing = [f for f, p in drift_curve.items() if p["call_accuracy"] < 0.9]
    record("contamination", "breakdown_fraction", min(failing) if failing else None)

    retained = [drift_curve[f]["raw_signal_retained"] for f in FRACTIONS]
    # The median reference absorbs tumour signal in proportion to contamination...
    assert all(b <= a + 0.02 for a, b in pairwise(retained)), retained
    assert drift_curve[0.1]["raw_signal_retained"] >= 0.8
    assert drift_curve[0.25]["raw_signal_retained"] >= 0.6
    # ...while a minority of bad reference cells still leaves calls and the diploid zero intact.
    for fraction in (0.0, 0.1, 0.25, 0.4):
        assert drift_curve[fraction]["call_accuracy"] >= 0.97, fraction
        assert drift_curve[fraction]["normal_offset"] < 0.03, fraction
