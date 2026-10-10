"""Checks for the opt-in posterior-Gamma KS breakpoint statistic."""

import numpy as np

from copykat_py.segmentation import _find_breakpoints_exact, _gamma_ks_distance


def test_gamma_ks_distance_is_zero_for_identical_posteriors() -> None:
    assert _gamma_ks_distance(4.0, 2.0, 4.0, 2.0) == 0.0


def test_gamma_ks_distance_detects_distinct_posteriors() -> None:
    forward = _gamma_ks_distance(3.0, 1.0, 9.0, 2.0)
    reverse = _gamma_ks_distance(9.0, 2.0, 3.0, 1.0)
    assert 0.0 < forward < 1.0
    assert np.isclose(forward, reverse, atol=1e-12)


def test_exact_breakpoints_do_not_depend_on_monte_carlo_seed() -> None:
    profile = np.r_[np.ones(50), np.full(50, 3.0)]
    assert _find_breakpoints_exact(profile, bins=10, cut_cor=0.1)
    assert not _find_breakpoints_exact(np.ones(100), bins=10, cut_cor=0.1)
