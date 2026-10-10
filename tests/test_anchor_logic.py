"""Opt-in marker reference selection and raw-input marker counting."""

import numpy as np
import pandas as pd

from copykat_py.anchor import count_markers, choose_anchor_cluster


def test_marker_counting_from_dataframe_and_csv(tmp_path) -> None:
    matrix = pd.DataFrame(
        [[1, 0], [1, 1], [1, 0], [0, 1]],
        index=["PTPRC", "LAPTM5", "CORO1A", "OTHER"],
        columns=["cell1", "cell2"],
    )
    expected = pd.Series([3, 1], index=["cell1", "cell2"])
    pd.testing.assert_series_equal(count_markers(matrix, ("PTPRC", "LAPTM5", "CORO1A")), expected)
    path = tmp_path / "counts.csv"
    matrix.to_csv(path)
    pd.testing.assert_series_equal(count_markers(str(path), ("PTPRC", "LAPTM5", "CORO1A")), expected)


def test_marker_enriched_cluster_overrides_sigma_cluster() -> None:
    labels = np.r_[np.ones(20, dtype=int), np.full(20, 2, dtype=int)]
    immune = np.r_[np.full(20, 4), np.zeros(20)]
    endothelial = np.zeros(40)
    selected, path = choose_anchor_cluster(labels, immune, endothelial, sigma_cluster=2)
    assert selected == 1
    assert path == "immune"
