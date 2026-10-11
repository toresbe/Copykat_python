"""The heatmap draws at most HEATMAP_MAX_ROWS evenly spaced cells, in order."""

import numpy as np

from copykat_py.plotting import _display_rows


def test_small_orders_are_kept_whole() -> None:
    order = np.array([3, 1, 2, 0])
    np.testing.assert_array_equal(_display_rows(order, max_rows=4), order)


def test_large_orders_are_subsampled_evenly_in_order() -> None:
    order = np.arange(10_000)[::-1]
    rows = _display_rows(order, max_rows=100)
    assert len(rows) == 100
    assert rows[0] == order[0] and rows[-1] == order[-1]
    assert np.all(np.diff(rows) < 0)  # same order as cell_order
