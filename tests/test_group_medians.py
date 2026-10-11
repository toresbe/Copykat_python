"""Threaded group medians must equal per-group np.median exactly."""

import numpy as np
import pytest

from copykat_py._medians import group_medians


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_group_medians_match_numpy(dtype: type, layout: str) -> None:
    rng = np.random.default_rng(3)
    mat = np.asarray(rng.normal(size=(301, 97)), dtype=dtype, order=layout)
    labels = rng.integers(1, 5, mat.shape[1])
    groups = [labels == k for k in sorted(set(labels))] + [np.arange(97) < 2]
    expected = [np.median(mat[:, g], axis=1) for g in groups]
    for n_cores in (1, 4):
        got = group_medians(mat, groups, n_cores=n_cores)
        for e, g in zip(expected, got, strict=True):
            assert g.dtype == e.dtype
            np.testing.assert_array_equal(g, e)
