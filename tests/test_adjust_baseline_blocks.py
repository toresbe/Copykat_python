"""Row-blocked baseline adjustment must match the whole-matrix computation exactly."""

import numpy as np
import pytest

from copykat_py.final_call import adjust_baseline_inplace


def _reference(mat: np.ndarray, diploid_mask: np.ndarray) -> np.ndarray:
    """The previous whole-matrix implementation."""
    mat -= mat[:, diploid_mask].mean(axis=1, keepdims=True)
    mat -= mat.mean(axis=0, keepdims=True)
    diploid = mat[:, diploid_mask]
    cf_h = np.std(diploid, axis=1)
    base = np.mean(diploid, axis=1)
    threshold = 0.25 * cf_h
    cell_means = mat.mean(axis=0, keepdims=True)
    noise = np.abs(mat - base[:, None]) <= threshold[:, None]
    np.copyto(mat, np.broadcast_to(cell_means, mat.shape), where=noise)
    mat -= mat.mean(axis=0, keepdims=True)
    return mat


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("chunk_elems", [7, 26, 50, 1000, 1 << 24])
def test_blocked_adjustment_is_identical(dtype: type, order: str, chunk_elems: int) -> None:
    rng = np.random.default_rng(11)
    mat = np.asarray(rng.normal(size=(157, 83)), dtype=dtype, order=order)
    mask = rng.random(83) < 0.3
    expected = _reference(mat.copy(order="K"), mask)
    got = adjust_baseline_inplace(mat.copy(order="K"), mask, chunk_elems=chunk_elems)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
