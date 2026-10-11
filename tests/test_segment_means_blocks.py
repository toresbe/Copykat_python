"""Blocked, threaded segment means must equal the whole-matrix computation."""

import itertools

import numpy as np
import pytest

from copykat_py import segmentation


def _reference(fttmat: np.ndarray, BR: list[int]) -> np.ndarray:
    """The previous whole-matrix implementation of cna_mcmc's step 3."""
    n_genes, n_cells = fttmat.shape
    norm_mat_all = np.exp(fttmat)
    cumsum = np.vstack([np.zeros((1, n_cells), dtype=norm_mat_all.dtype), np.cumsum(norm_mat_all, axis=0)])
    out = np.empty((n_genes, n_cells), dtype=np.float32)
    for left, right in itertools.pairwise(BR):
        seg_sum = cumsum[right + 1] - cumsum[left]
        seg_mean = np.maximum(seg_sum / max(1, right - left + 1), 1e-300)
        out[left : right + 1, :] = np.log(seg_mean).astype(np.float32, copy=False)
    return out


@pytest.mark.parametrize("n_cores", [1, 3])
@pytest.mark.parametrize("block_elems", [1, 500, 1 << 22])
def test_blocked_segment_means(monkeypatch: pytest.MonkeyPatch, n_cores: int, block_elems: int) -> None:
    rng = np.random.default_rng(2)
    profile = np.repeat(rng.normal(0, 0.3, size=6), 40)[:, None]
    fttmat = (profile + rng.normal(0, 0.2, size=(240, 50))).astype(np.float32)
    clusters = np.repeat([1, 2], 25)
    monkeypatch.setattr(segmentation, "_SEGMENT_BLOCK_ELEMS", block_elems)
    result = segmentation.cna_mcmc(clusters, fttmat, bins=10, cut_cor=0.1, n_cores=n_cores, mc_samples=200)
    np.testing.assert_array_equal(result.log_cna, _reference(fttmat, result.breakpoints))
