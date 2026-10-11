"""Streaming transform + smoothing must match the whole-matrix path exactly."""

import numpy as np
import pytest
from scipy import sparse

from copykat_py._pipeline import preprocessing
from copykat_py.smoothing import dlm_smooth


def _counts(n_genes: int, n_cells: int) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.poisson(rng.gamma(0.3, 2.0, size=(n_genes, 1)), size=(n_genes, n_cells)).astype(np.float64)


@pytest.mark.parametrize("as_sparse", [True, False])
@pytest.mark.parametrize("block_cells", [1, 37, 4096])
def test_streaming_matches_whole_matrix(monkeypatch: pytest.MonkeyPatch, as_sparse: bool, block_cells: int) -> None:
    counts = _counts(400, 150)
    chroms = np.repeat(np.arange(1, 5), 100)
    dense = np.asfortranarray(counts)
    ref_info: dict = {"steps": [], "parameters": {}}
    norm, seg_ref, keep_ref = preprocessing.transform_counts_inplace(
        dense.copy(order="F"), chroms, detection_rate=0.1, ngene_chr=1, runtime_info=ref_info
    )
    expected = dlm_smooth(norm, n_cores=2)

    monkeypatch.setattr(preprocessing, "_STREAM_BLOCK_CELLS", block_cells)
    info: dict = {"steps": [], "parameters": {}}
    source = sparse.csr_matrix(counts) if as_sparse else counts
    got, seg, keep, _ = preprocessing.transform_and_smooth(
        source, chroms, detection_rate=0.1, ngene_chr=1, n_cores=2, runtime_info=info
    )
    np.testing.assert_array_equal(seg, seg_ref)
    np.testing.assert_array_equal(keep, keep_ref)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    assert [s["step"] for s in info["steps"]] == ["freeman_tukey_transform"]
