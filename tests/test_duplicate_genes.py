"""Duplicate gene symbols: the r profile filters each row before summing, as CopyKAT-R does."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from copykat_py import profiles
from copykat_py.input import _prepare_input_matrix

N_CELLS = 10
LOW_DR = 0.3


def _counts() -> tuple[np.ndarray, list[str]]:
    rows = {
        "A": np.r_[1, 1, np.zeros(8)],  # a duplicate whose copies are each in 2 of 10 cells (0.2)
        "B": np.r_[np.ones(8), 0, 0],  # a duplicate with one passing copy (8 of 10) ...
        "C": np.ones(N_CELLS),
        "D": np.r_[0, 0, np.ones(8)],
    }
    mat = [rows["A"], np.r_[0, 0, 1, 1, np.zeros(6)], rows["B"], np.r_[0, 0, 0, 0, 0, 0, 0, 0, 5, 0], rows["C"], rows["D"]]
    return np.array(mat), ["A", "A", "B", "B", "C", "D"]


def _prepare(profile: str, as_sparse: bool) -> tuple[list[str], np.ndarray]:
    values, genes = _counts()
    cells = [f"c{i}" for i in range(N_CELLS)]
    with profiles.use(profiles.resolve(profile)):
        if as_sparse:
            raw = {"matrix": sparse.csc_matrix(values), "genes": np.array(genes, dtype=object), "barcodes": np.array(cells, dtype=object)}
        else:
            raw = pd.DataFrame(values, index=genes, columns=cells)
        prepared = _prepare_input_matrix(raw, 0, LOW_DR)
    matrix = prepared.matrix.toarray() if sparse.issparse(prepared.matrix) else np.asarray(prepared.matrix)
    return [str(g) for g in prepared.genes], matrix


@pytest.mark.parametrize("as_sparse", [True, False])
def test_r_profile_filters_duplicate_rows_one_by_one(as_sparse: bool) -> None:
    genes, matrix = _prepare("r", as_sparse)
    # A: both copies are under the cut, so it is gone (summed they would pass: 4 of 10 cells).
    # B: only the first copy passes, and it is kept alone, not added to the other copy's 5 counts.
    assert genes == ["B", "C", "D"]
    assert matrix[0].tolist() == [1] * 8 + [0, 0]


@pytest.mark.parametrize("as_sparse", [True, False])
def test_experimental_profile_sums_duplicates_first(as_sparse: bool) -> None:
    genes, matrix = _prepare("experimental", as_sparse)
    assert genes == ["A", "B", "C", "D"]
    assert matrix[1].tolist() == [1] * 8 + [5, 0]


def _counts_with_low_cells(n_genes_under_100: int) -> np.ndarray:
    """Cells with 300 genes, plus three with 150 (and optionally ones with 50)."""
    rng = np.random.default_rng(0)
    sizes = [300] * 20 + [150] * 3 + [50] * n_genes_under_100
    return np.vstack([np.r_[np.ones(k), np.zeros(400 - k)][rng.permutation(400)] for k in sizes]).T


@pytest.mark.parametrize("as_sparse", [True, False])
def test_r_1_1_0_keeps_cells_under_200_genes_unless_two_have_under_100(as_sparse: bool) -> None:
    def kept(values: np.ndarray, version: str) -> int:
        genes = [f"g{i}" for i in range(values.shape[0])]
        cells = [f"c{i}" for i in range(values.shape[1])]
        with profiles.use(profiles.resolve("r", version)):
            raw = (
                {"matrix": sparse.csc_matrix(values), "genes": np.array(genes, dtype=object), "barcodes": np.array(cells, dtype=object)}
                if as_sparse
                else pd.DataFrame(values, index=genes, columns=cells)
            )
            return len(_prepare_input_matrix(raw, 200, 0.0).barcodes)

    one_low = _counts_with_low_cells(1)  # 3 cells at 150 genes, 1 at 50
    two_low = _counts_with_low_cells(2)
    assert kept(one_low, "1.1.0") == one_low.shape[1]  # 1.1.0 removes nothing: only one cell is under 100
    assert kept(one_low, "1.2.5") == one_low.shape[1] - 4  # 1.2.5 removes all four under 200
    assert kept(two_low, "1.1.0") == two_low.shape[1] - 5
