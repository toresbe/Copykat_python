"""Grouped, cell-blocked bin medians must equal the per-bin computation exactly."""

import numpy as np
import pytest

from copykat_py import convert_bins
from copykat_py._types import Genome
from copykat_py.data_loader import load_full_anno
from tests import _convert_bins_reference as reference


@pytest.mark.parametrize("n_genes", [300, 3000])
def test_grouped_bins_match_reference(monkeypatch: pytest.MonkeyPatch, n_genes: int) -> None:
    rng = np.random.default_rng(n_genes)
    anno = load_full_anno(Genome.HG20)
    anno = anno[anno["chromosome_name"].between(1, 23)].drop_duplicates("hgnc_symbol")
    genes = anno.iloc[np.sort(rng.choice(len(anno), n_genes, replace=False))].reset_index(drop=True)
    genes = genes.iloc[np.argsort(genes["abspos"].to_numpy(), kind="mergesort")].reset_index(drop=True)
    values = rng.normal(0, 0.3, size=(n_genes, 70)).astype(np.float32)
    cells = [f"c{i}" for i in range(70)]
    monkeypatch.setattr(convert_bins, "_BIN_BLOCK_CELLS", 16)
    got = convert_bins.convert_to_bins(genes, n_cores=3, values=values, cell_names=cells)
    expected = reference.convert_to_bins(genes, n_cores=1, values=values, cell_names=cells)
    assert got.values.dtype == expected.values.dtype and got.values.flags.f_contiguous
    np.testing.assert_array_equal(got.values, expected.values)
    assert got.rna_table.equals(expected.rna_table)
