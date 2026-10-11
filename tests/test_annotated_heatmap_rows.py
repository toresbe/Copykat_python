"""The annotated heatmap subsamples rows consistently with its sidebars and groups."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from copykat_py import plotting


def test_annotated_heatmap_with_subsampled_rows(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plotting, "HEATMAP_MAX_ROWS", 50)
    rng = np.random.default_rng(4)
    n_bins, n_cells = 60, 300
    mat = rng.normal(0, 0.2, size=(n_bins, n_cells))
    cells = [f"c{i}" for i in range(n_cells)]
    meta = pd.DataFrame(
        {"cell": cells, "group": rng.choice(["A", "B", "C"], n_cells), "score": rng.normal(size=n_cells)}
    )
    meta.to_csv(tmp_path / "meta.csv", index=False)
    out = tmp_path / "annotated.png"
    plotting.plot_heatmap_annotated(
        mat=mat,
        cell_names=cells,
        chrom_info=np.repeat(np.arange(1, 4), n_bins // 3),
        meta_csv=str(tmp_path / "meta.csv"),
        options=plotting.AnnotatedHeatmapOptions(row_split_col="group", output_path=str(out), n_cores=1),
    )
    assert out.stat().st_size > 0


def test_display_positions_follow_the_module_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plotting, "HEATMAP_MAX_ROWS", 50)
    assert len(plotting._display_positions(300)) == 50
