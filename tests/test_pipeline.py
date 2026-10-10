"""Integration checks for stage wiring and genome-specific output contracts."""

import importlib
import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from copykat_py._pipeline import baseline, preprocessing, segmentation
from copykat_py._types import (
    AnchorPath,
    AnchorStrategy,
    BaselineWarning,
    CellLineMode,
    FinalCallStrategy,
    Genome,
    PredictionLabel,
)
from copykat_py.baseline import BaselineResult, SyntheticBaselineResult
from copykat_py.convert_bins import BinConversionResult
from copykat_py.segmentation import SegmentationResult

workflow = importlib.import_module("copykat_py.copykat")


@pytest.fixture
def pipeline_stubs(monkeypatch):
    """Small deterministic stage inputs; input filtering and final clustering run normally."""

    def annotation(genes, *, id_type, genome):
        count = len(genes)
        return pd.DataFrame(
            {
                "abspos": np.arange(count) * 1000,
                "chromosome_name": np.repeat([1, 2], count // 2),
                "start_position": np.arange(count) * 1000,
                "end_position": np.arange(count) * 1000 + 100,
                "ensembl_gene_id": list(genes),
                "hgnc_symbol" if genome is Genome.HG20 else "mgi_symbol": list(genes),
                "band": ["p"] * count,
            }
        ), np.arange(count)

    def auto_baseline(values, *, cell_names, anchor_selector=None, **kwargs):
        return BaselineResult(
            np.median(values[:, :6], axis=1),
            BaselineWarning.NONE,
            cell_names[:6],
            np.repeat([1, 2], values.shape[1] // 2),
            AnchorPath.IMMUNE if anchor_selector is not None else AnchorPath.SIGMA,
        )

    def synthetic(values, **kwargs):
        return SyntheticBaselineResult(
            values - np.median(values, axis=1, keepdims=True),
            np.repeat([1, 2], values.shape[1] // 2),
        )

    def segment(labels, values, **kwargs):
        return SegmentationResult(values.copy(order="F"), list(range(30)))

    def convert(annotation, *, genome, n_cores, values, cell_names):
        coords = pd.DataFrame(
            {
                "chrom": annotation["chromosome_name"].to_numpy(),
                "chrompos": annotation["end_position"].to_numpy(),
                "abspos": annotation["abspos"].to_numpy(),
            }
        )
        return BinConversionResult(coords.copy(), coords, values.copy(order="F"))

    stubs = {
        "annotate_gene_rows": annotation,
        "load_cyclegenes": lambda: [],
        "baseline_norm_cl": auto_baseline,
        "baseline_synthetic": synthetic,
        "cna_mcmc": segment,
        "convert_to_bins": convert,
        "dlm_smooth": lambda values, **kwargs: values.copy(order="F"),
    }
    for target, names in (
        (preprocessing, ("annotate_gene_rows", "load_cyclegenes")),
        (baseline, ("baseline_norm_cl", "baseline_synthetic")),
        (segmentation, ("cna_mcmc",)),
        (workflow, ("convert_to_bins", "dlm_smooth")),
    ):
        for name in names:
            monkeypatch.setattr(target, name, stubs[name])
    plotting = importlib.import_module("copykat_py.plotting")
    plot = Mock()
    annotated_plot = Mock()
    monkeypatch.setattr(plotting, "plot_heatmap", plot)
    monkeypatch.setattr(plotting, "plot_heatmap_annotated", annotated_plot)
    monkeypatch.setattr(workflow._anchor, "arm_correlation_calls", lambda *args: np.arange(12) >= 6)
    return stubs, plot, annotated_plot


def input_counts() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    counts = rng.poisson(4, size=(100, 13))
    counts[:, -1] = 0  # retained in the prediction output as not.defined
    return pd.DataFrame(counts, index=[f"GENE{i}" for i in range(100)], columns=[f"cell{i}" for i in range(13)])


@pytest.mark.parametrize("genome", [Genome.HG20, Genome.MM10])
@pytest.mark.parametrize("reference", ["known", "auto", "synthetic", "markers"])
def test_pipeline_output_contracts(tmp_path, pipeline_stubs, genome, reference) -> None:
    _, plot, annotated_plot = pipeline_stubs
    names = [f"cell{i}" for i in range(6)] if reference == "known" else ""
    meta = tmp_path / "metadata.csv"
    pd.DataFrame({"cell": [f"cell{i}" for i in range(13)], "group": ["sample"] * 13}).to_csv(meta, index=False)
    sample = str(tmp_path / "sample")
    result = workflow.copykat(
        input_counts(),
        genome=genome,
        min_gene_per_cell=5,
        ngene_chr=1,
        norm_cell_names=names,
        cell_line=CellLineMode.YES if reference == "synthetic" else CellLineMode.NO,
        anchor=AnchorStrategy.MARKERS if reference == "markers" else AnchorStrategy.SIGMA,
        plot_genes=True,
        output_seg=True,
        meta_csv=str(meta),
        sam_name=sample,
    )
    prefix = f"{sample}_copykat_"
    has_predictions = not (genome is Genome.HG20 and reference == "synthetic")
    assert ("prediction" in result) == has_predictions
    if has_predictions:
        assert result["prediction"].shape[0] == 13
        assert result["prediction"].iloc[-1]["copykat.pred"] == PredictionLabel.NOT_DEFINED
        saved = pd.read_csv(f"{prefix}prediction.txt", sep="\t")
        pd.testing.assert_frame_equal(saved, result["prediction"])
    annotation_columns = 3 if genome is Genome.HG20 else 7
    assert result["CNAmat"].shape == (100, annotation_columns + 12)
    assert result["CNAmat"].columns[annotation_columns:].tolist() == [f"cell{i}" for i in range(12)]
    with open(f"{prefix}runtime.json") as report:
        assert json.load(report) == result["runtime"]
    with open(f"{prefix}clustering_results.pkl", "rb") as stream:
        saved_clustering = pickle.load(stream)
    np.testing.assert_array_equal(saved_clustering["labels"], result["hclustering"]["labels"])
    assert result["runtime"]["parameters"]["UP_DR"] == 0.1
    assert result["runtime"]["parameters"]["UP_DR_effective"] == 0.05
    assert (
        result["runtime"]["reference"]["mode"]
        == {
            "known": "known_normal",
            "auto": "automatic",
            "synthetic": "synthetic",
            "markers": "automatic",
        }[reference]
    )
    plot.assert_called_once()
    assert plot.call_args.kwargs["genome"] is genome
    annotated_plot.assert_called_once()
    assert annotated_plot.call_args.kwargs["genome"] is genome
    assert annotated_plot.call_args.kwargs["cell_names"] == [f"cell{i}" for i in range(12)]
    assert bool(list(Path(tmp_path).glob("*.seg"))) == (genome is Genome.HG20)


def test_arm_calls_survive_baseline_adjustment(tmp_path, pipeline_stubs) -> None:
    result = workflow.copykat(
        input_counts(),
        min_gene_per_cell=5,
        ngene_chr=1,
        norm_cell_names=[f"cell{i}" for i in range(6)],
        final_call=FinalCallStrategy.ARM_CORRELATION,
        plot_genes=False,
        sam_name=str(tmp_path / "arm"),
    )
    assert result["runtime"]["final_call_path"] == "arm_correlation"
    assert result["prediction"]["copykat.pred"].tolist() == (
        [PredictionLabel.DIPLOID] * 6 + [PredictionLabel.ANEUPLOID] * 6 + [PredictionLabel.NOT_DEFINED]
    )


def test_transform_retains_backing_array_and_fortran_layout() -> None:
    counts = np.asfortranarray([[0.0, 4.0], [1.0, 9.0], [2.0, 0.0]])
    expected = np.log(np.sqrt(counts) + np.sqrt(counts + 1))
    expected -= expected.mean(axis=0, keepdims=True)
    transformed, genes, cells = preprocessing.transform_counts_inplace(
        counts,
        np.array([1, 1, 2]),
        detection_rate=0.75,
        ngene_chr=1,
        runtime_info={"steps": []},
    )
    assert transformed is counts
    assert transformed.flags.f_contiguous
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(genes, [False, True, False])
    np.testing.assert_array_equal(cells, [False, False])
