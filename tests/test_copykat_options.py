"""Regression checks for segmentation retries and genome-specific plotting."""

import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from copykat_py._types import BaselineWarning, DataQualityStatus, DistanceMetric, Genome, KSMethod
from copykat_py.segmentation import SegmentationResult

segmentation = importlib.import_module("copykat_py._pipeline.segmentation")
output = importlib.import_module("copykat_py._pipeline.output")


@pytest.mark.parametrize("success_attempt", [1, 2, 3, None])
def test_segmentation_retry_cutoffs_and_early_stop(monkeypatch, success_attempt) -> None:
    labels = np.array([1, 2])
    values = np.zeros((30, 2))
    options = segmentation._SegmentationOptions(window_size=25, ks_cutoff=0.1, ks_method=KSMethod.EXACT, n_cores=2)
    attempts = 3 if success_attempt is None else success_attempt
    results = [
        SegmentationResult(values, list(range(25 if attempt == success_attempt else 24)))
        for attempt in range(1, attempts + 1)
    ]
    segment = Mock(side_effect=results)
    monkeypatch.setattr(segmentation, "cna_mcmc", segment)

    if success_attempt is None:
        with pytest.raises(ValueError, match="Too few segments"):
            segmentation._segment_with_retries(labels, values, options=options)
    else:
        assert segmentation._segment_with_retries(labels, values, options=options) is results[-1]

    assert segment.call_count == attempts
    assert [call.kwargs["cut_cor"] for call in segment.call_args_list] == [0.1, 0.05, 0.025][:attempts]
    for call in segment.call_args_list:
        assert call.args[0] is labels
        assert call.args[1] is values
        assert call.kwargs["bins"] == 25
        assert call.kwargs["ks_method"] is KSMethod.EXACT
        assert call.kwargs["n_cores"] == 2


@pytest.mark.parametrize("genome", [Genome.HG20, Genome.MM10])
def test_heatmap_preserves_genome_and_data_references(monkeypatch, genome) -> None:
    plotting = importlib.import_module("copykat_py.plotting")
    plot = Mock()
    monkeypatch.setattr(plotting, "plot_heatmap", plot)
    values = np.zeros((30, 2))
    chromosomes = np.ones(30)
    predictions = {"cell": "diploid"}
    options = output._HeatmapOptions(
        sample_name="sample",
        distance=DistanceMetric.EUCLIDEAN,
        n_cores=2,
        genome=genome,
        output_path="heatmap.png",
    )

    output._run_plot_heatmap(
        values, chromosomes, predictions, DataQualityStatus.LOW, BaselineWarning.UNCLASSIFIED, options=options
    )

    plot.assert_called_once()
    assert plot.call_args.args[0] is values
    assert plot.call_args.args[1] is chromosomes
    assert plot.call_args.kwargs == {
        "predictions": predictions,
        "sample_name": "sample",
        "distance": DistanceMetric.EUCLIDEAN,
        "n_cores": 2,
        "WNS1": DataQualityStatus.LOW,
        "WNS": BaselineWarning.UNCLASSIFIED,
        "output_path": "heatmap.png",
        "genome": genome,
    }
    assert plot.call_args.kwargs["predictions"] is predictions


@pytest.mark.parametrize("genome, expected", [(Genome.HG20, ["20", "21"]), (Genome.MM10, ["X", "Y"])])
def test_annotated_heatmap_uses_explicit_genome(monkeypatch, tmp_path, genome, expected) -> None:
    import pandas as pd

    plotting = importlib.import_module("copykat_py.plotting")
    add_labels = plotting._add_chr_labels
    observed = []

    def capture_labels(ax, chrom_info, **kwargs):
        add_labels(ax, chrom_info, **kwargs)
        observed.extend(text.get_text() for text in ax.texts)

    monkeypatch.setattr(plotting, "_add_chr_labels", capture_labels)
    monkeypatch.setattr(plotting.plt, "savefig", Mock())
    metadata = tmp_path / "meta.csv"
    pd.DataFrame({"cell": ["a", "b"], "group": ["one", "two"]}).to_csv(metadata, index=False)
    plotting.plot_heatmap_annotated(
        np.zeros((4, 2)),
        ["a", "b"],
        np.array([20, 20, 21, 21]),
        str(metadata),
        plotting.AnnotatedHeatmapOptions(output_path=str(tmp_path / "plot.png")),
        genome=genome,
    )
    assert observed == expected
