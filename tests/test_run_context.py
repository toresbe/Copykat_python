"""Public keyword binding, configuration normalization, and stage boundaries."""

import importlib
from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock

import numpy as np
import pytest

from copykat_py import CopyKATArguments, RunContext, copykat
from copykat_py._types import (
    AnchorStrategy,
    CellLineMode,
    DistanceMetric,
    ExecutionBackend,
    GeneIdType,
    Genome,
    KSMethod,
)

workflow = importlib.import_module("copykat_py.copykat")


def test_public_keywords_match_context_constructor() -> None:
    configurable_fields = {f.name for f in fields(RunContext) if f.init}
    assert configurable_fields == set(CopyKATArguments.__optional_keys__)
    assert not CopyKATArguments.__required_keys__


def test_defaults_preserve_requested_and_resolved_settings() -> None:
    context = RunContext()
    assert context.genome is Genome.HG20
    assert context.random_seed == 1234
    assert context.sample_name == "_copykat_"
    assert context.pca_components is None
    assert context.baseline_options(256).pca_components == 256
    assert context.prediction_options(256).pca_components == 256
    assert context.pca_components is None
    assert context.filtering_options.upper_detection_rate == 0.1
    assert context.segmentation_options.ks_cutoff == 0.1
    with pytest.raises(FrozenInstanceError):
        context.sam_name = "changed"


def test_string_options_are_normalized_without_selecting_backend(monkeypatch) -> None:
    activate = Mock()
    monkeypatch.setattr(workflow.backend, "set_backend", activate)
    context = RunContext(
        id_type="Ensembl",
        cell_line="yes",
        distance="spearman",
        genome="mm10",
        backend_name="cpu",
        ks_method="exact",
        anchor="markers",
        sam_name="sample",
    )
    assert context.id_type is GeneIdType.ENSEMBL
    assert context.cell_line is CellLineMode.YES
    assert context.distance is DistanceMetric.SPEARMAN
    assert context.genome is Genome.MM10
    assert context.backend_name is ExecutionBackend.CPU
    assert context.ks_method is KSMethod.EXACT
    assert context.anchor is AnchorStrategy.MARKERS
    assert context.output_options.sample_name == "sample_copykat_"
    activate.assert_not_called()


def test_context_does_not_copy_or_report_reference_names() -> None:
    class ReferenceNames(list):
        def __deepcopy__(self, memo):
            raise AssertionError("Reference names should not be copied")

    names = ReferenceNames(["normal1", "normal2"])
    context = RunContext(norm_cell_names=names, n_cores=3, pca_components=64)
    assert context.norm_cell_names is names
    report = context.runtime_parameters()
    assert "norm_cell_names" not in report
    assert "sam_name" not in report
    assert "backend_name" not in report
    assert "pca_components" not in report
    assert report["backend"] is ExecutionBackend.CPU
    assert report["pca_components_requested"] == 64
    assert report["n_cores"] == 3
    assert report["random_seed"] == 1234
    assert "rawmat" not in {f.name for f in fields(context)}
    assert not hasattr(context.output_options, "norm_cell_names")


@pytest.mark.parametrize(
    "options, exception, message",
    [
        ({"n_core": 2}, TypeError, "n_core"),
        ({"random_seed": 42}, TypeError, "random_seed"),
        ({"genome": "unknown"}, ValueError, "unknown"),
        ({"genome": "mm10", "final_call": "arm_correlation"}, ValueError, "only supported for hg20"),
    ],
)
def test_invalid_keywords_fail_before_backend_or_input_work(monkeypatch, options, exception, message) -> None:
    activate = Mock()
    prepare = Mock()
    monkeypatch.setattr(workflow.backend, "set_backend", activate)
    monkeypatch.setattr(workflow, "_prepare_input_matrix", prepare)
    with pytest.raises(exception, match=message):
        copykat(np.zeros((2, 2)), **options)
    activate.assert_not_called()
    prepare.assert_not_called()


def test_options_after_matrix_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        copykat(np.zeros((2, 2)), GeneIdType.SYMBOL)
