"""Lightweight loading of run metadata and predictions (never the CNA matrix)."""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunReport:
    """A completed run and the files belonging to its sample prefix."""

    sample: str
    runtime: dict[str, Any]
    predictions: dict[str, int] | None
    files: tuple[Path, ...]
    images: tuple[Path, ...]


def load_report(run_dir: str | Path, sample_name: str | None = None) -> RunReport:
    """Select one runtime file, requiring a sample when the directory has several.

    Legacy runtime files are supported; absent metadata is left unavailable.
    Only existing, sample-prefixed analysis files are included. Prediction labels
    are preserved, including undefined and low-confidence labels.
    """
    directory = Path(run_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    if sample_name is not None:
        if Path(sample_name).name != sample_name or sample_name in {".", ".."}:
            raise ValueError("Sample name must be a filename prefix, not a path")
        candidates = [directory / f"{sample_name}_copykat_runtime.json"]
    else:
        candidates = sorted(directory.glob("*_copykat_runtime.json"))
    if not candidates or not candidates[0].is_file():
        raise ValueError("No runtime JSON found; supply a completed CopyKAT run directory")
    if len(candidates) != 1:
        raise ValueError("Multiple runs found; select one with --sample-name")
    runtime_path = candidates[0]
    with runtime_path.open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    if not isinstance(runtime, dict) or not isinstance(runtime.get("steps"), list):
        raise ValueError("Runtime JSON must contain an object with a steps list")
    if any(not isinstance(step, dict) for step in runtime["steps"]):
        raise ValueError("Every runtime step must be an object")
    for key in ("parameters", "versions", "reference"):
        if key in runtime and not isinstance(runtime[key], dict):
            raise ValueError(f"Runtime {key} must be an object")
    if "warnings" in runtime and (
        not isinstance(runtime["warnings"], list) or any(not isinstance(note, str) for note in runtime["warnings"])
    ):
        raise ValueError("Runtime warnings must be a list of strings")
    sample = runtime_path.name.removesuffix("_copykat_runtime.json")
    prefix = f"{sample}_copykat_"
    prediction_path = directory / f"{prefix}prediction.txt"
    predictions = None
    if prediction_path.is_file():
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"cell.names", "copykat.pred"}.issubset(reader.fieldnames or []):
                raise ValueError("Prediction table requires cell.names and copykat.pred columns")
            counts: Counter[str] = Counter()
            for row in reader:
                label = row.get("copykat.pred")
                if not row.get("cell.names") or not label or None in row:
                    raise ValueError("Malformed prediction row: expected a cell name and prediction")
                counts[label] += 1
            predictions = dict(sorted(counts.items()))
    # Explicit inventory avoids including stale reports or another sample's files.
    suffixes = (
        "runtime.json",
        "prediction.txt",
        "CNA_results.txt",
        "CNA_raw_results_gene_by_cell.txt",
        "clustering_results.pkl",
        "heatmap.png",
        "annotated_heatmap.png",
        "meta_with_pred.csv",
        "CNA_results.seg",
    )
    files = tuple(directory / f"{prefix}{suffix}" for suffix in suffixes if (directory / f"{prefix}{suffix}").is_file())
    images = tuple(path for path in files if path.suffix == ".png")
    return RunReport(sample, runtime, predictions, files, images)
