"""Runtime reports and progress timing for the CopyKAT pipeline."""

import json
import logging
import os
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from copykat_py._types import Genome, ParallelInfo, RuntimeInfo
from copykat_py.baseline import (
    AUTO_PCA_CELL_COUNT_CUTOFF,
    AUTO_PCA_LARGE_SAMPLE,
    AUTO_PCA_SMALL_SAMPLE,
    MOUSE_AUTO_PCA_LARGE_SAMPLE,
    MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF,
    MOUSE_AUTO_PCA_MEDIUM_SAMPLE,
    MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF,
    MOUSE_AUTO_PCA_SMALL_SAMPLE,
    resolve_adaptive_pca_components,
)

logger = logging.getLogger(__name__)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"


def _record_step(
    runtime_info: RuntimeInfo,
    step: str,
    start_time: float,
    parallel_info: ParallelInfo | None = None,
    extra: Mapping[str, Any] | None = None,
) -> float:
    elapsed = time.perf_counter() - start_time
    entry = {"step": step, "seconds": round(float(elapsed), 4)}
    if parallel_info:
        entry["parallel"] = bool(parallel_info.get("parallel", False))
        entry["requested_cores"] = int(parallel_info.get("requested_cores", 1))
        entry["effective_cores"] = int(parallel_info.get("effective_cores", 1))
        for key in ("tasks", "chunk_size", "mc_samples", "engine", "approximate"):
            if key in parallel_info:
                entry[key] = parallel_info[key]
    if extra:
        entry.update(extra)
    runtime_info["steps"].append(entry)
    return elapsed


def new_runtime_info(sample_name: str, n_cores: int, parameters: dict[str, Any]) -> RuntimeInfo:
    """Create the mutable report; parameters describe the requested settings."""
    runtime_info: RuntimeInfo = {
        "sample_name": sample_name,
        "requested_cores": int(n_cores),
        "available_cores": int(os.cpu_count() or 1),
        "steps": [],
        "parameters": parameters,
        "versions": {},
        "warnings": [],
    }
    for package in ("copykat-py", "numpy", "scipy", "pandas", "scikit-learn", "fastcluster"):
        try:
            runtime_info["versions"][package] = version(package)
        except PackageNotFoundError:
            runtime_info["versions"][package] = "unavailable (source checkout or package not installed)"

    return runtime_info


def select_pca_components(
    input_cell_count: int,
    *,
    requested: int | None,
    genome: Genome,
    runtime_info: RuntimeInfo,
) -> int:
    """Resolve the adaptive PCA cap and record the selection rule."""
    selected_pca_components = resolve_adaptive_pca_components(
        input_cell_count,
        pca_components=requested,
        genome=genome,
    )
    runtime_info["pca_components"] = int(selected_pca_components)
    runtime_info["pca_selection_mode"] = "manual" if requested is not None else "auto_by_input_cell_count"
    runtime_info["pca_selection_genome"] = str(genome)
    runtime_info["pca_selection_input_cells"] = input_cell_count
    if genome is Genome.MM10:
        runtime_info["pca_selection_rule"] = (
            f"<{MOUSE_AUTO_PCA_SMALL_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_SMALL_SAMPLE},"
            f"<{MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_MEDIUM_SAMPLE},"
            f">={MOUSE_AUTO_PCA_MEDIUM_CELL_COUNT_CUTOFF}->{MOUSE_AUTO_PCA_LARGE_SAMPLE}"
        )
    else:
        runtime_info["pca_selection_rule"] = (
            f"<{AUTO_PCA_CELL_COUNT_CUTOFF}->{AUTO_PCA_SMALL_SAMPLE},"
            f">={AUTO_PCA_CELL_COUNT_CUTOFF}->{AUTO_PCA_LARGE_SAMPLE}"
        )
    return selected_pca_components


def finish_runtime_report(runtime_info: RuntimeInfo, sample_name: str, start_time: float) -> None:
    """Write the completed timing report after all outputs have been generated."""
    runtime_info["total_seconds"] = round(time.perf_counter() - start_time, 4)
    with open(f"{sample_name}runtime.json", "w", encoding="utf-8") as report:
        json.dump(runtime_info, report, indent=2)
    logger.info(f"Done. Elapsed time: {_format_seconds(runtime_info['total_seconds'])}")
    logger.info(f"Runtime report saved to: {sample_name}runtime.json")
