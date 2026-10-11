"""Shared fixtures for the synthetic-truth tests.

Every test records the numbers it asserts on through the ``record`` fixture.
When ``SYNTHETIC_METRICS_JSON`` names a file, they are written there at the
end of the session, so CI can publish them and compare a branch against its
base (see ``report.py``).
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from metrics import Run, run_copykat
from synthetic_data import SimulationConfig, SyntheticDataset, simulate

from copykat_py._types import Genome

_METRICS: dict[str, Any] = {}
GENOMES = [Genome.HG20, Genome.MM10]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(pytest.mark.synthetic)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "synthetic: end-to-end runs on simulated data with planted copy-number truth")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    target = os.environ.get("SYNTHETIC_METRICS_JSON")
    if not target or not _METRICS:
        return
    import copykat_py

    payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "copykat_py": str(Path(copykat_py.__file__).parent),
            "exit_status": int(exitstatus),
        },
        "metrics": _METRICS,
    }
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@pytest.fixture(scope="session", params=GENOMES, ids=str)
def genome(request: pytest.FixtureRequest) -> Genome:
    """Every synthetic test runs once per reference genome."""
    return request.param


@pytest.fixture(scope="session")
def record(genome: Genome):
    """``record(section, key, value)`` keeps a metric for the CI summary, filed under the genome."""

    def _record(section: str, key: str, value: Any) -> None:
        _METRICS.setdefault(f"{genome}/{section}", {})[key] = value

    return _record


@pytest.fixture(scope="session")
def dataset(genome: Genome) -> SyntheticDataset:
    return simulate(SimulationConfig(genome=genome))


@pytest.fixture(scope="session")
def runs_dir(genome: Genome, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp(f"copykat_runs_{genome}")


@pytest.fixture(scope="session")
def baseline_run(dataset: SyntheticDataset, runs_dir: Path) -> Run:
    """The pipeline on the default dataset with automatic reference selection, run once per genome."""
    return run_copykat(dataset, runs_dir / "baseline")
