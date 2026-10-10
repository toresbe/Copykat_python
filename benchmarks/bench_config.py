"""Shared locations and interpreters for the benchmark scripts.

Everything is configurable through environment variables; the defaults work
from a fresh checkout without editing any script.

COPYKAT_BENCH_ROOT          performance working directory (snapshots, results, scratch)
COPYKAT_BENCH_ACCURACY_ROOT accuracy-study working directory
COPYKAT_BENCH_DATA          input data (README samples, Xenium, 3CA, ScPCA, ...)
COPYKAT_BENCH_PYTHON        interpreter used for CPU runs and reports
COPYKAT_BENCH_GPU_PYTHON    interpreter used for GPU runs (default: COPYKAT_BENCH_PYTHON)
COPYKAT_BENCH_ARCHIVE       where evidence tarballs are written
COPYKAT_BENCH_XENIUM        Xenium cell_feature_matrix.h5
COPYKAT_BENCH_SAMPLES       README samples directory
COPYKAT_BENCH_CACHE         parsed-matrix cache
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOCS = REPO / "docs"

_BASE = Path(os.getenv("COPYKAT_BENCH_HOME", Path.home() / "copykat_bench"))
ROOT = Path(os.getenv("COPYKAT_BENCH_ROOT", _BASE / "benchmark"))
ACCURACY_ROOT = Path(os.getenv("COPYKAT_BENCH_ACCURACY_ROOT", _BASE / "accuracy"))
DATA = Path(os.getenv("COPYKAT_BENCH_DATA", _BASE / "data"))
ARCHIVE_DIR = Path(os.getenv("COPYKAT_BENCH_ARCHIVE", _BASE / "archive"))
PYTHON = os.getenv("COPYKAT_BENCH_PYTHON", sys.executable)
GPU_PYTHON = os.getenv("COPYKAT_BENCH_GPU_PYTHON", PYTHON)
XENIUM_H5 = Path(os.getenv("COPYKAT_BENCH_XENIUM", DATA / "xenium/cell_feature_matrix.h5"))
SAMPLES_DIR = Path(os.getenv("COPYKAT_BENCH_SAMPLES", DATA / "copykat_readme_data/samples"))
CACHE_DIR = Path(os.getenv("COPYKAT_BENCH_CACHE", _BASE / "cache"))


def archive_path(name):
    """Path for an evidence tarball, creating the archive directory."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR / name
