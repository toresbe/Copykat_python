"""Optional dumps of intermediate results, for stage-by-stage comparison with CopyKAT-R.

Nothing is written unless a directory is configured (``copykat(..., dump_stages=DIR)``
or ``copykat-py --dump-stages DIR``). The layout matches what
``verify/r/run_copykat.R`` writes for R (see ``verify/stages.md``): one directory per
value, numbers as raw little-endian float64 in ``values.f64``, text one entry per line
in ``values.txt``, the shape in ``shape.txt``, and row/column names alongside.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_DIRECTORY: list[Path | None] = [None]


def configure(directory: str | Path | None) -> None:
    """Write stages under ``directory`` from now on; ``None`` turns dumping off."""
    _DIRECTORY[0] = None if directory is None else Path(directory)


def enabled() -> bool:
    return _DIRECTORY[0] is not None


def save(
    path: str,
    value: Any,
    *,
    rownames: Sequence[str] | None = None,
    colnames: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> None:
    """Save ``value`` as stage ``path`` (a no-op unless dumping is configured).

    Mappings become subdirectories. Matrices are written column-major, as R stores them.
    """
    root = _DIRECTORY[0]
    if root is None or value is None:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            save(f"{path}/{key}", item)
        return
    target = root / path
    target.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        (target / "values.txt").write_text("".join(f"{v}\n" for v in array.ravel(order="F")))
    else:
        array.astype("<f8").ravel(order="F").tofile(target / "values.f64")
    (target / "shape.txt").write_text(" ".join(str(n) for n in (array.shape or (1,))) + "\n")
    for filename, labels in (("rownames.txt", rownames), ("colnames.txt", colnames), ("names.txt", names)):
        if labels is not None:
            (target / filename).write_text("".join(f"{label}\n" for label in labels))


def load(directory: str | Path) -> Any:
    """Read a stage written by :func:`save` or by ``run_copykat.R``.

    Returns a dict for a directory of stages, otherwise a dict with ``values``
    (a NumPy array, Fortran-ordered to the saved shape) and any saved names.
    """
    directory = Path(directory)
    shape_file = directory / "shape.txt"
    if not shape_file.exists():
        return {child.name: load(child) for child in sorted(directory.iterdir()) if child.is_dir()}
    shape = tuple(int(n) for n in shape_file.read_text().split())
    if (directory / "values.txt").exists():
        values: np.ndarray = np.array((directory / "values.txt").read_text().splitlines(), dtype=object)
    else:
        values = np.fromfile(directory / "values.f64", dtype="<f8")
    stage: dict[str, Any] = {"values": values.reshape(shape, order="F")}
    for key in ("rownames", "colnames", "names"):
        file = directory / f"{key}.txt"
        if file.exists():
            stage[key] = file.read_text().splitlines()
    return stage
