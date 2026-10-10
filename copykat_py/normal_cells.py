"""Utilities for resolving user-supplied normal-cell identifiers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def normal_cells_to_names(
    normal_cells: str | bytes | Sequence[Any] | npt.NDArray[Any] | None,
    reference_cell_names: Sequence[str],
) -> set[str]:
    """Resolve normal-cell names and integer indices against reference cells.

    Integer values are treated as zero-based indices. Out-of-range indices are
    ignored; other values are converted to strings and treated as cell names.
    """
    if normal_cells is None:
        return set()

    if isinstance(normal_cells, str | bytes):
        return {str(normal_cells)}

    values = list(normal_cells)
    if not values:
        return set()

    names = []
    for value in values:
        if isinstance(value, int | np.integer):
            idx = int(value)
            if 0 <= idx < len(reference_cell_names):
                names.append(reference_cell_names[idx])
        else:
            names.append(str(value))
    return set(names)
