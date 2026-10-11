"""Row medians over groups of columns, computed in parallel threads."""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from copykat_py._types import BoolArray, FloatArray, GeneByCell

# Rows per task. Small blocks keep every thread busy when one group is much
# larger than the others; NumPy releases the GIL while gathering and partitioning.
_BLOCK_ROWS = 64


def group_medians(mat: GeneByCell, groups: Sequence[BoolArray], n_cores: int = 1) -> list[FloatArray]:
    """``[np.median(mat[:, g], axis=1) for g in groups]``, with identical results.

    A median selects values rather than accumulating them, so splitting the
    rows into blocks cannot change any result. Blocks of every group run in
    one thread pool.
    """
    max_cores = int(os.getenv("COPYKAT_MAX_CORES", str(os.cpu_count() or 1)))
    n_threads = max(1, min(int(n_cores), max_cores))
    if n_threads == 1:
        return [np.median(mat[:, g], axis=1) for g in groups]

    n_rows = mat.shape[0]
    columns = [np.flatnonzero(g) for g in groups]
    # np.median's result dtype for this input (float32 stays float32)
    out = [np.empty(n_rows, dtype=np.median(mat[:1, cols], axis=1).dtype) for cols in columns]

    def _block(task: tuple[int, int]) -> None:
        i, lo = task
        out[i][lo : lo + _BLOCK_ROWS] = np.median(mat[lo : lo + _BLOCK_ROWS][:, columns[i]], axis=1)

    tasks = [(i, lo) for i in range(len(columns)) for lo in range(0, n_rows, _BLOCK_ROWS)]
    with ThreadPoolExecutor(n_threads) as executor:
        list(executor.map(_block, tasks))
    return out
