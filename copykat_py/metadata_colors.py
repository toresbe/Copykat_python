"""Continuous metadata colors for annotated heatmaps."""

from dataclasses import dataclass

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import colormaps

from copykat_py._types import FloatArray


@dataclass(frozen=True)
class ContinuousAnnotation:
    colors: FloatArray
    norm: mcolors.Normalize
    ticks: list[float]
    has_missing: bool


def is_continuous(values: pd.Series) -> bool:
    """Detect numeric measurements while retaining small integer-coded categories."""
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        return False
    finite = values.to_numpy(dtype=float, na_value=np.nan)
    finite = finite[np.isfinite(finite)]
    return bool(len(finite) and (len(np.unique(finite)) > 20 or np.any(finite != np.floor(finite))))


def continuous_annotation(values: pd.Series) -> ContinuousAnnotation:
    """Map finite measurements to viridis, with missing/nonfinite values in grey.

    A constant measurement gets one centered color and one labeled tick. Missing
    metadata never expands the numeric range.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite_mask = np.isfinite(numeric)
    finite = numeric[finite_mask]
    ticks = []
    if len(finite):
        low, high = float(finite.min()), float(finite.max())
        ticks = [low] if low == high else [low, high]
        if low == high:
            margin = max(abs(low) * 1e-6, 0.5)
            low, high = low - margin, high + margin
    else:
        low, high = 0.0, 1.0
    norm = mcolors.Normalize(vmin=low, vmax=high)
    colors = np.full((len(numeric), 1, 3), mcolors.to_rgb("#cccccc"), dtype=np.float32)
    colors[finite_mask, 0, :] = colormaps["viridis"](norm(numeric[finite_mask]))[:, :3]
    return ContinuousAnnotation(colors, norm, ticks, bool((~finite_mask).any()))
