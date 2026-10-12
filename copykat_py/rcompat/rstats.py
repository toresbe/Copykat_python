"""R's own arithmetic for the few results that decide CopyKAT-R's random draws.

R's ``sum()`` and ``mean()`` accumulate in ``long double`` (80-bit on x86-64), and
``mean()`` adds a second, correcting pass. The posterior shape CopyKAT passes to
``rgamma`` is ``mean(y) + sum(y)``; a one-ulp difference there changes the draws
slightly, and can flip an accept/reject step and desynchronise the whole stream,
so these are reproduced exactly with NumPy's ``longdouble``.
"""

import numpy as np

if np.finfo(np.longdouble).nmant < 63:  # pragma: no cover - platform dependent
    import warnings

    warnings.warn(
        "numpy.longdouble is not 80-bit extended precision on this platform; R-compatible "
        "segmentation draws may differ from R in rare cases",
        stacklevel=2,
    )


def r_sum(x: np.ndarray) -> float:
    """R's ``sum(x)`` for doubles: a sequential long-double sum, rounded once."""
    return float(np.cumsum(np.asarray(x, dtype=np.longdouble))[-1]) if len(x) else 0.0


def r_mean(x: np.ndarray) -> float:
    """R's ``mean(x)`` for doubles (summary.c ``real_mean``)."""
    xl = np.asarray(x, dtype=np.longdouble)
    n = len(xl)
    s = np.cumsum(xl)[-1] / n
    if np.isfinite(s):
        t = np.cumsum(xl - s)[-1]
        s = s + t / n
    return float(s)


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """``ks.test(x, y)$statistic`` as R computes it on x86-64.

    R takes ``max(abs(cumsum(ifelse(order(w) <= n.x, 1/n.x, -1/n.y))))`` with a
    long-double ``cumsum``, keeping only positions where the sorted values change.
    """
    nx, ny = len(x), len(y)
    w = np.concatenate([x, y])
    order = np.argsort(w, kind="stable")
    steps = np.where(order < nx, 1.0 / nx, -1.0 / ny).astype(np.longdouble)
    z = np.cumsum(steps).astype(np.float64)  # R's cumsum rounds each long-double partial sum to double
    sorted_w = w[order]
    if len(np.unique(w)) < nx + ny:
        keep = np.concatenate([np.flatnonzero(np.diff(sorted_w) != 0), [nx + ny - 1]])
        z = z[keep]
    return float(np.max(np.abs(z)))
