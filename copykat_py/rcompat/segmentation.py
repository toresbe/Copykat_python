"""Step 5 as CopyKAT-R's ``CNA.MCMC`` runs it with ``n.cores = 1``, draw for draw.

``copykat()`` calls ``set.seed(1234)`` first, and nothing draws a random number
before step 5, so step 5 starts from that seed. For each cluster in turn (by number,
``min(clu):max(clu)``), R walks the consensus profile in windows of ``bins`` genes and
compares the Poisson-Gamma posteriors of neighbouring windows with 1,000 draws each
and ``ks.test``; a window boundary is a breakpoint when the KS statistic is above
``cut.cor``. Then, for every cell and every segment, it draws 1,000 more values and
keeps their mean.

The breakpoints here use the same generator (``rng.py``), the same arithmetic for the
posterior shapes (``rstats.py``), the same KS statistic and the same draw order, so
they are R's breakpoints. For the per-cell values, the `r` profiles replay R's
1,000 draws (``segment_values = "r_replay"``); ``"closed_form"`` instead computes
the posterior mean that those draws estimate. When too few breakpoints
are found R retries with a lower cut, after having made every per-cell draw, so the
generator is advanced past them (``burn``) before the retry.
"""

import logging
import math

import numpy as np
from numba import njit, prange

from copykat_py._medians import group_medians
from copykat_py._types import ClusterLabels, GeneByCell
from copykat_py.rcompat.rng import RRandom, rgamma_one
from copykat_py.rcompat.rstats import ks_statistic, r_mean, r_sum
from copykat_py.segmentation import SegmentationResult

logger = logging.getLogger(__name__)

# Cells per block when computing the per-cell posterior shapes in long double.
_SHAPE_BLOCK_CELLS = 256


@njit(parallel=True, cache=True)
def libm_exp(x):  # type: ignore[no-untyped-def]
    """``exp`` from the C library, as R uses (NumPy's SIMD ``exp`` can differ in the last bit)."""
    flat = x.ravel()
    out = np.empty(flat.shape[0])
    for i in prange(flat.shape[0]):
        out[i] = math.exp(flat[i])
    return out.reshape(x.shape)


@njit(parallel=True, cache=True)
def libm_log(x):  # type: ignore[no-untyped-def]
    """``log`` from the C library, as R uses."""
    flat = x.ravel()
    out = np.empty(flat.shape[0])
    for i in prange(flat.shape[0]):
        out[i] = math.log(flat[i])
    return out.reshape(x.shape)


def _window_bounds(n: int, bins: int) -> list[int]:
    """R's ``breks``, 0-based: ``c(seq(1, as.integer(n/bins - 1) * bins, bins), n)``."""
    return [*range(0, (n // bins - 1) * bins, bins), n - 1]


def _posterior_shape(y: np.ndarray) -> tuple[float, float]:
    """``MCpoissongamma(y, max(mean(y), 0.001), 1)``: shape ``alpha + sum(y)`` and rate ``1 + n``."""
    alpha = max(r_mean(y), 0.001)
    return alpha + r_sum(y), 1.0 + len(y)


def find_breakpoints(consensus: np.ndarray, bins: int, cut_cor: float, rng: RRandom) -> list[int]:
    """Breakpoints of one exponentiated consensus profile (0-based gene indices)."""
    breks = _window_bounds(len(consensus), bins)
    found = []
    for i in range(len(breks) - 2):
        shape1, rate1 = _posterior_shape(consensus[breks[i] : breks[i + 1] + 1])
        posterior1 = rng.rgamma(1000, shape1, rate1)
        shape2, rate2 = _posterior_shape(consensus[breks[i + 1] + 1 : breks[i + 2] + 1])
        posterior2 = rng.rgamma(1000, shape2, rate2)
        if ks_statistic(posterior1, posterior2) > cut_cor:
            found.append(breks[i + 1])
    return found


def _segment_shapes(fttmat: GeneByCell, BR: list[int], lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Posterior shapes and rates of every segment for cells ``lo:hi`` (segments x cells)."""
    values = libm_exp(np.asarray(fttmat[:, lo:hi], dtype=np.float64)).astype(np.longdouble)
    shapes = np.empty((len(BR) - 1, hi - lo))
    rates = np.empty(len(BR) - 1)
    for s in range(len(BR) - 1):
        y = values[BR[s] : BR[s + 1] + 1]
        n = y.shape[0]
        total = np.cumsum(y, axis=0)[-1]
        mean = total / n
        mean = mean + np.cumsum(y - mean, axis=0)[-1] / n
        alpha = np.maximum(mean.astype(np.float64), 0.001)
        shapes[s] = alpha + total.astype(np.float64)
        rates[s] = 1.0 + n
    return shapes, rates


@njit(cache=True)
def _draw_segments(mt, mti, shapes, rates, mc, out_means):  # type: ignore[no-untyped-def]
    """R's per-cell loop: ``mc`` draws per segment, cells in order; keeps each mean."""
    draws = np.empty(mc)
    for c in range(shapes.shape[1]):
        for s in range(shapes.shape[0]):
            scale = 1.0 / rates[s]
            for k in range(mc):
                draws[k] = rgamma_one(mt, mti, shapes[s, c], scale)
            # R's mean(): a sum, then a correcting second pass
            total = 0.0
            for k in range(mc):
                total += draws[k]
            m = total / mc
            correction = 0.0
            for k in range(mc):
                correction += draws[k] - m
            out_means[s, c] = m + correction / mc


def cna_mcmc(
    clu: ClusterLabels,
    fttmat: GeneByCell,
    *,
    bins: int,
    cut_cor: float,
    rng: RRandom,
    replay: bool,
    burn_if_retry: bool = True,
    n_cores: int = 1,
    out_dtype: type = np.float64,
) -> SegmentationResult:
    """``CNA.MCMC(clu, fttmat, bins, cut.cor, n.cores = 1)``.

    ``rng`` is advanced exactly as R's stream is. With ``replay`` the per-cell values
    are R's 1,000-draw means; otherwise they are the exact posterior means, and the
    per-cell draws are only made (and discarded) when a retry will follow
    (fewer than 25 breakpoints), so that the retry starts where R's does.
    """
    n_genes, n_cells = fttmat.shape
    clu = np.asarray(clu)
    lowest, highest = int(np.min(clu)), int(np.max(clu))
    labels = [i for i in range(lowest, highest + 1) if np.any(clu == i)]
    if len(labels) != highest - lowest + 1:
        logger.warning("  a step-4 cluster lost all its cells to filtering; CopyKAT-R would stop here")
    consensus = np.column_stack(group_medians(fttmat, [clu == i for i in labels], n_cores))
    consensus = libm_exp(np.asarray(consensus, dtype=np.float64))

    breakpoints: set[int] = set()
    for c in range(consensus.shape[1]):
        bre = find_breakpoints(consensus[:, c], bins, cut_cor, rng)
        breakpoints.update({0, *bre, n_genes - 1})
    BR = sorted(breakpoints)

    retry = len(BR) < 25
    logCNA: np.ndarray = np.empty((n_genes, n_cells), dtype=out_dtype)
    for lo in range(0, n_cells, _SHAPE_BLOCK_CELLS):
        hi = min(n_cells, lo + _SHAPE_BLOCK_CELLS)
        shapes, rates = _segment_shapes(fttmat, BR, lo, hi)
        if replay or (retry and burn_if_retry):
            means = np.empty_like(shapes)
            _draw_segments(rng.mt, rng.mti, shapes, rates, 1000, means)
            if not replay:
                means = shapes / rates[:, None]
        else:
            means = shapes / rates[:, None]
        # x[BR[i]:BR[i + 1]] <- value: later segments overwrite the shared boundary gene
        log_means = libm_log(means)
        for s in range(len(BR) - 1):
            logCNA[BR[s] : BR[s + 1] + 1, lo:hi] = log_means[s]
    return SegmentationResult(log_cna=logCNA, breakpoints=BR)
