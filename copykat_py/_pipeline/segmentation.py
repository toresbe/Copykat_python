"""Segmentation options and retry policy for the CopyKAT pipeline."""

import logging
from dataclasses import dataclass

from copykat_py import _stages, profiles
from copykat_py._types import ClusterLabels, FloatArray, KSMethod
from copykat_py.rcompat import segmentation as r_segmentation
from copykat_py.rcompat.rng import RRandom
from copykat_py.segmentation import SegmentationResult, cna_mcmc

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SegmentationOptions:
    """Requested segmentation settings; retry cutoffs are derived from these.

    Attributes
    ----------
    window_size : int
        Number of genes per window for MCMC segmentation; corresponds to the
        public ``win_size`` argument and the segmenter's ``bins`` argument.
    ks_cutoff : float
        Requested KS breakpoint threshold, from 0 to 1; corresponds to
        ``KS_cut``. If fewer than 25 breakpoints are found, retries use 50%
        and then 25% of this value without changing the requested setting.
    ks_method : KSMethod
        ``MONTE_CARLO`` compares sampled posterior distributions; ``EXACT``
        computes the posterior-Gamma KS statistic directly.
    n_cores : int
        Maximum number of CPU workers requested for segmentation.
    """

    window_size: int
    ks_cutoff: float
    ks_method: KSMethod
    n_cores: int


def _segment_with_retries(
    cluster_labels: ClusterLabels,
    relative_expression: FloatArray,
    *,
    options: _SegmentationOptions,
) -> SegmentationResult:
    """Retry sparse breakpoint results without copying the input matrix."""
    profile = profiles.active()
    # copykat() calls set.seed(1234) first and nothing draws before step 5.
    r_stream = RRandom(1234) if profile.segmentation == "r" else None
    for attempt, fraction in enumerate((1.0, 0.5, 0.25), start=1):
        if fraction < 1.0:
            logger.info(f"  too few breakpoints; decreased KS_cut to {fraction:.0%}")
        if r_stream is not None:
            result = r_segmentation.cna_mcmc(
                cluster_labels,
                relative_expression,
                bins=options.window_size,
                cut_cor=fraction * options.ks_cutoff,
                rng=r_stream,
                replay=profile.segment_values == "r_replay",
                burn_if_retry=attempt < 3,
                n_cores=options.n_cores,
                out_dtype=profiles.storage_dtype(),
            )
        else:
            result = cna_mcmc(
                cluster_labels,
                relative_expression,
                bins=options.window_size,
                cut_cor=fraction * options.ks_cutoff,
                n_cores=options.n_cores,
                ks_method=options.ks_method,
            )
        _stages.save(
            f"segmentation_{attempt}",
            {
                "breaks": [b + 1 for b in result.breakpoints],  # 1-based, as in R
                "logCNA": result.log_cna,
                "cut.cor": [fraction * options.ks_cutoff],
            },
        )
        if len(result.breakpoints) >= 25:
            return result
        # Release this attempt's large result before allocating the next one.
        del result
    raise ValueError("Too few segments; try decreasing KS_cut or improving data quality")
