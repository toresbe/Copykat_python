"""Numerics profiles: which implementation's results a run reproduces.

Most of this package's speed and memory work leaves results unchanged and is used by
every profile. Where it does change results, the choice is one field of
:class:`NumericsProfile`, and a profile is a set of choices:

``r`` (default)
    Reproduce CopyKAT-R as closely as possible, including its known quirks, so that a
    run can be checked against R step by step (``verify/``). ``r_version`` selects
    1.2.5 (default) or 1.1.0, which differ in the final call.
``experimental``
    This fork's fastest settings, including changes that move some calls.

``docs/profiles.md`` lists what each field changes and how ``verify/`` checks it.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericsProfile:
    """One choice per place where CopyKAT-R and this package's fast path differ."""

    name: str
    r_version: Literal["1.2.5", "1.1.0"] | None
    """CopyKAT-R release whose cluster-size rule and final call (steps 7-8) are used."""

    storage: Literal["float64", "float32"]
    """Precision of the smoothed, segmented and binned matrices (statistics are float64)."""

    input_filters: Literal["r", "fast"]
    """Step 1. ``r``: a single cell under ``min_gene_per_cell`` is kept, as R's
    ``sum(genes.raw < min.gene.per.cell) > 1`` does; ``fast`` removes it."""

    baseline: Literal["r", "fast"]
    """Step 4. ``r`` (``rcompat/baseline.py``): Ward on every gene, cutree's cluster numbers,
    mixtools' EM, ``sd()`` with n - 1, the silhouette over every cell, every cluster tied for
    the lowest sigma, and R's per-cell GMM fallback. ``fast``: PCA above 2,000 cells, a
    subsampled silhouette and a guarded fallback (``baseline.py``)."""

    min_cluster_cells: int
    """Step-4 clusters must have more than this many cells (R 1.2.5: 10; R 1.1.0: 5)."""

    segmentation: Literal["r", "fast"]
    """Step 5. ``r`` (``rcompat/segmentation.py``): R's generator from ``set.seed(1234)``,
    R's KS statistic, 1,000 draws, clusters in R's order, draw for draw as R with
    ``n.cores = 1``. ``fast``: NumPy draws, SciPy's KS, 500-1,000 draws."""

    segment_values: Literal["closed_form", "r_replay"]
    """Per-cell segment values: the exact posterior mean (R's 1,000-draw average converges to
    it), or R's own draws replayed (serial; matches an R run with ``n.cores = 1``)."""

    final_call: Literal["r1.2.5", "r1.1.0", "fast"]
    """Steps 7-8. ``r1.2.5``/``r1.1.0`` (``rcompat/final_call.py``): ``hclust(ward.D)`` on the
    full bins matrix with ``distance``, ``sd()`` with n - 1, and that release's rule for
    which cells are diploid. ``fast``: Ward (ward.D2) with PCA when wide, k = 2."""

    def replace(self, **changes: object) -> "NumericsProfile":
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]


R_1_2_5 = NumericsProfile(
    name="r",
    r_version="1.2.5",
    storage="float64",
    input_filters="r",
    baseline="r",
    min_cluster_cells=10,
    segmentation="r",
    segment_values="closed_form",
    final_call="r1.2.5",
)

R_1_1_0 = R_1_2_5.replace(r_version="1.1.0", min_cluster_cells=5, final_call="r1.1.0")

EXPERIMENTAL = NumericsProfile(
    name="experimental",
    r_version=None,
    storage="float32",
    input_filters="fast",
    baseline="fast",
    min_cluster_cells=5,
    segmentation="fast",
    segment_values="closed_form",
    final_call="fast",
)

PROFILES: dict[str, NumericsProfile] = {
    "r": R_1_2_5,
    "experimental": EXPERIMENTAL,
}
DEFAULT_PROFILE = "r"


def resolve(profile: "str | NumericsProfile", r_version: str | None = None) -> NumericsProfile:
    """Look up a profile by name; ``r_version="1.1.0"`` selects R 1.1.0's rules for ``r``."""
    if isinstance(profile, NumericsProfile):
        return profile
    try:
        resolved = PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}") from None
    if r_version is not None and r_version != resolved.r_version:
        if resolved.name != "r":
            raise ValueError("r_version applies only to the 'r' profile")
        if r_version == "1.1.0":
            return R_1_1_0
        if r_version != "1.2.5":
            raise ValueError(f"unsupported r_version {r_version!r}; expected '1.2.5' or '1.1.0'")
    return resolved


def storage_dtype() -> type[np.floating]:
    """dtype for the smoothed, segmented and binned matrices under the active profile."""
    return np.float64 if active().storage == "float64" else np.float32


_ACTIVE: ContextVar[NumericsProfile] = ContextVar("copykat_numerics_profile", default=EXPERIMENTAL)


def active() -> NumericsProfile:
    """The profile of the running ``copykat()`` call.

    Library functions called on their own, outside ``copykat()``, keep their
    previous behaviour, which is the ``experimental`` profile.
    """
    return _ACTIVE.get()


@contextlib.contextmanager
def use(profile: NumericsProfile) -> Iterator[NumericsProfile]:
    token = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(token)
