"""Compute backend selection.

``cpu`` (default) is the reference implementation. ``gpu`` runs the numeric
steps on a CUDA device and, freed from the CPU's cost constraints, uses the
exact versions of steps the CPU path approximates for speed:

- Ward clustering on the full feature matrix instead of a PCA projection,
- the silhouette over all cells instead of a stratified subsample,
- FP64 segment sums instead of differences of an FP32 running total,
- a full Ward dendrogram for the heatmap at any cell count (reusing the
  step-8 tree) instead of k-means block ordering above 3,000 cells.

``gpu-compat`` runs on the GPU but keeps the CPU path's approximations
(PCA, silhouette subsample), to separate the speed of the port from the
effect of the exact algorithms.
"""

_STATE = {"name": "cpu"}

BACKENDS = ("cpu", "gpu", "gpu-compat")


def set_backend(name):
    name = str(name or "cpu").lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {BACKENDS}")
    if name != "cpu":
        try:
            import torch
            import cupy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(f"backend {name!r} needs torch and cupy with CUDA: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"backend {name!r} needs a CUDA device")
    _STATE["name"] = name


def get_backend():
    return _STATE["name"]


def use_gpu():
    return _STATE["name"] != "cpu"


def exact_algorithms():
    """True when the backend replaces the CPU path's approximations with exact versions."""
    return _STATE["name"] == "gpu"
