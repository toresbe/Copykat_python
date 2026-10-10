"""Compute backend selection for the experimental CUDA port.

This execution branch retains the CPU PCA and silhouette sampling policies.
The separate numerical-policy branch enables full-feature Ward, all-cell
silhouette, FP64 segment accumulation, and full-tree heatmap ordering.
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
    """Whether the separately reviewed precision policy is enabled."""
    return _STATE["name"] == "gpu"
