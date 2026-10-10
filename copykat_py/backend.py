"""Compute backend selection for the experimental CUDA port.

This execution branch retains the CPU PCA and silhouette sampling policies.
The separate numerical-policy branch enables full-feature Ward, all-cell
silhouette, FP64 segment accumulation, and full-tree heatmap ordering.
"""

from copykat_py._types import ExecutionBackend

_STATE: dict[str, ExecutionBackend] = {"name": ExecutionBackend.CPU}
BACKENDS = tuple(ExecutionBackend)


def set_backend(name: ExecutionBackend | str) -> None:
    name = ExecutionBackend(str(name or ExecutionBackend.CPU).lower())
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {BACKENDS}")
    if name is not ExecutionBackend.CPU:
        try:
            import cupy  # noqa: F401
            import torch
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(f"backend {name!r} needs torch and cupy with CUDA: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"backend {name!r} needs a CUDA device")
    _STATE["name"] = name


def get_backend() -> ExecutionBackend:
    return _STATE["name"]


def use_gpu() -> bool:
    return _STATE["name"] is not ExecutionBackend.CPU


def exact_algorithms() -> bool:
    """Whether the separately reviewed precision policy is enabled."""
    return _STATE["name"] is ExecutionBackend.GPU
