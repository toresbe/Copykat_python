"""Elementwise and scan kernels for the GPU backend (PyTorch + CuPy RawKernels).

The consumer GPUs this targets have FP64 throughput at 1/64 of FP32, but these
kernels are memory bound, so they keep FP64 arithmetic wherever the CPU path
uses it (the Kalman recursion, the Freeman-Tukey transform) at no real cost.
"""

import cupy as cp
import numpy as np
import torch

_DLM_SRC = r"""
extern "C" __global__
void dlm_smooth(const double* __restrict__ y, const double* __restrict__ K,
                const double* __restrict__ B, double* __restrict__ m,
                float* __restrict__ out, const long n, const long n_cells)
{
    // One thread per cell; y, m, out are row-major (genes x cells) so
    // consecutive threads touch consecutive addresses.
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cells) return;
    double prev = 0.0;  // m[0]: prior mean
    for (long t = 0; t < n; ++t) {
        prev = prev + K[t] * (y[t * n_cells + j] - prev);
        m[t * n_cells + j] = prev;  // m[t] holds the filtered mean m_{t+1}
    }
    // Rauch-Tung-Striebel smoother; s_n = m_n
    double s = prev;
    double total = s;
    for (long t = n - 1; t >= 1; --t) {
        double mt = m[(t - 1) * n_cells + j];
        s = mt + B[t] * (s - mt);
        m[(t - 1) * n_cells + j] = s;
        total += s;
    }
    double mean = total / (double)n;
    for (long t = 0; t < n; ++t) {
        out[t * n_cells + j] = (float)(m[t * n_cells + j] - mean);
    }
}
"""

_dlm_kernel = None


def _cp(t):
    return cp.from_dlpack(t)


def dlm_smooth_gpu(y, K, B, chunk_cells=None):
    """Shared-gain local-level Kalman smoother on the GPU.

    ``y`` is a (genes x cells) float64 CUDA tensor (row-major). Returns the
    centred smoothed states as a float32 CUDA tensor of the same shape, with
    the same recursion as ``smoothing._dlm_smooth_blocks``.
    """
    global _dlm_kernel
    if _dlm_kernel is None:
        _dlm_kernel = cp.RawKernel(_DLM_SRC, "dlm_smooth")
    n, n_cells = y.shape
    dev = y.device
    Kt = torch.as_tensor(np.ascontiguousarray(K), dtype=torch.float64, device=dev)
    Bt = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float64, device=dev)
    out = torch.empty((n, n_cells), dtype=torch.float32, device=dev)
    if chunk_cells is None:
        free, _ = torch.cuda.mem_get_info(dev)
        chunk_cells = max(256, int(0.4 * free // (8 * 2 * max(n, 1))))
    for lo in range(0, n_cells, chunk_cells):
        hi = min(n_cells, lo + chunk_cells)
        yc = y[:, lo:hi].contiguous()
        m = torch.empty_like(yc)
        oc = torch.empty((n, hi - lo), dtype=torch.float32, device=dev)
        threads = 128
        blocks = (hi - lo + threads - 1) // threads
        _dlm_kernel((blocks,), (threads,), (_cp(yc), _cp(Kt), _cp(Bt), _cp(m), _cp(oc), np.int64(n), np.int64(hi - lo)))
        out[:, lo:hi] = oc
        del yc, m, oc
    return out
