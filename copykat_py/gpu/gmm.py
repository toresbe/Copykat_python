"""Batched 3-component shared-variance Gaussian mixture EM on the GPU.

Same EM iteration, initialisation and stopping rule as
``baseline._fit_gmm_3component`` (FP64 throughout); each series is fitted by
one thread block that runs every iteration inside a single kernel launch.
Only the summation order differs from the CPU version.
"""

import cupy as cp
import numpy as np
import torch

_SRC = r"""
__device__ inline double block_sum(double v, double* sh) {
    // warp reduce then cross-warp via shared memory
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        v = (lane < NT / 32) ? sh[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    double r = sh[0];
    __syncthreads();
    return r;
}

extern "C" __global__
void gmm3(const double* __restrict__ X, const long n, const double* __restrict__ sigma0,
          const double mu0, const double mu1, const double mu2,
          const int max_iter, const double tol,
          double* __restrict__ out_means, double* __restrict__ out_weights,
          double* __restrict__ out_sigma, int* __restrict__ out_iters,
          double* __restrict__ scratch)
{
    __shared__ double sh[32];
    const double* x = X + (long)blockIdx.x * n;
    const double TINY = 2.2250738585072014e-308;
    const double SQ2PI = 2.5066282746310002;
    double m0 = mu0, m1 = mu1, m2 = mu2;
    double w0 = 1.0 / 3.0, w1 = 1.0 / 3.0, w2 = 1.0 / 3.0;
    double s = fmax(sigma0[blockIdx.x], 1e-8);
    double prev = -__longlong_as_double(0x7ff0000000000000LL);
    int it = 0;
    for (; it < max_iter; ++it) {
        double inv = 1.0 / s, norm = 1.0 / (s * SQ2PI);
        double a_n0 = 0, a_n1 = 0, a_n2 = 0, a_x0 = 0, a_x1 = 0, a_x2 = 0, a_ll = 0;
        for (long i = threadIdx.x; i < n; i += NT) {
            double xi = x[i];
            double z0 = (xi - m0) * inv, z1 = (xi - m1) * inv, z2 = (xi - m2) * inv;
            double d0 = exp(-0.5 * z0 * z0) * norm * w0;
            double d1 = exp(-0.5 * z1 * z1) * norm * w1;
            double d2 = exp(-0.5 * z2 * z2) * norm * w2;
            double rs = d0 + d1 + d2;
            if (rs <= 0) rs = TINY;
            double r0 = d0 / rs, r1 = d1 / rs, r2 = d2 / rs;
            double* sc = scratch + 3 * ((long)blockIdx.x * n + i);
            sc[0] = r0; sc[1] = r1; sc[2] = r2;
            a_n0 += r0; a_n1 += r1; a_n2 += r2;
            a_x0 += r0 * xi; a_x1 += r1 * xi; a_x2 += r2 * xi;
            a_ll += log(rs);
        }
        double nk0 = block_sum(a_n0, sh), nk1 = block_sum(a_n1, sh), nk2 = block_sum(a_n2, sh);
        double sx0 = block_sum(a_x0, sh), sx1 = block_sum(a_x1, sh), sx2 = block_sum(a_x2, sh);
        double ll = block_sum(a_ll, sh);
        if (nk0 <= 0) nk0 = TINY;
        if (nk1 <= 0) nk1 = TINY;
        if (nk2 <= 0) nk2 = TINY;
        double nm0 = sx0 / nk0, nm1 = sx1 / nk1, nm2 = sx2 / nk2;
        // second pass: variance around the new means with the same responsibilities
        double a_v = 0;
        for (long i = threadIdx.x; i < n; i += NT) {
            double xi = x[i];
            const double* sc = scratch + 3 * ((long)blockIdx.x * n + i);
            double e0 = xi - nm0, e1 = xi - nm1, e2 = xi - nm2;
            a_v += sc[0] * e0 * e0 + sc[1] * e1 * e1 + sc[2] * e2 * e2;
        }
        double var = block_sum(a_v, sh) / (double)n;
        w0 = nk0 / (double)n; w1 = nk1 / (double)n; w2 = nk2 / (double)n;
        m0 = nm0; m1 = nm1; m2 = nm2;
        s = sqrt(fmax(var, 1e-12));
        if (fabs(ll - prev) < tol * (fabs(prev) + tol)) { ++it; break; }
        prev = ll;
    }
    if (threadIdx.x == 0) {
        long b = blockIdx.x;
        out_means[3 * b] = m0; out_means[3 * b + 1] = m1; out_means[3 * b + 2] = m2;
        out_weights[3 * b] = w0; out_weights[3 * b + 1] = w1; out_weights[3 * b + 2] = w2;
        out_sigma[b] = s;
        out_iters[b] = it;
    }
}
"""

_kernels = {}


def _get_kernel(nt):
    if nt not in _kernels:
        _kernels[nt] = cp.RawKernel(_SRC, "gmm3", options=(f"-DNT={nt}",))
    return _kernels[nt]


def fit_gmm_3component_batch(X, sigma_init=None, mu_init=(-0.2, 0.0, 0.2), max_iter=500, tol=1e-8):
    """Fit one 3-component GMM per row of ``X`` (b x n).

    Accepts a NumPy array or CUDA tensor; returns NumPy (means (b,3),
    weights (b,3), sigma (b,)).
    """
    if isinstance(X, torch.Tensor):
        Xt = X.to(device="cuda", dtype=torch.float64).contiguous()
    else:
        Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float64), device="cuda")
    if Xt.ndim == 1:
        Xt = Xt[None, :]
    b, n = Xt.shape
    if sigma_init is None:
        sigma_init = torch.clamp(0.5 * Xt.std(dim=1, unbiased=False), min=0.05)
    sig = torch.as_tensor(
        np.broadcast_to(
            np.asarray(
                sigma_init.cpu().numpy() if isinstance(sigma_init, torch.Tensor) else sigma_init, dtype=np.float64
            ),
            (b,),
        ).copy(),
        device="cuda",
    )
    means = cp.empty((b, 3), dtype=cp.float64)
    weights = cp.empty((b, 3), dtype=cp.float64)
    sigma = cp.empty(b, dtype=cp.float64)
    iters = cp.empty(b, dtype=cp.int32)
    scratch = cp.empty(b * n * 3, dtype=cp.float64)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    nt = 512 if b < 2 * sm_count else 256
    _get_kernel(nt)(
        (b,),
        (nt,),
        (
            cp.from_dlpack(Xt),
            np.int64(n),
            cp.from_dlpack(sig),
            np.float64(mu_init[0]),
            np.float64(mu_init[1]),
            np.float64(mu_init[2]),
            np.int32(max_iter),
            np.float64(tol),
            means,
            weights,
            sigma,
            iters,
            scratch,
        ),
    )
    return cp.asnumpy(means), cp.asnumpy(weights), cp.asnumpy(sigma)
