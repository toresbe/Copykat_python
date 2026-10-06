"""Fused Triton kernels for the Ward nearest-neighbour search.

Given the fp32 GEMM block G = S[rows] @ S.T, each kernel turns G into Ward
cost lower bounds on the fly,

    lower[i, j] = f(s_i, s_j) * max(n_i + n_j - 2 G[i, j] - err_i, 0),
    f(a, b) = a b / (a + b),   lower[i, rows[i]] = +inf,

and reduces each row without writing the bounds out: ``row_argmin`` returns
the smallest bound and its column (lowest column on ties), ``row_reach``
counts the columns whose bound is <= a per-row threshold and records the
lowest ``cap`` of them. Each reads G once; the unfused PyTorch version made
about six passes over it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lower_tile(G_ptr, row, cols, mask, stride_g, nrm_r, err_r, size_r, nrm_ptr, size_ptr, self_col):
    g = tl.load(G_ptr + row * stride_g + cols, mask=mask, other=0.0)
    nc = tl.load(nrm_ptr + cols, mask=mask, other=0.0)
    sc = tl.load(size_ptr + cols, mask=mask, other=1.0)
    d2 = nrm_r + nc - 2.0 * g - err_r
    d2 = tl.maximum(d2, 0.0)
    low = (size_r * sc / (size_r + sc)) * d2
    low = tl.where(mask & (cols != self_col), low, float("inf"))
    return low


@triton.jit
def _row_argmin_kernel(G_ptr, stride_g, rows_ptr, nrm_ptr, size_ptr, err_ptr, m,
                       out_val_ptr, out_idx_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    self_col = tl.load(rows_ptr + i)
    nrm_r = tl.load(nrm_ptr + self_col)
    size_r = tl.load(size_ptr + self_col)
    err_r = tl.load(err_ptr + i)
    best_v = tl.full([BLOCK], float("inf"), tl.float32)
    best_j = tl.full([BLOCK], 2147483647, tl.int32)
    for start in range(0, m, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < m
        low = _lower_tile(G_ptr, i, cols, mask, stride_g, nrm_r, err_r, size_r, nrm_ptr, size_ptr, self_col)
        better = low < best_v  # strict: earlier (lower) columns win ties per lane
        best_v = tl.where(better, low, best_v)
        best_j = tl.where(better, cols, best_j)
    v = tl.min(best_v, axis=0)
    j = tl.min(tl.where(best_v == v, best_j, 2147483647), axis=0)
    tl.store(out_val_ptr + i, v)
    tl.store(out_idx_ptr + i, j)


@triton.jit
def _row_reach_kernel(G_ptr, stride_g, rows_ptr, nrm_ptr, size_ptr, err_ptr, thr_ptr, m,
                      out_cnt_ptr, out_cols_ptr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    self_col = tl.load(rows_ptr + i)
    nrm_r = tl.load(nrm_ptr + self_col)
    size_r = tl.load(size_ptr + self_col)
    err_r = tl.load(err_ptr + i)
    thr = tl.load(thr_ptr + i)
    count = 0
    for start in range(0, m, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < m
        low = _lower_tile(G_ptr, i, cols, mask, stride_g, nrm_r, err_r, size_r, nrm_ptr, size_ptr, self_col)
        hit = low <= thr
        n_hit = tl.sum(hit.to(tl.int32), axis=0)
        if n_hit > 0:
            # record hits in column order, up to CAP per row
            pos = count + tl.cumsum(hit.to(tl.int32), axis=0) - 1
            keep = hit & (pos < CAP)
            tl.store(out_cols_ptr + i * CAP + pos, cols, mask=keep)
            count += n_hit
    tl.store(out_cnt_ptr + i, count)


def row_argmin(G, rows, nrm32, size32, err):
    q, m = G.shape
    val = torch.empty(q, dtype=torch.float32, device=G.device)
    idx = torch.empty(q, dtype=torch.int32, device=G.device)
    _row_argmin_kernel[(q,)](G, G.stride(0), rows, nrm32, size32, err, m, val, idx, BLOCK=1024, num_warps=4)
    return val, idx.long()


def row_reach(G, rows, nrm32, size32, err, thr, cap=16):
    q, m = G.shape
    cnt = torch.empty(q, dtype=torch.int32, device=G.device)
    cols = torch.full((q, cap), -1, dtype=torch.int32, device=G.device)
    _row_reach_kernel[(q,)](G, G.stride(0), rows, nrm32, size32, err, thr, m, cnt, cols,
                            CAP=cap, BLOCK=1024, num_warps=4)
    return cnt, cols.long()
