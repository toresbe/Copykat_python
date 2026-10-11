"""Exact Ward linkage by parallel reciprocal-nearest-neighbour merging, on the CPU.

A port of the GPU engine's algorithm (``copykat_py.gpu.ward``). Ward's
criterion is reducible: merging two clusters never brings the merged cluster
closer to a third one than the nearer of its parts was. So any pair of
reciprocal nearest neighbours (i's nearest neighbour is j and j's is i) can be
merged at any time without changing the hierarchy, and each round merges every
such pair at once. Only the merged clusters, and clusters whose nearest
neighbour was merged, need a new nearest-neighbour search. Memory is O(n * d):
no n^2 distance matrix.

Nearest neighbours come from an FP64 GEMM (||a||^2 + ||b||^2 - 2 a.b) that is
turned into a lower bound on every Ward cost by subtracting a rounding-error
bound. Each row's smallest bound is re-scored exactly from coordinate
differences, then every column whose bound could still beat or tie the best
exact cost is re-scored too, so the result is the exact nearest neighbour
(lowest cluster index on ties) without any fallback scan.

Merge costs are computed from cluster centroids instead of fastcluster's
Lance-Williams distance updates, so the merge tree is the same but heights
can differ in the last bits.
"""

import numba
import numpy as np
from numba import njit, prange
from threadpoolctl import threadpool_limits

from copykat_py._types import CellByFeature, FloatArray, IntArray, LinkageMatrix

# Bytes of GEMM output per nearest-neighbour block.
_BLOCK_BYTES = 256 << 20


@njit(inline="always")
def _sq_dist(C: FloatArray, r: int, j: int) -> float:
    acc = 0.0
    for k in range(C.shape[1]):
        t = C[r, k] - C[j, k]
        acc += t * t
    return acc


@njit(parallel=True)
def _nearest_from_gemm(
    G: FloatArray,
    rows: IntArray,
    C: FloatArray,
    size: FloatArray,
    nrm: FloatArray,
    gamma: float,
    out_nn: IntArray,
    out_w: FloatArray,
) -> None:
    """Exact Ward nearest neighbour of each row in ``rows`` given ``G = C[rows] @ C.T``."""
    m = C.shape[0]
    for b in prange(rows.shape[0]):
        r = rows[b]
        sr = size[r]
        nr = nrm[r]
        # Smallest lower bound (first index on ties)
        j1 = -1
        lb1 = np.inf
        for j in range(m):
            if j == r:
                continue
            s = nr + nrm[j]
            lb = (s - 2.0 * G[b, j] - gamma * s) * (sr * size[j] / (sr + size[j]))
            if lb < lb1:
                lb1 = lb
                j1 = j
        best_j = j1
        best = sr * size[j1] / (sr + size[j1]) * _sq_dist(C, r, j1)
        # Every column whose lower bound reaches the best exact cost could beat or tie it.
        for j in range(m):
            if j in (r, j1):
                continue
            s = nr + nrm[j]
            f = sr * size[j] / (sr + size[j])
            lb = (s - 2.0 * G[b, j] - gamma * s) * f
            if lb <= best:
                w = f * _sq_dist(C, r, j)
                if w < best or (w == best and j < best_j):
                    best = w
                    best_j = j
        out_nn[b] = best_j
        out_w[b] = best


def _nearest_neighbours(
    C: FloatArray, size: FloatArray, nrm: FloatArray, query: IntArray, gamma: float
) -> tuple[IntArray, FloatArray]:
    m = C.shape[0]
    nn = np.empty(len(query), dtype=np.int64)
    w = np.empty(len(query))
    block = max(1, min(len(query), _BLOCK_BYTES // (8 * m)))
    buffer = np.empty(block * m)  # reused: fresh pages for every block cost more than the GEMM
    for lo in range(0, len(query), block):
        rows = query[lo : lo + block]
        G = buffer[: len(rows) * m].reshape(len(rows), m)
        np.matmul(C[rows], C.T, out=G)
        _nearest_from_gemm(G, rows, C, size, nrm, gamma, nn[lo : lo + block], w[lo : lo + block])
    return nn, w


def ward_linkage_rnn(data: CellByFeature, n_threads: int = 1) -> LinkageMatrix:
    """Exact Ward linkage of the rows of ``data`` (Euclidean), in SciPy's linkage format."""
    X = np.asarray(data, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        return np.zeros((0, 4))
    # Ward is translation invariant; centring shrinks the norms and so the GEMM error bound.
    C = X - X.mean(axis=0)
    # Rounding-error bound on ||a||^2 + ||b||^2 - 2 a.b relative to ||a||^2 + ||b||^2
    # (each dot product within (d + 2) unit roundoffs of the norms), with 2x slack.
    gamma = 8.0 * (d + 4) * np.finfo(np.float64).eps
    nrm = np.einsum("ij,ij->i", C, C)

    n_threads = max(1, int(n_threads))
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(n_threads, user_api="blas"):
            merges = _rnn_merges(C, nrm, gamma)
    finally:
        numba.set_num_threads(previous_threads)
    return _to_scipy_linkage(n, *merges)


def _rnn_merges(
    C: FloatArray, nrm: FloatArray, gamma: float
) -> tuple[list[IntArray], list[IntArray], list[FloatArray], list[FloatArray], list[IntArray]]:
    """Merge reciprocal nearest neighbours round by round; returns the merges in provisional order.

    Merged clusters get ids n, n + 1, ... in the order they are created.
    """
    n = C.shape[0]
    size = np.ones(n)
    pid = np.arange(n)
    child_a: list[IntArray] = []
    child_b: list[IntArray] = []
    delta: list[FloatArray] = []
    counts: list[FloatArray] = []
    rounds: list[IntArray] = []
    next_id = n
    rnd = 0
    nn, nnw = _nearest_neighbours(C, size, nrm, np.arange(n), gamma)
    while C.shape[0] > 1:
        m = C.shape[0]
        idx = np.arange(m)
        a = idx[(nn[nn] == idx) & (idx < nn)]
        if len(a) == 0:  # only possible through exact ties; merge the globally cheapest pair
            i = int(np.argmin(nnw))
            a = np.array([min(i, int(nn[i]))])
        b = nn[a]
        sa, sb = size[a], size[b]
        snew = sa + sb
        child_a.append(pid[a])
        child_b.append(pid[b])
        delta.append(nnw[a])
        counts.append(snew)
        rounds.append(np.full(len(a), rnd))
        C[a] = (C[a] * sa[:, None] + C[b] * sb[:, None]) / snew[:, None]
        nrm[a] = np.einsum("ij,ij->i", C[a], C[a])
        size[a] = snew
        pid[a] = np.arange(next_id, next_id + len(a))
        next_id += len(a)
        # By reducibility a cluster keeps its nearest neighbour unless that was merged.
        touched = np.zeros(m, dtype=bool)
        touched[a] = True
        touched[b] = True
        stale = touched[nn]
        stale[a] = True
        keep = np.ones(m, dtype=bool)
        keep[b] = False
        new_index = np.cumsum(keep) - 1
        C = C[keep]
        nrm = nrm[keep]
        size = size[keep]
        pid = pid[keep]
        nn = new_index[nn[keep]]
        nnw = nnw[keep]
        stale = stale[keep]
        rnd += 1
        if C.shape[0] > 1:
            query = np.flatnonzero(stale)
            if len(query):
                nn[query], nnw[query] = _nearest_neighbours(C, size, nrm, query, gamma)
    return child_a, child_b, delta, counts, rounds


def _to_scipy_linkage(
    n: int,
    child_a: list[IntArray],
    child_b: list[IntArray],
    delta: list[FloatArray],
    counts: list[FloatArray],
    rounds: list[IntArray],
) -> LinkageMatrix:
    """Order merges by height like SciPy/fastcluster and renumber the merged clusters."""
    ca = np.concatenate(child_a)
    cb = np.concatenate(child_b)
    h = np.sqrt(2.0 * np.maximum(np.concatenate(delta), 0.0))
    cnt = np.concatenate(counts)
    rd = np.concatenate(rounds)
    n_merges = n - 1
    # A parent is never lower than its children (guards against rounding).
    for i in range(n_merges):
        for c in (ca[i], cb[i]):
            if c >= n and h[c - n] > h[i]:
                h[i] = h[c - n]
    order = np.lexsort((np.arange(n_merges), rd, h))
    final_of = np.empty(n_merges, dtype=np.int64)
    final_of[order] = np.arange(n_merges)

    def remap(c: IntArray) -> IntArray:
        c = c.copy()
        merged = c >= n
        c[merged] = n + final_of[c[merged] - n]
        return c

    ca, cb = remap(ca), remap(cb)
    Z = np.empty((n_merges, 4))
    Z[:, 0] = np.minimum(ca, cb)[order]
    Z[:, 1] = np.maximum(ca, cb)[order]
    Z[:, 2] = h[order]
    Z[:, 3] = cnt[order]
    return Z
