"""Exact Ward linkage by parallel reciprocal-nearest-neighbour merging, on the CPU.

A port of the GPU engine's algorithm (``copykat_py.gpu.ward``). Ward's
criterion is reducible: merging two clusters never brings the merged cluster
closer to a third one than the nearer of its parts was. So any pair of
reciprocal nearest neighbours (i's nearest neighbour is j and j's is i) can be
merged at any time without changing the hierarchy, and each round merges every
such pair at once. Only the merged clusters, and clusters whose nearest
neighbour was merged, need a new nearest-neighbour search. Memory is O(n * d):
no n^2 distance matrix.

Nearest neighbours come from an FP32 GEMM (||a||^2 + ||b||^2 - 2 a.b) that is
turned into a lower bound on every Ward cost by subtracting a rounding-error
bound. Each row's smallest bound is re-scored exactly from coordinate
differences, then every column whose bound could still beat or tie the best
exact cost is re-scored too, so the result is the exact nearest neighbour
(lowest cluster index on ties).

A full scan also remembers each cluster's 16 best candidates and a lower
bound on its cost to every other cluster. Reducibility keeps both valid as
clusters merge (a merged cluster is no cheaper than the cheaper of its
parts), so a cluster whose nearest neighbour was merged re-scores its
remembered candidates exactly, and only needs a new scan when the best of
them does not undercut that bound.

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
# Precision of the candidate GEMM. FP32 is twice as fast; its larger rounding
# error only widens the lower bounds, so more candidates are re-scored
# exactly in FP64 and the result is unchanged.
_SEARCH_DTYPE = np.float32


@njit(inline="always")
def _sq_dist(C: FloatArray, r: int, j: int) -> float:
    acc = 0.0
    for k in range(C.shape[1]):
        t = C[r, k] - C[j, k]
        acc += t * t
    return acc


# Candidates remembered per cluster between full scans, and the relative
# margin by which a re-scored candidate must undercut the bound on all others.
_N_CANDIDATES = 16
_CERTIFY_MARGIN = 1e-9


@njit(parallel=True)
def _scan_from_gemm(
    G: FloatArray,
    rows: IntArray,
    C: FloatArray,
    size: FloatArray,
    nrm: FloatArray,
    gamma: float,
    out_nn: IntArray,
    out_w: FloatArray,
    cand: IntArray,
    cbound: FloatArray,
    tau: FloatArray,
) -> None:
    """Exact Ward nearest neighbour of each row in ``rows`` given ``G = S[rows] @ S.T``.

    Also records the row's ``K`` best candidates (``cand``, with lower bounds
    or exact costs in ``cbound``) and ``tau``, a lower bound on the cost to
    every cluster not listed.
    """
    m = C.shape[0]
    K = cand.shape[1]
    for b in prange(rows.shape[0]):
        r = rows[b]
        sr = size[r]
        nr = nrm[r]
        # The K + 1 smallest lower bounds, ascending (earlier index first on ties)
        tid = np.full(K + 1, -1, dtype=np.int64)
        tv = np.full(K + 1, np.inf)
        for j in range(m):
            if j == r:
                continue
            s = nr + nrm[j]
            lb = (s - 2.0 * G[b, j] - gamma * s) * (sr * size[j] / (sr + size[j]))
            if lb < tv[K]:
                pos = K
                while pos > 0 and tv[pos - 1] > lb:
                    tv[pos] = tv[pos - 1]
                    tid[pos] = tid[pos - 1]
                    pos -= 1
                tv[pos] = lb
                tid[pos] = j
        # Re-score candidates in bound order until no bound can beat or tie the best.
        best = np.inf
        best_j = -1
        certified = False
        for t in range(K + 1):
            j = tid[t]
            if j < 0 or tv[t] > best:
                certified = True
                break
            w = sr * size[j] / (sr + size[j]) * _sq_dist(C, r, j)
            if t < K:
                tv[t] = w  # exact costs are bounds too, and tighter
            if w < best or (w == best and j < best_j):
                best = w
                best_j = j
        if not certified:
            # More than K + 1 columns can reach the best: check every column.
            for j in range(m):
                if j == r:
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
        for t in range(K):
            cand[r, t] = tid[t]
            cbound[r, t] = tv[t]
        tau[r] = tv[K] if tid[K] >= 0 else np.inf


@njit(parallel=True)
def _refresh_rows(
    rows: IntArray,
    C: FloatArray,
    size: FloatArray,
    cand: IntArray,
    cbound: FloatArray,
    tau: FloatArray,
    margin: float,
    out_nn: IntArray,
    out_w: FloatArray,
    ok: np.ndarray,
) -> None:
    """Nearest neighbour of each row among its remembered candidates, certified against ``tau``."""
    K = cand.shape[1]
    for b in prange(rows.shape[0]):
        r = rows[b]
        sr = size[r]
        best = np.inf
        best_j = -1
        for t in range(K):
            j = cand[r, t]
            if j < 0 or j == r:
                continue
            w = sr * size[j] / (sr + size[j]) * _sq_dist(C, r, j)
            cbound[r, t] = w
            if w < best or (w == best and j < best_j):
                best = w
                best_j = j
        out_nn[b] = best_j
        out_w[b] = best
        ok[b] = best_j >= 0 and best < tau[r] * (1.0 - margin)


@njit(parallel=True)
def _merge_lists(a: IntArray, b: IntArray, cand: IntArray, cbound: FloatArray, tau: FloatArray) -> None:
    """Candidate list of each merged cluster (stored in row ``a``) from the lists of its parts.

    By reducibility the cost from a merged cluster to any cluster j is at
    least the smaller of its parts' costs to j, so j's bound is the smaller of
    the parts' bounds (a part that does not list j bounds it by its ``tau``).
    """
    K = cand.shape[1]
    for p in prange(a.shape[0]):
        ra = a[p]
        rb = b[p]
        ids = np.full(2 * K, -1, dtype=np.int64)
        vals = np.full(2 * K, np.inf)
        n_u = 0
        for side in range(2):
            r, other = (ra, rb) if side == 0 else (rb, ra)
            for t in range(K):
                j = cand[r, t]
                if j < 0 or j in (ra, rb):
                    continue
                seen = False
                for u in range(n_u):
                    if ids[u] == j:
                        seen = True
                        break
                if seen:
                    continue
                bound_r = np.inf
                for u in range(K):
                    if cand[r, u] == j and cbound[r, u] < bound_r:
                        bound_r = cbound[r, u]
                bound_o = tau[other]
                for u in range(K):
                    if cand[other, u] == j and cbound[other, u] < bound_o:
                        bound_o = cbound[other, u]
                ids[n_u] = j
                vals[n_u] = min(bound_r, bound_o)
                n_u += 1
        order = np.argsort(vals[:n_u], kind="mergesort")
        new_tau = min(tau[ra], tau[rb])
        if n_u > K:
            new_tau = min(new_tau, vals[order[K]])
        for t in range(K):
            if t < n_u:
                cand[ra, t] = ids[order[t]]
                cbound[ra, t] = vals[order[t]]
            else:
                cand[ra, t] = -1
                cbound[ra, t] = np.inf
        tau[ra] = new_tau


def _nearest_neighbours(
    C: FloatArray,
    S: FloatArray,
    size: FloatArray,
    nrm: FloatArray,
    query: IntArray,
    gamma: float,
    cand: IntArray,
    cbound: FloatArray,
    tau: FloatArray,
) -> tuple[IntArray, FloatArray]:
    """Exact nearest neighbours of ``query`` rows by full scans; ``S`` is ``C`` in the search precision."""
    m = C.shape[0]
    nn = np.empty(len(query), dtype=np.int64)
    w = np.empty(len(query))
    block = max(1, min(len(query), _BLOCK_BYTES // (S.itemsize * m)))
    buffer = np.empty(block * m, dtype=S.dtype)  # reused: fresh pages for every block cost more than the GEMM
    for lo in range(0, len(query), block):
        rows = query[lo : lo + block]
        G = buffer[: len(rows) * m].reshape(len(rows), m)
        np.matmul(S[rows], S.T, out=G)
        _scan_from_gemm(G, rows, C, size, nrm, gamma, nn[lo : lo + block], w[lo : lo + block], cand, cbound, tau)
    return nn, w


def ward_linkage_rnn(data: CellByFeature, n_threads: int = 1) -> LinkageMatrix:
    """Exact Ward linkage of the rows of ``data`` (Euclidean), in SciPy's linkage format."""
    X = np.asarray(data, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        return np.zeros((0, 4))
    # Ward is translation invariant; centring shrinks the norms and so the GEMM error bound.
    C = X - X.mean(axis=0)
    # Error bound on ||a||^2 + ||b||^2 - 2 a.b (exact FP64 norms, dot products from
    # the search-precision copy) relative to ||a||^2 + ||b||^2: rounding the
    # coordinates costs about 3 unit roundoffs and the GEMM d + 2; 2x slack.
    gamma = 2.0 * (d + 8) * float(np.finfo(_SEARCH_DTYPE).eps)
    nrm = np.einsum("ij,ij->i", C, C)

    n_threads = max(1, int(n_threads))
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(n_threads, user_api="blas"):
            merges = _rnn_merges(C, C.astype(_SEARCH_DTYPE), nrm, gamma)
    finally:
        numba.set_num_threads(previous_threads)
    return _to_scipy_linkage(n, *merges)


def _rnn_merges(
    C: FloatArray, S: FloatArray, nrm: FloatArray, gamma: float
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
    K = min(_N_CANDIDATES, max(1, n - 1))
    cand = np.full((n, K), -1, dtype=np.int64)
    cbound = np.full((n, K), np.inf)
    tau = np.full(n, np.inf)
    nn, nnw = _nearest_neighbours(C, S, size, nrm, np.arange(n), gamma, cand, cbound, tau)
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
        _merge_lists(a, b, cand, cbound, tau)
        C[a] = (C[a] * sa[:, None] + C[b] * sb[:, None]) / snew[:, None]
        S[a] = C[a]
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
        # Old position -> new position; a merged-away cluster now lives in its partner's slot.
        remap = new_index.copy()
        remap[b] = new_index[a]
        C = C[keep]
        S = S[keep]
        nrm = nrm[keep]
        size = size[keep]
        pid = pid[keep]
        nn = new_index[nn[keep]]
        nnw = nnw[keep]
        stale = stale[keep]
        cand = cand[keep]
        cand = np.where(cand >= 0, remap[np.maximum(cand, 0)], -1)
        cbound = cbound[keep]
        tau = tau[keep]
        rnd += 1
        if C.shape[0] > 1:
            query = np.flatnonzero(stale)
            if len(query):
                q_nn = np.empty(len(query), dtype=np.int64)
                q_w = np.empty(len(query))
                ok = np.empty(len(query), dtype=np.bool_)
                _refresh_rows(query, C, size, cand, cbound, tau, _CERTIFY_MARGIN, q_nn, q_w, ok)
                nn[query[ok]] = q_nn[ok]
                nnw[query[ok]] = q_w[ok]
                rescan = query[~ok]
                if len(rescan):
                    nn[rescan], nnw[rescan] = _nearest_neighbours(C, S, size, nrm, rescan, gamma, cand, cbound, tau)
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
