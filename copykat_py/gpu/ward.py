"""Exact Ward linkage on the GPU via parallel reciprocal-nearest-neighbour merging.

Ward's criterion is *reducible*: merging two clusters never brings the merged
cluster closer to a third one than the nearer of its two parts was. For
reducible linkages any pair of reciprocal nearest neighbours (RNN: i's nearest
neighbour is j and j's is i) can be merged at any time without changing the
final hierarchy, and disjoint RNN pairs stay RNN pairs after the others merge.
So each round finds every cluster's nearest neighbour at once (a GEMM) and
merges all RNN pairs together. Rounds shrink the active set geometrically, so
the cost is a handful of all-pairs passes instead of n sequential NN-chain
steps, and memory stays O(n * d) (no n^2 distance matrix).

Nearest-neighbour candidates come from an FP32 GEMM
(||a||^2 + ||b||^2 - 2 a.b); the top candidates are then re-scored exactly in
FP64 from coordinate differences, and a row is accepted only when the next
candidate's error-adjusted lower bound cannot beat the exact winner. Rows that
fail that check are recomputed exactly against every cluster. Ties go to the
lowest cluster index.

For wide inputs the candidate search can run on an orthogonal projection of
the data (``search_dim``): projecting never increases a Euclidean distance,
so projected distances are still valid lower bounds, while candidates are
re-scored on the full-width coordinates. The result is the exact full-width
Ward tree; the projection only decides how often the certification falls
back to a full-width scan.

Returns a SciPy-format linkage matrix (heights are Euclidean Ward distances,
sqrt(2 * delta_SSE), the same convention as scipy/fastcluster).
"""

import numpy as np
import torch

_EPS32 = 2.0 ** -24


def _ward_factor(si, sj):
    return si * sj / (si + sj)


def _exact_rows(C, size, rows):
    """Exact FP64 Ward costs from ``rows`` to every cluster; returns (argmin, min).

    Squared distances are accumulated over column blocks of FP64 differences
    (never via the cancellation-prone norm expansion).
    """
    best_j = torch.empty(len(rows), dtype=torch.long, device=C.device)
    best_w = torch.empty(len(rows), dtype=torch.float64, device=C.device)
    m, d = C.shape
    budget = int(2.5e8)
    col = max(1, min(d, budget // max(1, m)))
    for i, r in enumerate(rows.tolist()):
        acc = torch.zeros(m, dtype=torch.float64, device=C.device)
        for c0 in range(0, d, col):
            diff = C[:, c0:c0 + col].double() - C[r, c0:c0 + col].double()
            acc += (diff * diff).sum(1)
        w = _ward_factor(size[r], size) * acc
        w[r] = float("inf")
        best_w[i], best_j[i] = w.min(0)
    return best_j, best_w


def _nearest_neighbours(C, size, query=None, k=8, mem_bytes=1 << 30, S=None):
    """Exact Ward nearest neighbour of rows ``query`` (default: all) of C (m x d, float32).

    ``S`` (m x d', float32) are optional search coordinates whose pairwise
    distances never exceed those of C (an orthogonal projection of C); the
    GEMM candidate search runs on S and the exact re-score on C. Rows whose
    winner cannot be certified from ``k`` candidates are retried with 4x as
    many, and finally scanned exactly.
    """
    m = C.shape[0]
    dev = C.device
    if query is None:
        query = torch.arange(m, device=dev)
    nn, nnw, bad = _candidate_search(C, size, query, k, mem_bytes, S)
    n_retry = 0
    while bad.numel():
        n_retry += int(bad.numel())
        k *= 4
        if k >= m // 2:
            nn[bad], nnw[bad] = _exact_rows(C, size, query[bad])
            break
        qnn, qnnw, qbad = _candidate_search(C, size, query[bad], k, mem_bytes, S)
        nn[bad] = qnn
        nnw[bad] = qnnw
        bad = bad[qbad]
    return nn, nnw, n_retry


def _candidate_search(C, size, query, k, mem_bytes, S):
    """One certified top-k pass; returns (nn, nnw, positions in query not certified)."""
    m, d = C.shape
    dev = C.device
    if S is None:
        S = C
    q = len(query)
    k = min(k, m - 1)
    nrm = (S.double() ** 2).sum(1)  # fp64 norms of the fp32 search coordinates
    nrm32 = nrm.float()
    # Error bound on an fp32 GEMM squared distance, relative to the norms
    # (plus slack for a projection basis that is orthonormal only to fp32).
    gamma = 4.0 * _EPS32 * (S.shape[1] ** 0.5) + 8.0 * _EPS32 + (1e-5 if S is not C else 0.0)
    max_nrm = float(nrm.max())
    nn = torch.empty(q, dtype=torch.long, device=dev)
    nnw = torch.empty(q, dtype=torch.float64, device=dev)
    uncertified = []
    block = max(1, min(q, int(mem_bytes // (4 * 3 * m))))
    size32 = size.float()
    for lo in range(0, q, block):
        hi = min(q, lo + block)
        rows = query[lo:hi]
        G = S[rows] @ S.T  # fp32 GEMM
        D2 = (nrm32[rows, None] + nrm32[None, :]) - 2.0 * G
        del G
        D2 -= gamma * (nrm32[rows, None] + max_nrm)
        D2.clamp_(min=0.0)
        D2 *= _ward_factor(size32[rows, None], size32[None, :])
        lower = D2
        lower[torch.arange(hi - lo, device=dev), rows] = float("inf")
        cand_w, cand = torch.topk(lower, k + 1 if k + 1 <= m - 1 else k, dim=1, largest=False, sorted=True)
        del lower, D2
        kk = min(k, cand.shape[1])
        cidx = cand[:, :kk]
        # Exact FP64 re-score of the candidates, in chunks of rows so the
        # (rows x k x d) differences stay within the memory budget
        w = torch.empty(cidx.shape, dtype=torch.float64, device=dev)
        sub = max(1, int(mem_bytes // (8 * 2 * kk * d)))
        for s0 in range(0, hi - lo, sub):
            r = rows[s0:s0 + sub]
            ci = cidx[s0:s0 + sub]
            diff = C[r].double()[:, None, :] - C[ci].double()
            w[s0:s0 + sub] = _ward_factor(size[r, None], size[ci]) * (diff * diff).sum(-1)
            del diff
        # Tie-break on the lowest index: order candidates by index first
        order = torch.argsort(cidx, dim=1)
        cidx = torch.gather(cidx, 1, order)
        w = torch.gather(w, 1, order)
        bw, bpos = w.min(dim=1)
        nn[lo:hi] = torch.gather(cidx, 1, bpos[:, None])[:, 0]
        nnw[lo:hi] = bw
        if cand.shape[1] > kk:
            # Next candidate's lower bound must not beat the exact winner
            # (ties included, so the lowest-index rule stays exact).
            bad = torch.nonzero(cand_w[:, kk].double() <= bw)[:, 0]
            if bad.numel():
                uncertified.append(bad + lo)
    bad = torch.cat(uncertified) if uncertified else torch.empty(0, dtype=torch.long, device=dev)
    return nn, nnw, bad


def _compact_rows_(T, keep_idx, chunk_rows):
    """T[:len(keep_idx)] = T[keep_idx] in place; keep_idx must be increasing.

    Each destination row i reads from keep_idx[i] >= i, so copying chunks in
    increasing order never overwrites a row a later chunk still needs, and
    no full-size temporary is allocated.
    """
    m = len(keep_idx)
    for lo in range(0, m, chunk_rows):
        hi = min(m, lo + chunk_rows)
        T[lo:hi] = T[keep_idx[lo:hi]]
    return T[:m]


def _projection_basis(C, dim, seed=1234):
    """Orthonormal (d x dim) basis of the leading principal subspace of C (centred)."""
    torch.manual_seed(seed)
    _, _, V = torch.svd_lowrank(C, q=dim, niter=3)
    Q, _ = torch.linalg.qr(V)  # re-orthonormalise
    return Q.contiguous()


def ward_linkage(X, device="cuda", k=8, return_stats=False, search_dim=None):
    """Exact Ward linkage of the rows of ``X`` (Euclidean) on the GPU.

    Parameters
    ----------
    X : array-like or torch.Tensor, shape (n, d)
    k : int
        Number of GEMM candidates re-scored exactly per row.
    search_dim : int or None
        When set and smaller than d, run the candidate search on a
        ``search_dim``-wide principal projection (the tree is still exact).

    Returns
    -------
    Z : np.ndarray, shape (n - 1, 4)
        SciPy linkage matrix.
    """
    if isinstance(X, torch.Tensor):
        C = X.to(device=device, dtype=torch.float32).contiguous()  # centred in place below
    else:
        C = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    n, d = C.shape
    if n < 2:
        return np.zeros((0, 4))
    # Ward is translation invariant; centring shrinks norms and GEMM error.
    # Done in place (fp64 column sums over row chunks) to avoid a full copy.
    copy_rows = max(1, int(2.5e8 // max(1, d)))
    colsum = torch.zeros(d, dtype=torch.float64, device=device)
    for lo in range(0, n, copy_rows):
        colsum += C[lo:lo + copy_rows].double().sum(0)
    C -= (colsum / n).float()
    S = None
    if search_dim is not None and search_dim < d and n > search_dim:
        S = (C @ _projection_basis(C, search_dim)).contiguous()
        k = max(k, 16)
    size = torch.ones(n, dtype=torch.float64, device=device)
    pid = torch.arange(n, dtype=torch.long, device=device)  # provisional ids of active clusters

    child_a, child_b, delta, counts, rounds = [], [], [], [], []
    next_id = n
    rnd = 0
    total_fallback = 0
    total_queries = n
    nn, nnw, nfb = _nearest_neighbours(C, size, k=k, S=S)
    total_fallback += nfb
    while C.shape[0] > 1:
        m = C.shape[0]
        idx = torch.arange(m, device=device)
        a = idx[(nn[nn] == idx) & (idx < nn)]
        if a.numel() == 0:  # cannot happen without exact ties cycling; merge the global best
            a = torch.argmin(nnw)[None]
            a = torch.minimum(a, nn[a])
        b = nn[a]
        sa, sb = size[a], size[b]
        snew = sa + sb
        cnew = ((C[a].double() * sa[:, None] + C[b].double() * sb[:, None]) / snew[:, None]).float()
        p = a.numel()
        new_ids = torch.arange(next_id, next_id + p, device=device)
        next_id += p
        child_a.append(pid[a]); child_b.append(pid[b]); delta.append(nnw[a]); counts.append(snew)
        rounds.append(torch.full((p,), rnd, device=device))
        # a's slot becomes the merged cluster; b's slot is dropped
        C[a] = cnew
        if S is not None:
            S[a] = ((S[a].double() * sa[:, None] + S[b].double() * sb[:, None]) / snew[:, None]).float()
        size[a] = snew
        pid[a] = new_ids
        keep = torch.ones(m, dtype=torch.bool, device=device)
        keep[b] = False
        # By reducibility, a cluster whose nearest neighbour was not merged
        # keeps it; only the merged clusters and rows that pointed into a
        # merged pair need a new search.
        stale = torch.zeros(m, dtype=torch.bool, device=device)
        stale[a] = True
        touched = torch.zeros(m, dtype=torch.bool, device=device)
        touched[a] = True
        touched[b] = True
        stale |= touched[nn]
        new_index = torch.cumsum(keep, 0) - 1
        keep_idx = torch.nonzero(keep)[:, 0]
        C = _compact_rows_(C, keep_idx, copy_rows)
        if S is not None:
            S = _compact_rows_(S, keep_idx, copy_rows)
        size = size[keep]
        pid = pid[keep]
        nn = new_index[nn[keep]]
        nnw = nnw[keep]
        stale = stale[keep]
        rnd += 1
        if C.shape[0] > 1:
            query = torch.nonzero(stale)[:, 0]
            total_queries += query.numel()
            if query.numel():
                qnn, qw, nfb = _nearest_neighbours(C, size, query=query, k=k, S=S)
                nn[query] = qnn
                nnw[query] = qw
                total_fallback += nfb

    ca = torch.cat(child_a).cpu().numpy()
    cb = torch.cat(child_b).cpu().numpy()
    h = np.sqrt(2.0 * np.maximum(torch.cat(delta).cpu().numpy(), 0.0))
    cnt = torch.cat(counts).cpu().numpy()
    rd = torch.cat(rounds).cpu().numpy()

    # Enforce monotone heights against fp noise (children precede parents in
    # provisional order, so one forward pass propagates maxima).
    nm = n - 1
    for i in range(nm):
        for c in (ca[i], cb[i]):
            if c >= n:
                ch = h[c - n]
                if ch > h[i]:
                    h[i] = ch
    order = np.lexsort((np.arange(nm), rd, h))
    final_of = np.empty(nm, dtype=np.int64)
    final_of[order] = np.arange(nm)

    def remap(c):
        c = c.copy()
        big = c >= n
        c[big] = n + final_of[c[big] - n]
        return c

    ca, cb = remap(ca), remap(cb)
    Z = np.empty((nm, 4))
    Z[:, 0] = np.minimum(ca, cb)[order]
    Z[:, 1] = np.maximum(ca, cb)[order]
    Z[:, 2] = h[order]
    Z[:, 3] = cnt[order]
    if return_stats:
        return Z, {"rounds": rnd, "fallback_rows": total_fallback, "queries": total_queries}
    return Z
