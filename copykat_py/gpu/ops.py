"""GPU implementations of the CopyKAT numeric steps (PyTorch).

Each function takes and returns host (NumPy / SciPy) arrays so it can stand in
for the CPU code inside the shared ``copykat()`` control flow. Work that is
independent per cell streams over cell chunks sized to the free device
memory, so matrices larger than the GPU (e.g. 170k cells x 12k bins) work;
host<->device copies are a small fraction of the work they replace.
"""

import numpy as np
import torch

from copykat_py.gpu.kernels import dlm_smooth_gpu
from copykat_py.gpu.ward import ward_linkage

DEVICE = "cuda"
# Width of the principal projection used to find Ward nearest-neighbour
# candidates for wide inputs (see ward.ward_linkage(search_dim=...)).
PROJECTED_SEARCH_DIM = 512
# Below this many cells a full-width search is as fast (11k x 9.6k: 1.5 s).
PROJECTED_SEARCH_MIN_CELLS = 30000


def free_bytes():
    """Device memory available to this process, counting blocks PyTorch's
    caching allocator has reserved but not handed out."""
    free, _ = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


def _chunk(n_items, bytes_per_item, fraction=0.25, minimum=1):
    """Items per chunk so a chunk uses at most ``fraction`` of free device memory."""
    return max(minimum, min(n_items, int(fraction * free_bytes() // max(1, bytes_per_item))))


def _upload(a, dtype=torch.float32):
    """Host array -> contiguous device tensor of ``dtype``.

    Transfers in the source dtype and layout and converts / transposes on
    the device: host-side conversions and F->C transposes are single-threaded
    and far slower than the PCIe copy.
    """
    a = np.asarray(a)
    if a.ndim == 2 and not a.flags.c_contiguous and a.strides[0] < a.strides[1]:
        # column-major-ish (e.g. a row slice of an F-ordered matrix, or the
        # transpose of a C-ordered one): a.T gathers contiguous runs
        return torch.from_numpy(np.ascontiguousarray(a.T)).to(DEVICE).to(dtype).T.contiguous()
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE).to(dtype)


def _download_into(dst, M):
    """dst[...] = M (device tensor), matching dst's dtype and memory layout."""
    M = M.to(torch.from_numpy(np.empty(0, dtype=dst.dtype)).dtype)
    if dst.ndim == 2 and not dst.flags.c_contiguous and dst.strides[0] < dst.strides[1]:
        dst.T[...] = M.T.contiguous().cpu().numpy()
    else:
        dst[...] = M.cpu().numpy()


def _center_rows_inplace(X):
    """Subtract the column mean (accumulated in fp64, in row chunks) from X in place."""
    n = X.shape[0]
    acc = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    step = _chunk(n, 8 * X.shape[1], 0.1)
    for lo in range(0, n, step):
        acc += X[lo:lo + step].double().sum(0)
    X -= (acc / n).to(X.dtype)
    return X


def median(x, dim):
    """np.median semantics (mean of the two middle values) along ``dim``."""
    n = x.shape[dim]
    s, _ = torch.sort(x, dim=dim)
    lo = s.narrow(dim, (n - 1) // 2, 1)
    hi = s.narrow(dim, n // 2, 1)
    return ((lo + hi) / 2).squeeze(dim)


# ---------------------------------------------------------------------------
# Pre-smoothing filters on sparse counts
# ---------------------------------------------------------------------------

class DeviceCounts:
    """A sparse (genes x cells) count matrix held on the device as COO.

    Replaces the host-side sparse conversions and per-chromosome coverage
    matmuls between annotation and smoothing. Indices are int32 and counts
    float32 (exact for integers below 2**24).
    """

    def __init__(self, mat):
        from scipy import sparse

        csr = sparse.csr_matrix(mat)
        n_genes, n_cells = csr.shape
        indptr = torch.from_numpy(csr.indptr.astype(np.int64)).to(DEVICE)
        self.row = torch.repeat_interleave(
            torch.arange(n_genes, dtype=torch.int32, device=DEVICE), torch.diff(indptr))
        self.col = torch.from_numpy(csr.indices.astype(np.int32)).to(DEVICE)
        self.val = torch.from_numpy(csr.data).to(DEVICE).float()
        nz = self.val != 0  # explicit zeros are not detections
        if not bool(nz.all()):
            self.row, self.col, self.val = self.row[nz], self.col[nz], self.val[nz]
        self.shape = (n_genes, n_cells)
        self._col_ptr = None

    def keep_cells_by_chr_coverage(self, chroms, ngene_chr, row_mask=None):
        """Same rule as copykat._keep_cells_by_chr_coverage (nonzero counts per chromosome)."""
        import pandas as pd

        n_genes, n_cells = self.shape
        rows = np.arange(n_genes) if row_mask is None else np.flatnonzero(row_mask)
        codes, uniq = pd.factorize(np.asarray(chroms)[rows], sort=False)
        code_of_row = np.full(n_genes, -1, dtype=np.int64)
        code_of_row[rows] = codes
        code_of_row = torch.as_tensor(code_of_row, device=DEVICE)
        n_chrom = len(uniq)
        counts = torch.zeros(n_chrom * n_cells, dtype=torch.int64, device=DEVICE)
        step = 1 << 26
        for lo in range(0, len(self.row), step):
            c = code_of_row[self.row[lo:lo + step].long()]
            ok = c >= 0
            key = c[ok] * n_cells + self.col[lo:lo + step][ok].long()
            counts += torch.bincount(key, minlength=n_chrom * n_cells)
        counts = counts.view(n_chrom, n_cells)
        keep = (counts.sum(0) >= 5) & (counts > 0).all(0) & (counts.min(0).values >= ngene_chr)
        return keep.cpu().numpy()

    def select_cells(self, keep):
        keep_t = torch.as_tensor(np.asarray(keep, dtype=bool), device=DEVICE)
        new_col = torch.cumsum(keep_t, 0, dtype=torch.int32) - 1
        sel = keep_t[self.col.long()]
        self.row, self.col, self.val = self.row[sel], new_col[self.col[sel].long()], self.val[sel]
        self.shape = (self.shape[0], int(keep_t.sum()))
        self._col_ptr = None

    def positive_per_gene(self):
        return torch.bincount(self.row[self.val > 0].long(), minlength=self.shape[0]).cpu().numpy()

    def dense_columns(self, lo, hi):
        """Columns [lo, hi) as a dense fp64 (genes x (hi - lo)) device tensor."""
        if self._col_ptr is None:
            order = torch.argsort(self.col, stable=True)
            self.row, self.col, self.val = self.row[order], self.col[order], self.val[order]
            self._col_ptr = torch.searchsorted(
                self.col, torch.arange(self.shape[1] + 1, dtype=torch.int32, device=DEVICE)).cpu().numpy()
        a, b = int(self._col_ptr[lo]), int(self._col_ptr[hi])
        out = torch.zeros((self.shape[0], hi - lo), dtype=torch.float64, device=DEVICE)
        out[self.row[a:b].long(), (self.col[a:b] - lo).long()] = self.val[a:b].double()
        return out


# ---------------------------------------------------------------------------
# Step 3: Freeman-Tukey + centring + DLM smoothing
# ---------------------------------------------------------------------------

def _dense_columns(counts, lo, hi):
    """Columns [lo, hi) of a dense or sparse (genes x cells) matrix as an fp64 device tensor."""
    from scipy import sparse

    if isinstance(counts, DeviceCounts):
        return counts.dense_columns(lo, hi)
    if sparse.issparse(counts):
        block = counts[:, lo:hi].tocsc()
        t = torch.sparse_csc_tensor(
            torch.from_numpy(block.indptr.astype(np.int64)),
            torch.from_numpy(block.indices.astype(np.int64)),
            torch.from_numpy(block.data.astype(np.float64)),
            size=block.shape, dtype=torch.float64,
        ).to(DEVICE)
        return t.to_dense()
    return _upload(counts[:, lo:hi], torch.float64)


def freeman_tukey_smooth(counts, K, B):
    """Freeman-Tukey transform, per-cell centring and DLM smoothing on the GPU.

    ``counts`` is the (genes x cells) raw count matrix: dense, SciPy sparse,
    or a ``DeviceCounts`` (sparse input is densified chunk by chunk on the
    device). Returns the float32 smoothed matrix as NumPy.
    """
    from scipy import sparse

    if sparse.issparse(counts):
        counts = counts.tocsc()
    n_genes, n_cells = counts.shape
    out = np.empty((n_genes, n_cells), dtype=np.float32)
    chunk = _chunk(n_cells, n_genes * 8 * 4, 0.5, minimum=256)
    for lo in range(0, n_cells, chunk):
        hi = min(n_cells, lo + chunk)
        y = _dense_columns(counts, lo, hi)
        y = torch.log(torch.sqrt(y) + torch.sqrt(y + 1))
        y -= y.mean(dim=0, keepdim=True)
        out[:, lo:hi] = dlm_smooth_gpu(y.contiguous(), K, B).cpu().numpy()
        del y
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _upload_collapsed(data):
    """Upload the rows of ``data`` (n x d host array) as float32, collapsing runs
    of identical adjacent columns like ``baseline._collapse_repeated_features``
    (each kept column scaled by sqrt(run length); distances are unchanged).

    Rows are streamed in chunks into one device tensor; returns (tensor,
    collapsed width or None).
    """
    n, d = data.shape
    X = torch.empty((n, d), dtype=torch.float32, device=DEVICE)
    changes = torch.zeros(max(d - 1, 0), dtype=torch.bool, device=DEVICE)
    step = _chunk(n, 8 * d, 0.1)
    for lo in range(0, n, step):
        blk = _upload(data[lo:lo + step])
        X[lo:lo + step] = blk
        if d > 1:
            changes |= (blk[:, 1:] != blk[:, :-1]).any(dim=0)
        del blk
    if d < 2:
        return X, None
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=DEVICE), torch.nonzero(changes)[:, 0] + 1])
    if len(starts) == d:
        return X, None
    runs = torch.diff(torch.cat([starts, torch.tensor([d], device=DEVICE)]))
    scale = torch.sqrt(runs.double())
    out = torch.empty((n, len(starts)), dtype=torch.float32, device=DEVICE)
    for lo in range(0, n, step):
        out[lo:lo + step] = (X[lo:lo + step, starts].double() * scale).float()
    del X
    return out, out.shape[1]


def pca_reduce(X, n_components, seed=1234):
    """Randomised PCA scores (n x k) of rows of X, on the GPU (centres X in place)."""
    torch.manual_seed(seed)
    _center_rows_inplace(X)
    U, S, V = torch.svd_lowrank(X, q=n_components + 10, niter=4)
    return X @ V[:, :n_components]


def ward_cluster(data, reduce_to=None, collapse=True):
    """Ward linkage of the rows of ``data`` on the GPU.

    ``reduce_to``: optional PCA width applied (after collapsing repeated
    feature runs) only when the collapsed width exceeds it. Returns
    (Z, engine string).
    """
    engine = "gpu.ward_rnn"
    if collapse:
        X, width = _upload_collapsed(data)
        if width is not None:
            engine = f"dedup{width}+{engine}"
    else:
        X = _upload(data)
    if reduce_to is not None and X.shape[1] > reduce_to:
        X = pca_reduce(X, reduce_to)
        engine = f"gpu.pca{reduce_to}+{engine}"
    search_dim = None
    n, d = X.shape
    if reduce_to is None and d > 4 * PROJECTED_SEARCH_DIM and n > PROJECTED_SEARCH_MIN_CELLS:
        # Exact tree; the projection only narrows the candidate search.
        search_dim = PROJECTED_SEARCH_DIM
        engine = f"{engine}+search{search_dim}"
    Z = ward_linkage(X, search_dim=search_dim, overwrite_input=True)
    return Z, engine


# ---------------------------------------------------------------------------
# Baseline estimation helpers
# ---------------------------------------------------------------------------

def cluster_medians(mat, labels, cluster_ids):
    """Per-cluster median profiles of the columns of ``mat`` (genes x cells).

    Streams gene (row) blocks so the matrix never has to fit on the device.
    """
    n_genes, n_cells = mat.shape
    lab = np.asarray(labels)
    members = [torch.as_tensor(np.flatnonzero(lab == cid), device=DEVICE) for cid in cluster_ids]
    out = [np.empty(n_genes, dtype=mat.dtype) for _ in cluster_ids]
    step = _chunk(n_genes, 4 * 3 * n_cells, 0.3)
    for lo in range(0, n_genes, step):
        hi = min(n_genes, lo + step)
        M = _upload(mat[lo:hi], torch.float32 if mat.dtype == np.float32 else torch.float64)
        for o, idx in zip(out, members):
            o[lo:hi] = median(M[:, idx], dim=1).cpu().numpy()
        del M
    return out


def silhouette(data, labels, block_rows=None):
    """Exact mean silhouette (Euclidean) of the rows of ``data`` on the GPU."""
    X = _upload(data)
    _center_rows_inplace(X)
    n = X.shape[0]
    lab = np.asarray(labels)
    uniq, inv = np.unique(lab, return_inverse=True)
    k = len(uniq)
    inv_t = torch.as_tensor(inv, device=DEVICE)
    onehot = torch.zeros((n, k), dtype=torch.float64, device=DEVICE)
    onehot[torch.arange(n, device=DEVICE), inv_t] = 1.0
    counts = onehot.sum(0)
    nrm = torch.empty(n, dtype=torch.float32, device=DEVICE)
    step = _chunk(n, 8 * X.shape[1], 0.1)
    for lo in range(0, n, step):
        nrm[lo:lo + step] = (X[lo:lo + step].double() ** 2).sum(1).float()
    if block_rows is None:
        block_rows = _chunk(n, 4 * 3 * n, 0.5)
    sums = torch.empty((n, k), dtype=torch.float64, device=DEVICE)
    for lo in range(0, n, block_rows):
        hi = min(n, lo + block_rows)
        D = X[lo:hi] @ X.T
        D.mul_(-2.0).add_(nrm[lo:hi, None]).add_(nrm[None, :]).clamp_(min=0.0).sqrt_()
        D[torch.arange(hi - lo, device=DEVICE), torch.arange(lo, hi, device=DEVICE)] = 0.0
        sums[lo:hi] = (D @ onehot.float()).double()
        del D
    own = counts[inv_t]
    a = sums[torch.arange(n, device=DEVICE), inv_t] / torch.clamp(own - 1, min=1)
    mean_other = sums / counts[None, :]
    mean_other[torch.arange(n, device=DEVICE), inv_t] = float("inf")
    b = mean_other.min(1).values
    s = (b - a) / torch.maximum(a, b)
    s = torch.where(own > 1, s, torch.zeros_like(s))
    s = torch.nan_to_num(s, nan=0.0)
    return float(s.mean())


# ---------------------------------------------------------------------------
# Step 5: segmentation
# ---------------------------------------------------------------------------

def segment_log_means(fttmat, breaks):
    """log(mean(exp(x))) over each segment [BR[i], BR[i+1]] per cell, in FP64.

    Mirrors the closed-form path of ``cna_mcmc`` (segments share boundary
    genes, later segments overwrite them) but differences an FP64 running
    total instead of an FP32 one (whose spacing near 1e4 is ~1e-3).
    """
    n_genes, n_cells = fttmat.shape
    out = np.empty((n_genes, n_cells), dtype=np.float32)
    BR = list(breaks)
    # gene -> segment index (later segments win on shared boundaries)
    seg_of_gene = np.empty(n_genes, dtype=np.int64)
    for s, (left, right) in enumerate(zip(BR[:-1], BR[1:])):
        seg_of_gene[left:right + 1] = s
    left = np.asarray(BR[:-1])
    right = np.asarray(BR[1:])
    seg_len = np.maximum(1, right - left + 1).astype(np.float64)
    # Inclusive segment sums from an fp64 prefix sum over genes: within a
    # cell chunk the running total is exact enough in fp64 (unlike fp32).
    left_t = torch.as_tensor(left, device=DEVICE)
    right_t = torch.as_tensor(right, device=DEVICE)
    seg_idx = torch.as_tensor(seg_of_gene, device=DEVICE)
    inv_len = torch.as_tensor(1.0 / seg_len, device=DEVICE)
    chunk = _chunk(n_cells, n_genes * 8 * 3, 0.25, minimum=64)
    for lo in range(0, n_cells, chunk):
        hi = min(n_cells, lo + chunk)
        E = torch.exp(_upload(fttmat[:, lo:hi], torch.float64))
        csum = torch.cat([torch.zeros((1, hi - lo), dtype=torch.float64, device=DEVICE), torch.cumsum(E, 0)])
        del E
        sums = csum[right_t + 1] - csum[left_t]  # (n_seg x cells)
        del csum
        logm = torch.log(torch.clamp(sums * inv_len[:, None], min=1e-300))
        out[:, lo:hi] = logm[seg_idx].float().cpu().numpy()
        del sums, logm
    return out


# ---------------------------------------------------------------------------
# Step 6: genomic bins
# ---------------------------------------------------------------------------

def bin_medians(values, bin_gene_indices, n_bins, source_bin):
    """Median over each bin's gene rows for every cell, as a host fp64 F-order array.

    ``values`` (genes x cells) float32. Output row ``i`` holds the medians of
    bin ``source_bin[i]`` (lets empty bins copy their nearest valid bin).
    Bins are grouped by gene count so each group is one gather + sort; cells
    are streamed in chunks.
    """
    n_genes, n_cells = values.shape
    by_len = {}
    for b, rows in enumerate(bin_gene_indices):
        if rows:
            by_len.setdefault(len(rows), []).append(b)
    groups = [
        (L, torch.as_tensor(np.array([bin_gene_indices[b] for b in bins], dtype=np.int64), device=DEVICE),
         torch.as_tensor(np.array(bins), device=DEVICE))
        for L, bins in by_len.items()
    ]
    src = torch.as_tensor(np.asarray(source_bin), device=DEVICE)
    result = np.empty((n_bins, n_cells), dtype=np.float64, order="F")
    # per cell: input column, bin medians, gathered output, plus a sort buffer
    chunk = _chunk(n_cells, 4 * (n_genes + 2 * n_bins) + 8 * n_bins, 0.3, minimum=32)
    for lo in range(0, n_cells, chunk):
        hi = min(n_cells, lo + chunk)
        V = _upload(values[:, lo:hi])
        out = torch.zeros((n_bins, hi - lo), dtype=torch.float32, device=DEVICE)
        for L, idx, bt in groups:
            step = _chunk(len(bt), 4 * 2 * L * (hi - lo), 0.2)
            for b0 in range(0, len(bt), step):
                g = V[idx[b0:b0 + step]]  # (nb x L x cells)
                out[bt[b0:b0 + step]] = median(g, dim=1) if L > 1 else g[:, 0]
                del g
        _download_into(result[:, lo:hi], out[src])
        del V, out
    return result


def adjust_baseline(mat, diploid_mask):
    """GPU version of copykat._adjust_baseline_inplace; overwrites ``mat``.

    Same arithmetic as the CPU version. The per-row diploid statistics need
    passes over all cells: (1) diploid row means, (2) re-centre and
    accumulate diploid means, (3) diploid SDs, (4) flatten noise and
    re-centre each cell. The matrix stays on the device when it fits (fp64,
    else fp32 storage with fp64 reductions) and each pass walks it in cell
    chunks so temporaries stay small; otherwise fp64 cell chunks are streamed
    from the host on every pass.
    """
    n_bins, n_cells = mat.shape
    dm = np.asarray(diploid_mask, dtype=bool)
    n_dip = int(dm.sum())
    free = free_bytes()
    work = _chunk(n_cells, n_bins * 8 * 4, 0.15, minimum=64)  # cells per pass chunk
    R = None
    if 1.2 * n_bins * n_cells * 8 < 0.7 * free:
        R = torch.empty((n_cells, n_bins), dtype=torch.float64, device=DEVICE).T
    elif 1.2 * n_bins * n_cells * 4 < 0.7 * free:
        R = torch.empty((n_cells, n_bins), dtype=torch.float32, device=DEVICE).T
    spans = [(lo, min(n_cells, lo + work)) for lo in range(0, n_cells, work)]
    if R is not None:
        for lo, hi in spans:
            R[:, lo:hi] = _upload(mat[:, lo:hi], torch.float64).to(R.dtype)

    def load(lo, hi):
        return R[:, lo:hi] if R is not None else _upload(mat[:, lo:hi], torch.float64)

    def store(lo, hi, M):
        if R is None:
            _download_into(mat[:, lo:hi], M)

    def dip_cols(lo, hi):
        return torch.as_tensor(np.flatnonzero(dm[lo:hi]), device=DEVICE)

    def col_means(M):
        return M.mean(dim=0, dtype=torch.float64, keepdim=True)

    acc = torch.zeros(n_bins, dtype=torch.float64, device=DEVICE)
    for lo, hi in spans:
        acc += load(lo, hi)[:, dip_cols(lo, hi)].sum(1, dtype=torch.float64)
    dip_mean = acc / n_dip

    acc.zero_()
    for lo, hi in spans:
        M = load(lo, hi)
        M -= dip_mean[:, None].to(M.dtype)
        M -= col_means(M).to(M.dtype)
        acc += M[:, dip_cols(lo, hi)].sum(1, dtype=torch.float64)
        store(lo, hi, M)
    base = acc / n_dip

    sq = torch.zeros(n_bins, dtype=torch.float64, device=DEVICE)
    for lo, hi in spans:
        sq += ((load(lo, hi)[:, dip_cols(lo, hi)].double() - base[:, None]) ** 2).sum(1)
    thr = 0.25 * torch.sqrt(sq / n_dip)

    for lo, hi in spans:
        M = load(lo, hi)
        cell_means = col_means(M).to(M.dtype)
        noise = (M - base[:, None].to(M.dtype)).abs() <= thr[:, None].to(M.dtype)
        M = torch.where(noise, cell_means.expand_as(M), M)
        M -= col_means(M).to(M.dtype)
        _download_into(mat[:, lo:hi], M)
        del M, noise
    return mat
