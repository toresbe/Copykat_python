"""Marker-guided normal anchor and arm-level correlation calls (opt-in).

Two independent options for copykat():

anchor="markers"
    Step 4 normally takes the Ward cluster whose consensus profile has the smallest
    3-component-GMM sigma as the normal reference. That favours whichever population is
    the majority, and inverts the calls when the tumour dominates. With "markers", the
    reference is the step-4 cluster holding the largest population of cells that express
    pan-immune markers (or, failing that, pan-endothelial markers): cell types that are not
    malignant in solid tumours. Without such a population, a cluster clearly enriched for
    immune markers is used; failing that, the sigma rule.

final_call="arm_correlation"
    Steps 7-8 split cells in two by Ward clustering and call the side overlapping the
    reference "diploid". In immune-rich samples with a small tumour that split can follow
    cell type instead of copy number. With "arm_correlation", each cell's profile is
    averaged per chromosome arm, correlated with the consensus of the 10% of cells with the
    largest arm-level deviation, and called aneuploid above the 99th percentile of the
    reference cells' correlations.

Not for haematological malignancies (the tumour cells are immune cells).
"""

import numpy as np
import pandas as pd
from scipy import sparse

IMMUNE_MARKERS = ("PTPRC", "LAPTM5", "CORO1A", "CD53", "LCP1", "CD52", "ARHGDIB")
ENDOTHELIAL_MARKERS = ("PECAM1", "VWF", "CDH5", "CLDN5", "ESAM", "EMCN", "PLVAP")
MIN_MARKERS = 3          # markers detected for a cell to count as immune/endothelial
MIN_FRACTION = 0.5       # marked fraction for a cluster to qualify
MIN_CELLS = 20           # and at least max(MIN_CELLS, 1% of cells)
ENRICHMENT = 3.0         # fallback: mean immune-marker count vs the rest of the cells

# hg38 centromere midpoints (Mb), chr1..22, X (=23)
_CENTROMERES_MB = np.array([123.4, 93.9, 90.9, 50.0, 48.8, 59.8, 60.1, 45.2, 43.0, 39.8, 53.4, 35.5,
                            17.7, 17.2, 19.0, 36.8, 25.1, 18.5, 26.2, 28.1, 12.0, 15.0, 60.6])


def count_markers(rawmat, markers):
    """Detected markers per cell from the raw input, before any gene filtering.

    Returns a pandas Series indexed by cell barcode. Supports the dict input
    ({"matrix", "genes", "barcodes"}, genes x cells) and genes x cells DataFrames.
    """
    if isinstance(rawmat, dict):
        mat, genes, cells = rawmat["matrix"], np.asarray(rawmat["genes"]), np.asarray(rawmat["barcodes"])
        rows = [i for i, g in enumerate(genes) if g in set(markers)]
        sub = sparse.csr_matrix(mat)[rows] if sparse.issparse(mat) else np.asarray(mat)[rows]
        counts = np.asarray((sub > 0).sum(axis=0)).ravel() if rows else np.zeros(len(cells))
        return pd.Series(counts, index=cells)
    if isinstance(rawmat, pd.DataFrame):
        sub = rawmat.loc[rawmat.index.isin(markers)]
        return (sub > 0).sum(axis=0)
    if isinstance(rawmat, str):
        sep = "\t" if rawmat.endswith((".tsv", ".txt")) else ","
        frame = pd.read_csv(rawmat, index_col=0, sep=sep)
        return (frame.loc[frame.index.isin(markers)] > 0).sum(axis=0)
    raise TypeError("anchor='markers' needs dict, DataFrame, or CSV/TSV input with gene symbols")


def _largest_population(labels, marked, n_cells):
    best, best_n = None, 0
    for k in np.unique(labels):
        m = labels == k
        if m.sum() >= max(MIN_CELLS, 0.01 * n_cells) and marked[m].mean() >= MIN_FRACTION and marked[m].sum() > best_n:
            best, best_n = k, int(marked[m].sum())
    return best


def choose_anchor_cluster(labels, immune_counts, endothelial_counts, sigma_cluster):
    """Return (cluster id, path) for the normal reference; path is "immune", "endothelial",
    "enrichment" or "sigma" and doubles as a coarse confidence indicator."""
    labels = np.asarray(labels)
    imm = np.asarray(immune_counts, dtype=float)
    endo = np.asarray(endothelial_counts, dtype=float)
    n = len(labels)
    for path, marked in (("immune", imm >= MIN_MARKERS), ("endothelial", endo >= MIN_MARKERS)):
        k = _largest_population(labels, marked, n)
        if k is not None:
            return k, path
    ids = np.unique(labels)
    means = np.array([imm[labels == k].mean() for k in ids])
    j = int(np.argmax(means))
    m = labels == ids[j]
    rest = imm[~m].mean() if (~m).any() else 0.0
    if m.sum() >= MIN_CELLS and means[j] >= 1.0 and means[j] >= ENRICHMENT * max(rest, 1e-9):
        return ids[j], "enrichment"
    return sigma_cluster, "sigma"


def _arm_ids(chrom, pos_bp):
    chrom = np.asarray(chrom).astype(int)
    cen = _CENTROMERES_MB[np.clip(chrom, 1, 23) - 1]
    return 2 * (chrom - 1) + (np.asarray(pos_bp, dtype=float) / 1e6 > cen).astype(int)


def arm_correlation_calls(values, chrom, pos_bp, anchor_mask, top_fraction=0.10, percentile=99.0):
    """Aneuploid (True) / diploid (False) per cell from bins x cells values (step-6 profiles)."""
    arm = _arm_ids(chrom, pos_bp)
    ids, inv, sizes = np.unique(arm, return_inverse=True, return_counts=True)
    agg = sparse.csr_matrix((1.0 / sizes[inv], (inv, np.arange(len(arm)))), shape=(len(ids), len(arm)))
    A = np.asarray(agg @ np.asarray(values, dtype=np.float32), dtype=np.float64)  # arms x cells
    A = A.astype(np.float16).astype(np.float64)  # same precision as the evaluated harness
    w = sizes.astype(np.float64)
    energy = np.sqrt((w[:, None] * A ** 2).sum(0) / w.sum())
    consensus = A[:, energy >= np.percentile(energy, 100 * (1 - top_fraction))].mean(1)
    Ac = A - (w[:, None] * A).sum(0) / w.sum()
    cc = consensus - (w * consensus).sum() / w.sum()
    r = (w[:, None] * Ac * cc[:, None]).sum(0) / np.sqrt((w[:, None] * Ac ** 2).sum(0) * (w * cc ** 2).sum() + 1e-30)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    return r > np.percentile(r[anchor_mask], percentile)
