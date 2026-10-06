#!/usr/bin/env python
"""Subsample a 10x Xenium cell_feature_matrix.h5 into a genes x cells CSV for R-vs-Python comparison.

R COMPATIBILITY validation helper.  CopyKAT-R cannot cluster ~170k cells (quadratic ``hclust``), so the
Xenium Atera comparison in R_COMPAT.md uses a subset: cells are sampled proportionally across the
Xenium graph-based clusters (at least 30 per cluster, fixed RNG seed 0), keeping only
"Gene Expression" features.  Needs ``h5py``, ``numpy``, ``pandas`` and ``scipy``.

Usage:
    python scripts/subsample_xenium.py cell_feature_matrix.h5 \
        analysis/clustering/gene_expression_graphclust/clusters.csv 10000 out_prefix
Writes ``out_prefix_counts.csv`` (genes x cells) and ``out_prefix_meta.csv`` (cluster per cell).
"""
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(h5_path, clusters_csv, n_target, prefix):
    n_target = int(n_target)
    f = h5py.File(h5_path)["matrix"]
    bc = np.array([b.decode() for b in f["barcodes"][:]])
    n_feat, n_cell = (int(v) for v in f["shape"][:])
    m = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=(n_feat, n_cell))
    ftype = np.array([x.decode() for x in f["features/feature_type"][:]])
    name = np.array([x.decode() for x in f["features/name"][:]])
    cl = pd.read_csv(clusters_csv, index_col=0)["Cluster"].reindex(bc)

    rng = np.random.RandomState(0)
    idx = []
    for _, g in pd.Series(np.arange(n_cell), index=cl.values).groupby(level=0):
        k = max(30, int(round(n_target * len(g) / n_cell)))
        idx += list(rng.choice(g.values, size=min(k, len(g)), replace=False))
    idx = np.sort(np.array(idx))

    genes = np.where(ftype == "Gene Expression")[0]
    sub = m[:, idx].tocsr()[genes]
    sym = name[genes]
    uniq, inv = np.unique(sym, return_inverse=True)  # duplicate symbols are summed
    agg = sp.csr_matrix((np.ones(len(sym)), (inv, np.arange(len(sym)))), shape=(len(uniq), len(sym))) @ sub
    pd.DataFrame(agg.toarray().astype(np.int32), index=uniq, columns=bc[idx]).to_csv(f"{prefix}_counts.csv")
    pd.DataFrame({"cluster": "c" + cl.iloc[idx].astype(int).astype(str).values}, index=bc[idx]).to_csv(f"{prefix}_meta.csv")
    print(f"wrote {len(idx)} cells x {len(uniq)} genes")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
