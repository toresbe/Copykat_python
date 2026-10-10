"""Compare two sweep output directories sample by sample.

Usage: python compare.py RUN_A RUN_B

For each sample present in both: speedup (wall A / wall B), fraction of cells
with the same aneuploid/diploid call, and the difference between the final
CNA matrices (float16 copies saved by run_sample.py): Pearson r and the 99.9th
percentile / max absolute difference over the cells both runs kept.
"""

import json
import os
import sys

import numpy as np
import pandas as pd


def _load(run):
    rows = {}
    with open(os.path.join(run, "results.jsonl")) as f:
        for line in f:
            if line.startswith("FAILED"):
                continue
            d = json.loads(line)
            rows[d["sample"]] = d
    return rows


def _calls(run, sample):
    p = pd.read_csv(os.path.join(run, sample, "prediction.tsv"), sep="\t", dtype=str)
    s = p.set_index("cell.names")["copykat.pred"]
    return s[~s.str.contains("not.defined")].str.contains("aneuploid")


def main(a, b):
    A, B = _load(a), _load(b)
    print(f"{'sample':30s} {'speedup':>8s} {'agree':>7s} {'r':>9s} {'p99.9|d|':>9s} {'max|d|':>8s}")
    for s in A:
        if s not in B:
            continue
        sp = A[s]["wall_seconds"] / B[s]["wall_seconds"]
        ca, cb = _calls(a, s), _calls(b, s)
        common = ca.index.intersection(cb.index)
        agree = float((ca.loc[common] == cb.loc[common]).mean())
        ma = np.load(os.path.join(a, s, "cna.npy")).astype(np.float32)
        mb = np.load(os.path.join(b, s, "cna.npy")).astype(np.float32)
        na = np.load(os.path.join(a, s, "cna_cells.npy"), allow_pickle=True)
        nb = np.load(os.path.join(b, s, "cna_cells.npy"), allow_pickle=True)
        ia = pd.Index(na)
        ib = pd.Index(nb)
        both = ia.intersection(ib)
        xa = ma[:, ia.get_indexer(both)]
        xb = mb[:, ib.get_indexer(both)]
        r = np.corrcoef(xa.ravel(), xb.ravel())[0, 1] if xa.size else float("nan")
        d = np.abs(xa - xb)
        print(f"{s[:30]:30s} {sp:8.1f} {agree:7.4f} {r:9.6f} {np.quantile(d, 0.999):9.5f} {d.max():8.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
