"""Ward linkage: GPU (copykat_py.gpu.ward) vs fastcluster on real step-4 input.

Uses a saved step-4 matrix (cells x genes, smoothed expression; see
diag_step4/precision captures) and random cell subsets of it. CPU timings
use the CPU path's fastest exact engine (threaded pdist + fastcluster).
Usage: python ward_bench.py STEP4.npy
"""

import sys
import time

import fastcluster
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

from copykat_py.baseline import _pdist_euclidean
from copykat_py.gpu.ward import ward_linkage


def main(path):
    X = np.load(path)
    rng = np.random.default_rng(0)
    ward_linkage(X[:300])
    print("| cells | genes | fastcluster (32 threads) | GPU | speedup | ARI k=2/6/50 |")
    print("|---:|---:|---:|---:|---:|---|")
    for n in (2000, 5000, X.shape[0]):
        Y = X[np.sort(rng.choice(X.shape[0], n, replace=False))] if n < X.shape[0] else X
        t = time.perf_counter()
        Zc = fastcluster.linkage(_pdist_euclidean(Y.astype(np.float64), n_cores=32), method="ward", preserve_input=False)
        tc = time.perf_counter() - t
        torch.cuda.synchronize()
        t = time.perf_counter()
        Zg = ward_linkage(Y)
        torch.cuda.synchronize()
        tg = time.perf_counter() - t
        aris = "/".join(f"{adjusted_rand_score(fcluster(Zc, k, 'maxclust'), fcluster(Zg, k, 'maxclust')):.3f}" for k in (2, 6, 50))
        print(f"| {n} | {Y.shape[1]} | {tc:.1f} s | {tg:.2f} s | {tc / tg:.0f}x | {aris} |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
