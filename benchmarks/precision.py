"""Quantify the CPU path's approximations against exact computations.

For each sample, runs copykat() with the GPU backend while capturing the
step-4 clustering input and the segmentation input, then measures:

pca_ari_k2/k6  ARI between the CPU path's step-4 partition (sklearn randomized
               PCA to 256 components + Ward) and the exact full-width Ward
               partition, at the 2- and 6-cluster cuts.
sil_sub/exact  Silhouette of the 2-cluster cut on the CPU path's stratified
               subsample vs on all cells (decision threshold: 0.15).
cumsum_err     Max |logCNA| error of the FP32 running-total segment means
               against the FP64 ones.
mc_seed_flip   Breakpoints (union over cluster consensus profiles) that
               differ when only the Monte Carlo seed changes.
mc_vs_exact    Breakpoints that differ between the MC KS test and the exact
               KS distance.

Usage: python precision.py OUT.jsonl [SAMPLE ...]
"""

import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_sample, sample_names  # noqa: E402

ck = importlib.import_module("copykat_py.copykat")
bl = importlib.import_module("copykat_py.baseline")
sg = importlib.import_module("copykat_py.segmentation")


class _Stop(Exception):
    pass


def _capture(sample):
    rawmat, _ = load_sample(sample)
    cap = {}
    orig_norm_cl = bl.baseline_norm_cl
    orig_mcmc = ck.cna_mcmc

    def norm_cl(norm_mat_smooth, **kw):
        cap["smooth"] = np.array(norm_mat_smooth, dtype=np.float32)
        return orig_norm_cl(norm_mat_smooth, **kw)

    def mcmc(clu, fttmat, **kw):
        cap["clu"] = np.asarray(clu).copy()
        cap["ftt"] = np.array(fttmat, dtype=np.float32)
        cap["kw"] = kw
        raise _Stop

    bl.baseline_norm_cl = norm_cl
    ck.baseline_norm_cl = norm_cl
    ck.cna_mcmc = mcmc
    try:
        ck.copykat(rawmat, sam_name="prec", n_cores=32, plot_genes=False, backend_name="gpu")
    except _Stop:
        pass
    finally:
        bl.baseline_norm_cl = orig_norm_cl
        ck.baseline_norm_cl = orig_norm_cl
        ck.cna_mcmc = orig_mcmc
    return cap


def _breakpoints(ftt, clu, bins, cut, method, seed_offset=0):
    CON = np.column_stack([np.median(ftt[:, clu == c], axis=1) for c in sorted(set(clu))])
    E = np.exp(CON)
    BR = set()
    n = ftt.shape[0]
    for c in range(E.shape[1]):
        if method == "exact":
            bre = sg._find_breakpoints_exact(E[:, c], bins, cut)
        else:
            bre = sg._find_breakpoints_for_cluster(E[:, c], bins, cut, rng_seed=42 + c + seed_offset, mc_samples=1000)
        BR.update([0] + bre + [n - 1])
    return BR


def analyse(sample):
    from fastcluster import linkage as fc_linkage
    from scipy.cluster.hierarchy import fcluster
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from copykat_py import backend
    from copykat_py.gpu import ops

    cap = _capture(sample)
    out = {"sample": sample}
    X = cap["smooth"].T  # cells x genes
    n = X.shape[0]

    backend.set_backend("gpu")
    Z_exact, _ = ops.ward_cluster(X)
    pcs = PCA(n_components=min(256, n - 1, X.shape[1]), svd_solver="randomized", random_state=1234).fit_transform(X)
    Z_pca = fc_linkage(pcs.astype(np.float64), method="ward")
    for k in (2, 6):
        out[f"pca_ari_k{k}"] = float(adjusted_rand_score(fcluster(Z_exact, k, "maxclust"), fcluster(Z_pca, k, "maxclust")))

    labels_2 = fcluster(Z_exact, 2, "maxclust")
    out["sil_exact"] = ops.silhouette(X, labels_2)
    if n > 3000:
        rng = np.random.RandomState(1234)
        target = max(3000, min(int(0.20 * n), 20000))
        idx = np.concatenate([
            rng.choice(np.where(labels_2 == c)[0], size=min(len(np.where(labels_2 == c)[0]),
                       max(200, int(target * (labels_2 == c).sum() / n))), replace=False)
            for c in np.unique(labels_2)
        ])
        out["sil_sub"] = float(silhouette_score(X[idx], labels_2[idx]))
    else:
        out["sil_sub"] = out["sil_exact"]

    # Segment means: FP32 running total (CPU path) vs FP64
    ftt, clu = cap["ftt"], cap["clu"]
    bins = cap["kw"].get("bins", 25)
    cut = cap["kw"].get("cut_cor", 0.1)
    BR = sorted(_breakpoints(ftt, clu, bins, cut, "mc"))
    e32 = np.exp(ftt)
    cs = np.vstack([np.zeros((1, e32.shape[1]), np.float32), np.cumsum(e32, axis=0)])
    worst = 0.0
    for left, right in zip(BR[:-1], BR[1:]):
        m32 = np.log(np.maximum((cs[right + 1] - cs[left]) / (right - left + 1), 1e-300))
        m64 = np.log(np.exp(ftt[left:right + 1].astype(np.float64)).mean(axis=0))
        worst = max(worst, float(np.max(np.abs(m32 - m64))))
    out["cumsum_err"] = worst

    br_mc = _breakpoints(ftt, clu, bins, cut, "mc")
    br_mc2 = _breakpoints(ftt, clu, bins, cut, "mc", seed_offset=1000)
    br_ex = _breakpoints(ftt, clu, bins, cut, "exact")
    out["n_breaks_mc"] = len(br_mc)
    out["n_breaks_exact"] = len(br_ex)
    out["mc_seed_flip"] = len(br_mc ^ br_mc2)
    out["mc_vs_exact"] = len(br_mc ^ br_ex)
    return out


def main(path, samples):
    os.makedirs(os.path.expanduser("~/.cache/copykat_bench/diag/precision"), exist_ok=True)
    os.chdir(os.path.expanduser("~/.cache/copykat_bench/diag/precision"))
    for s in samples or sample_names():
        res = analyse(s)
        print(json.dumps(res), flush=True)
        with open(path, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), sys.argv[2:])
