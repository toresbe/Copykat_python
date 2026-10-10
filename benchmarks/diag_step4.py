"""Diagnose step 4 (normal-anchor selection) for one sample and backend.

Runs copykat() up to baseline_norm_cl by monkeypatching it, then prints, for
each of the step-4 Ward clusters: size, GMM sigma (the selection criterion),
and the fraction of truly malignant cells. Usage:
    python diag_step4.py SAMPLE BACKEND
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contextlib
import importlib

from datasets import load_sample

ck = importlib.import_module("copykat_py.copykat")
bl = importlib.import_module("copykat_py.baseline")


class _Stop(Exception):
    pass


def main(sample, backend_name):
    rawmat, truth = load_sample(sample)
    orig = bl.baseline_norm_cl
    gmm = bl._fit_gmm_3component

    def spy(norm_mat_smooth, min_cells=5, n_cores=1, cell_names=None, pca_components=None, genome="hg20"):
        res = orig(
            norm_mat_smooth,
            min_cells=min_cells,
            n_cores=n_cores,
            cell_names=cell_names,
            pca_components=pca_components,
            genome=genome,
        )
        labels = res["cl"]
        names = np.asarray(cell_names, dtype=object)
        mal = truth.reindex(names).to_numpy()
        print(f"WNS={res['WNS']!r} engine={bl.get_last_cluster_info()['engine']}")
        for cid in sorted(set(labels)):
            m = labels == cid
            cons = np.median(norm_mat_smooth[:, m], axis=1)
            sx = max(0.05, 0.5 * np.std(cons))
            sigma = gmm(cons, sigma_init=sx, max_iter=5000)[2]
            print(f"  cluster {cid}: n={m.sum():6d} sigma={sigma:.5f} malignant={np.nanmean(mal[m].astype(float)):.3f}")
        pre = set(res["preN"])
        print(f"  preN n={len(pre)} malignant={np.nanmean(truth.reindex(list(pre)).astype(float)):.3f}")
        raise _Stop

    ck.baseline_norm_cl = spy
    os.chdir(os.path.expanduser("~/.cache/copykat_bench/diag"))
    with contextlib.suppress(_Stop):
        ck.copykat(rawmat, sam_name="diag", n_cores=32, plot_genes=False, backend_name=backend_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
