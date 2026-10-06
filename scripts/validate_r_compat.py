#!/usr/bin/env python
"""Compare CopyKAT-Py against CopyKAT-R on one count matrix.

R COMPATIBILITY validation harness.  Runs the R package (via ``Rscript``) and
CopyKAT-Py twice -- default mode and ``r_compat=True`` -- on the same
genes x cells CSV, then reports how many cells get the same label.

Usage:
    python scripts/validate_r_compat.py counts.csv --rscript /path/to/Rscript \
        --sample-name demo --n-cores 8 [--workdir out_dir]

The R installation must have the ``copykat`` package (>= 1.2.5).  The expected
outcome is: r_compat agreement ~100% (a few borderline cells may differ because
R's segmentation is Monte Carlo); default-mode agreement is lower on samples
where the documented deviations matter (see copykat_py/r_compat.py).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

R_SCRIPT = """
library(copykat)
x <- read.csv("{counts}", row.names = 1, check.names = FALSE)
res <- copykat(rawmat = as.matrix(x), id.type = "S", sam.name = "{name}",
               genome = "hg20", n.cores = {cores}, plot.genes = FALSE)
"""


def run_r(counts, name, rscript, cores, workdir):
    os.makedirs(workdir, exist_ok=True)
    script = os.path.join(workdir, "run_r.R")
    with open(script, "w") as fh:
        fh.write(R_SCRIPT.format(counts=os.path.abspath(counts), name=name, cores=cores))
    subprocess.run([rscript, script], cwd=workdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pd.read_csv(os.path.join(workdir, f"{name}_copykat_prediction.txt"), sep="\t", index_col=0).iloc[:, 0]


def run_py(counts, name, cores, workdir, r_compat):
    from copykat_py import copykat

    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        x = pd.read_csv(os.path.join(cwd, counts) if not os.path.isabs(counts) else counts, index_col=0)
        res = copykat(rawmat=x, id_type="S", sam_name=name, genome="hg20", n_cores=cores,
                      r_compat=r_compat, plot_genes=False)
    finally:
        os.chdir(cwd)
    return res["prediction"].set_index("cell.names")["copykat.pred"]


def name_key(name):
    # R COMPATIBILITY: R's data.frame() mangles cell names -- every character that is not
    # alphanumeric, "_" or "." becomes "." and purely numeric names get an "X" prefix.  Apply the
    # same mangling (and drop that prefix) on both sides so cells can be matched by name.
    k = re.sub(r"[^A-Za-z0-9_.]", ".", str(name))
    return k[1:] if re.fullmatch(r"X\d+", k) else k


def norm(label):
    # R COMPATIBILITY: R only emits "low.conf" suffixed labels when WNS says so; compare on the
    # diploid/aneuploid/not.defined core so a confidence flag alone is not a disagreement.
    for core in ("aneuploid", "diploid", "not.defined"):
        if core in label:
            return core
    return label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counts", help="genes x cells CSV with gene symbols as row names")
    ap.add_argument("--rscript", required=True)
    ap.add_argument("--sample-name", default="sample")
    ap.add_argument("--n-cores", type=int, default=4)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--min-agreement", type=float, default=0.995, help="exit non-zero if r_compat agreement is lower")
    a = ap.parse_args()

    wd = a.workdir or tempfile.mkdtemp(prefix="copykat_validate_")
    pred_r = run_r(a.counts, a.sample_name, a.rscript, a.n_cores, os.path.join(wd, "R"))
    out = {}
    for mode, flag in (("default", False), ("r_compat", True)):
        p = run_py(a.counts, a.sample_name, a.n_cores, os.path.join(wd, mode), flag)
        r_k = pred_r.rename(index=name_key)
        p_k = p.rename(index=name_key)
        j = pd.concat([r_k.rename("R"), p_k.rename("Py")], axis=1, join="inner")
        if len(j) < len(pred_r):
            print(f"  warning: only {len(j)} of {len(pred_r)} R cells could be matched by name")
        j = j.map(norm)
        agree = float((j["R"] == j["Py"]).mean())
        out[mode] = agree
        print(f"\n== {mode}: agreement with R = {agree:.4f}")
        print(pd.crosstab(j["R"], j["Py"]))
    print(f"\nwork directory: {wd}")
    sys.exit(0 if out["r_compat"] >= a.min_agreement else 1)


if __name__ == "__main__":
    main()
