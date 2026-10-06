"""Markdown tables from sweep outputs.

Usage: python report.py RUNS_DIR MODE=dir [MODE=dir ...]
The first MODE is the reference for speedups and agreement. Prints
per-sample wall times and speedups, per-step totals, classification
metrics against the malignant labels, and agreement with the reference.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import _calls  # noqa: E402

STEP_GROUPS = [
    ("load+filter", ["read_and_filter", "annotate_genes", "cell_filter_pre_smoothing", "freeman_tukey_transform"]),
    ("DLM smoothing", ["dlm_smoothing"]),
    ("baseline (step 4)", ["baseline_estimation"]),
    ("segmentation", ["cell_filter_pre_segmentation", "segmentation"]),
    ("bins", ["convert_to_bins"]),
    ("steps 7+8", ["baseline_adjustment", "final_prediction"]),
    ("write outputs", ["write_gene_level_output", "write_final_outputs"]),
    ("heatmap", ["plot_heatmap"]),
]


def load(run):
    rows = {}
    path = os.path.join(run, "results.jsonl")
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if line.startswith("FAILED"):
            continue
        d = json.loads(line)
        rows[d["sample"]] = d
    return rows


def main(runs_dir, specs):
    modes = []
    for spec in specs:
        name, d = spec.split("=", 1)
        modes.append((name, os.path.join(runs_dir, d), load(os.path.join(runs_dir, d))))
    ref_name, ref_dir, ref = modes[0]
    samples = [s for s in ref if all(s in m[2] for m in modes)]

    print("### Wall time per sample (s)\n")
    print("| sample | cells | " + " | ".join(m[0] for m in modes) + " | best speedup |")
    print("|---|---:|" + "---:|" * len(modes) + "---:|")
    totals = np.zeros(len(modes))
    for s in samples:
        walls = [m[2][s]["wall_seconds"] for m in modes]
        totals += walls
        n = modes[0][2][s]["metrics"]["n_scored"] if modes[0][2][s]["metrics"] else ""
        print(f"| {s} | {n} | " + " | ".join(f"{w:.1f}" for w in walls) + f" | {walls[0] / min(walls[1:] or walls):.1f}x |")
    print("| **total** | | " + " | ".join(f"**{t:.0f}**" for t in totals) + f" | **{totals[0] / (totals[1:].min() if len(totals) > 1 else totals[0]):.1f}x** |")

    print("\n### Time per pipeline stage, summed over samples (s)\n")
    print("| stage | " + " | ".join(m[0] for m in modes) + " |")
    print("|---|" + "---:|" * len(modes))
    for label, keys in STEP_GROUPS:
        vals = [sum(m[2][s]["steps"].get(k, 0.0) for s in samples for k in keys) for m in modes]
        print(f"| {label} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    if all(modes[0][2][s]["metrics"] for s in samples):
        print("\n### Balanced accuracy vs. malignant labels (low-confidence calls marked *)\n")
        print("| sample | " + " | ".join(m[0] for m in modes) + " |")
        print("|---|" + "---:|" * len(modes))
        means = np.zeros(len(modes))
        for s in samples:
            cells = []
            for i, m in enumerate(modes):
                met = m[2][s]["metrics"]
                means[i] += met["balanced_accuracy"]
                cells.append(f"{met['balanced_accuracy']:.3f}{'*' if met['low_conf'] else ''}")
            print(f"| {s} | " + " | ".join(cells) + " |")
        print("| **mean** | " + " | ".join(f"**{v / len(samples):.3f}**" for v in means) + " |")

    print(f"\n### Agreement with {ref_name}: same call (fraction of cells) / CNA Pearson r\n")
    print("| sample | " + " | ".join(m[0] for m in modes[1:]) + " |")
    print("|---|" + "---|" * (len(modes) - 1))
    for s in samples:
        cells = []
        ca = _calls(ref_dir, s)
        ma = np.load(os.path.join(ref_dir, s, "cna.npy")).astype(np.float32)
        na = pd.Index(np.load(os.path.join(ref_dir, s, "cna_cells.npy"), allow_pickle=True))
        for name, d, _ in modes[1:]:
            cb = _calls(d, s)
            common = ca.index.intersection(cb.index)
            agree = float((ca.loc[common] == cb.loc[common]).mean())
            mb = np.load(os.path.join(d, s, "cna.npy")).astype(np.float32)
            nb = pd.Index(np.load(os.path.join(d, s, "cna_cells.npy"), allow_pickle=True))
            both = na.intersection(nb)
            r = np.corrcoef(ma[:, na.get_indexer(both)].ravel(), mb[:, nb.get_indexer(both)].ravel())[0, 1]
            cells.append(f"{agree:.4f} / {r:.4f}")
        print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
