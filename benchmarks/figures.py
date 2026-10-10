"""Build the audit figures from measured results (no synthetic datasets)."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import bench_config as cfg
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score

ROOT = cfg.ROOT
RESULTS = ROOT / "results"
ASSETS = cfg.DOCS / "benchmark-assets"
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def result(name):
    return json.loads((RESULTS / (name + ".json")).read_text())


def topology(z):
    # Rooted clades as exact leaf sets: insensitive to child order and node numbering.
    n = len(z) + 1
    membership = [1 << i for i in range(n)]
    clades = set()
    for a, b, *_ in z:
        leaves = membership[int(a)] | membership[int(b)]
        membership.append(leaves)
        clades.add(leaves)
    return clades


def compare_trees(a, b):
    ca, cb = topology(a), topology(b)
    return {
        "changed_clades_A": len(ca - cb),
        "changed_clades_B": len(cb - ca),
        "merge_rows_identical": bool(np.array_equal(a[:, [0, 1, 3]], b[:, [0, 1, 3]])),
        "height_rank_max_absolute_difference": float(np.max(np.abs(np.sort(a[:, 2]) - np.sort(b[:, 2])))),
        "ARI": {
            str(k): float(
                adjusted_rand_score(fcluster(a, k, criterion="maxclust"), fcluster(b, k, criterion="maxclust"))
            )
            for k in (2, 6, 50)
        },
    }


summary = {}
if (RESULTS / "kidney_gpu.cna.npz").exists():
    names = ["kidney_optimistic", "kidney_gpu", "kidney_compat", "kidney_exactks"]
    baseline = np.load(RESULTS / "kidney_optimistic.cna.npz")
    cells = baseline["cells"]
    reference = baseline["values"]
    za = np.load(RESULTS / "kidney_optimistic.linkage.npy")
    summary["kidney"] = {}
    for name in names[1:]:
        with np.load(RESULTS / (name + ".cna.npz")) as z:
            assert np.array_equal(z["annotation"], baseline["annotation"])
            index = {c: i for i, c in enumerate(z["cells"])}
            common = [(i, index[c]) for i, c in enumerate(cells) if c in index]
            # Keep comparisons at original stored precision, aligned by barcode.
            ia, ib = np.asarray(common).T
            delta = z["values"][:, ib] - reference[:, ia]
            flat = delta.ravel()
            summary["kidney"][name] = {
                "shared_cells": len(common),
                "max_abs_CNA_difference": float(np.max(np.abs(flat))),
                "abs_CNA_quantiles": {str(q): float(np.quantile(np.abs(flat), q)) for q in (0.5, 0.99, 0.999)},
                "rms_CNA_difference": float(np.sqrt(np.mean(flat * flat))),
            }
            del delta, flat
        summary["kidney"][name]["tree"] = compare_trees(za, np.load(RESULTS / (name + ".linkage.npy")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for ax, name, title in zip(
        axes[:2], names[:2], ["Main-equivalent five-PR bundle", "GPU full-feature policy"], strict=False
    ):
        z = np.load(RESULTS / (name + ".linkage.npy"))
        dendrogram(
            z,
            truncate_mode="lastp",
            p=16,
            show_leaf_counts=True,
            ax=ax,
            color_threshold=0,
            above_threshold_color="#234a6b",
            leaf_font_size=7,
            leaf_rotation=45,
        )
        ax.set_title(title)
        ax.set_xlabel("Terminal aggregate (labels show cell counts)")
        ax.set_ylabel("Ward merge height")
    values = [result(n)["metrics"]["balanced_accuracy"] for n in names]
    axes[2].bar(
        ["Main-equivalent", "GPU", "GPU compat", "GPU + exact KS"],
        values,
        color=["#234a6b", "#25856b", "#bc4b44", "#8c6cac"],
    )
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Balanced accuracy vs malignant labels")
    for i, v in enumerate(values):
        axes[2].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Bi2021 Kidney P90: geometry and orientation are separate review questions", fontsize=13)
    fig.text(
        0.01,
        0.01,
        (
            "Real 3,726-cell final trees, truncated to 16 aggregates; trees have independent "
            "leaf order. Malignant labels are a proxy, not a CNA oracle."
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    fig.savefig(ASSETS / "kidney-dendrograms.png", dpi=170)
    plt.close(fig)
    # Distribution plot uses actual full-precision differences; breakpoint data shown separately.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["GPU", "GPU compat", "GPU + exact KS"]
    x = np.arange(3)
    for offset, q, color in [(-0.23, ".5", "#234a6b"), (0, ".99", "#25856b"), (0.23, ".999", "#8c6cac")]:
        # JSON string keys are 0.5, 0.99, 0.999.
        vals = [summary["kidney"][n]["abs_CNA_quantiles"][str(float(q))] for n in names[1:]]
        axes[0].bar(x + offset, vals, width=0.23, label=f"p{100 * float(q):g}", color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Absolute final CNA difference from main-equivalent")
    axes[0].legend()
    for row, n in enumerate(["kidney_optimistic", "kidney_gpu", "kidney_exactks"]):
        d = result(n)
        breaks = d.get("breakpoint_attempts", [[]])[-1]
        axes[1].scatter(breaks, [row] * len(breaks), marker="|", s=90, label=n)
    axes[1].set_yticks([0, 1, 2], ["Main-equivalent MC", "GPU MC", "GPU exact KS"])
    axes[1].set_xlabel("Breakpoint index in ordered retained-gene matrix")
    axes[1].set_title("Positions are specific to each retained-gene matrix")
    fig.suptitle("Kidney: full-precision CNA deviations and selected breakpoints")
    fig.tight_layout()
    fig.savefig(ASSETS / "kidney-cna-and-breakpoints.png", dpi=170)
    plt.close(fig)
    baseline.close()
    del reference

if (RESULTS / "t989_collapse.linkage.npy").exists():
    a = np.load(RESULTS / "t989_ward.linkage.npy")
    b = np.load(RESULTS / "t989_collapse.linkage.npy")
    summary["T989"] = compare_trees(a, b)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, z, title in zip(
        axes[:2], [a, b], ["Before collapse: float32 PCA", "After collapse: weighted float64 features"], strict=False
    ):
        dendrogram(
            z,
            truncate_mode="lastp",
            p=16,
            show_leaf_counts=True,
            ax=ax,
            color_threshold=0,
            above_threshold_color="#234a6b",
            leaf_font_size=7,
            leaf_rotation=45,
        )
        ax.set_title(title)
        ax.set_xlabel("Terminal aggregate cell counts")
        ax.set_ylabel("Ward height")
    difference = np.sort(b[:, 2]) - np.sort(a[:, 2])
    axes[2].plot(np.arange(len(difference)), difference, lw=0.7, color="#bc4b44")
    axes[2].set_xlabel("Merge height rank (not matched node identity)")
    axes[2].set_ylabel("Difference in sorted merge heights")
    metrics = summary["T989"]
    fig.suptitle(
        f"T989 mouse: {metrics['changed_clades_A']} clades replaced; ARI at k=2/6/50 = "
        + "/".join(f"{metrics['ARI'][str(k)]:.3f}" for k in (2, 6, 50)),
        fontsize=12,
    )
    fig.text(
        0.01,
        0.01,
        (
            "Full-dataset trees truncated to 16 aggregates. Here topology matches; linkage "
            "ordering and small merge-height differences remain observable."
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(ASSETS / "t989-dendrograms.png", dpi=170)
    plt.close(fig)

(RESULTS / "scientific_comparisons.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
