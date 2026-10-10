"""Summarize completed and censored audit runs without discarding failures."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from labels import LABELS, SCOPE

ROOT = Path("/home/toresbe/cancer_research/benchmark_2026-10-09")
REPO = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = REPO / "docs/benchmark-assets"
ASSETS.mkdir(exist_ok=True)


def load(name):
    path = RESULTS / (name + ".json")
    return json.loads(path.read_text()) if path.exists() else None


def seconds(d):
    if not d:
        return "not attempted"
    if d["status"] == "ok":
        return f"{d['pipeline_s']:.1f}"
    return f"{d['status']} at {d['elapsed_process_s']:.0f}s process time"


def equal(a, b, key):
    return a.get(key) is not None and a.get(key) == b.get(key)


rows = []
for cells, stem in [(10000, "xenium10k"), (30000, "xenium30k"), (170057, "xenium170k")]:
    runs = {v: load(stem + "_" + v) for v in ["main", "optimistic", "cpu_stack", "gpu"]}
    kept = runs["gpu"]["cna_shape"][1] - 3 if runs["gpu"] else "?"
    rows.append(f"| {cells:,} | {kept:,} | " + " | ".join(seconds(runs[v]) for v in runs) + " |")

lines = [
    "# Cursory benchmark evidence for the proposed changes",
    "",
    "Follow-up: [uncapped CPU-accounted 10k / 30k / 170k runs](benchmark-cpu.md). "
    "The capped observations below are retained as original evidence.",
    "",
    "Fresh measurements on 2026-10-09; one cold pipeline run per mode/size, 8 requested cores. "
    "These runs prioritize reviews, not statistically stable performance estimates. "
    "Source snapshots were exported from Git without changing branches. "
    "GPU access required execution outside the sandbox; the RTX 5080 driver works normally there. "
    "“Main” in the measurements means upstream at ea1a15c, not your proposed experimental fork main.",
    "",
    "CPU/GPU modes use recorded Python environments and dependencies. Inputs and calculation scratch are on the SSD. "
    "Xenium subsets are nested deterministic subsets (seed 20261009). "
    "Pipeline time excludes input loading and audit hashing/compression, includes heatmaps and table formatting. "
    "Scaling tables go to `/dev/null`, so these timings exclude persistent output storage throughput. "
    "Kidney output files were actually written to SSD and hashed. "
    "Peak memory is sampled sum of RSS across the process tree, including input loading and audit work; "
    "it double-counts shared pages and is not PSS or GPU VRAM.",
    "",
    "## 10k / 30k / full 170k scaling",
    "",
    (
        "| Input cells | Final cells (GPU) | Upstream main, s | Earlier five-PR subset, "
        "s | CPU experimental (no verification), s | GPU experimental (no verification), "
        "s |"
    ),
    "|---:|---:|---|---|---|---|",
    *rows,
    "",
    "![Fresh Xenium scaling](benchmark-assets/scaling.png)",
    "",
    SCOPE + " "
    "GPU = `hardcore-optimization` at `14aae8d`, `backend=gpu`, default Monte Carlo KS. "
    "These integration timings do not isolate individual candidates or justify "
    "classifying the larger stacks as bit-identical.",
    "",
    "At 170k, no fresh actual-main or five-candidate run was attempted: the saved pre-GPU 170k log already spends "
    "47m10s in baseline estimation, 69m29s in baseline adjustment and 58m24s in "
    "final prediction, before finishing the heatmap. "
    "That older log is from d23282b, not actual main, and is not a completed runtime or a fresh result. "
    "The fresh larger-CPU-stack run is bounded to 180 seconds. Any timeout is a "
    "censored observation, not a completed timing. "
    "No 170k speedup ratio is inferred from that partial log.",
    "",
    "## What the conservative bundle preserves",
    "",
    (
        "| Dataset | Main, s | Bundle, s | Speedup | Main / bundle peak tree RSS, GB | "
        "Numeric CNA / calls / merge structure / heights |"
    ),
    "|---|---:|---:|---:|---|---|",
]
for name, label in [("kidney", "Bi2021 Kidney P90"), ("xenium10k", "Xenium 10k")]:
    a, b = load(name + "_main"), load(name + "_optimistic")
    if not a or not b:
        continue
    keys = ["cna_numeric_sha256", "prediction_sha256", "tree_structure_sha256", "tree_heights_sha256"]
    checks = " / ".join("identical" if equal(a, b, k) else "DIFFERENT" for k in keys)
    lines.append(
        f"| {label} | {a['pipeline_s']:.1f} | {b['pipeline_s']:.1f} | {a['pipeline_s'] / b['pipeline_s']:.2f}x | "
        f"{a['tree_peak_rss_gb']:.1f} / {b['tree_peak_rss_gb']:.1f} | {checks} |"
    )
lines += [
    "",
    "On Kidney, both CNA TSV files, prediction text, clustering pickle and heatmap PNG are byte-identical. "
    "On Xenium 10k, heatmap/prediction/pickle bytes are identical; the large TSVs were sent to `/dev/null`, "
    "so their bytes were not checked. Original-dtype CNA hashes were checked; no float16 downcast was used. "
    "This sample pair does not exercise every known-normal/fallback branch. Broader "
    "historical validation is separately labeled.",
    "",
    "## Changes that must remain visible to a reviewer",
    "",
    "| Kidney mode | Pipeline, s | Balanced accuracy vs malignant labels | Final cells |",
    "|---|---:|---:|---:|",
]
for name, label in [
    ("kidney_main", "main"),
    ("kidney_optimistic", "five-candidate bundle"),
    ("kidney_gpu", "gpu"),
    ("kidney_compat", "gpu-compat"),
    ("kidney_exactks", "gpu + exact KS"),
]:
    d = load(name)
    if d:
        lines.append(
            f"| {label} | {d['pipeline_s']:.1f} | {d['metrics']['balanced_accuracy']:.4f} | {d['cna_shape'][1] - 3:,} |"
        )
lines += [
    "",
    "`gpu-compat` reproduces a near-complete label inversion despite retaining the approximation policy. "
    "ARI can remain high under a label swap, so partition similarity alone is insufficient. "
    "The full-policy GPU and exact-KS variants also change numeric CNA and many tree clades; "
    "their fast runtime does not make those changes byte-preserving or establish biological correctness.",
    "",
    "![Kidney dendrograms and label orientation](benchmark-assets/kidney-dendrograms.png)",
    "",
    "![Kidney CNA differences and breakpoints](benchmark-assets/kidney-cna-and-breakpoints.png)",
    "",
    "The trees above are final post-adjustment trees comparing whole modes, not a causal isolation of step-4 PCA. "
    "They are truncated to 16 aggregates, with independent leaf orders. "
    "Breakpoint positions refer to retained ordered-gene indices; exact lists are in the run JSON. "
    "These figures motivate the proposed separate calculation changes, rather than "
    "attributing every difference to one kernel.",
]
scientific = (
    json.loads((RESULTS / "scientific_comparisons.json").read_text())
    if (RESULTS / "scientific_comparisons.json").exists()
    else {}
)
if "kidney" in scientific:
    lines += [
        "",
        (
            "| Whole GPU mode vs main-equivalent bundle | Median / p99.9 / max absolute "
            "final CNA deviation | Replaced final-tree clades | ARI k=2 / 6 / 50 |"
        ),
        "|---|---|---:|---|",
    ]
    for name, d in scientific["kidney"].items():
        q, t = d["abs_CNA_quantiles"], d["tree"]
        lines.append(
            f"| {name} | {q['0.5']:.6g} / {q['0.999']:.6g} / {d['max_abs_CNA_difference']:.6g} | "
            f"{t['changed_clades_A']:,} | " + " / ".join(f"{t['ARI'][str(k)]:.3f}" for k in (2, 6, 50)) + " |"
        )
    lines += [
        "",
        "These are original-precision, barcode-aligned differences over all 3,726 shared cells and 12,167 bins. "
        "They combine changes in reference selection, segmentation and arithmetic; they "
        "are not an estimate of isolated FP64 summation error.",
    ]
if "T989" in scientific:
    d = scientific["T989"]
    a, b = load("t989_ward"), load("t989_collapse")
    lines += [
        "",
        "## Real mouse linkage-byte counterexample",
        "",
        f"Fresh T989 replay: {a['pipeline_s']:.1f}s before versus {b['pipeline_s']:.1f}s "
        "after repeated-bin collapse/PCA bypass. "
        f"{d['changed_clades_A']} rooted clades replaced; max difference between sorted merge heights "
        f"{d['height_rank_max_absolute_difference']:.6g}. ARI k=2/6/50: "
        + "/".join(f"{d['ARI'][str(k)]:.6g}" for k in (2, 6, 50))
        + ".",
        "",
        "Final original-dtype CNA hashes: " + ("identical." if equal(a, b, "cna_numeric_sha256") else "different."),
        "Prediction hashes: " + ("identical." if equal(a, b, "prediction_sha256") else "different."),
        "",
        "![T989 dendrograms and height changes](benchmark-assets/t989-dendrograms.png)",
        "",
        "The pre-collapse and collapse snapshots are adjacent revisions `350c1c0` and `5768793`. "
        "This comparison isolates that original bundled change, including its PCA bypass. "
        "Saved separate replay evidence identifies float32 PCA as the mechanism: casting "
        "its input to float64 restores the matching linkage rows/heights to float64 "
        "precision in that replay. "
        "C3 and C4 should therefore be separate reviews. Matching the final two-way "
        "calls does not imply matching the hierarchy.",
    ]

lines += [
    "",
    "## Which independent PRs look promising",
    "",
    "| Candidate | Evidence | Priority interpretation |",
    "|---|---|---|",
    (
        "| Shared DLM gains | Fresh conservative pipeline: Kidney smoothing 2.94s → "
        "0.97s. Saved isolated warm sweeps also show substantial stage gains. | High for "
        "a small invariant-based review; pipeline impact is limited by other stages. |"
    ),
    (
        "| Thread pools | Saved independent function outputs are hash-identical across "
        "1/8/32 cores; cluster-fit worker PSS drops sharply. Conservative bundle peak "
        "tree RSS also drops ~44% on Kidney. | High memory/value ratio; do not attribute "
        "the entire bundle reduction to this one change. |"
    ),
    (
        "| Heatmap interpolation | Identical fresh Kidney/Xenium PNGs. Kidney heatmap "
        "46.3s → 42.9s in the conservative bundle; saved larger cases show "
        "rendering/memory gains. | Small review and useful memory reduction; ordering "
        "can dominate total plot time. |"
    ),
    (
        "| Skip unused GMM clustering | Saved outputs support equivalence; saved draft "
        "explicitly lacks isolated performance measurements. | Promising conditional "
        "shortcut; needs one isolated fallback benchmark before a speed claim. |"
    ),
    (
        "| Known-normal set | Exact membership shortcut; fresh default-mode samples do "
        "not exercise it. | Tiny review; benefit conditional on known-normal input. |"
    ),
    (
        "| Arrow gene writer | Saved independent SCPCL001108 run: writer 22.91s → 2.68s; "
        "pipeline 108.66s → 88.48s. Values preserved, text differs. | Strong immediate "
        "value, but outside strict byte-preserving wave. |"
    ),
    (
        "| Storage/Ward/collapse series | Fresh 10k larger CPU stack: 152.3s → 45.7s; "
        "full CNA and calls match, height bytes differ. | Largest CPU promise; isolate "
        "component effects and make numerical deviations explicit. |"
    ),
    (
        "| Repeated-row writer | Saved post-CPU-foundation 11-sample stage totals: "
        "output 63.6s → 7.3s. Fresh signed-zero edge test fails byte identity. | Strong "
        "CPU-reusable candidate after fixing run detection; fresh isolated performance "
        "benchmark still pending. |"
    ),
    (
        "| GPU full-policy backend | Fresh 10k/30k/170k: 9.8/19.0/98.8s. | Largest "
        "scaling benefit; bundled algorithm/label deviations require their own audit. |"
    ),
    "",
    "Historical rows are evidence from previous local measurements, not fresh isolated PR benchmarks. "
    "The conservative integration has only five candidates; memory/Ward/output changes remain outside it.",
]
edge = load("writer_signed_zero")
if edge:
    lines += [
        "",
        "Fresh repeated-row writer counterexample: the reference writes `"
        + edge["reference_row1"]
        + "`, while the repeated writer writes `"
        + edge["repeated_writer_row1"]
        + "`. Numerically equal `+0.0` and `-0.0` are merged despite their distinct text representations. "
        "The helper must be fixed before a general byte-preservation claim; see `writer_edges.py`.",
    ]
for name in ["smoothing_main_8", "smoothing_optimistic_8"]:
    d = load(name)
    if d and d.get("warm_s"):
        lines += [
            "",
            f"Fresh isolated DLM {d['variant']} on captured SCPCL001108 input {d['shape']}: cold {d['cold_s']:.4f}s, "
            f"median of three warm calls {statistics.median(d['warm_s']):.4f}s; "
            f"original float32 output SHA256 `{d['sha256']}`.",
        ]
lines += [
    "",
    "## Reproduction and provenance",
    "",
    "Raw results, progress, logs, linkage matrices, original-precision compressed CNA comparisons "
    f"and source manifests: `{ROOT}`. "
    "The NAS archive contains those small evidence files, source snapshots and "
    "review scripts; calculation scratch is excluded.",
    "",
    "Use `prepare.py` to reconstruct snapshots from the recorded Git objects. "
    "The benchmark driver accepts `COPYKAT_REVIEW_ROOT`, `COPYKAT_REVIEW_PYTHON`, "
    "`COPYKAT_REVIEW_GPU_PYTHON` and `COPYKAT_BENCH_XENIUM`. "
    "Stage T989 and the captured DLM input on your SSD before the corresponding follow-up runs. "
    "The shell sweeps document exact calls for this machine. Results include "
    "censored runs and their last completed stages. "
    "Snapshot and input manifests pin the implementations, seed and Xenium input checksum.",
    "",
    "Before upstream performance claims, repeat shortlisted timings in alternating "
    "order with matched dependencies and controlled thread counts. "
    "Before scientific approval, add the per-change intermediate oracles and "
    "held-out validation specified in the proposal. "
    "This cursory integration audit is not a substitute for those focused review packets.",
]
(REPO / "docs/benchmark-scaling.md").write_text("\n".join(lines) + "\n")

# Scaling visualization; timeouts shown as explicitly censored process observations.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
colors = {"main": "#234a6b", "optimistic": "#729ab7", "cpu_stack": "#ba8241", "gpu": "#25856b"}
labels = LABELS
for variant in colors:
    xs, ys, rss = [], [], []
    for cells, stem in [(10000, "xenium10k"), (30000, "xenium30k"), (170057, "xenium170k")]:
        d = load(stem + "_" + variant)
        if d and d["status"] == "ok":
            xs.append(cells)
            ys.append(d["pipeline_s"])
            rss.append(d["tree_peak_rss_gb"])
        elif d:
            axes[0].scatter([cells], [d["elapsed_process_s"]], marker="^", color=colors[variant], s=60)
            offset = {"main": 7, "optimistic": 21, "cpu_stack": 7, "gpu": 7}[variant]
            axes[0].annotate(
                f"{labels[variant]}: process censored",
                (cells, d["elapsed_process_s"]),
                xytext=(0, offset),
                textcoords="offset points",
                fontsize=7,
            )
    if xs:
        axes[0].plot(xs, ys, "o-", color=colors[variant], label=labels[variant])
        axes[1].plot(xs, rss, "o-", color=colors[variant], label=labels[variant])
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([10000, 30000, 170057], ["10k", "30k", "170k"])
    ax.set_xlabel("Input cells")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("")
axes[0].set_title("Wall-clock time (seconds)", pad=14)
axes[0].set_yscale("log")
lower, upper = axes[0].get_ylim()
axes[0].set_ylim(lower, upper * 1.5)
axes[1].set_ylabel("")
axes[1].set_title("System memory usage (GB)", pad=14)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),
    ncol=2,
    fontsize=8,
    frameon=False,
)
fig.suptitle("Fresh Xenium scaling: measured integration modes, 8 CPU cores")
fig.text(
    0.01,
    0.01,
    (
        "TSVs formatted to /dev/null; input loading excluded from pipeline time. "
        "Triangles are process-time caps, not completed pipeline measurements."
    ),
    fontsize=7,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.78))
fig.savefig(ASSETS / "scaling.png", dpi=170)
plt.close(fig)
print("Report and scaling figure updated.")
