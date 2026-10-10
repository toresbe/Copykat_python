"""Report the uncapped CPU-accounted sweep; partial observations are not timings."""

import fcntl
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/toresbe/cancer_research/navin_review_2026-10-09/scratch/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from navin_review_labels import LABELS, SCOPE
from navin_review_projection import project

ROOT = Path("/home/toresbe/cancer_research/navin_review_2026-10-09")
report_lock = open(ROOT / "results/cpuaccount_report.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
fcntl.flock(report_lock, fcntl.LOCK_EX)
DOCS = Path(__file__).resolve().parent
variants = {v: label for v, label in LABELS.items() if v != "optimistic"}
manifest_path = ROOT / "expanded_sweep_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
sizes = manifest.get("sizes", [10000, 30000, 170057])
lines = [
    "# Uncapped CPU-accounted scaling",
    "",
    "Follow-up: [two serial repeat rounds at 2k–40k](navin-review-serial-benchmarks.md), "
    "queued after this sweep completes.",
    "",
    "Last refreshed: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + ".",
    "",
    SCOPE,
    "",
    "These follow-up runs have no elapsed-time limit. Each uses eight requested cores. "
    "The RAM scheduler reserves future growth and at least 11 GB of available memory; "
    "each supervisor stops a run if available RAM falls below 10 GB or SSD space below 2 GB. "
    "New runs are pinned to CPUs 0–7 or 8–15: eight distinct physical cores on the "
    "5950X, with no SMT sibling shared between new pinned jobs. At most two jobs run at once, "
    "with only one new job while the surviving unpinned 170k run finishes. That original "
    "run is preserved without changing its affinity. Memory bandwidth, CPU boost and "
    "cache effects can still interact; timings are exploratory. The schedule records "
    "overlap, and four isolated 10k anchors run after unpinned work finishes. "
    "Fresh worker processes reuse filesystem/input/JIT caches; those caches are not flushed. "
    "CPU core-seconds count user "
    "and system CPU consumed by all threads and descendant workers in a dedicated Linux cgroup, "
    "measured at pipeline boundaries. They exclude loading and audit hashing. CPU time does "
    "not include GPU device work; use elapsed time for CPU/GPU latency comparisons. "
    "This sweep disables heatmap rendering consistently across every mode and size "
    "(plot=false); table formatting remains included, with large tables directed to "
    "/dev/null. The earlier sweep included heatmaps, so its times are not directly "
    "comparable to these calculation-focused measurements. Peak tree RSS sums "
    "processes and can count shared pages more than once; it excludes GPU VRAM.",
    "",
    "The previous 170k CPU point was censored by an explicit 180-second process-time cap "
    "during baseline estimation, not by an observed memory failure. This sweep replaces "
    "that cap with completion or an explicitly reported resource/error outcome. "
    "Original evidence is retained.",
    "",
    (
        "| Input cells | Implementation | Status | Elapsed seconds | CPU core-seconds | "
        "Peak tree RSS, GB | CPU placement |"
    ),
    "|---:|---|---|---:|---:|---:|---|",
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
data = {}
colors = {"main": "#1f77b4", "optimistic": "#ff7f0e", "proposed6": "#9467bd", "cpu_stack": "#2ca02c", "gpu": "#d62728"}
for variant, label in variants.items():
    done = []
    for n in sizes:
        name = f"{variant}_xenium{n}_cpuaccount"
        if variant == "proposed6":
            name = f"proposed6_xenium{n}_prcomplete_r1"
        if variant == "gpu" and (ROOT / "results" / f"gpu_xenium{n}_vram.json").exists():
            name = f"gpu_xenium{n}_vram"
        path = ROOT / "results" / (name + ".json")
        live = ROOT / "results" / (name + ".live.json")
        if path.exists():
            result = json.loads(path.read_text())
            data[variant, n] = result
            status = result["status"]
            wall, cpu = result.get("pipeline_s"), result.get("pipeline_cpu_s")
            rss = result.get("tree_peak_rss_gb")
            if status == "ok" and wall is not None and cpu is not None:
                done.append((n, wall, cpu, result.get("scheduling_policy", "").startswith("physical8-")))
            placement = (
                "pinned physical8"
                if result.get("scheduling_policy", "").startswith("physical8-")
                else "original unpinned"
            )
            lines.append(
                f"| {n:,} | {label} | {status} | "
                + " | ".join(f"{x:.2f}" if x is not None else "—" for x in (wall, cpu, rss))
                + " | "
                + placement
                + " |"
            )
        elif live.exists():
            result = json.loads(live.read_text())
            lines.append(
                f"| {n:,} | {label} | running; {result.get('wall_elapsed_s', 0) / 60:.1f} min "
                f"elapsed | — | — | — | recorded on completion |"
            )
        else:
            lines.append(f"| {n:,} | {label} | queued | — | — | — | — |")
    for ax, column in zip(axes[:1], (1,), strict=False):
        if done:
            (plotted,) = ax.plot(
                [d[0] for d in done], [d[column] for d in done], "-", label=label, color=colors[variant]
            )
            for d in done:
                ax.scatter(
                    d[0],
                    d[column],
                    s=40,
                    zorder=3,
                    edgecolors=plotted.get_color(),
                    facecolors=plotted.get_color() if d[3] else "white",
                )
    memory = [
        (n, r["tree_peak_rss_gb"])
        for (v, n), r in sorted(data.items())
        if v == variant and r.get("status") == "ok" and r.get("tree_peak_rss_gb") is not None
    ]
    if memory:
        axes[1].plot(*zip(*memory, strict=False), "-o", color=colors[variant], label=label, markersize=4)
projections = [
    project(data, v)
    for v in ["main", "optimistic", "proposed6"]
    if (v, 170057) in data and data.get((v, 80000), {}).get("status") == "ok"
]
projections = [p for p in projections if p is not None]
projection_path = ROOT / "results/170k_projections.json"
temporary = projection_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(projections, indent=2, allow_nan=False) + "\n")
temporary.replace(projection_path)
for p in projections:
    v = p["variant"]
    color = colors[v]
    target = p["target_cells"]
    anchor = data[v, 80000]
    for ax, key, bounds, anchor_key in zip(
        axes[:2],
        ["elapsed_s", "tree_peak_rss_gb"],
        ["elapsed_model_range_s", "memory_model_range_gb"],
        ["pipeline_s", "tree_peak_rss_gb"],
        strict=False,
    ):
        value = p[key]
        lo, hi = p[bounds]
        ax.plot([80000, target], [anchor[anchor_key], value], "--", color=color, alpha=0.7)
        ax.errorbar(
            target,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt="X",
            color=color,
            markersize=8,
            capsize=5,
            zorder=5,
            label=variants[v] + " estimate",
        )
    axes[0].annotate(
        f"≈{p['elapsed_s'] / 60:.0f} min",
        (target, p["elapsed_s"]),
        xytext=(-8, {"main": 30, "optimistic": -32, "proposed6": -55}[v]),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=color,
        arrowprops={"arrowstyle": "-", "color": color},
    )
    axes[1].annotate(
        f"≈{p['tree_peak_rss_gb']:.0f} GB",
        (target, p["tree_peak_rss_gb"]),
        xytext=(-8, -27 if v == "main" else 17),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=color,
    )
vram = []
for n in sizes:
    p = ROOT / "results" / f"gpu_xenium{n}_vram.json"
    if p.exists():
        r = json.loads(p.read_text())
        if r.get("status") == "ok" and all(
            r.get(k) is not None for k in ["gpu_peak_vram_gb", "torch_peak_allocated_gb", "torch_peak_reserved_gb"]
        ):
            allocated = r["torch_peak_allocated_gb"]
            reserved = r["torch_peak_reserved_gb"]
            process = r["gpu_peak_vram_gb"]
            if not 0 <= allocated <= reserved:
                raise ValueError(f"Peak counters cannot form an ordered envelope at {n} cells")
            vram.append((n, allocated, reserved - allocated, process))
if vram:
    axes[2].stackplot(
        [d[0] for d in vram],
        *[[d[i] for d in vram] for i in [1, 2]],
        labels=["Live tensor peak", "Additional allocator reservation"],
        colors=["#d62728", "#f49b82"],
        alpha=0.85,
    )
    axes[2].plot(
        [d[0] for d in vram],
        [d[3] for d in vram],
        "o-",
        color="#442222",
        markersize=3,
        label="Sampled process peak (NVML)",
    )
else:
    axes[2].text(
        0.5,
        0.5,
        "Isolated GPU memory repeats queued\nafter accuracy completes",
        ha="center",
        va="center",
        transform=axes[2].transAxes,
        fontsize=9,
    )
for ax, ylabel in zip(
    axes, ["Wall-clock time (seconds)", "System memory usage (GB)", "GPU memory usage (GB)"], strict=False
):
    ax.set_xscale("log")
    ax.set_xlim(1800, 200000)
    ticks = [2000, 5000, 10000, 20000, 40000, 80000, 170057] if manifest else sizes
    ax.set_xticks(ticks, [f"{n / 1000:g}k" if n != 170057 else "170k" for n in ticks])
    ax.set_xlabel("Input cells")
    ax.set_ylabel("")
    if ax is axes[1]:
        ax.set_title(ylabel, pad=30)
        ax.text(
            0.5,
            1.025,
            "Peak process-tree RSS",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    elif ax is axes[2]:
        ax.set_title(ylabel, pad=30)
        ax.text(
            0.5,
            1.025,
            "Independent peak counters",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    else:
        ax.set_title(ylabel, pad=30)
    ax.grid(alpha=0.25)
axes[0].set_yscale("log")
axes[1].set_ylim(0, 185)
axes[1].set_yticks([0, 40, 80, 120, 160])
axes[2].set_ylim(0, 18)
axes[2].set_yticks([0, 4, 8, 12, 16])
handles = [
    Line2D(
        [],
        [],
        color=colors[v],
        marker="o",
        label=label
        + (
            " (queued)"
            if v == "proposed6" and not any(k[0] == v and r.get("status") == "ok" for k, r in data.items())
            else ""
        ),
    )
    for v, label in variants.items()
]
if projections:
    handles += [Line2D([], [], color="gray", linestyle="--", marker="X", label="170k model estimate")]
for ax in axes[:2]:
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
if vram:
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.suptitle("Xenium scaling — measured runs and conditional 170k estimates (X; dashed)")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(DOCS / "navin-review-assets/cpu-accounted-scaling.png", dpi=180)
plt.close(fig)
lines += [
    "",
    "![Measured scaling and conditional 170k estimates](navin-review-assets/cpu-accounted-scaling.png)",
    "",
    "X markers and dashed lines are extrapolated 170k points, not completed runs. "
    "Error bars show model sensitivity, not statistical confidence intervals. "
    "Other queued/running/resource-stopped runs have no completed point in this figure. "
    "The table states their status. Single cold runs establish cursory scaling, "
    "not a stable estimate of isolated PR effects. Filled markers are new physical-core "
    "pinned runs; hollow markers are original unpinned runs. Lines connect sample sizes "
    "and do not imply a single unchanged clustering engine across those sizes.",
]
lines += [
    "",
    "System RAM usage means the sampled sum of benchmark process-tree RSS, "
    "not total host RAM occupancy or unique physical pages. It excludes GPU VRAM. "
    "GPU memory is sampled separately per benchmark process with "
    "[NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html), "
    "excluding desktop/other applications. Polling every 0.25s can miss brief peaks; "
    "The GPU stack compares independent peak counters: the live tensor peak and "
    "the reserved peak minus the live peak, summing to the allocator reserved peak. "
    "The sampled total process peak is overlaid as a separate line. These counters "
    "need not reach their peaks together. The NVML samples can miss a short-lived "
    "peak: at 120k the sampled total is below the allocator reserved high-water mark. "
    "The reserved-minus-allocated band reflects allocator headroom, including cached "
    "blocks, but cannot quantify the cache at the instant of peak process usage. "
    "The reruns are serial "
    "after accuracy, so another review calculation cannot force smaller GPU batches. "
    "Fresh GPU points replace the older GPU timings as they complete. Raw old results "
    "remain preserved.",
    "",
    "## Why the earlier five-PR timings were similar",
    "",
    "The snapshot was upstream plus the five recorded PR patches, not the wrong source. "
    "At 40k, the first serial main/subset runs took 1266.89/1261.79s. Shared gains reduced "
    "DLM smoothing from 9.27s to 1.95s, while the three clustering stages still totaled "
    "about 1038s and gene-table formatting took about 166s. The Arrow writer PR #5 was "
    "missing from this subset. Heatmap interpolation is outside these plot=false timings; "
    "known-normal membership and discarded fallback clustering help conditional paths, "
    "not every default run. The new six-PR sweep retains the real writer and directs "
    "its output to /dev/null, preserving formatting work while excluding SSD throughput.",
]
if projections:
    lines += [
        "",
        "## Conditional 170k projections",
        "",
        (
            "| Implementation | Elapsed estimate, min | Time model range, min | Host CPU "
            "estimate, core-min | Peak tree RSS estimate, GB | Memory model range, GB | "
            "Observed stop, min / GB |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in projections:
        lines.append(
            f"| {variants[p['variant']]} | {p['elapsed_s'] / 60:.1f} | "
            + "–".join(f"{x / 60:.1f}" for x in p["elapsed_model_range_s"])
            + f" | {p['host_cpu_s'] / 60:.1f} | {p['tree_peak_rss_gb']:.1f} | "
            + "–".join(f"{x:.1f}" for x in p["memory_model_range_gb"])
            + " | "
            + (
                f"{p['observed_stop_elapsed_s'] / 60:.1f} / {p['observed_peak_rss_lower_bound_gb']:.1f}"
                if p["observed_stop_elapsed_s"] is not None
                else "not run; skipped by request"
            )
            + " |"
        )
    lines += [
        "",
        "These estimates assume sufficient RAM, the same algorithms and CPU placement, "
        "and no substantial swapping. They are not predictions of completion time under "
        "the current memory guard or heavy paging. Already completed stages in the stopped "
        "170k run retain their observed time. The six-PR 170k run was skipped by "
        "user request: every stage is extrapolated from smaller inputs, with no "
        "measured 170k stage or memory lower bound. Missing clustering stages use completed "
        "80k/120k stages with the same PCA128/fastcluster engine: a power-law fit when "
        "two observations exist, otherwise a quadratic assumption. Model ranges compare "
        "that estimate with quadratic and n·log(n) stage scaling. Other missing stages "
        "scale linearly from the largest completed stage. Completed stages from stopped "
        "120k runs are usable stage evidence; their incomplete totals never enter the fit. "
        "Host CPU estimates apply the completed 80k CPU/elapsed ratio to the stage-based "
        "elapsed estimate; they are weaker extrapolations than the stage timings.",
        "",
        "Memory uses a linear fit with intercept to successful pinned 20k–80k tree-RSS "
        "peaks. Its sensitivity range compares that fit with the last-two-point linear "
        "fit and proportional scaling from 80k. Stopped 120k/170k peaks are lower bounds "
        "and are not treated as completed peaks. Tree RSS can count shared pages more "
        "than once; these values are not exact physical-RAM requirements. The recovered "
        "CPU-stack 170k run has no comparable whole-run tree-RSS history, so its different "
        "cgroup-memory measurement is not plotted as tree RSS. All inputs, stage estimates "
        "and model parameters are saved in `" + str(projection_path) + "`.",
    ]
lines += [
    "",
    "## Output checks",
    "",
    (
        "| Cells | CPU experimental (no verification) vs upstream main | Currently "
        "proposed changes (6 PRs) vs upstream main |"
    ),
    "|---:|---|---|",
]
keys = ["cna_numeric_sha256", "prediction_sha256", "tree_structure_sha256", "tree_heights_sha256"]
for n in sizes:
    base = data.get(("main", n), {})
    comparisons = []
    for v in ["cpu_stack", "proposed6"]:
        other = data.get((v, n), {})
        if base.get("status") != "ok" or other.get("status") != "ok":
            comparisons.append("awaiting completed pair")
        else:
            comparisons.append(
                "; ".join(
                    f"{key.removesuffix('_sha256')}: "
                    + (
                        "identical"
                        if base.get(key) is not None and base[key] == other.get(key)
                        else "different/missing"
                    )
                    for key in keys
                )
            )
    lines.append(f"| {n:,} | " + " | ".join(comparisons) + " |")
lines += [
    "",
    "## Isolated 10k checks",
    "",
    "These diagnostic checks run sequentially on CPUs 0–7 with no other review benchmark "
    "running. Comparing against the earlier 10k observations mixes changes in affinity, "
    "cache state and contention; it cannot attribute a difference to contention alone.",
    "",
    "| Implementation | Status | Elapsed seconds | CPU core-seconds | Original 10k CPU / isolated CPU |",
    "|---|---|---:|---:|---:|",
]
for (variant, n), result in data.items():
    if result.get("recovered_supervision"):
        peak = result.get("cgroup_memory_peak_gb")
        lines.insert(
            -4,
            f"Recovered {variants[variant]} {n:,}: original pipeline/cgroup CPU readings "
            "were retained. Original supervisor RSS-peak history was lost; "
            + (f"whole-run cgroup memory peak is {peak:.2f} GB. " if peak is not None else "")
            + "Cgroup memory includes charged cache and is a different metric from tree RSS.",
        )
for variant, label in variants.items():
    path = ROOT / "results" / f"{variant}_xenium10000_isolated8.json"
    control = json.loads(path.read_text()) if path.exists() else {}
    original = data.get((variant, 10000), {})
    ratio = original.get("pipeline_cpu_s", 0) / control["pipeline_cpu_s"] if control.get("pipeline_cpu_s", 0) else None
    values = [control.get("pipeline_s"), control.get("pipeline_cpu_s"), ratio]
    lines.append(
        "| "
        + label
        + " | "
        + control.get("status", "queued")
        + " | "
        + " | ".join(f"{value:.2f}" if value is not None else "—" for value in values)
        + " |"
    )
lines += [
    "",
    "## Clustering paths actually used",
    "",
    "| Input cells | Implementation | Baseline estimation | Baseline adjustment | Final prediction |",
    "|---:|---|---|---|---|",
]
for (variant, n), result in data.items():
    steps = {step["step"]: step for step in result.get("steps", [])}
    engines = [
        steps.get(name, {}).get("engine", "—")
        for name in ["baseline_estimation", "baseline_adjustment", "final_prediction"]
    ]
    lines.append(f"| {n:,} | {variants[variant]} | " + " | ".join(engines) + " |")
lines += [
    "",
    f"Raw evidence: `{ROOT}/results`. "
    "The scheduler JSON records every admission, concurrent cohort, RAM estimate and completion. "
    "Source/input manifests and the deterministic nested sampling seed are shared with "
    "[the initial review report](navin-review-benchmarks.md). "
    "Calculation I/O and joblib memory maps use the SSD; archival evidence uses the NAS.",
]
full_gpu = data.get(("gpu", 170057))
previous_gpu_path = ROOT / "results/xenium170k_gpu.json"
if full_gpu and full_gpu.get("status") == "ok" and previous_gpu_path.exists():
    previous_gpu = json.loads(previous_gpu_path.read_text())
    checks = "; ".join(
        key.removesuffix("_sha256")
        + ": "
        + (
            "identical"
            if previous_gpu.get(key) is not None and previous_gpu[key] == full_gpu.get(key)
            else "different/missing"
        )
        for key in keys
    )
    lines += ["", "Full GPU repeat vs the earlier completed 170k run: " + checks + "."]
archive = Path("/mnt/nas/cancer_research/navin_review_2026-10-09/navin-review-cpuaccount-evidence.tar.gz")
lines += [
    "",
    "Completed-sweep NAS archive target: `" + str(archive) + "`. "
    "It is created after every queued run has a recorded result, with a SHA256 sidecar; "
    "the original archive is preserved.",
]
(DOCS / "navin-review-cpu-benchmarks.md").write_text("\n".join(lines) + "\n")
