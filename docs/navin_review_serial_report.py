"""Serial-repeat averages, spread and repeat consistency; no pooled parallel data."""

import fcntl
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/toresbe/cancer_research/navin_review_2026-10-09/scratch/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navin_review_labels import LABELS, SCOPE

ROOT = Path("/home/toresbe/cancer_research/navin_review_2026-10-09")
report_lock = open(ROOT / "results/serial8_report.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
fcntl.flock(report_lock, fcntl.LOCK_EX)
DOCS = Path(__file__).resolve().parent
manifest_path = ROOT / "serial_sweep_manifest.json"
if not manifest_path.exists():
    raise SystemExit(0)
manifest = json.loads(manifest_path.read_text())
labels = {v: label for v, label in LABELS.items() if v != "optimistic"}
schedule = ROOT / "results/serial8_schedule.json"
events = json.loads(schedule.read_text()) if schedule.exists() else []
started = any(e["event"] == "serial_phase_started" for e in events)
lines = [
    "# Serial 2k–40k repeat benchmarks",
    "",
    "Last refreshed: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + ".",
    "",
    SCOPE,
    "",
    "Status: "
    + ("serial phase started" if started else "queued; waiting for the original sweep and its archive")
    + ".",
    "",
    "Two new runs per implementation and size, strictly one review calculation at a time, "
    "using CPUs 0–7 (eight physical cores), eight requested threads, the same nested inputs "
    "and seed, and unchanged source snapshots. Round two reverses both size and implementation "
    "order. Fresh processes reuse filesystem and compilation caches; caches are not flushed. "
    "No heatmaps; table formatting remains included with large TSVs directed to /dev/null. "
    "Input loading and audit hashing are excluded from pipeline timings. Other host activity "
    "can still affect results. CPU core-seconds exclude GPU device work.",
    "",
    "Means use only successful runs from this serial phase. Parallel measurements and "
    "the earlier isolated anchors are not pooled. Sample standard deviation describes "
    "two observations, not a confidence interval or a precise estimate of variability. "
    "Failed attempts are shown and never averaged as completed runtimes.",
    "",
    (
        "| Input cells | Implementation | Successful / planned | Status | Elapsed mean ± "
        "SD, s | CPU mean ± SD, core-s | Elapsed range, s |"
    ),
    "|---:|---|---:|---|---:|---:|---:|",
]
fig, ax = plt.subplots(figsize=(8, 5))
axes = [ax]
summary = []
keys = ["cna_numeric_sha256", "prediction_sha256", "tree_structure_sha256", "tree_heights_sha256"]
consistency = []
for variant, label in labels.items():
    plotted = []
    for n in manifest["sizes"]:
        runs = []
        states = []
        for rep in [1, 2]:
            name = f"{variant}_xenium{n}_serial8_r{rep}"
            if variant == "proposed6":
                name = f"proposed6_xenium{n}_prcomplete_r{rep}"
            p = ROOT / "results" / (name + ".json")
            live = ROOT / "results" / (name + ".live.json")
            if p.exists():
                d = json.loads(p.read_text())
                states.append(d["status"])
                if d["status"] == "ok" and d.get("pipeline_cpu_s") is not None:
                    runs.append(d)
            elif live.exists():
                states.append(json.loads(live.read_text()).get("status", "running"))
            else:
                states.append("queued")
        row = {"variant": variant, "cells": n, "successful": len(runs), "planned": 2, "statuses": states}
        displays = []
        for field in ["pipeline_s", "pipeline_cpu_s"]:
            values = [d[field] for d in runs]
            row[field] = (
                {
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "minimum": min(values),
                    "maximum": max(values),
                }
                if values
                else None
            )
            if len(values) == 2:
                displays.append(f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}")
            elif values:
                displays.append(f"{values[0]:.2f} (one completed)")
            else:
                displays.append("—")
        ranges = f"{row['pipeline_s']['minimum']:.2f}–{row['pipeline_s']['maximum']:.2f}" if runs else "—"
        lines.append(
            f"| {n:,} | {label} | {len(runs)} / 2 | "
            + ", ".join(states)
            + " | "
            + " | ".join(displays)
            + " | "
            + ranges
            + " |"
        )
        if len(runs) == 2:
            row["repeat_hash_equality"] = {k: runs[0].get(k) is not None and runs[0][k] == runs[1].get(k) for k in keys}
            row["affinity_consistent"] = all(d.get("cpu_affinity") == list(range(8)) for d in runs)
            consistency.append(
                f"| {n:,} | {label} | "
                + " / ".join("identical" if row["repeat_hash_equality"][k] else "different/missing" for k in keys)
                + " |"
            )
            plotted.append((n, row))
        summary.append(row)
    for ax, field in zip(axes, ["pipeline_s"], strict=False):
        if plotted:
            ax.errorbar(
                [n for n, row in plotted],
                [row[field]["mean"] for n, row in plotted],
                yerr=[
                    [row[field]["mean"] - row[field]["minimum"] for n, row in plotted],
                    [row[field]["maximum"] - row[field]["mean"] for n, row in plotted],
                ],
                fmt="o-",
                capsize=3,
                label=label,
            )
for ax, ylabel in zip(axes, ["Wall-clock time (seconds)"], strict=False):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2000, 5000, 10000, 20000, 40000], ["2k", "5k", "10k", "20k", "40k"])
    ax.set_xlabel("Input cells")
    ax.set_ylabel("")
    ax.set_title(ylabel, pad=14)
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Serial repeats — mean and observed range; pairs completed only")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(DOCS / "navin-review-assets/serial8-scaling.png", dpi=180)
plt.close(fig)
lines += [
    "",
    "![Serial averages and range](navin-review-assets/serial8-scaling.png)",
    "",
    "Error bars show the observed minimum–maximum, not confidence intervals. "
    "Only pairs with two completed runs appear in this figure.",
    "",
    "## Repeat output consistency",
    "",
    "| Input cells | Implementation | CNA / predictions / merge structure / heights |",
    "|---:|---|---|",
    *consistency,
    "",
    "Individual JSON results, source/input manifests and serial admission events are "
    f"under `{ROOT}`. Full-precision output hashes are retained. "
    "A separate NAS archive is created when the serial phase completes. "
    "[Original scaling and output checks](navin-review-cpu-benchmarks.md).",
]
summary_path = ROOT / "results/serial8_summary.json"
temporary = summary_path.with_suffix(".tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n")
temporary.replace(summary_path)
report = DOCS / "navin-review-serial-benchmarks.md"
temporary = report.with_suffix(".md.tmp")
temporary.write_text("\n".join(lines) + "\n")
temporary.replace(report)
