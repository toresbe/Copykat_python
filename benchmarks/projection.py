"""Auditable conditional 170k projections; censored totals never enter fits."""

import numpy as np

CLUSTER_STEPS = {"baseline_estimation", "baseline_adjustment", "final_prediction"}


def project(data, variant, target=170057):
    observed = data[variant, target]
    if observed.get("status") not in {"memory_guard", "skipped_by_user"}:
        return None
    # Preserve stages that actually finished in the stopped target run.
    target_steps = {s["step"]: s for s in observed.get("steps", [])}
    anchor = data[variant, 80000]
    anchor_steps = {s["step"]: s for s in anchor["steps"]}
    estimates = []
    for name, step in anchor_steps.items():
        if name in target_steps:
            value = float(target_steps[name]["seconds"])
            estimates.append(
                {
                    "step": name,
                    "method": "observed completed target stage",
                    "seconds": value,
                    "model_range_s": [value, value],
                }
            )
            continue
        candidates = []
        for (v, n), result in sorted(data.items()):
            if v != variant or n >= target or n < 80000:
                continue
            s = next((s for s in result.get("steps", []) if s["step"] == name), None)
            if s is None:
                continue
            # Partial runs contribute only completed stages with matching engines.
            if s.get("engine") != step.get("engine"):
                continue
            candidates.append((n, float(s["seconds"])))
        if not candidates:
            raise ValueError("Missing completed calibration stage " + name)
        n, y = candidates[-1]
        ratio = target / n
        if name in CLUSTER_STEPS:
            power = (
                float(np.polyfit(np.log([x[0] for x in candidates]), np.log([x[1] for x in candidates]), 1)[0])
                if len(candidates) > 1
                else 2.0
            )
            central = y * ratio**power
            alternatives = [central, y * ratio**2, y * ratio * np.log(target) / np.log(n)]
            method = (
                "same-engine stage power fit"
                if len(candidates) > 1
                else "quadratic stage assumption (one same-engine anchor)"
            )
        else:
            power = 1.0
            central = y * ratio
            alternatives = [central]
            method = "linear stage scaling"
        estimates.append(
            {
                "step": name,
                "method": method,
                "power": power,
                "calibration": candidates,
                "seconds": float(central),
                "model_range_s": [float(min(alternatives)), float(max(alternatives))],
            }
        )
    wall = sum(s["seconds"] for s in estimates)
    wall_range = [sum(s["model_range_s"][i] for s in estimates) for i in [0, 1]]
    # CPU extrapolation is explicitly only the largest completed run's host ratio.
    cpu_ratio = anchor["pipeline_cpu_s"] / anchor["pipeline_s"]
    memory = []
    for (v, n), r in sorted(data.items()):
        if (
            v == variant
            and 20000 <= n <= 80000
            and r.get("status") == "ok"
            and r.get("scheduling_policy", "").startswith("physical8-")
            and r.get("tree_peak_rss_gb") is not None
        ):
            memory.append((n, r["tree_peak_rss_gb"]))
    x, y = np.array(memory).T
    slope, intercept = np.polyfit(x, y, 1)
    rss = float(intercept + slope * target)
    tail_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    tail = float(y[-1] + tail_slope * (target - x[-1]))
    proportional = float(y[-1] * target / x[-1])
    memory_models = [rss, tail, proportional]
    lower = observed.get("tree_peak_rss_gb")
    if (lower is not None and rss < lower) or wall < (observed.get("pipeline_elapsed_at_stop_s") or 0):
        raise ValueError("Projection contradicts observed lower bound")
    return {
        "variant": variant,
        "target_cells": target,
        "elapsed_s": wall,
        "elapsed_model_range_s": wall_range,
        "host_cpu_s": wall * cpu_ratio,
        "host_cpu_model_range_s": [t * cpu_ratio for t in wall_range],
        "host_cpu_ratio": cpu_ratio,
        "tree_peak_rss_gb": rss,
        "memory_model_range_gb": [min(memory_models), max(memory_models)],
        "target_status": observed["status"],
        "observed_stop_elapsed_s": observed.get("pipeline_elapsed_at_stop_s"),
        "observed_peak_rss_lower_bound_gb": lower,
        "completed_stage_evidence": estimates,
        "memory_calibration": memory,
        "memory_slope_gb_per_cell": float(slope),
        "memory_intercept_gb": float(intercept),
        "memory_alternatives_gb": {"tail_linear": tail, "proportional": proportional},
        "interpretation": (
            "Conditional on enough RAM and unchanged algorithms/placement, without "
            "substantial swapping; model sensitivity ranges, not confidence intervals. Peak "
            "tree RSS sums processes and can double-count shared pages; excludes GPU VRAM."
        ),
    }
