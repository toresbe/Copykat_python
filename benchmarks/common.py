"""Helpers shared by the benchmark driver and the campaign scripts."""

import json
import os
from pathlib import Path

import bench_config as cfg

ROOT = cfg.ROOT


def save(path, obj):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def proc_rss(pid):
    # RSS over the process tree. Shared pages are counted for each process.
    try:
        stat = Path(f"/proc/{pid}/status").read_text()
        rss = next(int(x.split()[1]) * 1024 for x in stat.splitlines() if x.startswith("VmRSS:"))
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(proc_rss(int(c)) for c in children)
    except (OSError, StopIteration):
        return 0


def tree_metrics(pid, observed):
    """CPU ticks include all threads; c{u,s}time adds reaped children.

    Remember processes by PID/start tick to avoid losing exited workers or
    double-counting their CPU after a parent reaps them. Self time and
    aggregated child time are compared against explicitly observed subtree
    time, so a pool child's work is counted once.
    """
    hz = os.sysconf("SC_CLK_TCK")
    current = []
    pending = [pid]
    rss = 0
    while pending:
        p = pending.pop()
        try:
            raw = Path(f"/proc/{p}/stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            key = f"{p}:{fields[19]}"
            row = {
                "pid": p,
                "parent": int(fields[1]),
                "self_s": (int(fields[11]) + int(fields[12])) / hz,
                "children_s": (int(fields[13]) + int(fields[14])) / hz,
            }
            if key not in observed:
                row["parent_key"] = next((k for k, v in observed.items() if v["pid"] == row["parent"]), None)
            else:
                row["parent_key"] = observed[key]["parent_key"]
            observed[key] = row
            current.append(key)
            status = Path(f"/proc/{p}/status").read_text().splitlines()
            rss += next(int(x.split()[1]) * 1024 for x in status if x.startswith("VmRSS:"))
            pending.extend(map(int, Path(f"/proc/{p}/task/{p}/children").read_text().split()))
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    by_parent = {}
    for key, row in observed.items():
        by_parent.setdefault(row["parent_key"], []).append(key)

    def total(key):
        row = observed[key]
        return row["self_s"] + max(row["children_s"], sum(total(k) for k in by_parent.get(key, [])))

    roots = [key for key in observed if observed[key]["parent_key"] not in observed]
    return rss, sum(total(key) for key in roots)


def plain_log_axes(ax, y_subs=(1, 2, 5)):
    """Label log axes with plain numbers (3000, not 3 x 10^3) at a fixed set of ticks.

    Matplotlib's default log formatter switches between plain and scientific
    labels depending on the data range, which makes figures look inconsistent.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=y_subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
