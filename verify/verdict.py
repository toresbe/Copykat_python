#!/usr/bin/env python
"""Summarise comparison tables from verify/compare.py: did both runs decide the same?

Usage: verify/verdict.py TABLE.tsv [TABLE.tsv ...]

CopyKAT makes a handful of decisions; everything else is arithmetic on top of
them. A table passes when every decision is the same in both runs:

  steps 1-3  the same cells survive filtering
  step 4     the same normal reference cells, clusters and low-confidence warning
  step 5     the same number of segmentation attempts and the same breakpoints
  step 8     the same final clusters and the same call for every cell

The last column shows how far apart the genomic-bin matrices are: about 1e-10
when this package replays R's draws (the default), and Monte Carlo noise with
--exact-segment-means or between R runs with n.cores > 1. Exits with status 1 if
a comparison against this package (not R against itself) has a different decision.
"""

import csv
import sys
from pathlib import Path


def read(path: Path) -> dict[tuple[str, str], str]:
    with path.open() as handle:
        rows = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        next(rows)  # header
        return {(step, metric): value for step, metric, value in rows}


def same_triplet(value: str) -> bool:
    """'294 294 294' (A, B, shared): A, B and their overlap are all equal."""
    a, b, shared = value.split()
    return a == b == shared


def decisions(t: dict[tuple[str, str], str]) -> dict[str, bool]:
    warning_a, warning_b = t["4 baseline", "warning_A_B"].split(" ", 1)
    low_a, low_b = t["8 calls", "low_confidence_flag_A_B"].split()
    attempts_a, attempts_b = t["5 segmentation", "attempts_A_B"].split()
    return {
        "cells": t["1-3 filtering", "cells_only_in_A"] == t["1-3 filtering", "cells_only_in_B"] == "0",
        "normal cells": same_triplet(t["4 baseline", "normal_cells_A_B_shared"]),
        "step-4 clusters": float(t["4 baseline", "clusters_ARI"]) == 1.0,
        "warning": warning_a == warning_b,
        "attempts": attempts_a == attempts_b,
        "breakpoints": same_triplet(t["5 segmentation", "breakpoints_A_B_shared"]),
        "final clusters": float(t["8 final", "clusters_ARI"]) == 1.0,
        "calls": t["8 calls", "discordant_cells"] == "0" and low_a == low_b,
    }


def main() -> int:
    failed = False
    print(
        "| comparison | verdict | differing decisions | calls differing | breakpoints (A B shared) | max diff, bins |"
    )
    print("|---|---|---|---|---|---|")
    for name in sys.argv[1:]:
        path = Path(name)
        table = read(path)
        checks = decisions(table)
        differing = [check for check, ok in checks.items() if not ok]
        verdict = "same decisions" if not differing else "DIFFERENT"
        r_vs_r = "4cores" in path.name
        failed |= bool(differing) and not r_vs_r
        print(
            f"| {path.parent.name}: {path.stem} | {verdict} | {', '.join(differing) or '-'} | "
            f"{table['8 calls', 'discordant_cells']} of {table['8 calls', 'cells_compared']} | "
            f"{table['5 segmentation', 'breakpoints_A_B_shared']} | {table['6 bins', 'max_abs_diff']} |"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
