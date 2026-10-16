#!/usr/bin/env python
"""Compare two CopyKAT runs step by step.

Usage: verify/compare.py RUN_A RUN_B > comparison.tsv

Each RUN is an output folder of verify/r/run_copykat.R or verify/run_port.sh; both
write their intermediate results to RUN/stages/ in the same format (verify/README.md).
The output is one line per measurement: step, metric, value. Cells are matched by
name, so a different cell order is not a difference.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from copykat_py._stages import load

Row = tuple[str, str, object]


def columns(stage: dict, cells: list[str], colnames: list[str] | None = None) -> np.ndarray:
    """The columns of a saved (genes or bins) x cells matrix, in the order of ``cells``."""
    # R's data.frame step turns "-" in barcodes into "." in the bins matrix, so match on that spelling
    index = {name.replace("-", "."): i for i, name in enumerate(colnames or stage["colnames"])}
    return stage["values"][:, [index[c.replace("-", ".")] for c in cells]]


def by_cell(stage: dict) -> dict[str, object]:
    return dict(zip(stage["names"], stage["values"].ravel(), strict=True))


def first_value(stage: dict) -> object:
    return stage["values"].ravel()[0]


def core_label(label: str) -> str:
    """'c1:diploid:low.conf' -> 'diploid'; the low-confidence flag is reported separately."""
    for core in ("aneuploid", "diploid", "not.defined"):
        if core in label:
            return core
    return label


def matrix_rows(step: str, a: np.ndarray, b: np.ndarray) -> list[Row]:
    if a.shape != b.shape:
        return [(step, "shape_A_vs_B", f"{a.shape} vs {b.shape}")]
    diff = np.abs(a - b)
    per_cell = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]
    return [
        (step, "max_abs_diff", f"{diff.max():.3g}"),
        (step, "p99.9_abs_diff", f"{np.quantile(diff, 0.999):.3g}"),
        (step, "median_cell_correlation", f"{np.median(per_cell):.6f}"),
        (step, "min_cell_correlation", f"{np.min(per_cell):.6f}"),
    ]


def ari(a: dict, b: dict, cells: list[str]) -> str:
    return f"{adjusted_rand_score([a[c] for c in cells], [b[c] for c in cells]):.6f}"


def last_segmentation(stages: dict) -> tuple[int, dict]:
    """The last segmentation attempt (R retries with a lower KS cut when breakpoints are few)."""
    attempts = sorted(int(k.split("_")[1]) for k in stages if k.startswith("segmentation_"))
    return attempts[-1], stages[f"segmentation_{attempts[-1]}"]


def compare(run_a: Path, run_b: Path) -> list[Row]:
    a, b = load(run_a / "stages"), load(run_b / "stages")
    rows: list[Row] = []

    # Steps 1-3: which cells and genes survive filtering, and the smoothed matrix.
    cells_a, cells_b = a["smoothed"]["colnames"], b["smoothed"]["colnames"]
    shared = [c for c in cells_a if c in set(cells_b)]
    genes_a, genes_b = a["smoothed"]["values"].shape[0], b["smoothed"]["values"].shape[0]
    rows.append(("1-3 filtering", "cells_only_in_A", len(cells_a) - len(shared)))
    rows.append(("1-3 filtering", "cells_only_in_B", len(cells_b) - len(shared)))
    rows.append(("1-3 filtering", "genes_A_vs_B", f"{genes_a} vs {genes_b}"))
    rows += matrix_rows("3 smoothed", columns(a["smoothed"], shared), columns(b["smoothed"], shared))

    # Step 4: the normal-cell reference and the clusters handed to segmentation.
    rel_a, rel_b = a["relative"], b["relative"]
    seg_cols_a = rel_a["norm.mat.relat"]["colnames"]
    seg_cols_b = rel_b["norm.mat.relat"]["colnames"]
    cells = [c for c in seg_cols_a if c in set(seg_cols_b)]
    rows.append(("4 baseline", "clusters_ARI", ari(by_cell(rel_a["CL"]), by_cell(rel_b["CL"]), cells)))
    normal_a = set(rel_a["preN"]["values"].ravel()) if "preN" in rel_a else set()
    normal_b = set(rel_b["preN"]["values"].ravel()) if "preN" in rel_b else set()
    rows.append(
        ("4 baseline", "normal_cells_A_B_shared", f"{len(normal_a)} {len(normal_b)} {len(normal_a & normal_b)}")
    )
    rows.append(("4 baseline", "warning_A_B", f"{first_value(rel_a['WNS'])!r} {first_value(rel_b['WNS'])!r}"))
    rows += matrix_rows(
        "4 relative expression",
        columns(rel_a["norm.mat.relat"], cells),
        columns(rel_b["norm.mat.relat"], cells),
    )

    # Step 5: breakpoints, and the per-cell segment values (before each cell is centred).
    attempt_a, seg_a = last_segmentation(a)
    attempt_b, seg_b = last_segmentation(b)
    rows.append(("5 segmentation", "attempts_A_B", f"{attempt_a} {attempt_b}"))
    breaks_a = set(seg_a["breaks"]["values"].ravel().astype(int))
    breaks_b = set(seg_b["breaks"]["values"].ravel().astype(int))
    rows.append(
        ("5 segmentation", "breakpoints_A_B_shared", f"{len(breaks_a)} {len(breaks_b)} {len(breaks_a & breaks_b)}")
    )
    rows += matrix_rows(
        "5 segment values",
        columns(seg_a["logCNA"], cells, seg_cols_a),
        columns(seg_b["logCNA"], cells, seg_cols_b),
    )

    # Step 6: genomic bins.
    rows += matrix_rows("6 bins", columns(a["bins"], cells), columns(b["bins"], cells))

    # Steps 7-8: clusters, the re-centred matrix, and the calls.
    fin_a, fin_b = a["final"], b["final"]
    if "step7_clusters" in fin_a and "step7_clusters" in fin_b:
        step7 = ari(by_cell(fin_a["step7_clusters"]), by_cell(fin_b["step7_clusters"]), cells)
        rows.append(("7 final", "clusters_ARI", step7))
    rows += matrix_rows("7 adjusted matrix", columns(fin_a["mat.adj"], cells), columns(fin_b["mat.adj"], cells))
    step8 = ari(by_cell(fin_a["step8_clusters"]), by_cell(fin_b["step8_clusters"]), cells)
    rows.append(("8 final", "clusters_ARI", step8))
    pred_a, pred_b = by_cell(fin_a["prediction"]), by_cell(fin_b["prediction"])
    call_a = {c: core_label(pred_a[c]) for c in cells}
    call_b = {c: core_label(pred_b[c]) for c in cells}
    rows.append(("8 calls", "cells_compared", len(cells)))
    rows.append(("8 calls", "discordant_cells", sum(call_a[c] != call_b[c] for c in cells)))
    rows.append(("8 calls", "aneuploid_A", sum(v == "aneuploid" for v in call_a.values())))
    rows.append(("8 calls", "aneuploid_B", sum(v == "aneuploid" for v in call_b.values())))
    low_a = any("low.conf" in pred_a[c] for c in cells)
    low_b = any("low.conf" in pred_b[c] for c in cells)
    rows.append(("8 calls", "low_confidence_flag_A_B", f"{low_a} {low_b}"))
    for side, fin in (("A", fin_a), ("B", fin_b)):
        if "consensus_correlation" in fin:
            value = f"{first_value(fin['consensus_correlation']):.4f}"
            rows.append(("8 calls", f"consensus_correlation_{side}", value))
    return rows


def main() -> None:
    run_a, run_b = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"# A = {run_a}\n# B = {run_b}")
    print("step\tmetric\tvalue")
    for step, metric, value in compare(run_a, run_b):
        print(f"{step}\t{metric}\t{value}")


if __name__ == "__main__":
    main()
