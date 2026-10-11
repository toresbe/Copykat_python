"""Render synthetic-test metrics as Markdown (for the CI job summary) and a PNG of the key curves.

    python tests/synthetic/report.py head.json [--base base.json] --markdown out.md --figure out.png

With ``--base``, every metric gets the base value and the change next to it,
so a pull request shows how it moves recovery, drift and stability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Categorical slots 1 and 2 of the reference data-viz palette (validated pair, light surface).
HEAD_COLOR, BASE_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

SECTIONS = {
    "recovery": "Planted-CNV recovery",
    "null": "Pure-diploid null",
    "perturbation": "Injected copy-number shift",
    "contamination": "Reference contamination",
    "invariance": "Input-order invariance",
    "subsampling": "Subsampling stability",
    "depth": "Sequencing depth",
    "purity": "Tumour purity",
}


def _load(path: str | None) -> dict[str, Any] | None:
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text())["metrics"]


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f"`{json.dumps(value)}`" if isinstance(value, dict) else str(value)
    if isinstance(value, int):
        return str(value)
    # Small nonzero values (float rounding in the invariance checks) would print as 0.0000.
    return f"{value:.2e}" if 0 < abs(value) < 1e-3 else f"{value:.4f}"


def _delta(head: Any, base: Any) -> str:
    if base is None:
        return "new"
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (head, base)):
        return "" if head == base else "changed"
    diff = head - base
    if diff == 0:
        return "0"
    return f"{diff:+.2e}" if abs(diff) < 1e-3 else f"{diff:+.4f}"


def _by_genome(metrics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Split ``"genome/section"`` keys into ``{genome: {section: rows}}``."""
    grouped: dict[str, dict[str, Any]] = {}
    for key, rows in (metrics or {}).items():
        genome, _, section = key.rpartition("/")
        grouped.setdefault(genome or "hg20", {})[section] = rows
    return grouped


def markdown(head: dict[str, Any], base: dict[str, Any] | None, figure_note: str | None) -> str:
    lines = ["## Synthetic-truth tests", ""]
    if base is None:
        lines += ["No base-branch metrics to compare against; showing this run only.", ""]
    else:
        lines += ["Each metric for this branch, the base branch run on the same tests, and the change.", ""]
    if figure_note:
        lines += [figure_note, ""]
    base_genomes = _by_genome(base)
    for genome, sections in _by_genome(head).items():
        lines += [f"### {genome}", ""]
        for section, title in SECTIONS.items():
            rows = sections.get(section)
            if not rows:
                continue
            lines += [f"#### {title}", ""]
            if base is None:
                lines += ["| Metric | Value |", "| --- | ---: |"]
                lines += [f"| {key} | {_fmt(value)} |" for key, value in rows.items()]
            else:
                base_rows = base_genomes.get(genome, {}).get(section, {})
                lines += ["| Metric | This branch | Base | Change |", "| --- | ---: | ---: | ---: |"]
                for key, value in rows.items():
                    old = base_rows.get(key)
                    lines.append(f"| {key} | {_fmt(value)} | {_fmt(old)} | {_delta(value, old)} |")
            lines.append("")
    return "\n".join(lines)


def _contamination_curve(metrics: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    points = sorted(
        (float(name.split()[0]), value)
        for name, value in metrics.get("contamination", {}).items()
        if name.endswith(f" {key}") and value is not None
    )
    return [p[0] for p in points], [p[1] for p in points]


def _dose_response(metrics: dict[str, Any]) -> tuple[list[float], list[float]]:
    points = []
    for name, value in metrics.get("perturbation", {}).items():
        parts = name.split()
        if len(parts) == 3 and parts[1].startswith("x") and parts[2] == "shift":
            points.append((math.log2(float(parts[1][1:])), value))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def figure(head: dict[str, Any], base: dict[str, Any] | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (
            "Reference contamination:\nraw tumour signal retained",
            "Tumour share of reference",
            lambda m: _contamination_curve(m, "raw_signal_retained"),
        ),
        (
            "Reference contamination:\ncall accuracy",
            "Tumour share of reference",
            lambda m: _contamination_curve(m, "call_accuracy"),
        ),
        ("Injected shift on chr5:\nCNA response", "Injected log2 fold change", _dose_response),
    ]
    head_genomes, base_genomes = _by_genome(head), _by_genome(base)
    genomes = list(head_genomes)
    fig, axes = plt.subplots(len(genomes), 3, figsize=(12, 3.6 * len(genomes)), facecolor=SURFACE, squeeze=False)
    cells = [(axes[row][col], genome, panel) for row, genome in enumerate(genomes) for col, panel in enumerate(panels)]
    for ax, genome, (title, xlabel, extract) in cells:
        ax.set_facecolor(SURFACE)
        # Base first and dashed, so this branch draws on top and identical curves still show both.
        runs = [("This branch", head_genomes[genome], HEAD_COLOR, "-")]
        if genome in base_genomes:
            runs.insert(0, ("Base", base_genomes[genome], BASE_COLOR, "--"))
        for label, metrics, color, style in runs:
            x, y = extract(metrics)
            if x:
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=2,
                    linestyle=style,
                    marker="o",
                    markersize=7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    label=label,
                )
        ax.set_title(f"{genome} · {title}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    if base_genomes:
        axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("head")
    parser.add_argument("--base")
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--figure")
    parser.add_argument("--figure-note", help="Line placed above the tables, e.g. where to find the figure")
    args = parser.parse_args()
    head = _load(args.head)
    if head is None:
        raise SystemExit(f"no metrics at {args.head}")
    base = _load(args.base)
    Path(args.markdown).write_text(markdown(head, base, args.figure_note))
    if args.figure:
        figure(head, base, Path(args.figure))


if __name__ == "__main__":
    main()
