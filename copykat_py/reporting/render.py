"""TXT, Markdown, self-contained HTML, and JSON from the same run summary."""

import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from copykat_py.reporting.html import render_html
from copykat_py.reporting.images import preview_issue
from copykat_py.reporting.model import RunReport

FORMATS = {"txt": ".txt", "markdown": ".md", "html": ".html", "json": ".json"}


def report_sections(report: RunReport) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Common content, with explicit unknowns rather than invented legacy data."""
    runtime = report.runtime
    sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Run",
            [
                ("Sample", report.sample or "(unnamed)"),
                ("Elapsed seconds", runtime.get("total_seconds", "Unavailable")),
            ],
        ),
        ("Versions", list(runtime.get("versions", {}).items()) or [("Versions", "Unavailable for this run")]),
        ("Parameters", list(runtime.get("parameters", {}).items()) or [("Parameters", "Unavailable for this run")]),
        (
            "Reference cells",
            list(runtime.get("reference", {}).items()) or [("Reference details", "Unavailable for this run")],
        ),
    ]
    filtering = [
        (f"{step.get('step', 'unknown')}: {key}", value)
        for step in runtime["steps"]
        for key, value in step.items()
        if key
        in {
            "filtered_cells",
            "filtered_gene_rows",
            "genes_after_annotation",
            "cells_after_filter",
            "genes_after_filter",
        }
    ]
    if "pca_selection_input_cells" in runtime:
        filtering.insert(0, ("Input cells", runtime["pca_selection_input_cells"]))
    sections.append(("Filtering", filtering or [("Filtering counts", "Unavailable for this run")]))
    sections.append(
        (
            "Predictions",
            list(report.predictions.items())
            if report.predictions is not None
            else [("Predictions", "No prediction table available (e.g. cell-line mode)")],
        )
    )
    notes = list(
        dict.fromkeys(
            [*runtime.get("warnings", []), *(str(step["warning"]) for step in runtime["steps"] if step.get("warning"))]
        )
    )
    sections.append(
        (
            "Warnings and analysis notes",
            [("Note", note) for note in notes]
            or [("Notes", "No warnings recorded in the run metadata; consult the run log for other messages")],
        )
    )
    sections.append(
        (
            "Step timings (seconds)",
            [(str(step.get("step", "unknown")), step.get("seconds", "Unavailable")) for step in runtime["steps"]],
        )
    )
    plot_notes = [(path.name, issue) for path in report.images if (issue := preview_issue(path)) is not None]
    if plot_notes:
        sections.append(("Heatmap preview warnings", plot_notes))
    return sections


def _value(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list | bool) or value is None else str(value)
    )


def _md(value: Any) -> str:
    text = html.escape(_value(value), quote=False)
    for char in "\\`*_{}[]|":
        text = text.replace(char, "\\" + char)
    return text.replace("\n", " ").replace("\r", " ")


def _href(path: Path, output_dir: Path) -> str:
    # Absolute file URIs also work when the report is written outside the run directory.
    return quote(path.name, safe="") if path.parent == output_dir else path.as_uri()


def render_report(report: RunReport, fmt: str, output_dir: Path) -> str:
    """Render without executing code or loading large matrix files."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown report format: {fmt}")
    sections = report_sections(report)
    if fmt == "json":
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_name": report.sample,
                    "runtime": report.runtime,
                    "prediction_counts": report.predictions,
                    "outputs": [str(p) for p in report.files],
                    "plot_warnings": {
                        path.name: issue for path in report.images if (issue := preview_issue(path)) is not None
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    if fmt == "txt":
        lines = ["CopyKAT-Python run report"]
        for title, rows in sections:
            lines.extend(["", title, *(f"  {key}: {_value(value)}" for key, value in rows)])
        lines.extend(["", "Output files", *(f"  {path}" for path in report.files)])
        return "\n".join(lines) + "\n"
    if fmt == "markdown":
        blocks = ["# CopyKAT-Python run report"]
        for title, rows in sections:
            blocks.append(f"## {title}\n\n" + "\n".join(f"- {_md(key)}: {_md(value)}" for key, value in rows))
        blocks.append(
            "## Output files\n\n"
            + "\n".join(f"- [{_md(path.name)}]({_href(path, output_dir)})" for path in report.files)
        )
        blocks.append(
            "## Heatmaps\n\n"
            + (
                "\n\n".join(
                    f"[{_md(path.name)}]({_href(path, output_dir)}) — {_md(issue)}"
                    if (issue := preview_issue(path)) is not None
                    else f"![{_md(path.name)}]({_href(path, output_dir)})"
                    for path in report.images
                )
                or "No heatmaps available."
            )
        )
        return "\n\n".join(blocks) + "\n"
    return render_html(report, sections, output_dir)


def write_reports(report: RunReport, formats: Iterable[str], output_dir: str | Path) -> list[Path]:
    """Write selected formats; only report files are created or replaced."""
    selected = list(dict.fromkeys(formats))
    if not selected or any(fmt not in FORMATS for fmt in selected):
        raise ValueError(f"Select one or more formats: {', '.join(FORMATS)}")
    directory = Path(output_dir).resolve()
    # Render everything before writing, so a missing image does not leave partial reports.
    rendered = [(fmt, render_report(report, fmt, directory)) for fmt in selected]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt, content in rendered:
        path = directory / f"{report.sample}_copykat_report{FORMATS[fmt]}"
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths
