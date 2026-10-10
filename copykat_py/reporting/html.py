"""Self-contained HTML presentation for a saved CopyKAT run."""

import base64
import html
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

from copykat_py.metadata_labels import legend_label, warning_caption
from copykat_py.reporting.charts import COLORS, confidence_status, prediction_pie
from copykat_py.reporting.images import preview_issue
from copykat_py.reporting.model import RunReport

STYLE = """
:root{color-scheme:light;--ink:#1c303a;--muted:#61717b;--line:#dce4e8;--accent:#146b72}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f7f8;
color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}
.page{max-width:1320px;margin:auto;padding:36px 32px 56px}header{padding:0 0 24px;
border-bottom:1px solid var(--line)}.eyebrow{text-transform:uppercase;letter-spacing:.14em;
font-size:11px;font-weight:700;color:var(--accent)}h1{font-size:34px;line-height:1.2;
letter-spacing:-.025em;margin:8px 0}h2{font-size:21px;line-height:1.3;margin:0 0 16px}
h3{font-size:16px;margin:0 0 12px}.muted,.subtitle{color:var(--muted)}p{margin:8px 0}
nav{display:flex;flex-wrap:wrap;gap:24px;margin-top:20px}a{color:var(--accent);
text-underline-offset:3px}nav a{font-size:13px;font-weight:600;text-decoration:none}
a:hover{text-decoration:underline}a:focus-visible,summary:focus-visible{outline:2px solid
var(--accent);outline-offset:4px}.section{margin-top:32px;scroll-margin-top:24px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;align-items:stretch}
.panel{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;
min-width:0}.notes ul{margin:0;padding-left:20px}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;
margin:20px 0 0}.metric strong{display:block;
font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.metric span{font-size:12px;
color:var(--muted)}table{border-collapse:collapse;table-layout:fixed;width:100%;font-size:13px}
.classifications>.grid{grid-template-columns:minmax(0,3fr) minmax(220px,2fr);align-items:center;gap:40px}
.classification-table th{width:75%}.classification-table td{text-align:right}
.classification-table>p{margin:16px 0 0;font-size:12px}
.prediction-pie{display:block;width:100%;max-width:240px;height:auto;margin:0}
.chart-panel{display:flex;align-items:center;justify-content:center;min-width:0}.notes{margin-top:24px}
.class-label{display:grid;grid-template-columns:10px minmax(0,1fr);gap:8px;align-items:baseline}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px}
th,td{text-align:left;vertical-align:top;padding:9px 0;border-bottom:1px solid #edf1f3;
overflow-wrap:anywhere}th{font-weight:500;color:var(--muted);width:56%;padding-right:20px}
td{font-variant-numeric:tabular-nums}.numeric{text-align:right}
tr:last-child th,tr:last-child td{border-bottom:0}
.raw{display:block;color:var(--muted);font-size:11px;font-family:ui-monospace,monospace}
.figure{margin:20px 0 0;background:white;border:1px solid var(--line);border-radius:10px;
overflow:hidden}.figure-head{padding:20px 24px 12px}.figure-head h3{margin:0}
img{display:block;width:100%;height:auto}.figure figcaption{padding:16px 24px;
border-top:1px solid #edf1f3;color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.files{list-style:none;margin:0;padding:0}.files li{padding:10px 0;border-bottom:1px solid
#edf1f3;overflow-wrap:anywhere}.files li:last-child{border:0}.files a{font-family:
ui-monospace,monospace;font-size:12px}.detail h3{margin-bottom:12px}footer{margin-top:28px;
font-size:12px;color:var(--muted)}
@media(max-width:760px){.page{padding:24px 16px}.grid,.classifications>.grid{grid-template-columns:1fr}
.classifications>.grid{gap:24px}.metrics{gap:16px}.metric strong{font-size:21px}
.panel{padding:18px}h1{font-size:28px}nav{gap:16px}.figure-head{padding:18px}}
@media(max-width:420px){.metrics{grid-template-columns:1fr;gap:12px}}
@media print{body{background:white}.page{padding:0;max-width:none}nav{display:none}
.panel,.figure{break-inside:avoid}.grid{display:block}.panel{margin-bottom:16px}}
"""


def _escape(value: Any) -> str:
    if isinstance(value, dict | list | bool) or value is None:
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value), quote=True)


def _table(rows: list[tuple[str, Any]], predictions: bool = False) -> str:
    body = []
    for key, value in rows:
        label = legend_label(key) if predictions else key
        raw = f'<span class="raw">{_escape(key)}</span>' if label != key else ""
        swatch = (
            f'<span class="swatch" aria-hidden="true" style="background:{COLORS.get(key, "#778899")}"></span>'
            if predictions
            else ""
        )
        heading = (
            f'<span class="class-label">{swatch}<span>{_escape(label)}{raw}</span></span>'
            if predictions
            else _escape(label)
        )
        numeric = ' class="numeric"' if isinstance(value, int | float) and not isinstance(value, bool) else ""
        body.append(f'<tr><th scope="row">{heading}</th><td{numeric}>{_escape(value)}</td></tr>')
    return "<table>" + "".join(body) + "</table>"


def _duration(seconds: Any) -> str:
    if not isinstance(seconds, int | float) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        return "Unavailable"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m {secs}s" if minutes else f"{secs}s"


def render_html(report: RunReport, sections: list[tuple[str, list[tuple[str, Any]]]], output_dir: Path) -> str:
    """Prioritize calls, analysis notes, and figures before reproducibility details."""
    by_title = dict(sections)
    sample = _escape(report.sample or "(unnamed)")
    cell_count = f"{sum(report.predictions.values()):,}" if report.predictions is not None else "Unavailable"
    duration = _escape(_duration(report.runtime.get("total_seconds")))
    confidence = confidence_status(report.predictions, report.runtime)
    blocks = [
        f'<header><div class="eyebrow">CopyKAT Python · Run report</div><h1>{sample}</h1>'
        '<p class="subtitle">Copy-number inference from single-cell expression data</p>'
        '<div class="metrics">'
        f'<div class="metric"><strong>{cell_count}</strong><span>Cells in prediction table</span></div>'
        f'<div class="metric"><strong>{duration}</strong><span>Analysis runtime</span></div>'
        f'<div class="metric" title="Categorical CopyKAT flag, not a probability. High means classified calls '
        f'without a low-confidence flag."><strong>{confidence}</strong>'
        "<span>Classification confidence</span></div></div>"
        '<nav aria-label="Report sections"><a href="#heatmaps">Heatmaps</a>'
        '<a href="#details">Run details</a><a href="#outputs">Output files</a></nav></header>',
        '<section class="section" id="overview" aria-label="Cell classifications">'
        '<article class="panel classifications"><h3>Cell classifications</h3><div class="grid">'
        '<div class="classification-table">'
        + _table(by_title["Predictions"], predictions=True)
        + '<p class="muted">Counts include cells without a final classification.</p></div>'
        + '<div class="chart-panel" aria-label="Classification proportions">'
        + prediction_pie(report.predictions)
        + "</div></div></article>",
    ]
    notes = []
    for _, value in by_title["Warnings and analysis notes"]:
        readable = warning_caption(str(value))
        raw = f'<span class="raw">{_escape(value)}</span>' if readable != str(value) else ""
        notes.append(f"<li>{_escape(readable)}{raw}</li>")
    blocks.append(
        '<article class="panel notes"><h3>Analysis notes</h3><ul>' + "".join(notes) + "</ul></article></section>"
    )
    blocks.append(
        '<section class="section" id="heatmaps"><h2>CNA heatmaps</h2>'
        '<p class="muted">Rows represent cells; columns follow genomic order. '
        "The red and blue scale shows relative CNA signal.</p>"
    )
    for path in report.images:
        issue = preview_issue(path)
        href = quote(path.name, safe="") if path.parent == output_dir else path.as_uri()
        if issue:
            blocks.append(f'<p class="panel"><a href="{_escape(href)}">{_escape(path.name)}</a>: {_escape(issue)}</p>')
            continue
        title = "Annotated CNA heatmap" if "annotated_heatmap" in path.name else "CNA heatmap with clustering"
        image = base64.b64encode(path.read_bytes()).decode()
        blocks.append(
            f'<figure class="figure"><div class="figure-head"><h3>{title}</h3></div>'
            f'<img alt="{_escape(path.name)}" src="data:image/png;base64,{image}" loading="lazy">'
            f"<figcaption>{_escape(path.name)}</figcaption></figure>"
        )
    if not report.images:
        blocks.append('<p class="panel">No heatmaps available.</p>')
    blocks.append('</section><section class="section" id="details"><h2>Run details</h2><div class="grid">')
    for title, rows in sections:
        if title in {"Predictions", "Warnings and analysis notes"}:
            continue
        blocks.append(f'<article class="panel detail"><h3>{_escape(title)}</h3>{_table(rows)}</article>')
    blocks.append(
        '</div></section><section class="section" id="outputs"><h2>Output files</h2>'
        '<div class="panel"><ul class="files">'
    )
    for path in report.files:
        href = quote(path.name, safe="") if path.parent == output_dir else path.as_uri()
        blocks.append(f'<li><a href="{_escape(href)}">{_escape(path.name)}</a></li>')
    blocks.append(
        "</ul></div></section><footer>Generated from saved CopyKAT outputs. "
        "Original prediction labels are retained alongside their display names.</footer>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{sample} · CopyKAT Python run report</title><style>{STYLE}</style></head>"
        '<body><main class="page">' + "".join(blocks) + "</main></body></html>\n"
    )
