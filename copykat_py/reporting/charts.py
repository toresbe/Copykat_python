"""Small standalone SVG charts and categorical confidence summaries."""

import html
import math
from collections.abc import Mapping
from typing import Any

from copykat_py.metadata_labels import legend_label

COLORS = {
    "diploid": "#3A87C8",
    "aneuploid": "#E8601C",
    "c1:diploid:low.conf": "#9EC8E8",
    "c2:aneuploid:low.conf": "#F4A86A",
    "not.defined": "#B0B0B0",
    "unknown": "#D4D4D4",
}


def confidence_status(predictions: Mapping[str, int] | None, runtime: Mapping[str, Any]) -> str:
    """Summarize saved classification flags, never invent confidence for missing calls."""
    if any(
        step.get("warning") == "unclassified.prediction"
        for step in runtime.get("steps", [])
        if step.get("step") == "final_prediction"
    ):
        return "Low"
    calls = predictions or {}
    if any(calls.get(state, 0) > 0 for state in ("c1:diploid:low.conf", "c2:aneuploid:low.conf")):
        return "Low"
    if any(calls.get(state, 0) > 0 for state in ("diploid", "aneuploid")):
        return "High"
    return "Unavailable"


def prediction_pie(predictions: Mapping[str, int] | None) -> str:
    """Draw count-proportional pie sectors, retaining all prediction states."""
    entries = [(label, count) for label, count in (predictions or {}).items() if count > 0]
    total = sum(count for _, count in entries)
    if not total:
        return '<p class="muted">No prediction counts available.</p>'
    description = "; ".join(f"{legend_label(label)}: {count:,} ({count / total:.1%})" for label, count in entries)
    sectors = []
    angle = -math.pi / 2
    for label, count in entries:
        color = COLORS.get(label, "#778899")
        tooltip = html.escape(f"{legend_label(label)}: {count:,} ({count / total:.1%})")
        if count == total:
            sectors.append(f'<circle cx="120" cy="120" r="100" fill="{color}"><title>{tooltip}</title></circle>')
            continue
        end = angle + 2 * math.pi * count / total
        x1, y1 = 120 + 100 * math.cos(angle), 120 + 100 * math.sin(angle)
        x2, y2 = 120 + 100 * math.cos(end), 120 + 100 * math.sin(end)
        large = int(count / total > 0.5)
        sectors.append(
            f'<path d="M120 120 L{x1:.4f} {y1:.4f} A100 100 0 {large} 1 {x2:.4f} {y2:.4f} Z" '
            f'fill="{color}" stroke="white" stroke-width="1.5"><title>{tooltip}</title></path>'
        )
        angle = end
    return (
        '<svg class="prediction-pie" viewBox="0 0 240 240" role="img" '
        f'aria-label="{html.escape(description, quote=True)}">'
        "<title>Cell classification proportions</title>" + "".join(sectors) + "</svg>"
    )
