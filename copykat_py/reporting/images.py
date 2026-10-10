"""Inspect PNG headers without decoding potentially enormous saved plots."""

import struct
from pathlib import Path

# Normal CopyKAT figures are only a few thousand pixels in either dimension.
# Limits apply to report previews, not the underlying analytical files.
MAX_PREVIEW_DIMENSION = 12_000
MAX_PREVIEW_PIXELS = 40_000_000


def preview_issue(path: Path) -> str | None:
    """Return an explanatory note for a PNG that should not be embedded."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return "Saved file has no valid PNG header; preview omitted."
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        return "Saved PNG has invalid dimensions; preview omitted."
    if max(width, height) > MAX_PREVIEW_DIMENSION or width * height > MAX_PREVIEW_PIXELS:
        return (
            f"Saved heatmap is {width:,} × {height:,} pixels and exceeds report preview limits; "
            "preview omitted. Inspect or regenerate the source plot."
        )
    return None
