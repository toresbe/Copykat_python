"""Borderless, separately titled categorical annotation legends."""

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.patches import Patch


def draw_metadata_legends(
    ax: Axes,
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    start: float = 1.0,
) -> list[Legend]:
    """Stack titled swatch groups using the available axes height in points."""
    height_points = ax.get_window_extent().height * 72 / ax.figure.dpi
    legends = []
    y = start
    for title, entries in groups:
        if not entries:
            continue
        legend = ax.legend(
            handles=[Patch(facecolor=color, edgecolor="none", label=label) for color, label in entries],
            title=title,
            title_fontproperties={"weight": "bold", "size": 11},
            loc="upper left",
            bbox_to_anchor=(0.2, y),
            alignment="left",
            fontsize=10,
            frameon=False,
            handlelength=1.2,
            handleheight=0.9,
            borderaxespad=0,
            labelspacing=0.6,
        )
        # Include long labels outside the legend axes in tight PNG exports.
        legend.set_clip_on(False)
        ax.add_artist(legend)
        legends.append(legend)
        y -= (len(entries) + 3) * 16 / max(height_points, 1)
    return legends
