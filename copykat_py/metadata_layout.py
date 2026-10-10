"""Column placement for annotated heatmaps."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from copykat_py.metadata_labels import is_prediction_column


@dataclass(frozen=True)
class AnnotationLayout:
    annotation_columns: dict[str, int]
    heatmap: int
    colorbar: int
    legend: int
    width_ratios: tuple[float, ...]


def annotation_layout(columns: Sequence[str], continuous: Collection[str]) -> AnnotationLayout:
    """Place predictions immediately before numeric measurements on the right.

    Preserve the supplied order within each side. Column zero is reserved for
    row-group labels; the CNA scale and legends follow the numeric strips.
    """
    predictions = [column for column in columns if is_prediction_column(column, []) and column not in continuous]
    left = [column for column in columns if column not in continuous and column not in predictions]
    right = predictions + [column for column in columns if column in continuous]
    heatmap = 1 + len(left)
    positions = {column: 1 + index for index, column in enumerate(left)}
    positions.update({column: heatmap + 1 + index for index, column in enumerate(right)})
    colorbar = heatmap + 1 + len(right)
    return AnnotationLayout(
        positions,
        heatmap,
        colorbar,
        colorbar + 1,
        (1.5, *([1.0] * len(left)), 35.0, *([1.0] * len(right)), 0.8, 8.0),
    )
