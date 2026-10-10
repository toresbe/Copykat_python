"""Display labels for metadata legends; source values remain unchanged."""

from collections.abc import Iterable

from copykat_py._types import BaselineWarning, DataQualityStatus, PredictionLabel

_PREDICTION_LABELS = {
    PredictionLabel.DIPLOID: "Diploid",
    PredictionLabel.ANEUPLOID: "Aneuploid",
    PredictionLabel.DIPLOID_LOW_CONFIDENCE: "Diploid (low confidence)",
    PredictionLabel.ANEUPLOID_LOW_CONFIDENCE: "Aneuploid (low confidence)",
    # copykat.py adds this for original cells without a final prediction.
    PredictionLabel.NOT_DEFINED: "Not classified",
    # plotting.py creates this when aligning absent/empty cell metadata.
    PredictionLabel.UNKNOWN: "Missing annotation",
}


def annotation_title(column: str) -> str:
    """Readable headings for known columns; custom metadata keeps its name."""
    return {
        "copykat_pred_py": "CopyKAT Python",
        "copykat_pred_r": "CopyKAT R",
        "copykat_pred": "CopyKAT",
        "copykat.pred": "CopyKAT",
        "n_umi": "UMI count per cell",
    }.get(column.lower(), column)


def warning_caption(*warnings: str) -> str:
    """Readable plot subtitles for runtime warning/status strings."""
    labels = {
        DataQualityStatus.OK: "Data quality: OK",
        DataQualityStatus.LOW: "Low data quality",
        BaselineWarning.UNCLASSIFIED: "Low-confidence classification",
        BaselineWarning.KNOWN_NORMAL: "Supplied normal-cell reference",
        BaselineWarning.CELL_LINE: "Cell-line mode",
    }
    return " · ".join(labels.get(value, value) for value in warnings if value)


def is_prediction_column(column: str, values: Iterable[object]) -> bool:
    """Recognize standard CopyKAT columns or an unambiguous set of calls."""
    if column.lower() in {"copykat.pred", "copykat_pred", "copykat_pred_py", "copykat_pred_r"}:
        return True
    categories = {str(value) for value in values}
    calls = set(_PREDICTION_LABELS) - {PredictionLabel.NOT_DEFINED, PredictionLabel.UNKNOWN}
    return categories <= set(_PREDICTION_LABELS) and bool(categories & calls)


def legend_label(value: object, *, prediction: bool = True) -> str:
    """Format known prediction states without altering unrelated metadata."""
    label = str(value)
    return _PREDICTION_LABELS.get(label, label) if prediction else label
