"""Reports from completed CopyKAT runs; inference is never repeated."""

from copykat_py.reporting.model import load_report
from copykat_py.reporting.render import write_reports

__all__ = ["load_report", "write_reports"]
