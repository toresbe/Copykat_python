"""Checks for scientific content ordering and the standalone HTML presentation."""

import unittest
from html.parser import HTMLParser
from pathlib import Path

from copykat_py.reporting.html import _duration
from copykat_py.reporting.model import RunReport
from copykat_py.reporting.render import render_report


class Elements(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = []
        self.targets = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "a" and attributes.get("href", "").startswith("#"):
            self.targets.append(attributes["href"][1:])


class HtmlPresentationTests(unittest.TestCase):
    def report(self, sample="T989"):
        return RunReport(
            sample,
            {"steps": [], "total_seconds": 3661, "warnings": ["unclassified.prediction"]},
            {"c1:diploid:low.conf": 1200, "not.defined": 30},
            (),
            (),
        )

    def test_results_and_figures_precede_run_details_and_navigation_resolves(self):
        content = render_report(self.report(), "html", Path("/tmp"))
        elements = Elements(content)
        self.assertEqual(elements.ids, ["overview", "heatmaps", "details", "outputs"])
        self.assertEqual(elements.targets, ["heatmaps", "details", "outputs"])
        self.assertEqual(len(elements.ids), len(set(elements.ids)))
        self.assertIn("1,230", content)
        self.assertIn("Cells in prediction table", content)
        self.assertIn("1h 1m", content)
        self.assertNotIn("Results at a glance", content)
        self.assertIn('<svg class="prediction-pie"', content)
        self.assertIn("Classification confidence", content)

    def test_labels_and_warning_provenance_are_both_present(self):
        content = render_report(self.report(), "html", Path("/tmp"))
        for text in [
            "Diploid (low confidence)",
            "c1:diploid:low.conf",
            "Not classified",
            "not.defined",
            "Low-confidence classification",
            "unclassified.prediction",
        ]:
            self.assertIn(text, content)

    def test_sample_is_escaped_in_title_and_heading(self):
        content = render_report(self.report('<script>alert("x")</script>'), "html", Path("/tmp"))
        self.assertNotIn("<script>", content)
        self.assertIn("&lt;script&gt;", content)
        self.assertNotIn('<link rel="stylesheet"', content)
        self.assertIn("@media(max-width:760px)", content)
        self.assertIn("@media print", content)

    def test_missing_predictions_and_invalid_runtime_are_explicit(self):
        report = RunReport("", {"steps": []}, None, (), ())
        content = render_report(report, "html", Path("/tmp"))
        self.assertIn("(unnamed)", content)
        self.assertIn("Unavailable", content)
        self.assertIn("No prediction table available", content)
        for value in [None, "12", True, -1, float("inf"), float("nan")]:
            self.assertEqual(_duration(value), "Unavailable")
        self.assertEqual(_duration(65), "1m 5s")
