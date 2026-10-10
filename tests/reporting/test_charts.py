"""Tests for count proportions, SVG accessibility, and confidence provenance."""

import unittest
from xml.etree import ElementTree

from copykat_py.reporting.charts import confidence_status, prediction_pie


class ChartTests(unittest.TestCase):
    def test_proportions_include_unclassified_cells(self):
        svg = ElementTree.fromstring(prediction_pie({"diploid": 3, "aneuploid": 1, "not.defined": 2}))
        self.assertEqual(svg.attrib["role"], "img")
        self.assertEqual(len(svg.findall("path")), 3)
        self.assertIn("Diploid: 3 (50.0%)", svg.attrib["aria-label"])
        self.assertIn("Not classified: 2 (33.3%)", svg.attrib["aria-label"])
        self.assertIn("A100 100 0 0 1", svg.findall("path")[0].attrib["d"])

    def test_single_state_draws_full_circle_and_zero_counts_are_omitted(self):
        svg = ElementTree.fromstring(prediction_pie({"diploid": 4, "aneuploid": 0}))
        self.assertEqual(len(svg.findall("circle")), 1)
        self.assertEqual(len(svg.findall("path")), 0)
        self.assertEqual(svg.find("circle").attrib["fill"], "#3A87C8")
        self.assertNotIn("Aneuploid", svg.attrib["aria-label"])
        self.assertIn("No prediction counts available", prediction_pie(None))
        self.assertIn("No prediction counts available", prediction_pie({"diploid": 0}))

    def test_majority_arc_and_unrecognized_labels_are_safe(self):
        svg = ElementTree.fromstring(prediction_pie({"diploid": 9, '<script>"&': 1}))
        self.assertIn("A100 100 0 1 1", svg.findall("path")[0].attrib["d"])
        self.assertEqual(len(svg.findall("script")), 0)
        self.assertIn('<script>"&', svg.attrib["aria-label"])

    def test_confidence_uses_actual_classifications_and_final_warning(self):
        self.assertEqual(confidence_status({"diploid": 10, "not.defined": 5}, {}), "High")
        self.assertEqual(confidence_status({"c1:diploid:low.conf": 10}, {}), "Low")
        runtime = {"steps": [{"step": "final_prediction", "warning": "unclassified.prediction"}]}
        self.assertEqual(confidence_status({"diploid": 10}, runtime), "Low")
        for calls in [None, {}, {"not.defined": 5}, {"diploid": 0}, {"future_state": 10}]:
            self.assertEqual(confidence_status(calls, {}), "Unavailable")
