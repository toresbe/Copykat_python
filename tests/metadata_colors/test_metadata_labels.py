"""Tests for the new display label module."""

import unittest

from copykat_py.metadata_labels import annotation_title, is_prediction_column, legend_label, warning_caption


class MetadataLabelTests(unittest.TestCase):
    def test_warning_caption_preserves_unknown_statuses_and_omits_empty(self):
        self.assertEqual(warning_caption("data quality is ok", "unclassified.prediction"),
                         "Data quality: OK · Low-confidence classification")
        self.assertEqual(warning_caption("", "future warning"), "future warning")
        self.assertEqual(warning_caption("", ""), "")

    def test_known_headings_and_custom_metadata(self):
        self.assertEqual(annotation_title("copykat_pred_py"), "CopyKAT Python")
        self.assertEqual(annotation_title("copykat_pred_R"), "CopyKAT R")
        self.assertEqual(annotation_title("n_umi"), "UMI count per cell")
        self.assertEqual(annotation_title("CellType"), "CellType")

    def test_low_confidence_calls_are_readable(self):
        self.assertEqual(legend_label("c1:diploid:low.conf"), "Diploid (low confidence)")
        self.assertEqual(legend_label("c2:aneuploid:low.conf"), "Aneuploid (low confidence)")

    def test_other_metadata_is_preserved(self):
        for value in ["diploid", "aneuploid", "not.defined", "unknown", "T cell", "c1", "low.conf"]:
            self.assertEqual(legend_label(value, prediction=False), value)
        self.assertEqual(legend_label(7), "7")

    def test_all_other_prediction_states(self):
        self.assertEqual(legend_label("diploid"), "Diploid")
        self.assertEqual(legend_label("aneuploid"), "Aneuploid")
        self.assertEqual(legend_label("not.defined"), "Not classified")
        self.assertEqual(legend_label("unknown"), "Missing annotation")
        self.assertEqual(legend_label("future_state"), "future_state")

    def test_prediction_column_recognition_including_unclassified_only(self):
        self.assertTrue(is_prediction_column("copykat_pred_py", ["not.defined", "unknown"]))
        self.assertTrue(is_prediction_column("copykat_pred_R", ["diploid"]))
        self.assertTrue(is_prediction_column("custom_calls", ["c1:diploid:low.conf", "unknown"]))
        self.assertFalse(is_prediction_column("CellType", ["unknown", "T cell"]))
        self.assertFalse(is_prediction_column("other", ["not.defined"]))
        self.assertFalse(is_prediction_column("other", ["diploid", "T cell"]))
