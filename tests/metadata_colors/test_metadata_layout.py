"""Tests for the new annotation layout module."""

import unittest

from copykat_py.metadata_layout import annotation_layout


class MetadataLayoutTests(unittest.TestCase):
    def test_predictions_precede_numeric_annotations_on_right(self):
        layout = annotation_layout(["n_umi", "copykat_pred_py", "type"], {"n_umi"})
        self.assertEqual(layout.annotation_columns, {"type": 1, "copykat_pred_py": 3, "n_umi": 4})
        self.assertEqual(layout.heatmap, 2)
        self.assertEqual(layout.colorbar, 5)

    def test_prediction_only_is_right_of_heatmap(self):
        layout = annotation_layout(["copykat.pred"], set())
        self.assertEqual(layout.heatmap, 1)
        self.assertEqual(layout.annotation_columns, {"copykat.pred": 2})

    def test_mixed_metadata_stays_ordered_and_flanks_heatmap(self):
        layout = annotation_layout(["type", "umi", "call", "score"], {"umi", "score"})
        self.assertEqual(layout.annotation_columns, {"type": 1, "call": 2, "umi": 4, "score": 5})
        self.assertEqual(layout.heatmap, 3)
        self.assertEqual(layout.colorbar, 6)
        self.assertEqual(layout.legend, 7)
        self.assertEqual(len(layout.width_ratios), 8)
        self.assertEqual(layout.width_ratios[layout.heatmap], 35)

    def test_categorical_only_keeps_original_placement(self):
        layout = annotation_layout(["type", "call"], set())
        self.assertEqual(layout.annotation_columns, {"type": 1, "call": 2})
        self.assertEqual(layout.heatmap, 3)
        self.assertEqual(layout.colorbar, 4)

    def test_numeric_only_and_no_annotations_have_valid_columns(self):
        numeric = annotation_layout(["umi"], {"umi"})
        self.assertEqual(numeric.heatmap, 1)
        self.assertEqual(numeric.annotation_columns, {"umi": 2})
        empty = annotation_layout([], set())
        self.assertEqual(empty.width_ratios, (1.5, 35, 0.8, 8))
        self.assertEqual(empty.legend, 3)
