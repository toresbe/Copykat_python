"""Tests for the new numeric annotation module, not the existing plot code."""

import unittest

import numpy as np
import pandas as pd

from copykat_py.metadata_colors import continuous_annotation, is_continuous


class MetadataColorTests(unittest.TestCase):
    def test_detection_preserves_integer_clusters_and_boolean_categories(self):
        self.assertFalse(is_continuous(pd.Series([0, 1, 2, 0])))
        self.assertFalse(is_continuous(pd.Series([True, False])))
        self.assertFalse(is_continuous(pd.Series(["T", "B"])))
        self.assertTrue(is_continuous(pd.Series(range(100))))
        self.assertTrue(is_continuous(pd.Series([0.1, 0.2])))

    def test_continuous_colors_share_range_and_are_ordered_by_values(self):
        annotation = continuous_annotation(pd.Series([0, 50, 100]))
        self.assertEqual(annotation.ticks, [0, 100])
        self.assertEqual(annotation.colors.shape, (3, 1, 3))
        self.assertFalse(annotation.has_missing)
        self.assertFalse(np.array_equal(annotation.colors[0], annotation.colors[-1]))
        reverse = continuous_annotation(pd.Series([100, 50, 0]))
        np.testing.assert_array_equal(reverse.colors[::-1], annotation.colors)

    def test_missing_and_nonfinite_values_are_grey_and_do_not_change_range(self):
        annotation = continuous_annotation(pd.Series([10, 30, "unknown", None, np.inf, -np.inf]))
        self.assertTrue(annotation.has_missing)
        self.assertEqual(annotation.ticks, [10, 30])
        np.testing.assert_allclose(annotation.colors[2:], 0.8)

    def test_constant_and_all_missing_columns_have_finite_colors(self):
        annotation = continuous_annotation(pd.Series([7, 7, 7]))
        self.assertEqual(annotation.ticks, [7])
        np.testing.assert_array_equal(annotation.colors[0], annotation.colors[1])
        missing = continuous_annotation(pd.Series([None, "unknown"]))
        self.assertEqual(missing.ticks, [])
        self.assertTrue(np.isfinite(missing.colors).all())

    def test_nullable_numeric_metadata(self):
        values = pd.Series([0.5, pd.NA, 2.5], dtype="Float64")
        self.assertTrue(is_continuous(values))
        annotation = continuous_annotation(values)
        self.assertEqual(annotation.ticks, [0.5, 2.5])
        self.assertTrue(annotation.has_missing)


if __name__ == "__main__":
    unittest.main()
