"""Tests for the new coordinate helper, using independent annotation fixtures."""

import unittest

import numpy as np
import pandas as pd

from copykat_py.genomic_coordinates import annotation_order, chromosome_label, split_cna_table


class GenomicCoordinateTests(unittest.TestCase):
    def test_mouse_order_uses_gene_position_despite_tied_chromosome_offsets(self):
        annotation = pd.DataFrame(
            {
                "chromosome_name": [2, 1, 1, 1, 20],
                "start_position": [5, 100, 10, 10, 1],
                "abspos": [2000, 1000, 1000, 1000, 3000],
            }
        )
        order = annotation_order(annotation, "mm10")
        np.testing.assert_array_equal(order, [2, 3, 1, 0, 4])
        # The same permutation must select annotation and corresponding expression rows.
        expression = np.array([205, 1100, 1010, 1011, 20001])
        np.testing.assert_array_equal(expression[order], [1010, 1011, 1100, 205, 20001])

    def test_numeric_strings_sort_numerically_and_human_order_stays_absolute(self):
        annotation = pd.DataFrame(
            {"chromosome_name": ["10", "2", "2"], "start_position": ["1", "20", "3"], "abspos": [1, 3, 3]}
        )
        np.testing.assert_array_equal(annotation_order(annotation, "mm10"), [2, 1, 0])
        np.testing.assert_array_equal(annotation_order(annotation, "hg20"), [0, 1, 2])

    def test_invalid_coordinates_are_rejected(self):
        annotation = pd.DataFrame({"chromosome_name": [1], "start_position": [np.nan]})
        with self.assertRaisesRegex(ValueError, "finite"):
            annotation_order(annotation, "mm10")

    def test_mouse_and_human_sex_chromosomes_are_distinct(self):
        self.assertEqual(chromosome_label(20, "mm10"), "X")
        self.assertEqual(chromosome_label("21", "mm10"), "Y")
        self.assertEqual(chromosome_label(20, "hg20"), "20")
        self.assertEqual(chromosome_label(23, "hg20"), "X")
        self.assertEqual(chromosome_label(24, "hg20"), "Y")
        self.assertEqual(chromosome_label("X", "mm10"), "X")

    def test_mouse_table_excludes_all_seven_annotation_columns(self):
        frame = pd.DataFrame(
            {
                "abspos": [100],
                "chromosome_name": [20],
                "start_position": [5],
                "end_position": [6],
                "ensembl_gene_id": ["gene1"],
                "mgi_symbol": ["A"],
                "band": ["q"],
                "cell1": [0.2],
                "cell2": [-0.3],
            }
        )
        values, names, chrom, genome = split_cna_table(frame)
        self.assertEqual(names, ["cell1", "cell2"])
        self.assertEqual(genome, "mm10")
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(values, [[0.2, -0.3]])
        np.testing.assert_array_equal(chrom, [20])

    def test_human_table_and_unrecognized_layout(self):
        frame = pd.DataFrame({"chrom": [1], "chrompos": [50], "abspos": [50], "cell1": [0.1]})
        values, names, _, genome = split_cna_table(frame)
        self.assertEqual(values.shape, (1, 1))
        self.assertEqual(names, ["cell1"])
        self.assertEqual(genome, "hg20")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            split_cna_table(pd.DataFrame({"cell1": [1]}))
