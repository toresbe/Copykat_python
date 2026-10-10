"""Tests for the new legend renderer."""

import unittest

import matplotlib.pyplot as plt

from copykat_py.metadata_legend import draw_metadata_legends


class MetadataLegendTests(unittest.TestCase):
    def test_groups_are_titled_borderless_and_stacked(self):
        fig, ax = plt.subplots()
        try:
            groups = [
                ("CopyKAT Python", [("blue", "Diploid"), ("orange", "Aneuploid")]),
                ("Cell type", [("green", "T cell")]),
            ]
            legends = draw_metadata_legends(ax, groups, start=0.8)
            self.assertEqual(len(legends), 2)
            self.assertEqual(legends[0].get_title().get_text(), "CopyKAT Python")
            self.assertEqual(legends[0].get_title().get_fontweight(), "bold")
            self.assertEqual([text.get_text() for text in legends[0].get_texts()], ["Diploid", "Aneuploid"])
            self.assertFalse(legends[0].get_frame_on())
            self.assertFalse(legends[0].get_clip_on())
            self.assertLess(legends[1].get_bbox_to_anchor().y1, legends[0].get_bbox_to_anchor().y1)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_empty_groups_are_skipped(self):
        fig, ax = plt.subplots()
        try:
            self.assertEqual(draw_metadata_legends(ax, [("Empty", [])]), [])
        finally:
            plt.close(fig)
