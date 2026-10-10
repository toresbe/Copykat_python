"""Tests of the new report modules using small saved-run fixtures only."""

import base64
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from copykat_py.reporting.cli import main
from copykat_py.reporting.model import load_report
from copykat_py.reporting.render import render_report, write_reports


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.runtime = {
            "sample_name": "sample",
            "total_seconds": 12.5,
            "pca_selection_input_cells": 4,
            "parameters": {"UP_DR": 0.1, "UP_DR_effective": 0.05},
            "versions": {"copykat-py": "1.0.0"},
            "reference": {"mode": "known_normal", "matched_supplied_count": 2},
            "warnings": ["Low data quality"],
            "steps": [
                {"step": "read_and_filter", "seconds": 1.5, "filtered_cells": 1},
                {"step": "baseline_estimation", "seconds": 2, "warning": "run with known normal"},
            ],
        }
        self.save_runtime()
        (self.directory / "sample_copykat_prediction.txt").write_text(
            "cell.names\tcopykat.pred\na\tdiploid\nb\taneuploid\nc\tnot.defined\nd\tc1:low.confidence\n"
        )
        self.png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
        )
        (self.directory / "sample_copykat_heatmap.png").write_bytes(self.png)

    def save_runtime(self, sample="sample"):
        (self.directory / f"{sample}_copykat_runtime.json").write_text(json.dumps(self.runtime), encoding="utf-8")

    def test_load_preserves_all_labels_and_only_sample_outputs(self):
        (self.directory / "other_copykat_heatmap.png").write_bytes(self.png)
        (self.directory / "sample_copykat_report.html").write_text("stale")
        # Large CNA files are inventoried without parsing or reading their content.
        (self.directory / "sample_copykat_CNA_results.txt").write_bytes(b"\xff")
        report = load_report(self.directory)
        self.assertEqual(report.predictions, {"aneuploid": 1, "c1:low.confidence": 1, "diploid": 1, "not.defined": 1})
        self.assertEqual(len(report.images), 1)
        self.assertFalse(any("other" in p.name or "report" in p.name for p in report.files))

    def test_formats_have_consistent_content_and_preserve_sources(self):
        before = {p: p.read_bytes() for p in self.directory.iterdir()}
        report = load_report(self.directory)
        paths = write_reports(report, ["txt", "markdown", "html", "json", "html"], self.directory)
        self.assertEqual(len(paths), 4)
        for path in paths:
            content = path.read_text()
            if path.suffix == ".md":
                content = content.replace("\\_", "_")
            self.assertIn("UP_DR_effective", content)
            self.assertIn("not.defined", content)
            self.assertIn("known_normal", content)
        self.assertEqual(json.loads(paths[-1].read_text())["prediction_counts"], report.predictions)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_html_escapes_metadata_and_embeds_images(self):
        self.runtime["warnings"] = ['<script>alert("x")</script>']
        self.save_runtime()
        content = render_report(load_report(self.directory), "html", self.directory)
        self.assertNotIn("<script>", content)
        self.assertIn("&lt;script&gt;", content)
        self.assertIn("data:image/png;base64," + base64.b64encode(self.png).decode(), content)

    def test_markdown_escapes_metadata_and_links_images_with_spaces(self):
        self.runtime["warnings"] = ["[x](javascript:evil)\n<script>bad</script>"]
        self.save_runtime("a sample")
        (self.directory / "sample_copykat_runtime.json").unlink()
        (self.directory / "a sample_copykat_heatmap.png").write_bytes(self.png)
        report = load_report(self.directory)
        content = render_report(report, "markdown", self.directory)
        self.assertIn("a%20sample_copykat_heatmap.png", content)
        self.assertIn("\\[x\\]", content)
        self.assertNotIn("<script>", content)

    def test_external_output_directory_and_html_portability(self):
        report = load_report(self.directory)
        destination = self.directory / "reports"
        paths = write_reports(report, ["markdown", "html"], destination)
        self.assertIn(report.images[0].as_uri(), paths[0].read_text())
        self.assertIn("data:image/png;base64,", paths[1].read_text())

    def test_legacy_run_and_no_plots_or_predictions(self):
        self.runtime = {"sample_name": "sample", "steps": []}
        self.save_runtime()
        (self.directory / "sample_copykat_prediction.txt").unlink()
        (self.directory / "sample_copykat_heatmap.png").unlink()
        report = load_report(self.directory)
        self.assertIsNone(report.predictions)
        content = render_report(report, "html", self.directory)
        self.assertIn("Unavailable for this run", content)
        self.assertIn("No heatmaps available", content)
        self.assertIn("No prediction table available", content)

    def test_multiple_runs_require_selection(self):
        self.save_runtime("second")
        with self.assertRaisesRegex(ValueError, "Multiple runs"):
            load_report(self.directory)
        self.assertEqual(load_report(self.directory, "sample").sample, "sample")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            load_report(self.directory, "../sample")
        with self.assertRaisesRegex(ValueError, "No runtime"):
            load_report(self.directory, "missing")
        (self.directory / "sample_copykat_runtime.json").write_text('{"steps": [null]}')
        with self.assertRaisesRegex(ValueError, "step must be an object"):
            load_report(self.directory)
        self.save_runtime()
        (self.directory / "sample_copykat_prediction.txt").write_text("bad\tcolumns\n")
        with self.assertRaisesRegex(ValueError, "requires"):
            load_report(self.directory)

    def test_invalid_formats_do_not_create_output_directory(self):
        report = load_report(self.directory)
        destination = self.directory / "invalid"
        for formats in ([], ["html", "pdf"], [""]):
            with self.assertRaises(ValueError):
                write_reports(report, formats, destination)
        self.assertFalse(destination.exists())

    def test_cli_generates_selected_formats_and_reports_bad_arguments(self):
        with redirect_stdout(io.StringIO()):
            main(["--run-dir", str(self.directory), "--formats", "txt,markdown"])
        self.assertTrue((self.directory / "sample_copykat_report.md").is_file())
        with self.assertRaises(SystemExit) as error, redirect_stderr(io.StringIO()):
            main(["--run-dir", str(self.directory), "--formats", "pdf"])
        self.assertEqual(error.exception.code, 2)

    def test_malformed_optional_metadata_is_rejected(self):
        for key, value in (("parameters", []), ("versions", None), ("reference", "bad"), ("warnings", [{}])):
            with self.subTest(key=key):
                original = self.runtime[key]
                self.runtime[key] = value
                self.save_runtime()
                with self.assertRaisesRegex(ValueError, key):
                    load_report(self.directory)
                self.runtime[key] = original

    def test_malformed_prediction_row_is_rejected(self):
        (self.directory / "sample_copykat_prediction.txt").write_text("cell.names\tcopykat.pred\ncell\n")
        with self.assertRaisesRegex(ValueError, "Malformed prediction row"):
            load_report(self.directory)

    def test_unnamed_run(self):
        self.save_runtime("")
        report = load_report(self.directory, "")
        self.assertEqual(report.sample, "")
        self.assertIn("(unnamed)", render_report(report, "txt", self.directory))

    def test_oversized_plot_is_flagged_in_all_formats_and_not_embedded(self):
        path = self.directory / "sample_copykat_annotated_heatmap.png"
        header = self.png[:16] + struct.pack(">II", 2587, 154474)
        path.write_bytes(header)
        report = load_report(self.directory)
        for fmt in ("txt", "markdown", "html", "json"):
            content = render_report(report, fmt, self.directory)
            self.assertIn("154,474", content)
            self.assertIn("preview omitted", content)
            self.assertNotIn(base64.b64encode(header).decode(), content)
        self.assertEqual(path.read_bytes(), header)
        self.assertIn("data:image/png;base64,", render_report(report, "html", self.directory))

    def test_invalid_png_is_flagged_without_embedding(self):
        (self.directory / "sample_copykat_heatmap.png").write_bytes(b"not an image")
        content = render_report(load_report(self.directory), "html", self.directory)
        self.assertIn("no valid PNG header", content)
        self.assertNotIn("data:image/png;base64,", content)


if __name__ == "__main__":
    unittest.main()
