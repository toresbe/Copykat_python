"""Standalone report command for saved runs."""

import argparse
from collections.abc import Sequence

from copykat_py.reporting.model import load_report
from copykat_py.reporting.render import FORMATS, write_reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate reports from a completed CopyKAT run without inference")
    parser.add_argument("--run-dir", required=True, help="Directory containing saved runtime JSON and results")
    parser.add_argument("--sample-name", default=None, help="Required when the directory contains multiple runs")
    parser.add_argument(
        "--formats", default="html", help=f"Comma-separated formats: {', '.join(FORMATS)} (default: html)"
    )
    parser.add_argument("--output-dir", default=None, help="Report destination (default: run directory)")
    args = parser.parse_args(argv)
    try:
        report = load_report(args.run_dir, args.sample_name)
        paths = write_reports(report, [fmt.strip() for fmt in args.formats.split(",")], args.output_dir or args.run_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
