"""Stage one matrix group independently of the inference dispatcher."""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import traceback

from accuracy_data import ROOT, save, stage_group

p = argparse.ArgumentParser()
p.add_argument("--cases", required=True)
p.add_argument("--name", required=True)
a = p.parse_args()
ids = a.cases.split(",")
cases = [c for c in json.loads((ROOT / "dataset_manifest.json").read_text())["cases"] if c["id"] in ids]
try:
    assert len(cases) == len(ids)
    stage_group(cases)
    result = {"status": "ok", "cases": ids}
except Exception:
    result = {"status": "input_error", "cases": ids, "error": traceback.format_exc()}
save(ROOT / "results" / (a.name + ".json"), result)
