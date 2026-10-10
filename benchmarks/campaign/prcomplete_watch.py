"""Refresh follow-up plots until the six-PR/VRAM evidence is archived."""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess
import time

import bench_config as cfg

r = cfg.ROOT / "results/prcomplete_schedule.json"
python = cfg.PYTHON
while True:
    for name in ["cpu_report.py", "serial_report.py"]:
        subprocess.run([python, "-B", str(cfg.HERE / "reports" / name)], check=True)
    if r.exists() and any(e["event"] == "archived" for e in json.loads(r.read_text())):
        break
    time.sleep(60)
