"""Refresh scientific results while the durable accuracy coordinator runs."""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess
import time
from pathlib import Path

import bench_config as cfg

root = cfg.ACCURACY_ROOT
report = Path(__file__).with_name("accuracy_report.py")
python = cfg.GPU_PYTHON
while True:
    subprocess.run([python, str(report)], check=True)
    schedule = root / "results/accuracy_schedule.json"
    if schedule.exists() and any(e["event"] == "archived" for e in json.loads(schedule.read_text())):
        break
    time.sleep(60)
