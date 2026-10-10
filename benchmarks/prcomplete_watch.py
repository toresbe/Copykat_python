"""Refresh follow-up plots until the six-PR/VRAM evidence is archived."""

import json
import subprocess
import time
from pathlib import Path

import bench_config as cfg

r = cfg.ROOT / "results/prcomplete_schedule.json"
python = cfg.PYTHON
while True:
    for name in ["cpu_report.py", "serial_report.py"]:
        subprocess.run([python, "-B", str(Path(__file__).with_name(name))], check=True)
    if r.exists() and any(e["event"] == "archived" for e in json.loads(r.read_text())):
        break
    time.sleep(60)
