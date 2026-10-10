"""Refresh follow-up plots until the six-PR/VRAM evidence is archived."""

import json
import subprocess
import time
from pathlib import Path

r = Path("/home/toresbe/cancer_research/benchmark_2026-10-09/results/prcomplete_schedule.json")
python = "/home/toresbe/envs/copykat_py_main/bin/python"
while True:
    for name in ["cpu_report.py", "serial_report.py"]:
        subprocess.run([python, "-B", str(Path(__file__).with_name(name))], check=True)
    if r.exists() and any(e["event"] == "archived" for e in json.loads(r.read_text())):
        break
    time.sleep(60)
