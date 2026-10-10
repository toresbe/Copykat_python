"""Refresh scientific results while the durable accuracy coordinator runs."""

import json
import subprocess
import time
from pathlib import Path

root = Path("/home/toresbe/cancer_research/navin_accuracy_2026-10-10")
report = Path(__file__).with_name("navin_accuracy_report.py")
python = "/home/toresbe/envs/copykat_py_gpu/bin/python"
while True:
    subprocess.run([python, str(report)], check=True)
    schedule = root / "results/accuracy_schedule.json"
    if schedule.exists() and any(e["event"] == "archived" for e in json.loads(schedule.read_text())):
        break
    time.sleep(60)
