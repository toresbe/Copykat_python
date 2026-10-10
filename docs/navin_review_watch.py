"""Keep the visible report fresh while the existing benchmark scheduler runs."""

import json
import subprocess
import time
from pathlib import Path

root = Path("/home/toresbe/cancer_research/navin_review_2026-10-09")
report = Path(__file__).with_name("navin_review_cpu_report.py")
while True:
    subprocess.run(["/home/toresbe/envs/copykat_py_main/bin/python", str(report)], check=True)
    serial_manifest = root / "serial_sweep_manifest.json"
    if serial_manifest.exists():
        subprocess.run(
            ["/home/toresbe/envs/copykat_py_main/bin/python", str(report.with_name("navin_review_serial_report.py"))],
            check=True,
        )
    schedule = json.loads((root / "results/parallel_cpuaccount_schedule.json").read_text())
    final_schedule = (
        root / "results/serial8_schedule.json"
        if serial_manifest.exists()
        else root / "results/parallel_cpuaccount_schedule.json"
    )
    finished = json.loads(final_schedule.read_text()) if final_schedule.exists() else []
    if any(row["event"] == "archived" for row in finished):
        break
    time.sleep(60)
