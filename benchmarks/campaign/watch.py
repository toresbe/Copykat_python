"""Keep the visible report fresh while the existing benchmark scheduler runs."""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json
import subprocess
import time

import bench_config as cfg

root = cfg.ROOT
report = cfg.HERE / "reports/cpu_report.py"
while True:
    subprocess.run([cfg.PYTHON, str(report)], check=True)
    serial_manifest = root / "serial_sweep_manifest.json"
    if serial_manifest.exists():
        subprocess.run(
            [cfg.PYTHON, str(cfg.HERE / "reports/serial_report.py")],
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
