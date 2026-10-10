"""Serial six-PR and isolated GPU-memory follow-up after accuracy finishes."""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import bench_config as cfg
from common import ROOT, save

DOCS = Path(__file__).resolve().parents[2] / "docs"
DRIVER = cfg.HERE / "benchmark.py"
PYTHON = cfg.PYTHON
ACCURACY = cfg.ACCURACY_ROOT / "results/accuracy_schedule.json"
SIZES = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 80000, 120000, 170057]
schedule = ROOT / "results/prcomplete_schedule.json"
events = json.loads(schedule.read_text()) if schedule.exists() else []
if any(e["event"] == "archived" for e in events):
    raise SystemExit(0)


def event(kind, **fields):
    row = dict(event=kind, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
    events.append(row)
    save(schedule, events)
    print(json.dumps(row), flush=True)


jobs = [{"variant": "gpu", "cells": n, "name": f"gpu_xenium{n}_vram", "gpu_memory": True, "repeat": 1} for n in SIZES]
jobs += [
    {"variant": "proposed6", "cells": n, "name": f"proposed6_xenium{n}_prcomplete_r1", "gpu_memory": False, "repeat": 1}
    for n in SIZES
]
jobs += [
    {"variant": "proposed6", "cells": n, "name": f"proposed6_xenium{n}_prcomplete_r2", "gpu_memory": False, "repeat": 2}
    for n in reversed(SIZES[:8])
]
save(
    ROOT / "prcomplete_manifest.json",
    {
        "jobs": jobs,
        "sizes": SIZES,
        "prs": [4, 5, 6, 7, 8, 9],
        "gate": "accuracy archived",
        "cores": 8,
        "serial": True,
        "gpu_memory_method": "per-benchmark process NVML, 0.25s polling; PyTorch allocator peaks separately",
        "plot": False,
        "null_outputs": True,
        "timeout_s": 0,
    },
)
if not events:
    event("waiting_for_accuracy_archive")
while not (ACCURACY.exists() and any(e["event"] == "archived" for e in json.loads(ACCURACY.read_text()))):
    time.sleep(30)
lock = open(ROOT / "results/scheduler.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
fcntl.flock(lock, fcntl.LOCK_EX)


def valid_pid(pid, name):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return str(DRIVER) in cmd and name in cmd


for job in jobs:
    output = ROOT / "results" / (job["name"] + ".json")
    if output.exists():
        continue
    old = next((e for e in reversed(events) if e["event"] == "started" and e["name"] == job["name"]), None)
    if old and valid_pid(old["pid"], job["name"]):
        event("adopted", name=job["name"], pid=old["pid"])
        while valid_pid(old["pid"], job["name"]) and not output.exists():
            time.sleep(5)
        if output.exists():
            continue
    # Never duplicate a surviving separately scoped worker after a wrapper crash.
    live = ROOT / "results" / (job["name"] + ".live.json")
    if live.exists():
        z = json.loads(live.read_text())
        if z.get("pid") and valid_pid(z["pid"], job["name"]):
            event("surviving_worker_needs_recovery", name=job["name"], pid=z["pid"])
            raise SystemExit(1)
    for suffix in [".worker.json", ".progress.json", ".live.json"]:
        p = ROOT / "results" / (job["name"] + suffix)
        if p.exists():
            p.rename(p.with_name(p.name + f".interrupted-{time.time_ns()}"))
    env = dict(
        os.environ,
        BENCH_COHORT="[]",
        BENCH_SCHEDULING="physical8-prcomplete-serial",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [
        PYTHON,
        "-B",
        str(DRIVER),
        "run",
        "--variant",
        job["variant"],
        "--sample",
        f"xenium{job['cells']}",
        "--name",
        job["name"],
        "--cores",
        "8",
        "--cpu-list",
        "0,1,2,3,4,5,6,7",
        "--timeout",
        "0",
        "--null-outputs",
        "--accounting-scope",
    ]
    if job["gpu_memory"]:
        command += ["--backend", "gpu", "--gpu-memory"]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    event("started", name=job["name"], pid=process.pid, job=job)
    process.wait()
    if not output.exists():
        raise RuntimeError("Supervisor exited without output " + job["name"])
    z = json.loads(output.read_text())
    event("finished", name=job["name"], status=z["status"])
    subprocess.run([PYTHON, "-B", str(cfg.HERE / "reports/cpu_report.py")], check=True)
    subprocess.run([PYTHON, "-B", str(cfg.HERE / "reports/serial_report.py")], check=True)
event("complete")
archive = cfg.archive_path("benchmark-six-pr-vram-evidence.tar.gz")
if not archive.exists():
    temp = archive.with_suffix(".tmp")
    with tarfile.open(temp, "w:gz") as tar:
        for name in [
            "results",
            "snapshots",
            "proposed6_snapshot_manifest.json",
            "prcomplete_manifest.json",
            "snapshot_manifest.json",
            "input_manifest.json",
        ]:
            tar.add(ROOT / name, arcname=name)
        tar.add(cfg.HERE, arcname="benchmarks", filter=lambda i: None if "__pycache__" in i.name else i)
        for p in DOCS.glob("benchmark-*"):
            tar.add(p, arcname="docs/" + p.name)
    temp.replace(archive)
h = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        h.update(block)
archive.with_name(archive.name + ".sha256").write_text(h.hexdigest() + "  " + archive.name + "\n")
event("archived", path=str(archive), sha256=h.hexdigest())
