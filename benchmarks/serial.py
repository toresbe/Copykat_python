"""Two serial repeat rounds, gated on completion of the existing expanded sweep."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import bench_config as cfg
from benchmark import ROOT, save

HERE = Path(__file__).resolve().parent
DOCS = Path(__file__).resolve().parents[1] / "docs"
DRIVER = HERE / "benchmark.py"
SIZES = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000]
VARIANTS = ["main", "optimistic", "cpu_stack", "gpu"]
jobs = [
    {"variant": v, "cells": n, "repeat": rep, "name": f"{v}_xenium{n}_serial8_r{rep}"}
    for rep in [1, 2]
    for n in (SIZES if rep == 1 else list(reversed(SIZES)))
    for v in (VARIANTS if rep == 1 else list(reversed(VARIANTS)))
]
manifest = {
    "sizes": SIZES,
    "repeats": 2,
    "jobs": jobs,
    "cpu_affinity": list(range(8)),
    "cores": 8,
    "maximum_concurrent_jobs": 1,
    "nested_subset_seed": 20261009,
    "plot": False,
    "timeout_s": 0,
    "gate": "Original 48-job expanded sweep recorded and its NAS archive completed",
    "ordering": "Ascending sizes/main-first round 1; descending sizes/GPU-first round 2",
    "caches": "Fresh processes; input/filesystem/compilation caches not flushed",
}
save(ROOT / "serial_sweep_manifest.json", manifest)
schedule = ROOT / "results/serial8_schedule.json"
events = json.loads(schedule.read_text()) if schedule.exists() else []


def event(kind, **fields):
    events.append(dict(event=kind, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields))
    save(schedule, events)
    print(json.dumps(events[-1]), flush=True)


if not events:
    event("waiting_for_original_sweep", jobs=len(jobs))
# No computation starts while the primary sweep or its archive is unfinished.
while True:
    original = json.loads((ROOT / "expanded_sweep_manifest.json").read_text())
    records = all((ROOT / "results" / (j["name"] + ".json")).exists() for j in original["jobs"])
    original_events = json.loads((ROOT / "results/parallel_cpuaccount_schedule.json").read_text())
    if records and any(e["event"] == "archived" for e in original_events):
        break
    time.sleep(30)
lock = open(ROOT / "results/scheduler.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
while True:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        time.sleep(5)
event("serial_phase_started")

for job in jobs:
    name = job["name"]
    final = ROOT / "results" / (name + ".json")
    if final.exists():
        continue
    previous = next((e for e in reversed(events) if e.get("name") == name and e["event"] == "started"), None)
    process = None
    adopted = False
    if previous:
        pid = previous["pid"]
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            cmd = ""
        if str(DRIVER) in cmd and name in cmd:
            adopted = True
            event("adopted", name=name, pid=pid)
        else:
            live = ROOT / "results" / (name + ".live.json")
            worker_pid = json.loads(live.read_text()).get("pid") if live.exists() else None
            try:
                worker_cmd = Path(f"/proc/{worker_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                worker_cmd = ""
            if name in worker_cmd and str(DRIVER) in worker_cmd:
                raise SystemExit(
                    f"Live scoped worker {worker_pid} survives supervisor loss for {name}; recover "
                    f"observer before serial work resumes"
                )
            event("interrupted", name=name)
            for suffix in [".progress.json", ".live.json", ".worker.json", ".log", ".supervisor.log"]:
                old = ROOT / "results" / (name + suffix)
                if old.exists():
                    old.rename(old.with_name(old.name + f".interrupted-{time.time_ns()}"))
    if not adopted:
        # Also guard against unrelated review workers outside the primary queue.
        while True:
            review_active = False
            for p in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    command = p.read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    continue
                if str(DRIVER) in command:
                    review_active = True
                    break
            available = next(
                int(line.split()[1]) / 1e6
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            if not review_active and available >= 15:
                break
            time.sleep(10)
        env = dict(os.environ, BENCH_COHORT="[]", BENCH_SCHEDULING="serial8-repeat")
        command = [
            sys.executable,
            str(DRIVER),
            "run",
            "--variant",
            job["variant"],
            "--sample",
            f"xenium{job['cells']}",
            "--name",
            name,
            "--cores",
            "8",
            "--cpu-list",
            "0,1,2,3,4,5,6,7",
            "--timeout",
            "0",
            "--null-outputs",
            "--accounting-scope",
        ]
        if job["variant"] == "gpu":
            command += ["--backend", "gpu"]
        log = open(ROOT / "results" / (name + ".supervisor.log"), "w")  # noqa: SIM115 - task owns this handle and closes it when its worker finishes.
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        pid = process.pid
        event("started", **job, pid=pid, cpu_affinity=list(range(8)), concurrent=[], mem_available_gb=available)
    while not final.exists():
        if process is not None and process.poll() is not None:
            break
        if process is None and not Path(f"/proc/{pid}").exists():
            break
        time.sleep(5)
    if process is not None:
        process.wait()
        log.close()
    if not final.exists():
        raise SystemExit("Supervisor ended without a recorded result; refusing to silently skip " + name)
    result = json.loads(final.read_text())
    event(
        "finished",
        name=name,
        status=result["status"],
        pipeline_s=result.get("pipeline_s"),
        pipeline_cpu_s=result.get("pipeline_cpu_s"),
    )
event("complete")
subprocess.run([cfg.PYTHON, str(HERE / "serial_report.py")], check=True)
archive = cfg.archive_path("benchmark-serial8-evidence.tar.gz")
if archive.exists():
    raise SystemExit("Refusing to overwrite " + str(archive))
temporary = archive.with_suffix(".tmp")
with tarfile.open(temporary, "w:gz") as tar:
    for relative in [
        "results",
        "snapshots",
        "snapshot_manifest.json",
        "input_manifest.json",
        "expanded_sweep_manifest.json",
        "serial_sweep_manifest.json",
    ]:
        tar.add(ROOT / relative, arcname=relative)
    for path in HERE.glob("*.py"):
        tar.add(path, arcname="benchmarks/" + path.name)
    for path in DOCS.glob("benchmark-*"):
        tar.add(path, arcname="docs/" + path.name)
temporary.replace(archive)
h = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        h.update(block)
archive.with_name(archive.name + ".sha256").write_text(h.hexdigest() + "  " + archive.name + "\n")
event("archived", path=str(archive), sha256=h.hexdigest())
