"""Restart-safe, RAM-gated expanded sweep; preserve every existing result/ref."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path("/home/toresbe/cancer_research/navin_review_2026-10-09")
DRIVER = Path(__file__).with_name("navin_review_benchmark.py")
SIZES = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 80000, 120000, 170057]
LANES = [list(range(8)), list(range(8, 16))]  # 5950X: exclude SMT siblings 16..31
lock = open(ROOT / "results/scheduler.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


schedule = ROOT / "results/parallel_cpuaccount_schedule.json"
events = json.loads(schedule.read_text()) if schedule.exists() else []


def event(kind, **data):
    row = dict(event=kind, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data)
    events.append(row)
    save(schedule, events)
    print(json.dumps(row), flush=True)


def estimate(variant, n):
    if variant == "gpu":
        return max(12, 8 + n / 10000)
    if variant == "cpu_stack":
        return max(12, 12 + n / 7000)
    return max(18, 18 + n / 2500 + (2 if variant == "optimistic" else 0))


def job(variant, n, control=False):
    return {
        "variant": variant,
        "cells": n,
        "control": control,
        "name": f"{variant}_xenium{n}_" + ("isolated8" if control else "cpuaccount"),
        "ram_estimate_gb": estimate(variant, n),
    }


# Finish cheap paired curves first; more costly large upstream runs come later.
jobs = [job(v, n) for n in SIZES[:7] for v in ["main", "optimistic", "cpu_stack", "gpu"]]
jobs += [job(v, n) for n in SIZES[7:10] for v in ["cpu_stack", "gpu"]]
jobs += [job(v, n) for n in SIZES[7:10] for v in ["main", "optimistic"]]
jobs += [job(v, 170057) for v in ["cpu_stack", "gpu", "optimistic", "main"]]
jobs += [job(v, 10000, True) for v in ["main", "optimistic", "cpu_stack", "gpu"]]
manifest = {
    "sizes": SIZES,
    "jobs": jobs,
    "lanes": LANES,
    "maximum_concurrent_jobs": 2,
    "memory_headroom_gb": 11,
    "emergency_available_ram_gb": 10,
    "policy": (
        "Eight distinct physical cores per new job; at most two jobs; one new job during "
        "surviving unpinned runs; isolated 10k anchors after those finish"
    ),
    "nested_subset_seed": 20261009,
    "plot": False,
    "timeout_s": 0,
}
save(ROOT / "expanded_sweep_manifest.json", manifest)


class ExistingProcess:
    def __init__(self, pid, name):
        self.pid, self.name, self.returncode = pid, name, None

    def poll(self):
        path = ROOT / "results" / (self.name + ".json")
        if path.exists():
            self.returncode = json.loads(path.read_text()).get("exit_code", 0)
        else:
            try:
                command = Path(f"/proc/{self.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                valid_driver = str(DRIVER) in command or str(DRIVER.with_name("navin_review_recover.py")) in command
                if not valid_driver or self.name not in command:
                    self.returncode = -1
            except OSError:
                self.returncode = -1
        return self.returncode


pending, running = [], []
for item in jobs:
    if (ROOT / "results" / (item["name"] + ".json")).exists():
        continue
    previous = next((e for e in reversed(events) if e.get("name") == item["name"] and e["event"] == "started"), None)
    if previous:
        process = ExistingProcess(previous["pid"], item["name"])
        if process.poll() is None:
            item.update(process=process, lane=previous.get("cpu_affinity"), log=None)
            running.append(item)
            event("adopted", name=item["name"], pid=process.pid, cpu_affinity=item["lane"])
            continue
        # A lost wrapper is not proof that its separately scoped worker stopped.
        # Preserve its evidence and wait for recovery rather than launch a duplicate.
        live_path = ROOT / "results" / (item["name"] + ".live.json")
        if live_path.exists():
            live_worker = json.loads(live_path.read_text()).get("pid")
            if live_worker:
                try:
                    worker_command = Path(f"/proc/{live_worker}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    worker_command = ""
                if item["name"] in worker_command and str(DRIVER) in worker_command:
                    raise SystemExit(
                        f"Live worker {live_worker} survives lost supervisor for {item['name']}; restore "
                        f"observer before resuming admissions"
                    )
        event(
            "interrupted",
            name=item["name"],
            pid=previous["pid"],
            reason="No completed result and original supervisor no longer exists",
        )
        # Keep the partial evidence rather than overwrite it on restart.
        for suffix in [".progress.json", ".live.json", ".log", ".supervisor.log", ".worker.json"]:
            old = ROOT / "results" / (item["name"] + suffix)
            if old.exists():
                old.rename(old.with_name(old.name + f".interrupted-{time.time_ns()}"))
    pending.append(item)

while pending or running:
    for item in list(running):
        if item["process"].poll() is not None:
            if item["log"]:
                item["log"].close()
            path = ROOT / "results" / (item["name"] + ".json")
            result = json.loads(path.read_text()) if path.exists() else {}
            event(
                "finished",
                name=item["name"],
                exit_code=item["process"].returncode,
                status=result.get("status", "crashed"),
                wall_s=result.get("pipeline_s"),
                cpu_s=result.get("pipeline_cpu_s"),
                peak_rss_gb=result.get("tree_peak_rss_gb"),
            )
            running.remove(item)
    legacy = any(item["lane"] is None for item in running)
    controls = [item for item in pending if item["control"]]
    # Isolated controls cannot be starved by continuously arriving primary work.
    eligible = controls if controls and not legacy else [item for item in pending if not item["control"]]
    for item in list(eligible):
        if len(running) >= 2 or any(j["control"] for j in running):
            break
        if item["control"] and running:
            break
        occupied = [j["lane"] for j in running if j["lane"] is not None]
        lane = next((lane for lane in LANES if lane not in occupied), None)
        if lane is None:
            break
        growth = 0
        for other in running:
            live = ROOT / "results" / (other["name"] + ".live.json")
            rss = json.loads(live.read_text()).get("rss_gb", 0) if live.exists() else 0
            growth += max(0, other["ram_estimate_gb"] - rss)
        free = next(
            int(x.split()[1]) / 1e6
            for x in Path("/proc/meminfo").read_text().splitlines()
            if x.startswith("MemAvailable:")
        )
        if running and free - growth - item["ram_estimate_gb"] < 11:
            continue
        if not running and free < 15:
            continue
        env = dict(
            os.environ,
            NAVIN_BENCH_COHORT=json.dumps([j["name"] for j in running]),
            NAVIN_BENCH_SCHEDULING="isolated8" if item["control"] else "physical8-parallel2",
        )
        command = [
            sys.executable,
            str(DRIVER),
            "run",
            "--variant",
            item["variant"],
            "--sample",
            f"xenium{item['cells']}",
            "--name",
            item["name"],
            "--cores",
            "8",
            "--cpu-list",
            ",".join(map(str, lane)),
            "--timeout",
            "0",
            "--null-outputs",
            "--accounting-scope",
        ]
        if item["variant"] == "gpu":
            command += ["--backend", "gpu"]
        log = open(ROOT / "results" / (item["name"] + ".supervisor.log"), "w")  # noqa: SIM115 - task owns this handle and closes it when its worker finishes.
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        item.update(process=process, log=log, lane=lane)
        running.append(item)
        pending.remove(item)
        event(
            "started",
            name=item["name"],
            pid=process.pid,
            ram_estimate_gb=item["ram_estimate_gb"],
            mem_available_gb=free,
            reserved_growth_gb=growth,
            cores=8,
            cpu_affinity=lane,
            legacy_unpinned_overlap=legacy,
            isolated_control=item["control"],
            timeout_s=0,
            concurrent=[j["name"] for j in running],
        )
    time.sleep(5)
event("complete", expanded=True)
subprocess.run(
    ["/home/toresbe/envs/copykat_py_main/bin/python", str(DRIVER.with_name("navin_review_cpu_report.py"))], check=True
)
archive = Path("/mnt/nas/cancer_research/navin_review_2026-10-09/navin-review-cpuaccount-evidence.tar.gz")
archive.parent.mkdir(exist_ok=True)
if archive.exists():
    raise SystemExit("Refusing to overwrite evidence archive: " + str(archive))
temporary = archive.with_suffix(".tmp")
with tarfile.open(temporary, "w:gz") as tar:
    for relative in [
        "results",
        "snapshots",
        "snapshot_manifest.json",
        "input_manifest.json",
        "expanded_sweep_manifest.json",
    ]:
        tar.add(ROOT / relative, arcname=relative)
    for path in DRIVER.parent.glob("navin*"):
        tar.add(path, arcname="docs/" + path.name)
temporary.replace(archive)
h = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        h.update(block)
archive.with_name(archive.name + ".sha256").write_text(h.hexdigest() + "  " + archive.name + "\n")
event("archived", path=str(archive), sha256=h.hexdigest(), bytes=archive.stat().st_size)
