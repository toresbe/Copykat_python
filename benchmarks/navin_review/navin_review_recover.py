"""Recover supervision of an existing scope without restarting its calculation."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from navin_review_benchmark import ROOT, save, tree_metrics

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--worker-pid", required=True, type=int)
a = p.parse_args()
name = a.name
if (ROOT / "results" / f"{name}.json").exists():
    sys.exit(0)
scope = "copykat-navin-" + name.replace("_", "-") + ".scope"
group = subprocess.check_output(
    ["systemctl", "--user", "show", scope, "-p", "ControlGroup", "--value"], text=True
).strip()
if not group:
    raise SystemExit("Original scope not present; refusing to restart computation")
cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
command = Path(f"/proc/{a.worker_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if name not in command or "navin_review_benchmark.py" not in command:
    raise SystemExit("Worker identity mismatch")
progress = ROOT / "results" / f"{name}.progress.json"
measured = ROOT / "results" / f"{name}.worker.json"
final = ROOT / "results" / f"{name}.json"
live = ROOT / "results" / f"{name}.live.json"
initial = json.loads(live.read_text())
process_start_ticks = int(Path(f"/proc/{a.worker_pid}/stat").read_text().rpartition(")")[2].split()[19])
process_started = process_start_ticks / os.sysconf("SC_CLK_TCK")
previous = json.loads(progress.read_text())
recovery_started = time.perf_counter()
# Register this observer before the coordinator starts and adopts it.
with open(ROOT / "results/scheduler.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    schedule = ROOT / "results/parallel_cpuaccount_schedule.json"
    events = json.loads(schedule.read_text())
    events.append(
        {
            "event": "started",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "name": name,
            "pid": os.getpid(),
            "worker_pid": a.worker_pid,
            "recovery": True,
            "cpu_affinity": None,
            "ram_estimate_gb": 36,
            "reason": "Original supervisor lost; original worker and cgroup preserved",
        }
    )
    save(schedule, events)
observed = {}
peak = initial.get("rss_gb", 0) * 1e9
min_available = next(
    int(line.split()[1]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
)
cpu = None
memory_peak = 0
reason = None
while not measured.exists():
    rss, sampled_cpu = tree_metrics(a.worker_pid, observed)
    peak = max(peak, rss)
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    min_available = min(min_available, available)
    try:
        counters = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())
        cpu = int(counters["usage_usec"]) / 1e6
        memory_peak = max(memory_peak, int((cgroup / "memory.peak").read_text()))
    except OSError:
        pass
    current = json.loads(progress.read_text()) if progress.exists() else previous
    start = current.get("pipeline_started_monotonic", recovery_started)
    save(
        live,
        {
            "name": name,
            "variant": current.get("variant"),
            "pid": a.worker_pid,
            "observer_pid": os.getpid(),
            "status": "running",
            "recovered_supervision": True,
            "pipeline_elapsed_s": time.perf_counter() - start,
            "wall_elapsed_s": time.perf_counter() - process_started,
            "tree_cpu_s": cpu,
            "cgroup_cpu_s": cpu,
            "rss_gb": rss / 1e9,
            "last_step": current.get("steps", [{}])[-1].get("step"),
        },
    )
    if available < 10e9:
        reason = "memory_guard"
    elif shutil.disk_usage(ROOT).free < 2e9:
        reason = "ssd_space_guard"
    elif not Path(f"/proc/{a.worker_pid}").exists():
        reason = "crashed"
    if reason:
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", scope], check=False)
        break
    time.sleep(1)
result = json.loads(measured.read_text()) if measured.exists() else dict(current, status=reason)
result.update(
    name=name,
    accounting_scope=scope,
    timeout_s=0,
    elapsed_process_s=time.perf_counter() - process_started,
    recovered_supervision=True,
    original_worker_pid=a.worker_pid,
    observer_pid=os.getpid(),
    exit_code=0 if measured.exists() else -1,
    process_cgroup_cpu_s_last_sample=cpu,
    cgroup_memory_peak_gb=memory_peak / 1e9,
    min_mem_available_after_recovery_gb=min_available / 1e9,
    tree_peak_rss_observed_after_recovery_gb=peak / 1e9,
    tree_peak_rss_gb=None,
    memory_measurement_note=(
        "Original supervisor peak-RSS history was lost. Whole-run cgroup memory.peak is "
        "retained; recovery RSS is a lower bound, not whole-run peak."
    ),
    cpu_measurement=(
        "Original dedicated cgroup intact; pipeline worker retains exact all-thread/all-descendant boundary readings"
    ),
    scheduling_policy="legacy unpinned; recovered observer",
    cpu_affinity=list(range(32)),
)
if not measured.exists():
    result["pipeline_elapsed_at_stop_s"] = time.perf_counter() - result["pipeline_started_monotonic"]
save(final, result)
save(
    live,
    {
        "name": name,
        "status": result["status"],
        "recovered_supervision": True,
        "wall_elapsed_s": time.perf_counter() - process_started,
    },
)
subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", scope], check=False)
shutil.rmtree(ROOT / "scratch" / f"{name}.joblib", ignore_errors=True)
print(
    json.dumps(
        {
            "name": name,
            "status": result["status"],
            "pipeline_s": result.get("pipeline_s"),
            "pipeline_cpu_s": result.get("pipeline_cpu_s"),
        }
    ),
    flush=True,
)
