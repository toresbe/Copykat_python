"""Accuracy-only parallel study, gated after both performance phases finish."""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

from navin_accuracy_data import ROOT, discover, save

PERF = Path("/home/toresbe/cancer_research/navin_review_2026-10-09")
HERE = Path(__file__).resolve().parent
DOCS = Path(__file__).resolve().parents[2] / "docs"
WORKER = HERE / "navin_accuracy_worker.py"
PYTHON = "/home/toresbe/envs/copykat_py_gpu/bin/python"  # same dependencies for both implementations
schedule = ROOT / "results/accuracy_schedule.json"
CPU_LANES = [list(range(i, i + 4)) for i in range(0, 28, 4)]
GPU_LANE = list(range(28, 32))
MAX_CPU = 7
MAX_GPU = 2
events = json.loads(schedule.read_text()) if schedule.exists() else []
if any(e["event"] == "archived" for e in events):
    raise SystemExit(0)


def event(kind, **fields):
    row = dict(event=kind, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
    events.append(row)
    save(schedule, events)
    print(json.dumps(row), flush=True)


if not events:
    event("waiting_for_serial_repeats")
while True:
    serial = PERF / "results/serial8_schedule.json"
    if serial.exists() and any(e["event"] == "archived" for e in json.loads(serial.read_text())):
        break
    time.sleep(30)
# Share the existing scheduler lock: no concurrent review phases.
lock = open(PERF / "results/scheduler.lock", "w")  # noqa: SIM115 - process-lifetime flock; descriptor must stay open until exit.
while True:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        time.sleep(5)
manifest = ROOT / "dataset_manifest.json"
cases = json.loads(manifest.read_text())["cases"] if manifest.exists() else discover()
if not cases:
    raise SystemExit("No eligible installed datasets; manifest records exclusions")
event("accuracy_phase_started", cases=len(cases), studies=len({c["study"] for c in cases}))
event(
    "concurrency_policy",
    max_cpu=MAX_CPU,
    max_gpu=MAX_GPU,
    threads=4,
    cpu_lanes=CPU_LANES,
    gpu_lane=GPU_LANE,
    note="Accuracy-only throughput; SMT sharing permitted; previous workers retain their original affinity",
)
# Every dataset has exactly the two requested endpoints.
tasks = [{"case": c, "variant": v, "name": c["id"] + "__" + v} for c in cases for v in ["main", "gpu_anchor"]]
running = []
pending = []


def alive(pid, name):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return str(WORKER) in command and name.split("__")[0] in command and name.split("__")[1] in command


for task in tasks:
    if (ROOT / "results" / (task["name"] + ".json")).exists():
        continue
    previous = next((e for e in reversed(events) if e.get("name") == task["name"] and e["event"] == "started"), None)
    if previous and alive(previous["pid"], task["name"]):
        task.update(pid=previous["pid"], process=None, log=None, lane=previous["cpu_affinity"])
        running.append(task)
        event("adopted", name=task["name"], pid=task["pid"])
        continue
    if previous:
        event("interrupted", name=task["name"])
        for suffix in [".log", ".live.json"]:
            p = ROOT / "results" / (task["name"] + suffix)
            if p.exists():
                p.rename(p.with_name(p.name + f".interrupted-{time.time_ns()}"))
    pending.append(task)

# Study matrices are staged once, then only small sample caches are reused.
groups = {}
for case in cases:
    groups.setdefault(case["path"], []).append(case)
STAGER = HERE / "navin_accuracy_stage.py"
staging = None
previous_stage = next((e for e in reversed(events) if e["event"] == "staging_started"), None)
if previous_stage:
    try:
        cmd = Path(f"/proc/{previous_stage['pid']}/cmdline").read_bytes().decode().replace("\0", " ")
        if str(STAGER) in cmd and previous_stage["name"] in cmd:
            staging = dict(previous_stage, process=None, log=None)
            event("staging_adopted", name=staging["name"], pid=staging["pid"])
    except OSError:
        pass


def available():
    return next(
        int(s.split()[1]) / 1e6 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")
    )


def rss(pid):
    total = 0
    seen = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text().splitlines()
            total += next(int(s.split()[1]) * 1024 for s in status if s.startswith("VmRSS:"))
            stack += list(map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split()))
        except (OSError, StopIteration):
            pass
    return total / 1e9


def estimate(task):
    return max(6, 4 + task["case"]["n_cells"] / 2000)


def cached(case):
    return (ROOT / "cache" / (case["id"] + ".npz")).exists() and (
        ROOT / "results" / (case["id"] + ".input.json")
    ).exists()


def finish(task, reason=None):
    if task["process"] is not None:
        task["process"].wait()
    if task["log"]:
        task["log"].close()
    p = ROOT / "results" / (task["name"] + ".json")
    if not p.exists():
        save(
            p,
            {
                "case": task["case"],
                "variant": task["variant"],
                "status": reason or "worker_crashed",
                "performance_measurements": False,
            },
        )
    result = json.loads(p.read_text())
    save(ROOT / "results" / (task["name"] + ".live.json"), {"name": task["name"], "status": result["status"]})
    event("finished", name=task["name"], status=result["status"])
    running.remove(task)
    shutil.rmtree(ROOT / "scratch" / (task["name"] + ".joblib"), ignore_errors=True)
    shutil.rmtree(ROOT / "scratch" / task["name"], ignore_errors=True)
    # Cache can be removed only after BOTH endpoint attempts are recorded.
    if all((ROOT / "results" / (task["case"]["id"] + "__" + v + ".json")).exists() for v in ["main", "gpu_anchor"]):
        (ROOT / "cache" / (task["case"]["id"] + ".npz")).unlink(missing_ok=True)


while pending or running or staging is not None:
    if staging is not None:
        receipt = ROOT / "results" / (staging["name"] + ".json")
        try:
            stage_cmd = Path(f"/proc/{staging['pid']}/cmdline").read_bytes().decode().replace("\0", " ")
            stage_alive = str(STAGER) in stage_cmd and staging["name"] in stage_cmd
        except OSError:
            stage_alive = False
        if receipt.exists() or not stage_alive:
            if staging["process"] is not None:
                staging["process"].wait()
            if staging["log"] is not None:
                staging["log"].close()
            outcome = (
                json.loads(receipt.read_text())
                if receipt.exists()
                else {"status": "input_error", "error": "Staging process exited without receipt"}
            )
            if outcome["status"] != "ok":
                failed = []
                for c in cases:
                    if c["id"] not in staging["cases"]:
                        continue
                    if cached(c):
                        continue
                    failed.append(c["id"])
                    for v in ["main", "gpu_anchor"]:
                        p = ROOT / "results" / (c["id"] + "__" + v + ".json")
                        if not p.exists():
                            save(
                                p,
                                {
                                    "case": c,
                                    "variant": v,
                                    "status": "input_error",
                                    "error": outcome["error"],
                                    "performance_measurements": False,
                                },
                            )
                pending = [t for t in pending if t["case"]["id"] not in failed]
            event(
                "staged" if outcome["status"] == "ok" else "staging_error",
                cases=staging["cases"],
                error=outcome.get("error"),
            )
            staging = None
    for task in list(running):
        ended = (
            task["process"].poll() is not None if task["process"] is not None else not alive(task["pid"], task["name"])
        )
        if ended:
            finish(task)
        else:
            save(
                ROOT / "results" / (task["name"] + ".live.json"),
                {"name": task["name"], "pid": task["pid"], "status": "running", "rss_gb": rss(task["pid"])},
            )
    if available() < 10 or shutil.disk_usage(ROOT).free < 2e9:
        if running:
            victim = running[-1]
            os.killpg(victim["pid"], signal.SIGKILL)
            while alive(victim["pid"], victim["name"]):
                time.sleep(1)
            finish(victim, "memory_guard" if available() < 10 else "ssd_space_guard")
        elif staging is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(staging["pid"], signal.SIGKILL)
        time.sleep(5)
        continue
    # Stage ahead only when memory can cover parser + existing future growth.
    growth = sum(max(0, estimate(t) - rss(t["pid"])) for t in running)
    if staging is None and available() - growth >= 32 and shutil.disk_usage(ROOT).free >= 6e9:
        needing = next((t for t in pending if not cached(t["case"])), None)
        if needing:
            group = [
                c
                for c in groups[needing["case"]["path"]]
                if any(t["case"]["id"] == c["id"] for t in pending) and not cached(c)
            ]
            ids = [c["id"] for c in group]
            name = "accuracy_stage_" + hashlib.sha256(",".join(ids).encode()).hexdigest()[:12]
            receipt = ROOT / "results" / (name + ".json")
            receipt.unlink(missing_ok=True)
            log = open(ROOT / "results" / (name + ".log"), "w")  # noqa: SIM115 - task owns this handle and closes it when its worker finishes.
            process = subprocess.Popen(
                [PYTHON, "-B", "-u", str(STAGER), "--cases", ",".join(ids), "--name", name],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            staging = {"name": name, "cases": ids, "pid": process.pid, "process": process, "log": log}
            event("staging_started", name=name, cases=ids, pid=process.pid, study=needing["case"]["study"])
    for task in list(pending):
        cpu_jobs = sum(t["variant"] == "main" for t in running)
        gpu_jobs = sum(t["variant"] == "gpu_anchor" for t in running)
        if (task["variant"] == "main" and cpu_jobs >= MAX_CPU) or (
            task["variant"] == "gpu_anchor" and gpu_jobs >= MAX_GPU
        ):
            continue
        if not cached(task["case"]):
            continue
        growth = sum(max(0, estimate(t) - rss(t["pid"])) for t in running)
        stage_growth = max(0, 16 - rss(staging["pid"])) if staging is not None else 0
        if available() - growth - stage_growth - estimate(task) < 12:
            continue
        if task["variant"] == "gpu_anchor":
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if probe.returncode or float(probe.stdout.splitlines()[0]) / 1024 < 2 + task["case"]["n_cells"] / 4000:
                continue
            lane = GPU_LANE
        else:
            occupied = [t["lane"] for t in running]
            lane = next((candidate for candidate in CPU_LANES if candidate not in occupied), None)
            if lane is None:
                continue
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(ROOT / "scratch/matplotlib"),
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            NUMBA_NUM_THREADS="4",
            JOBLIB_TEMP_FOLDER=str(ROOT / "scratch" / (task["name"] + ".joblib")),
        )
        Path(env["JOBLIB_TEMP_FOLDER"]).mkdir(exist_ok=True)
        command = [
            "taskset",
            "-c",
            ",".join(map(str, lane)),
            PYTHON,
            str(WORKER),
            "--case",
            task["case"]["id"],
            "--variant",
            task["variant"],
        ]
        log = open(ROOT / "results" / (task["name"] + ".log"), "w")  # noqa: SIM115 - task owns this handle and closes it when its worker finishes.
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        task.update(pid=process.pid, process=process, log=log, lane=lane)
        running.append(task)
        pending.remove(task)
        event("started", name=task["name"], pid=process.pid, cpu_affinity=lane, concurrent=[t["name"] for t in running])
    time.sleep(5)
event("complete")
subprocess.run([PYTHON, str(HERE / "navin_accuracy_report.py")], check=True)
archive = Path("/mnt/nas/cancer_research/navin_accuracy_2026-10-10/accuracy-evidence.tar.gz")
archive.parent.mkdir(exist_ok=True)
if not archive.exists():
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        for relative in ["results", "snapshots", "protocol.json", "dataset_manifest.json"]:
            tar.add(ROOT / relative, arcname=relative)
        for source in HERE.glob("navin_accuracy*"):
            tar.add(source, arcname="benchmarks/navin_review/" + source.name)
        for source in DOCS.glob("navin-accuracy*"):
            tar.add(source, arcname="docs/" + source.name)
    temporary.replace(archive)
else:
    # Recover the atomic rename -> audit-event gap without replacing evidence.
    with tarfile.open(archive, "r:gz") as tar:
        if not {"protocol.json", "dataset_manifest.json"} <= set(tar.getnames()):
            raise SystemExit("Existing archive lacks the expected evidence: " + str(archive))
h = hashlib.sha256()
with archive.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        h.update(block)
archive.with_name(archive.name + ".sha256").write_text(h.hexdigest() + "  " + archive.name + "\n")
event("archived", path=str(archive), sha256=h.hexdigest())
