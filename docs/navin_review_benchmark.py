"""Cursory local audit benchmarks; never changes Git refs.

Run with the existing CopyKAT Python environment. Source snapshots and all
calculation I/O live on the SSD. JSON retains censored runs and provenance.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.getenv("COPYKAT_REVIEW_ROOT", "/home/toresbe/cancer_research/navin_review_2026-10-09"))
REPO = Path(__file__).resolve().parents[1]
PYTHON = os.getenv("COPYKAT_REVIEW_PYTHON", "/home/toresbe/envs/copykat_py_main/bin/python")


def save(path, obj):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def usage_cpu():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_user_s": own.ru_utime,
        "self_system_s": own.ru_stime,
        "reaped_children_user_s": children.ru_utime,
        "reaped_children_system_s": children.ru_stime,
    }


def cgroup_cpu():
    if os.environ.get("NAVIN_BENCH_CGROUP") != "1":
        return None
    group = next(
        line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")
    )
    counters = dict(
        line.split() for line in (Path("/sys/fs/cgroup") / group.lstrip("/") / "cpu.stat").read_text().splitlines()
    )
    return int(counters["usage_usec"]) / 1e6


def sha_array(a):
    import numpy as np

    h = hashlib.sha256()
    for lo in range(0, len(a), 64):
        h.update(np.ascontiguousarray(a[lo : lo + 64]).tobytes())
    return h.hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_raw(sample):
    import numpy as np
    from scipy import sparse

    if sample.startswith("xenium"):
        import h5py

        with h5py.File(
            os.getenv("COPYKAT_BENCH_XENIUM", "/home/toresbe/cancer_research/xenium/cell_feature_matrix.h5")
        ) as f:
            m = f["matrix"]
            a = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
            keep = m["features/feature_type"][:].astype(str) == "Gene Expression"
            genes = m["features/name"][:].astype(str)[keep].astype(object)
            barcodes = m["barcodes"][:].astype(str).astype(object)
        a = a.tocsr()[keep].tocsc()
        n = int(sample[6:])
        permutation = np.random.default_rng(20261009).permutation(a.shape[1])
        cells = np.sort(permutation[:n])
        a, barcodes = a[:, cells], barcodes[cells]
        return {"matrix": a, "genes": genes, "barcodes": barcodes}, None, "hg20"
    if sample == "T989":
        p = ROOT / "scratch/T989_mm10.npz"
        z = np.load(p, allow_pickle=False)
        a = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return {"matrix": a, "genes": z["genes"].astype(object), "barcodes": z["barcodes"].astype(object)}, None, "mm10"
    sys.path.insert(1, str(REPO / ".claude/worktrees/gpu-acceleration-optimization-bfa6f0/benchmarks"))
    os.environ["COPYKAT_BENCH_CACHE"] = str(ROOT / "scratch/cache")
    from datasets import load_sample

    raw, truth = load_sample(sample)
    return raw, truth, "hg20"


def worker(args):
    import numpy as np

    source = ROOT / "snapshots" / args.variant
    sys.path.insert(0, str(source))
    ck = importlib.import_module("copykat_py.copykat")
    assert str(ck.__file__).startswith(str(source))
    raw, truth, genome = load_raw(args.sample)
    out = ROOT / "scratch" / args.name
    out.mkdir(exist_ok=True)
    os.chdir(out)
    result = {
        "variant": args.variant,
        "sample": args.sample,
        "input_shape": list(raw["matrix"].shape),
        "cores": args.cores,
        "plot": args.plot,
        "source": str(source),
        "steps": [],
    }
    original = ck._record_step
    result["source_copykat_sha256"] = sha_file(ck.__file__)

    def record(*a, **k):
        elapsed = original(*a, **k)
        result["steps"] = list(a[0]["steps"])
        save(ROOT / "results" / (args.name + ".progress.json"), result)
        return elapsed

    ck._record_step = record
    segmentation = ck.cna_mcmc
    result["breakpoint_attempts"] = []

    def segment(*a, **k):
        r = segmentation(*a, **k)
        result["breakpoint_attempts"].append([int(x) for x in r["breaks"]])
        return r

    ck.cna_mcmc = segment
    if args.mode == "capture":

        class Captured(Exception):
            pass

        def capture(y, **kwargs):
            np.save(ROOT / "scratch" / (args.sample + ".smoothing.npy"), y)
            result["smoothing_shape"] = list(y.shape)
            result["smoothing_dtype"] = str(y.dtype)
            result["smoothing_order"] = "F" if y.flags.f_contiguous else "C"
            raise Captured()

        ck.dlm_smooth = capture
    elif args.null_outputs:
        for suffix in ["CNA_raw_results_gene_by_cell.txt", "CNA_results.txt"]:
            os.symlink("/dev/null", out / ("audit_copykat_" + suffix))
    start = time.perf_counter()
    result["pipeline_cpu_start"] = usage_cpu()
    result["pipeline_cgroup_cpu_start_s"] = cgroup_cpu()
    result["pipeline_started_monotonic"] = start
    save(ROOT / "results" / (args.name + ".progress.json"), result)
    try:
        extra = {}
        if args.backend:
            extra.update(backend_name=args.backend, ks_method=args.ks_method)
        r = ck.copykat(raw, sam_name="audit", genome=genome, n_cores=args.cores, plot_genes=args.plot, **extra)
        result["status"] = "ok"
        result["pipeline_s"] = time.perf_counter() - start
        result["pipeline_cpu_end"] = usage_cpu()
        result["pipeline_cgroup_cpu_end_s"] = cgroup_cpu()
        if result["pipeline_cgroup_cpu_start_s"] is not None:
            result["pipeline_cpu_s"] = result["pipeline_cgroup_cpu_end_s"] - result["pipeline_cgroup_cpu_start_s"]
        result["pipeline_finished_monotonic"] = time.perf_counter()
        if args.gpu_memory and "torch" in sys.modules:
            torch = sys.modules["torch"]
            result["torch_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            result["torch_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        save(ROOT / "results" / (args.name + ".progress.json"), result)
        cna = r["CNAmat"]
        # Original dtype, all cells and bins; no float16 reduction.
        h = hashlib.sha256()
        for col in cna.columns[3:]:
            h.update(str(col).encode() + b"\0")
            h.update(np.ascontiguousarray(cna[col].to_numpy()).tobytes())
        result["cna_numeric_sha256"] = h.hexdigest()
        result["cna_shape"] = list(cna.shape)
        if truth is not None:
            np.savez_compressed(
                ROOT / "results" / (args.name + ".cna.npz"),
                values=cna.iloc[:, 3:].to_numpy(),
                cells=np.asarray(cna.columns[3:], dtype=str),
                annotation=cna.iloc[:, :3].astype(str).to_numpy(dtype=str),
            )
        pred = r["prediction"]
        pred.to_csv("predictions.tsv", sep="\t", index=False)
        result["prediction_sha256"] = sha_file("predictions.tsv")
        result["pred_counts"] = pred["copykat.pred"].value_counts().to_dict()
        import pickle

        with open("audit_copykat_clustering_results.pkl", "rb") as f:
            cl = pickle.load(f)
        z = np.asarray(cl["Z"])
        result["tree_structure_sha256"] = sha_array(z[:, [0, 1, 3]])
        result["tree_heights_sha256"] = sha_array(z[:, 2])
        np.save(ROOT / "results" / (args.name + ".linkage.npy"), z)
        result["labels_sha256"] = sha_array(np.asarray(cl["labels"]))
        result["file_sha256"] = {
            p.name: sha_file(p)
            for p in out.iterdir()
            if p.suffix in (".txt", ".png", ".seg", ".pkl") and not p.is_symlink()
        }
        result["output_bytes"] = sum(p.stat().st_size for p in out.iterdir() if not p.is_symlink())
        result["null_outputs"] = args.null_outputs
        if truth is not None:
            from datasets import score

            result["metrics"] = score(pred, truth)
    except Exception as e:
        result["pipeline_s"] = time.perf_counter() - start
        if args.mode == "capture" and type(e).__name__ == "Captured":
            result["status"] = "captured"
        else:
            result["status"] = "error"
            result["error"] = traceback.format_exc()
    result["process_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "benchmark_cohort": json.loads(os.getenv("NAVIN_BENCH_COHORT", "[]")),
        "scheduling_policy": os.getenv("NAVIN_BENCH_SCHEDULING", "legacy unpinned"),
        "packages": {
            name: getattr(sys.modules[name], "__version__", None)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "numba",
                "pyarrow",
                "fastcluster",
                "matplotlib",
                "sklearn",
                "torch",
                "cupy",
            ]
            if name in sys.modules
        },
    }
    save(ROOT / "results" / (args.name + ".worker.json"), result)
    shutil.rmtree(out)


def smoothing(args):
    import numpy as np

    path = ROOT / "snapshots" / args.variant / "copykat_py/smoothing.py"
    spec = importlib.util.spec_from_file_location("audit_smoothing", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    y = np.load(ROOT / "scratch" / (args.sample + ".smoothing.npy"), mmap_mode="r")
    start = time.perf_counter()
    out = m.dlm_smooth(y, n_cores=args.cores)
    elapsed = time.perf_counter() - start
    result = {
        "status": "ok",
        "variant": args.variant,
        "sample": args.sample,
        "stage": "dlm_smooth",
        "cores": args.cores,
        "shape": list(y.shape),
        "dtype": str(out.dtype),
        "cold_s": elapsed,
        "sha256": sha_array(out),
        "process_peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
    }
    del out
    warm = []
    for _ in range(args.reps):
        start = time.perf_counter()
        out = m.dlm_smooth(y, n_cores=args.cores)
        warm.append(time.perf_counter() - start)
        assert sha_array(out) == result["sha256"]
        del out
    result["warm_s"] = warm
    save(ROOT / "results" / (args.name + ".worker.json"), result)


def proc_rss(pid):
    # RSS over the process tree. Shared pages are counted for each process.
    try:
        stat = Path(f"/proc/{pid}/status").read_text()
        rss = next(int(x.split()[1]) * 1024 for x in stat.splitlines() if x.startswith("VmRSS:"))
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(proc_rss(int(c)) for c in children)
    except (OSError, StopIteration):
        return 0


def tree_metrics(pid, observed):
    """CPU ticks include all threads; c{u,s}time adds reaped children.

    Remember processes by PID/start tick to avoid losing exited workers or
    double-counting their CPU after a parent reaps them. Self time and
    aggregated child time are compared against explicitly observed subtree
    time, so a pool child's work is counted once.
    """
    hz = os.sysconf("SC_CLK_TCK")
    current = []
    pending = [pid]
    rss = 0
    while pending:
        p = pending.pop()
        try:
            raw = Path(f"/proc/{p}/stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            key = f"{p}:{fields[19]}"
            row = {
                "pid": p,
                "parent": int(fields[1]),
                "self_s": (int(fields[11]) + int(fields[12])) / hz,
                "children_s": (int(fields[13]) + int(fields[14])) / hz,
            }
            if key not in observed:
                row["parent_key"] = next((k for k, v in observed.items() if v["pid"] == row["parent"]), None)
            else:
                row["parent_key"] = observed[key]["parent_key"]
            observed[key] = row
            current.append(key)
            status = Path(f"/proc/{p}/status").read_text().splitlines()
            rss += next(int(x.split()[1]) * 1024 for x in status if x.startswith("VmRSS:"))
            pending.extend(map(int, Path(f"/proc/{p}/task/{p}/children").read_text().split()))
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    by_parent = {}
    for key, row in observed.items():
        by_parent.setdefault(row["parent_key"], []).append(key)

    def total(key):
        row = observed[key]
        return row["self_s"] + max(row["children_s"], sum(total(k) for k in by_parent.get(key, [])))

    roots = [key for key in observed if observed[key]["parent_key"] not in observed]
    return rss, sum(total(key) for key in roots)


def supervise(args):
    for suffix in [".json", ".worker.json", ".progress.json"]:
        existing = ROOT / "results" / (args.name + suffix)
        if existing.exists():
            raise SystemExit(f"Refusing to mix a new run with existing evidence: {existing}; choose a new name/root")
    env = dict(os.environ)
    env.update(MPLCONFIGDIR=str(ROOT / "scratch/matplotlib"), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    joblib = ROOT / "scratch" / (args.name + ".joblib")
    joblib.mkdir(exist_ok=True)
    env["JOBLIB_TEMP_FOLDER"] = str(joblib)
    env["PYTHONUNBUFFERED"] = "1"
    if args.cpu_list:
        selected = {int(x) for x in args.cpu_list.split(",")}
        if len(selected) != args.cores or not selected <= os.sched_getaffinity(0):
            raise SystemExit("CPU list must select exactly the requested number of available logical CPUs")
        os.sched_setaffinity(0, selected)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        env[key] = str(args.cores)
    runtime = (
        os.getenv("COPYKAT_REVIEW_GPU_PYTHON", "/home/toresbe/envs/copykat_py_gpu/bin/python")
        if args.backend
        else PYTHON
    )
    command = [
        runtime,
        str(Path(__file__).resolve()),
        args.mode,
        "--variant",
        args.variant,
        "--sample",
        args.sample,
        "--name",
        args.name,
        "--cores",
        str(args.cores),
    ]
    command.extend(["--reps", str(args.reps)])
    if args.plot:
        command.append("--plot")
    if args.null_outputs:
        command.append("--null-outputs")
    if args.backend:
        command.extend(["--backend", args.backend, "--ks-method", args.ks_method])
    if args.gpu_memory:
        command.append("--gpu-memory")
    unit = None
    if args.accounting_scope:
        unit = "copykat-navin-" + args.name.replace("_", "-") + ".scope"
        env["NAVIN_BENCH_CGROUP"] = "1"
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit=" + unit,
            "-p",
            "CPUAccounting=yes",
            "-p",
            "MemoryAccounting=yes",
            *command,
        ]
    logfile = ROOT / "results" / (args.name + ".log")
    started = time.perf_counter()
    peak = 0
    min_available = float("inf")
    reason = None
    observed = {}
    pipeline_cpu_start = None
    pipeline_cpu_end = None
    group_path = None
    group_cpu = None
    group_memory_peak = 0
    last_live = 0
    live_path = ROOT / "results" / (args.name + ".live.json")
    gpu_monitor = None
    gpu_peak = 0
    gpu_samples = 0
    if args.gpu_memory:
        from navin_review_gpu_memory import Monitor

        gpu_monitor = Monitor()
    with open(logfile, "w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            rss, cpu = tree_metrics(process.pid, observed)
            if gpu_monitor is not None:
                gpu_peak = max(gpu_peak, gpu_monitor.sample(process.pid))
                gpu_samples += 1
            if unit and group_path is None:
                probe = subprocess.run(
                    ["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    group_path = Path("/sys/fs/cgroup") / probe.stdout.strip().lstrip("/")
            if group_path is not None:
                try:
                    counters = dict(line.split() for line in (group_path / "cpu.stat").read_text().splitlines())
                    group_cpu = int(counters["usage_usec"]) / 1e6
                    group_memory_peak = max(group_memory_peak, int((group_path / "memory.peak").read_text()))
                except OSError:
                    pass
            peak = max(peak, rss)
            progress_path = ROOT / "results" / (args.name + ".progress.json")
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            if pipeline_cpu_start is None and "pipeline_started_monotonic" in progress:
                pipeline_cpu_start = cpu
            if pipeline_cpu_end is None and "pipeline_finished_monotonic" in progress:
                pipeline_cpu_end = cpu
            memory = Path("/proc/meminfo").read_text().splitlines()
            available = next(int(x.split()[1]) * 1024 for x in memory if x.startswith("MemAvailable:"))
            min_available = min(min_available, available)
            if available < 10e9:
                reason = "memory_guard"
            elif shutil.disk_usage(ROOT).free < 2e9:
                reason = "ssd_space_guard"
            elif args.timeout > 0 and time.perf_counter() - started > args.timeout:
                reason = "timeout"
            if time.perf_counter() - last_live > 5:
                last_live = time.perf_counter()
                save(
                    live_path,
                    {
                        "name": args.name,
                        "variant": args.variant,
                        "pid": process.pid,
                        "wall_elapsed_s": time.perf_counter() - started,
                        "tree_cpu_s": cpu,
                        "rss_gb": rss / 1e9,
                        "cgroup_cpu_s": group_cpu,
                        "last_step": progress.get("steps", [{}])[-1].get("step")
                        if progress.get("steps")
                        else "loading",
                        "status": "running",
                    },
                )
            if reason:
                os.killpg(process.pid, 9)
                process.wait()
                break
            time.sleep(0.25)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, 9)  # clean up any surviving pool workers
    measured = ROOT / "results" / (args.name + ".worker.json")
    result = json.loads(measured.read_text()) if measured.exists() else {"status": reason or "crashed"}
    result.update(
        name=args.name,
        elapsed_process_s=time.perf_counter() - started,
        tree_peak_rss_gb=peak / 1e9,
        min_mem_available_gb=min_available / 1e9,
        exit_code=process.returncode,
        command=command,
        timeout_s=args.timeout,
    )
    if gpu_monitor is not None:
        result.update(
            gpu_peak_vram_gb=gpu_peak / 1e9,
            gpu_memory_samples=gpu_samples,
            gpu_memory_method=(
                "NVML compute-process usedGpuMemory for worker descendants; 0.25s polling; "
                "excludes unrelated processes and display VRAM; sampled peaks may miss transients"
            ),
        )
    result["process_tree_cpu_s_sampled"] = tree_metrics(process.pid, observed)[1]
    result["pipeline_tree_cpu_s_sampled"] = (
        pipeline_cpu_end - pipeline_cpu_start
        if pipeline_cpu_end is not None and pipeline_cpu_start is not None
        else None
    )
    result["cpu_measurement"] = (
        "Linux process stat user+system ticks, all threads; observed and reaped "
        "descendants counted once; 0.25s boundary sampling"
    )
    result["accounting_scope"] = unit
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    result["benchmark_cohort"] = json.loads(env.get("NAVIN_BENCH_COHORT", "[]"))
    result["scheduling_policy"] = env.get("NAVIN_BENCH_SCHEDULING", "legacy unpinned")
    result["process_cgroup_cpu_s_last_sample"] = group_cpu
    result["cgroup_memory_peak_gb"] = group_memory_peak / 1e9 if unit else None
    if unit:
        result["cpu_measurement"] = (
            "Dedicated systemd cgroup cpu.stat usage_usec; exact all-thread/all-descendant "
            "pipeline boundary readings; process end sampled"
        )
    if not measured.exists():
        progress = ROOT / "results" / (args.name + ".progress.json")
        if progress.exists():
            result.update(json.loads(progress.read_text()))
            result["status"] = reason or "crashed"
            if "pipeline_started_monotonic" in result:
                result["pipeline_elapsed_at_stop_s"] = time.perf_counter() - result["pipeline_started_monotonic"]
        shutil.rmtree(ROOT / "scratch" / args.name, ignore_errors=True)
    save(ROOT / "results" / (args.name + ".json"), result)
    save(
        live_path,
        {
            "name": args.name,
            "status": result["status"],
            "wall_elapsed_s": result["elapsed_process_s"],
            "tree_cpu_s": result["process_tree_cpu_s_sampled"],
        },
    )
    shutil.rmtree(joblib, ignore_errors=True)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in ["name", "status", "pipeline_s", "cold_s", "tree_peak_rss_gb"]}
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "capture", "smoothing"])
    p.add_argument("--variant", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--null-outputs", action="store_true")
    p.add_argument("--backend", choices=["cpu", "gpu", "gpu-compat"])
    p.add_argument("--ks-method", choices=["mc", "exact"], default="mc")
    p.add_argument("--reps", type=int, default=0)
    p.add_argument("--accounting-scope", action="store_true")
    p.add_argument("--gpu-memory", action="store_true", help="Sample benchmark-process GPU VRAM using NVML")
    p.add_argument("--cpu-list", help="Comma-separated logical CPUs; inherited by every worker")
    args = p.parse_args()
    if os.environ.get("NAVIN_BENCH_CHILD") == "1":
        (smoothing if args.mode == "smoothing" else worker)(args)
    else:
        os.environ["NAVIN_BENCH_CHILD"] = "1"
        supervise(args)


if __name__ == "__main__":
    main()
