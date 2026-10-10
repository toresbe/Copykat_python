# Benchmarks

One harness for measuring CopyKAT-py speed and accuracy. Written reports and
figures live in [`docs/`](../docs) (`benchmark-*` and `accuracy-*`); the scripts
that produce them live here. Run everything from the repository root.

The harness compares pinned source snapshots (upstream main, proposed PR
stacks, the experimental CPU/GPU fork) on:

- **Scaling**: Xenium subsets of 10k / 30k / 170k cells, plus the README's labelled samples.
- **CPU-accounted and serial timing**, repeated rounds from 2k to 40k cells.
- **GPU memory** (NVML) and output-writer edge cases.
- **Accuracy**: 125 samples from 60 studies, scored against author malignant labels.

Timings are evidence for the pinned snapshots they were measured on, not for
the current commit.

## Scripts

| Group | Scripts | Report in `docs/` |
|---|---|---|
| Data | `datasets.py` (sample loaders and scoring), `prepare.py` (rebuild snapshots from recorded Git objects), `labels.py` | |
| Driver | `benchmark.py` (one run/variant, with provenance, timeouts, resource sampling), `recover.py` | |
| Scheduling | `serial.py`, `parallel.py`, `prcomplete_queue.py`, `sweep.sh`, `followup.sh`, `watch.py`, `prcomplete_watch.py` | |
| Reports | `scaling_report.py`, `cpu_report.py`, `serial_report.py`, `figures.py`, `projection.py` | [`benchmark-scaling.md`](../docs/benchmark-scaling.md), [`benchmark-cpu.md`](../docs/benchmark-cpu.md), [`benchmark-serial.md`](../docs/benchmark-serial.md) |
| Diagnostics | `gpu_memory.py`, `writer_edges.py`, `precision.py` (CPU-path approximations vs. exact computations), `diag_step4.py` (normal-anchor selection) | |
| Accuracy study | `accuracy_data.py`, `accuracy_stage.py`, `accuracy_worker.py`, `accuracy_queue.py`, `accuracy_check.py`, `accuracy_watch.py`, `accuracy_report.py` | [`accuracy-results.md`](../docs/accuracy-results.md), [`accuracy-plan.md`](../docs/accuracy-plan.md) |

Background: [`benchmark-proposal.md`](../docs/benchmark-proposal.md),
[`benchmark-expanded-plan.md`](../docs/benchmark-expanded-plan.md),
[`fork-integration-audit.md`](../docs/fork-integration-audit.md) and
`benchmark-manifest.json`.

## Configuration

Nothing is hard-coded to a machine: every location defaults to a path under
`~/copykat_bench` (see `bench_config.py`) and the current Python interpreter.
Override with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `COPYKAT_BENCH_HOME` | Base directory for the defaults below | `~/copykat_bench` |
| `COPYKAT_BENCH_ROOT` | Performance working directory (snapshots, results, scratch) | `$HOME_/benchmark` |
| `COPYKAT_BENCH_ACCURACY_ROOT` | Accuracy-study working directory | `$HOME_/accuracy` |
| `COPYKAT_BENCH_DATA` | Input data (README samples, Xenium, 3CA, ScPCA, Tabula Sapiens) | `$HOME_/data` |
| `COPYKAT_BENCH_SAMPLES` / `COPYKAT_BENCH_XENIUM` | README samples directory / Xenium `cell_feature_matrix.h5` | under `COPYKAT_BENCH_DATA` |
| `COPYKAT_BENCH_PYTHON` | Interpreter for CPU runs, schedulers and reports | the current interpreter |
| `COPYKAT_BENCH_GPU_PYTHON` | Interpreter for GPU runs | `COPYKAT_BENCH_PYTHON` |
| `COPYKAT_BENCH_ARCHIVE` | Where evidence tarballs are written | `$HOME_/archive` |
| `COPYKAT_BENCH_CACHE` | Parsed-matrix cache | `$HOME_/cache` |

(`$HOME_` is `COPYKAT_BENCH_HOME`.) The shell sweeps run from the repository
root regardless of where you call them. Reports are written to `docs/`.
The reports in `docs/` refer to these locations by variable name; the original
measurements were taken on one specific machine with specific datasets.
