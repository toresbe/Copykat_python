# Benchmarks

Tooling for measuring CopyKAT-py speed and accuracy. Written reports and
figures live in [`docs/`](../docs); the scripts that produce them live here.

There are two independent harnesses:

| Directory | Purpose | Inputs |
|---|---|---|
| `benchmarks/` (this level) | Quick per-sample runs on the README's 11 labelled tumour samples (plus an optional 170k-cell Xenium set): wall time per pipeline step, aneuploid-call accuracy, A/B comparison of two checkouts, CPU-path precision checks. | `COPYKAT_BENCH_SAMPLES`, `COPYKAT_BENCH_CACHE`, `COPYKAT_BENCH_XENIUM` (defaults under `~/cancer_research`, see `datasets.py`) |
| `benchmarks/navin_review/` | The 2026-10 Navin Lab review campaign: pinned source snapshots of upstream vs. the proposed PR stacks vs. the experimental/GPU fork, Xenium scaling (10k/30k/170k cells), CPU-accounted and serial timing, GPU memory, and a 125-sample accuracy study. | Environment variables listed below; paths default to the author's machine |

## Per-sample tools (`benchmarks/`)

| Script | What it does |
|---|---|
| `datasets.py` | Loads and caches the samples; `score()` compares calls with the `Malignant` labels in `metadata.csv`. |
| `run_sample.py SAMPLE OUTDIR` | Runs `copykat()` on one sample from whatever `copykat_py` is on `PYTHONPATH`; prints a `BENCH_JSON` line (per-step seconds, metrics) and saves the CNA matrix. |
| `sweep.sh CODE_ROOT OUT_ROOT` | Runs `run_sample.py` over all samples sequentially (waits for low machine load) and collects `results.jsonl`. |
| `summarize.py results.jsonl ...` | Prints a table of runtimes and metrics. |
| `compare.py RUN_A RUN_B` | Per-sample speedup, call agreement and CNA-matrix difference between two sweeps. |
| `precision.py OUT.jsonl` | Measures the CPU path's approximations (randomized PCA, subsampled silhouette, FP32 cumsum, Monte Carlo KS) against exact computations. |
| `diag_step4.py SAMPLE BACKEND` | Diagnoses normal-anchor selection (step 4). |

A/B example:

```bash
benchmarks/sweep.sh /path/to/checkout_a out_a
benchmarks/sweep.sh /path/to/checkout_b out_b
python benchmarks/compare.py out_a out_b
python benchmarks/summarize.py out_a/results.jsonl out_b/results.jsonl
```

## Navin review harness (`benchmarks/navin_review/`)

Scripts are prefixed `navin_review_*` (performance) or `navin_accuracy_*`
(accuracy study). Reports are written to `docs/` by the `*_report.py` scripts.

| Group | Scripts | Report in `docs/` |
|---|---|---|
| Setup | `navin_review_prepare.py` (rebuild snapshots from recorded Git objects), `navin_review_labels.py` | |
| Driver | `navin_review_benchmark.py` (one run/variant, with provenance, timeouts, resource sampling), `navin_review_recover.py` | |
| Scheduling | `navin_review_serial.py`, `navin_review_parallel.py`, `navin_review_prcomplete_queue.py`, `navin_review_sweep.sh`, `navin_review_followup.sh`, `navin_review_watch.py`, `navin_review_prcomplete_watch.py` | |
| Reports | `navin_review_report.py`, `navin_review_cpu_report.py`, `navin_review_serial_report.py`, `navin_review_figures.py`, `navin_review_projection.py` | [`navin-review-benchmarks.md`](../docs/navin-review-benchmarks.md), [`navin-review-cpu-benchmarks.md`](../docs/navin-review-cpu-benchmarks.md), [`navin-review-serial-benchmarks.md`](../docs/navin-review-serial-benchmarks.md) |
| Diagnostics | `navin_review_gpu_memory.py`, `navin_review_writer_edges.py` | |
| Accuracy study | `navin_accuracy_data.py`, `_stage.py`, `_worker.py`, `_queue.py`, `_check.py`, `_watch.py`, `_report.py` | [`navin-accuracy-results.md`](../docs/navin-accuracy-results.md), [`navin-accuracy-plan.md`](../docs/navin-accuracy-plan.md) |

Background and conclusions: [`navin-review-proposal.md`](../docs/navin-review-proposal.md),
[`navin-review-expanded-plan.md`](../docs/navin-review-expanded-plan.md),
[`fork-integration-audit.md`](../docs/fork-integration-audit.md) and
`navin-review-manifest.json`.

Configuration: `COPYKAT_REVIEW_ROOT` (working directory on fast storage),
`COPYKAT_REVIEW_PYTHON`, `COPYKAT_REVIEW_GPU_PYTHON`, `COPYKAT_BENCH_XENIUM`.
Some scheduler/report scripts and the `.sh` sweeps still hard-code
`/home/toresbe/...` interpreter and checkout paths; edit them for another
machine. Run them from the repository root.

Timings are evidence for the pinned snapshots they were measured on, not for
the current commit.
