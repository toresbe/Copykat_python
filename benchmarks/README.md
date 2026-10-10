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

Environment variables: `COPYKAT_REVIEW_ROOT` (working directory on fast
storage), `COPYKAT_REVIEW_PYTHON`, `COPYKAT_REVIEW_GPU_PYTHON`,
`COPYKAT_BENCH_SAMPLES`, `COPYKAT_BENCH_CACHE`, `COPYKAT_BENCH_XENIUM`.

Several scheduler/report scripts and the `.sh` sweeps still hard-code
`/home/toresbe/...` interpreter, checkout and data paths (including the
`benchmark_2026-10-09` and `accuracy_2026-10-10` working directories); edit
them for another machine.
