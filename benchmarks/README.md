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

## Data

Inputs are not in the repository. `python benchmarks/fetch_data.py check` shows
what is present under `COPYKAT_BENCH_DATA`. Which benchmarks need what:

| Benchmark | Input | How to get it |
|---|---|---|
| Xenium scaling (10k / 30k / 170k cells) | `xenium/cell_feature_matrix.h5` from the [Atera WTA FFPE human breast cancer](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer) dataset | Copy the download link for the cell-feature matrix (or the outs bundle) from the dataset page, then `python benchmarks/fetch_data.py xenium --url URL` (or `--file` for something already downloaded) |
| README samples (11 labelled tumours) | `copykat_readme_data/samples/<Study>_<sample>/` in 10x MTX layout plus `metadata.csv` | Download the 11 studies from the [Cancer Cell Atlas (3CA)](https://www.weizmann.ac.il/sites/3CA/) by hand (each cancer-type page has per-study "Download Data" and "Download Meta-data" links), unpack them under one directory, then `python benchmarks/fetch_data.py 3ca THAT_DIR`. The sample list is in `fetch_data.py` and the README table. |
| Accuracy study (125 samples) | The README samples above, more 3CA studies (`3ca/`), ScPCA libraries (`scpca_3ca/`), Tabula Sapiens h5ad files (`tabula_sapiens/`), `anchor_study/split.json` | Not scripted. `accuracy_data.py` selects whatever it finds under `COPYKAT_BENCH_DATA`, so a smaller selection runs fine, but the exact 125-sample set and `split.json` (a private development/holdout split) cannot be reproduced from this repository. |
| Mouse T989 comparison | `scratch/T989_mm10.npz` under `COPYKAT_BENCH_ROOT` | Not scripted; see `docs/mm10-order-validation.md`. |

Caveats on what was verified: the sandbox this was written in could not reach
10x Genomics or Weizmann hosts, so the download commands could not be run
against the real sites. `fetch_data.py 3ca` assumes the 3CA study layout that
`accuracy_data.py` already reads (`*UMI*.mtx` as genes x cells, `Cells*.csv`
with `cell_name`, `cell_type` and `sample`/`patient` columns, `Genes*.txt`), and
was tested only on synthetic data in that layout. The Atera dataset's direct
file URL was not confirmed, hence `--url`.

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
