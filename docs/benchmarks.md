# Performance benchmarks

Wall-clock time and memory for the CopyKAT-Python pipeline on a 170k-cell Xenium
breast cancer dataset, subsampled to 2k–170k cells. This is the entry point to
the benchmark reports; for how to run the harness yourself see
[`benchmarks/README.md`](../benchmarks/README.md).

![Measured scaling and conditional 170k estimates](benchmark-assets/cpu-accounted-scaling.png)

X markers and dashed lines are extrapolated 170k points, not completed runs.
Error bars show model sensitivity, not confidence intervals. Hollow markers
are runs that were not pinned to physical cores.

## What is compared

| Implementation | Meaning |
|---|---|
| Upstream main | The original implementation at commit `ea1a15c` |
| Currently proposed changes (6 PRs) | Upstream main plus the six performance PRs `#4`–`#9` proposed upstream |
| CPU experimental (no verification) | A broader CPU integration (Arrow output, memory/storage refactors, Ward-engine and repeated-bin changes) |
| GPU experimental (no verification) | The optional CUDA backend with default Monte Carlo KS |

"No verification" means outputs were not checked to be identical to upstream;
the experimental stacks change numerical details. See
[`benchmark-scaling.md`](benchmark-scaling.md) for what each stack preserves.

## Headline numbers

Single cold runs, eight requested cores, elapsed seconds (from
[`benchmark-cpu.md`](benchmark-cpu.md)):

| Input cells | Upstream main | Proposed (6 PRs) | CPU experimental | GPU experimental |
|---:|---:|---:|---:|---:|
| 10,000 | 134 | 86 | 24 | 8 |
| 40,000 | 1,310 | 1,127 | 515 | 22 |
| 80,000 | 3,337 | 2,081 | 2,088 | 41 |
| 170,057 | not completed (memory guard at 120k) | not completed | 9,909 | 97 |

Upstream main and the six-PR stack hit the memory guard (about 97 GB of
process-tree RSS) at 120,000 cells, so their 170k points in the figure are model
estimates, not measurements. Peak process-tree RSS at 80k cells was about 74 GB
for both, versus 12 GB for the CPU experimental stack and 12 GB for the GPU
stack. These are one-off measurements on one machine, not a stable estimate of
isolated per-change effects.

## Reports

| Report | Contents |
|---|---|
| [CPU-accounted scaling](benchmark-cpu.md) | Full 2k–170k sweep with CPU time, peak RSS, GPU memory, projections to 170k |
| [Serial repeats](benchmark-serial.md) | Two repeats per size from 2k to 40k, run one at a time, with output-consistency checks |
| [10k / 30k / 170k scaling and output checks](benchmark-scaling.md) | Earlier cursory sweep, kidney and mouse output comparisons, which PRs look promising |
| [Six-PR follow-up](benchmark-six-pr-followup.md) | Corrected six-PR performance and GPU-memory follow-up |
| [Accuracy study](accuracy-results.md) | Main vs. GPU + marker anchoring on 125 labelled samples (accuracy only, no timings) |

Background and method: [proposal](benchmark-proposal.md),
[expanded sweep plan](benchmark-expanded-plan.md),
[fork integration audit](fork-integration-audit.md) and the machine-readable
[manifest](benchmark-manifest.json).

## Reproducing

```bash
benchmarks/run.sh check                 # which inputs are present
benchmarks/run.sh xenium10000           # benchmark the committed HEAD at 10k cells
```

The published numbers come from the scheduling scripts in
`benchmarks/campaign/` and the report generators in `benchmarks/reports/`; they
were measured on one specific machine and have not been rerun since the
scripts were reorganised.
