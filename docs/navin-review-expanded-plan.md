# Expanded scaling sweep and crash recovery

The 2026-10-09 client crash left the original full CPU calculation and
scheduler alive. The replacement coordinator adopted its supervisor rather
than restarting the calculation. Its baseline stage completed in 56m17.5s;
the later stages remain uncapped. All completed original measurements remain
in place, including their original CPU placement.

## Samples and priority

Use nested, deterministic subsets of the same Xenium input, seed 20261009:

| Region | Input cells |
|---|---|
| Five new lower-size points | 2,000; 5,000; 15,000; 20,000; 25,000 |
| Original anchors | 10,000; 30,000; 170,057 |
| Three new upper-size points | 40,000; 80,000; 120,000 |

All eleven sizes compare upstream main, the five-candidate bundle, the
larger CPU stack and the full-policy GPU implementation: 44 primary jobs.
Completed jobs are reused, never overwritten. Four additional isolated 10k
checks diagnose differences in the original concurrent measurements.
The SSD `expanded_sweep_manifest.json` pins the complete queue and policy.

Smaller paired comparisons run first. Larger CPU-stack/GPU points precede
the more costly upstream/bundle points. The 40k point targets the neighborhood
of the CPU condensed-distance memory cutoff; retained cell counts determine
the actual switch. The report lists baseline, adjustment and prediction
engines instead of presenting all sizes as one unchanged calculation.
Small inputs also expose the full-matrix to PCA switch.

## Contention and resource policy

Each new job requests eight threads and is pinned to eight physical cores
on the Ryzen 5950X: logical CPUs 0–7 or 8–15. Their SMT siblings 16–31 are
not assigned to another pinned benchmark. There are at most two concurrent
jobs, and only one new job while the original unpinned full calculation is
running. That calculation's affinity is unchanged; its potential overlap
is explicitly recorded. New pinned jobs avoid sharing CPU cores with each
other, but still share memory bandwidth and boost power. Other host tasks,
including the unpinned calculation, can also affect caches.

Admission reserves predicted future memory growth plus 11 GB available
RAM. A per-run guard stops at less than 10 GB available RAM or 2 GB SSD
space; there is no clock cap. An oversized job can attempt to run alone,
so an estimate cannot silently prevent measurement forever. A genuine
resource stop is retained as a resource outcome, not a completed timing.

CPU core-seconds include user and system time from every thread and
descendant, using dedicated cgroup readings at pipeline boundaries. Loading
and audit hashing are excluded. Elapsed time remains available because CPU
time excludes GPU work and does not remove cache or bandwidth effects.
These remain exploratory single-process-start measurements; filesystem,
input and compilation caches are reused, not flushed.

Once the unpinned full calculation finishes, the next available idle slot
is reserved for four sequential isolated 10k checks on CPUs 0–7. No other
review job runs during each check. Comparisons with earlier 10k readings
mix affinity, caches and contention; they are diagnostics, not a causal
estimate of contention alone. Any surprising differences warrant focused
repeats rather than claiming greater precision from a denser curve.

All new runs consistently disable heatmap rendering, include large-table
formatting to `/dev/null`, and retain original-dtype CNA, prediction and
linkage hashes. Existing scientific dendrogram comparisons remain available.

## Runtime and recovery

The coordinator runs as user service `copykat-navin-expanded-scheduler.service`.
The low-priority report updater runs as `copykat-navin-expanded-report.service`.
They are independent of the ChatGPT client and restart on failure. A runtime
coordinator drop-in sets `KillMode=process`, preserving benchmark supervisors
when the coordinator restarts. The scheduler holds an exclusive lock and
adopts matching recorded supervisor PIDs; it verifies command lines before
trusting a PID. Interrupted evidence is renamed rather than overwritten.
These are user-session services; they are not a reboot-resume guarantee.

Read status with:

```sh
systemctl --user status copykat-navin-expanded-scheduler copykat-navin-expanded-report
```

To recreate either service after it is absent (these commands do not replace
an already existing unit):

```sh
systemd-run --user --unit=copykat-navin-expanded-scheduler --collect \
  --property=Restart=on-failure --property=RestartSec=30 --property=KillMode=process \
  --working-directory=/home/toresbe/Copykat_python \
  /home/toresbe/miniforge3/bin/python -u /home/toresbe/Copykat_python/benchmarks/navin_review/navin_review_parallel.py
systemd-run --user --unit=copykat-navin-expanded-report --collect \
  --property=Restart=on-failure --property=RestartSec=30 --property=Nice=19 \
  --working-directory=/home/toresbe/Copykat_python \
  /home/toresbe/miniforge3/bin/python -u /home/toresbe/Copykat_python/benchmarks/navin_review/navin_review_watch.py
```

The live report is `docs/navin-review-cpu-benchmarks.md`, refreshed every
minute. Detailed progress and overlap events are under
`/home/toresbe/cancer_research/navin_review_2026-10-09/results`.
Calculation inputs, output hashing and joblib memory maps stay on the SSD.
On completion, the coordinator archives snapshots, evidence, expanded
manifest and review scripts to a separate checksum-protected NAS archive.
No Git refs or PRs are changed by this workflow.

On the later resume, the original tool-session supervisor disappeared while
the separately scoped 170k worker (PID 600456) continued. A dedicated
`copykat-navin-recover170.service` observer restored memory/SSD guards and
result collection without restarting that worker or changing its cgroup.
The exact pipeline CPU boundary readings remain in the worker. Original
supervisor peak-RSS history is unavailable; the whole-run cgroup memory peak
is preserved and must be labeled separately. Early queue `crashed` and
`interrupted` events described the lost supervisor, not the surviving
calculation; subsequent recovery/adoption events record the correction.
The scheduler now refuses to rename evidence or restart work when its
recorded supervisor is gone but the recorded scoped worker is still alive.

## Serial repeats after this sweep

The later request adds two fresh, sequential repeat rounds at 2k, 5k, 10k,
15k, 20k, 25k, 30k and 40k: 64 jobs across the four implementations.
`copykat-navin-serial-repeats.service` waits until all 48 original attempts
have recorded outcomes and the original NAS archive is complete, then
acquires the same exclusive scheduler lock. No repeat begins early.
Every repeat uses CPUs 0–7, eight requested threads, unchanged source
snapshots and nested inputs, no heatmap, table formatting to `/dev/null`,
and the same cgroup CPU accounting and memory/SSD guards. The coordinator
also checks for review workers outside the original queue before launching
a repeat. This prevents overlap between review jobs; unrelated host work
remains a possible source of timing variation.

Round one visits sizes ascending and implementations main/bundle/CPU-stack/
GPU; round two reverses both orders to reduce systematic ordering effects.
Each attempt gets a fresh worker process; caches are reused, not flushed.
The primary and isolated-anchor timings are kept outside serial averages.
`serial_sweep_manifest.json` and `results/serial8_schedule.json` pin the
queue, admission times, repeats and outcomes. Completed repeat evidence is
reused on coordinator recovery, rather than overwritten.

`docs/navin-review-serial-benchmarks.md` reports elapsed and CPU means,
sample standard deviations, observed ranges, successful counts and repeat
hash consistency. The serial figure shows only completed pairs and uses
observed minimum–maximum error bars, not confidence intervals. A failed
attempt is retained and excluded from runtime averages; it is never
silently treated as a completed measurement. Two repeats provide a useful
descriptive check, not a precise estimate of timing uncertainty.
The low-priority report service continues through this second phase.
A distinct `navin-review-serial8-evidence.tar.gz` archive and SHA256 sidecar
are written on the NAS after the serial phase; earlier archives remain
unchanged. Serial coordinators preserve supervisors on restart with
`KillMode=process` and refuse to restart a still-alive scoped worker.

Serial service status:

```sh
systemctl --user status copykat-navin-serial-repeats
```
