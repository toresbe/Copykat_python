# CopyKAT-Py optimization history audit

This preserves the historical repository inventory from before proposal and
semantic branches were built. For the executed branch construction, current
dependencies, integration decisions, and updated measurements, see
[`fork-integration-audit.md`](fork-integration-audit.md) and
[`benchmark-manifest.json`](benchmark-manifest.json).

Inspected 2026-10-09 using local commit objects, branch refs, diffs, and
stable patch IDs. No remote fetch, history rewrite, or branch creation was
performed. Benchmark results below are evidence recorded by the original
work, not measurements repeated for this audit.

## Finding

There is a recoverable semantic history, but not a consistently atomic Git
history. `hardcore-optimization` has nine commits after `main`: six CPU
optimization commits followed by three GPU-era commits. The CPU work has
largely been extracted into existing `perf/*` branches. The GPU work still
bundles infrastructure, kernels, algorithm changes, CPU shortcuts, plotting,
CLI integration, and evaluation.

Accuracy work continues on `feature/anchor-allele-orientation`, five commits
beyond `hardcore-optimization`. It is absent from the hardcore branch tip.

## 1. Semantic changes

### CPU performance foundation

| Change | Original commit(s) | Semantic boundary |
|---|---|---|
| Shared DLM gains | `ffcfa3e`, `49b547c` | Compute data-independent Kalman/RTS gains once; apply them in Numba threads; disable disk cache. |
| Shared-memory task execution | `ffcfa3e` | Replace process pools for bin conversion and cluster GMM fits with threads; precompute consensus vectors. |
| Skip discarded fallback clustering | `ffcfa3e` | Add `baseline_gmm(cluster=False)` for the caller that only uses the baseline and normal cells. |
| Faster gene-level TSV output | `ffcfa3e`, `49b547c` | Use Arrow for full-precision, unquoted gene output; add Arrow to the environment. Text representation can differ from pandas. |
| Reduce host memory | `6a43d8c` | Keep input sparse through filtering/annotation, transform in place, release intermediates, preallocate bins, adjust baseline in place, stream output. |
| Distance-matrix Ward engine | `af463c0`, `8b15f85` | Use condensed distances plus fastcluster when feasible, with a low-memory fallback; replace a fixed 20k-cell cutoff with a configurable memory budget. |
| Collapse repeated features | `8b15f85` | Replace adjacent identical bin features with one weighted feature, preserving Euclidean distances mathematically; restrict use where PCA equivalence holds. |
| Thread Ward distances | `8b15f85` | Fill the condensed matrix using threaded row-block `cdist`. |
| Known-normal membership | `8b15f85` | Replace repeated list searches with set membership. |
| Parallel TSV formatting | `8b15f85` | Format bounded chunks in threads, write in order, release Arrow buffers. |
| Render colormap at image resolution | `d23282b` | Set `interpolation_stage="data"` on heatmaps; require matplotlib >=3.5. |

These are predominantly performance changes intended to preserve calls and
numeric results. They should not be described collectively as a guarantee
of bit identity: the original Arrow gene writer changes some number
spellings, and Ward merge heights can differ by rounding.

### GPU-era work on `hardcore-optimization`

| Change | Original commit(s) | Files / boundary |
|---|---|---|
| Backend selection and integration | `a0eaacf`, `0460d50`, `14aae8d` | `backend.py`; pipeline dispatch; Python/AnnData parameters; CLI flags; optional dependencies and README. Modes: `cpu`, `gpu-compat`, `gpu`. |
| Exact GPU Ward engine | `a0eaacf` | `gpu/ward.py`, clustering helpers in `gpu/ops.py`: reciprocal nearest-neighbour merges, FP32 candidate bounds, FP64 verification and full-scan fallback. |
| Ward kernel and memory refinement | `0460d50` | `gpu/nn_triton.py`, `gpu/ward.py`: fused reductions, candidate certification and tie handling, in-place centring/compaction, streamed upload. |
| GPU numeric pipeline | `a0eaacf`, `0460d50` | `gpu/kernels.py`, `gpu/gmm.py`, `gpu/ops.py`; hooks in baseline, bin conversion, segmentation and pipeline: sparse preparation, Freeman–Tukey/DLM, batched GMM, medians, binning, baseline adjustment, chunked memory management. |
| Remove CPU algorithm approximations | `a0eaacf` | `gpu` uses full-feature Ward instead of the CPU PCA projection, all-cell silhouette, FP64 segment sums, and full Ward heatmap ordering. `gpu-compat` retains the CPU approximation policy, but is not bit compatible. |
| Deterministic posterior KS segmentation | `a0eaacf`, CLI in `0460d50` | `segmentation.py`: exact distance between posterior Gamma CDFs instead of KS on Monte Carlo draws. Independently selected through `ks_method`; usable with CPU. |
| Format repeated output rows once | `a0eaacf` | `_row_run_starts`, `_csv_lines`, `_write_csv_repeated_rows` and the writer hook in `copykat.py`. CPU-reusable shortcut, independent of CUDA. |
| Reuse final Ward tree for heatmap | `a0eaacf` | Pass step-8 linkage to plotting when the metric and absence of PCA make reuse valid. CPU-reusable shortcut. |
| Large heatmap rendering | `a0eaacf` | `plotting.py`: bounded sampled raster, iterative dendrogram drawing, highest-link limit. Separate rendering behavior from the full-tree ordering policy. |
| Benchmark and precision evidence | `a0eaacf`, `14aae8d` | Dataset loaders, run/sweep/comparison tools, diagnostics, precision comparisons, Ward benchmark and `benchmarks/RESULTS.md`. |

The numeric port and algorithm policy are conceptually separate even though
they share functions and dispatch hooks. Extracting them into independent
commits requires editing hunks and interfaces, not simply cherry-picking
`a0eaacf`. `0460d50` also mixes Ward refinement, baseline memory management,
and public CLI integration.

The Ward projection used to search candidates is not the same change as
clustering a PCA-reduced matrix: the former verifies candidates against
full-width coordinates; the latter changes the clustering metric space.
Similarly, an exact Ward tree does not imply bit-identical end-to-end
results or better biological classification.

### Accuracy work beyond the hardcore tip

| Change | Commit(s) | Boundary |
|---|---|---|
| Marker-guided normal anchor | `d3f3831` | `anchor="markers"`: select a reference using immune/endothelial markers, with fallbacks. |
| Arm-level correlation calls | `d3f3831` | `final_call="arm_correlation"`: a separately selectable final classifier; bundled with the marker anchor. |
| Allele index/count infrastructure | `de8a6e3` | SNP k-mer index, GPU REF/ALT UMI counting, FASTQ helper and count loaders. |
| Phasing and imbalance model | `de8a6e3` | Eagle2 integration and phased-BAF HMM. |
| Allele orientation and interface | `de8a6e3` | Flip final call labels based on group imbalance evidence; separate CLI plus pipeline integration, dependency and usage documentation. |
| Research records and updated evidence | `2051057`, `8b10f1f`, `5401668` | Anchor/allele design and results logs, WGS-profile evaluation, revised accuracy report and mode table. |

This is an accuracy-method extension, not merely higher numerical precision.
The research logs contain earlier unsuccessful approaches as well as the
final implementation; not every logged research variant is shipped code.

## 2. Mapping to actual Git branches

### Standalone CPU extractions

Each branch below is one commit directly on `main` (`ea1a15c`), rather than
an ancestor of `hardcore-optimization`.

| Branch | Tip | Scope / qualification |
|---|---|---|
| `perf/thread-pools` | `24475ca` | Threaded bin conversion and cluster GMM execution. |
| `perf/skip-unused-gmm-clustering` | `1cf6437` | Skip unused fallback clustering. |
| `perf/dlm-shared-gains` | `516d719` | Shared smoother gains; incorporates the later no-disk-cache choice. |
| `perf/pyarrow-gene-output` | `92d7e51` | Reworked Arrow writer: requires Arrow >=22 and Python >=3.10, removes pandas fallback and standardizes unquoted output. This is broader than the original extraction. |
| `perf/known-normal-set` | `bb4abc5` | Set membership. Same stable patch ID as `4bda378` in the CPU stack. |
| `perf/heatmap-interpolation-stage` | `2215462` | Same stable patch ID as original `d23282b` and stacked `ff63f19`. |

### Cumulative CPU stack

| Branch | Tip | Commits above main | Increment over preceding row |
|---|---|---:|---|
| `perf/smoothing-and-parallel-overhead` | `49b547c` | 2 | Original bundled `ffcfa3e` plus cache/environment follow-up. |
| `perf/memory-footprint` | `6a43d8c` | 3 | Host-memory reduction. |
| `perf/clustering` | `350c1c0` | 4 | Revised distance-matrix Ward path and corrected peak-memory budget. |
| `perf/collapse-repeated-bins` | `5768793` | 5 | Weighted collapse of repeated features. |
| `perf/threaded-ward-distances` | `44c15db` | 6 | Threaded distance matrix. |
| `perf/parallel-output-writer` | `769aed3` | 8 | Includes `4bda378` set membership, then parallel TSV formatting. |
| `perf/exact-shortcuts` | `ff63f19` | 9 | Heatmap interpolation/dependency change. |

These branch names identify successive endpoints. A diff from `main` to
`perf/parallel-output-writer`, for example, contains eight commits, not
just output formatting. The first two rows are literal ancestors of the
hardcore branch; subsequent rows form a separate lineage.

The full tree diff `d23282b..ff63f19` changes only `baseline.py` and
`SINGULARITY_GUIDE.md`: the split stack counts the condensed distance matrix
**and fastcluster's working copy**. Under its 8 GB budget the threshold is
about 31.6k cells, versus about 44.7k in the original lineage. Thus the
CPU split is substantially complete, but the endpoints are not identical.

### GPU and accuracy branches

| Branch | Tip | Actual scope |
|---|---|---|
| `hardcore-optimization` | `14aae8d` | Original CPU foundation plus `a0eaacf` GPU bundle, `0460d50` refinement/CLI bundle, `14aae8d` documentation/benchmarks/dependency bundle. No finer GPU branches exist in the inspected refs. |
| `feature/anchor-allele-orientation` | `5401668` | Entire hardcore lineage plus five accuracy/research commits. Available locally and as `fork/feature/anchor-allele-orientation`. |
| Seven `claude/*` branches | `d23282b` | All point to the same pre-GPU CPU endpoint; names do not represent seven different changes. Even `claude/gpu-acceleration-optimization-bfa6f0` points here. |

The seven aliases are `gpu-acceleration-optimization-bfa6f0`,
`heuristic-tu-46585b`, `musing-mclean-10bf64`, `nostalgic-wescoff-78034f`,
`sad-ishizaka-89d494`, `sharp-khayyam-beb245`, and
`strange-haslett-2bef8c`, all under `claude/`.

`hardcore-optimization` has no corresponding remote-tracking ref in the
inspected checkout. This does not establish what currently exists on the
remote server. Its commits are also ancestors of the locally recorded
`fork/feature/anchor-allele-orientation` tip.

`dx/dependencies`, `dx/ruff`, `dx/typing`, and current `dx/logging` form a
developer-experience stack on the split CPU endpoint. They contain no GPU
commits. `r-compat-reference` branches independently from `main`; it is a
reference-method effort, not part of the hardcore lineage. The CLI-exit and
mm10 heatmap fixes are also separate branches from `main`.

## 3. Suggested branch decomposition for the remaining bundled work

These are proposed names, not branches created by this audit. Preserve the
original branches as provenance and build a new stack on the corrected CPU
foundation, reconciling overlapping hunks rather than restoring old files.

| Proposed branch | Scope | Dependency |
|---|---|---|
| `codex/perf-repeated-row-output` | Repeated-row writer from `a0eaacf`. | Corrected CPU foundation. |
| `codex/perf-reuse-heatmap-linkage` | Reuse the existing exact Euclidean tree. | Corrected CPU foundation. |
| `codex/feature-exact-gamma-ks` | CPU-capable exact posterior KS plus Python/CLI option. | Corrected CPU foundation; no CUDA required. |
| `codex/feature-cuda-backend` | Backend plumbing, numeric port, GPU Ward, compatibility policy, CLI and dependencies. | Corrected CPU foundation. |
| `codex/feature-gpu-precision` | Full-feature clustering, all-cell silhouette and explicit precision policy. | CUDA backend. |
| `codex/perf-gpu-ward-streaming` | Triton certification and Ward/baseline memory refinements. | CUDA backend. |
| `codex/perf-large-gpu-heatmaps` | Full-tree ordering policy and bounded rendering, with distinct commits. | CUDA/precision stack; linkage reuse where applicable. |
| `codex/bench-optimization-evidence` | Benchmark harness, precision diagnostics, results and usage docs. | Integrated optimization stack. |
| `codex/feature-marker-anchor` | Marker reference selection. | Integrated pipeline; preserve existing defaults. |
| `codex/feature-arm-correlation` | Independent final-call option. | Shared anchor/profile helpers extracted as needed. |
| `codex/feature-allele-orientation` | Distinct commits for index/counting, phasing/HMM, orientation, CLI/integration and evidence. | Integrated pipeline; marker/arm options for the documented combination. |

Keep focused smoke/equivalence checks with the change they validate; the
cross-mode benchmark report belongs at the integrated endpoint. Existing
GPU dispatch intertwines FP64 segmentation with the port, so separating
port from precision policy needs deliberate implementation decisions.

## 4. Evidence and naming caveats

- `benchmarks/RESULTS.md` calls `d23282b` "main"/`base`. Actual `main` is
  `ea1a15c`. The reported 7.9x and 11.3x speedups are against the
  **CPU-optimized pre-GPU endpoint**, not the repository's actual main tip.
- `gpu-compat` retains approximation choices, not exact CPU output. The
  report records substantial call differences, including a Kidney label
  inversion. Compatibility should be qualified explicitly.
- Recorded precision improvements do not establish classification gains.
  The hardcore report attributes major accuracy failures to normal-anchor
  selection; the descendant branch changes that decision and adds allele
  evidence.
- Statements in commit bodies about exactness, byte identity and speed are
  historical validation claims. This audit checked their code boundaries
  and Git provenance, but did not rerun the numerical experiments.
- No merges connect the extracted CPU stack to the hardcore lineage.
  Ancestry-only tools therefore miss equivalent changes with different
  hashes. Stable patch IDs establish the two duplicate patches above;
  the endpoint tree diff establishes the corrected Ward-budget difference.

Useful review ranges: `main..hardcore-optimization` (original nine commits),
`d23282b..hardcore-optimization` (three GPU-era commits),
`hardcore-optimization..feature/anchor-allele-orientation` (five accuracy
commits), and `d23282b..perf/exact-shortcuts` (CPU extraction differences).
