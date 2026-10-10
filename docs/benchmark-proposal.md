# Proposed benchmark and review cadence

Proposal dated 2026-10-09. No branches were switched, created, reset or
pushed, and no pull requests were submitted. Your later instruction makes fork `main` the experimental head in this proposal.
The source snapshots used
below are ordinary directories, not Git worktrees or branches.

## Recommended structure

Make review units follow **what changes in a scientific result**, rather
than GPU/CPU file boundaries or the original commit boundaries. Keep each
implementation change with its rationale, verification script, and small
before/after evidence. Keep formatting, typing, logging and dependency
cleanup outside the numerical review series unless a dependency is needed
by that specific change.

Use the upstream `main` at `ea1a15c` as the initial reference.
Your fork’s `main` is proposed as the experimental development head; it
is never the baseline for an independent upstream PR. The local
checkout is `dx/logging`; it is not the scientific reference. Existing
`hardcore-optimization` benchmarks incorrectly called the CPU-optimized
`d23282b` endpoint “main.” Every new benchmark must record the exact SHA,
patch manifest, environment, input cells, retained cells and output policy.

The proposed refs on your fork are:

| Ref | Contents and role |
|---|---|
| Upstream `main` / upstream-tracking ref | Stable scientific reference, pinned by SHA. In this checkout the recorded upstream ref is `origin/main`; a future `upstream/main` name can serve the same role. |
| `pr/<change>` | One independently reviewable candidate. Base on upstream main, never the experimental fork main; explicitly name the parent PR where required. |
| `proposed/bit-identical` | Integration of the independently reviewable, byte-preserving candidates. This is the conservative estimate of immediate value. |
| `with-proposed-changes` | Integration of all upstream candidates whose evidence packets are ready. It assumes approval of all included PRs, and is an optimistic scenario, not a prediction of approval. Record included/excluded candidates in a manifest. |
| Your fork’s `main` | Experimental development head: the integration branch plus opt-in precision, GPU and classification research, with developer-experience work kept in separate commits. Experimental method changes remain separately selectable. No separate experimental branch is needed. |
| Existing optimization/feature refs | Preserve provenance until the replacement history is validated; do not rewrite them to simulate an atomic history. |

`with-proposed-changes` should initially equal `proposed/bit-identical`.
Promote other candidates into it only when their audit packet is complete.
For each promotion, compare against both its immediate parent and the upstream
`main`; otherwise interactions between changes can conceal regressions.
Keep accepted changes synchronized with upstream, and remove the need for
their speculative PR refs as upstream adopts them. Annotated manifests or
release tags should identify measured integration versions.

The future experimental fork `main` should include the work now at
`feature/anchor-allele-orientation` (`5401668`), rebuilt on the corrected
CPU foundation and the newly separated commits. Simply renaming that tip
would retain the bundled history and old Ward memory budget. Do not fold
marker/allele behavior into the default merely because it is the latest
experiment. Keep the existing DX chain separate in this development history
so styling/typing changes do not obscure numerical diffs. Existing defaults should remain explicit in the manifest.

## Wave 1: independent, byte-preserving candidates against upstream main

These are **qualified claims over the tested domain and environment**, not
proofs for every future dependency version. “Bit-identical” must cover
numeric intermediates, predictions, tree structure and heights, tabular
bytes, `.seg` and image pixels as applicable. Timings and runtime metadata
are expected to differ. A matching float16 or float32-reduced hash is not
sufficient to establish equality of float64 values.

| Order / candidate | Existing extraction | Review rationale and evidence |
|---|---|---|
| A1. Heatmap interpolation stage | `perf/heatmap-interpolation-stage`, `2215462` | Normalize/colormap after nearest-neighbour image downsampling. Same sampled values and colors; substantially less time and memory on modern matplotlib. Verify both ordinary and annotated heatmaps, decoded pixels and PNG bytes. State the matplotlib >=3.5 requirement. |
| A2. Shared DLM gains | `perf/dlm-shared-gains`, `516d719` | Kalman and RTS gains depend on model variances and sequence length, not the observations. Preserve recurrence and summation order, output dtype and centring; test original-dtype intermediate equality at 1/8/32 cores, including zero/small matrices and actual preprocessing output. |
| A3. Shared-memory thread pools | `perf/thread-pools`, `24475ca` | Change transport/execution, keeping median inputs and per-cluster GMM arithmetic. Show consensus, fit and bin-output equality; report process-tree memory, including original workers. |
| A4. Skip unused fallback clustering | `perf/skip-unused-gmm-clustering`, `1cf6437` | Caller discards these cluster labels; avoid computing them. Preserve the public default and cover both low-confidence fallback outcomes and `RE_before`. Worth reviewing separately because it can have a large conditional benefit. |
| A5. Known-normal set membership | `perf/known-normal-set`, `bb4abc5` | Same membership result with a set; test empty/unknown/duplicate normal names and existing name conversion. Small standalone change; performance benefit is conditional on supplied known-normal cells. |

I constructed a **filesystem-only** optimistic snapshot of upstream `main` plus
A1–A5 for benchmarking. It corresponds to the proposed conservative
integration branch without changing Git state. Exact hashes and patch
order are in the snapshot manifest. Each existing candidate remains an
independent PR against upstream’s `main`; the integration order does not create
unnecessary PR dependencies.

Submit only two or three small reviews at a time after you approve the
structure. Lead with A1/A2, add A3, then the conditional A4/A5. Their review
packets should fit roughly one page each; the scripts and raw results are
attachments. Do not claim a large pipeline speedup from a microbenchmark
without reporting that stage's share of runtime.

## Wave 2: representation and memory improvements

These can deliver substantial value without changing the biological
model, but deserve separate contracts from the strict byte-preserving
wave. Their dependencies should not drag precision or classification
changes into otherwise simple performance reviews.

| Candidate | Dependencies | Required contract |
|---|---|---|
| B1. Arrow gene-output formatting | Independent of Wave 1; re-extract from `ffcfa3e` or explicitly review `92d7e51`. | Decoded values, annotations, headers and precision must agree. Text spellings differ from pandas; declare that deviation. `92d7e51` additionally raises Python/Arrow floors, removes the fallback and changes quoting, so split dependency/format policy from the speed change. Use full-precision parsed comparisons. |
| B2. Storage-only memory refactor | Wave 1 optional; existing `6a43d8c` is stacked on a bundle and is not an independent PR yet. | Separate sparse filtering/annotation, buffer lifetimes/preallocation, and in-place numerical transformations into distinct commits. Check original input mutation and array-layout invariants. Storage-only pieces can join the strict branch once intermediate equality is established. Any altered arithmetic or aliasing behavior gets its own numerical review. |
| B3. Parallel table formatting | B1/B2 output interfaces; source `769aed3`. | Same output bytes, bounded memory, deterministic ordered writing, same serial fallback behavior. No mathematical prerequisite. |
| B4. Repeated-row writer | B1/B2 output interfaces; source hunks in `a0eaacf`. | Format each unique numeric row once and combine with each row's annotations. Compare complete output bytes, including quoting/newlines, signed zero, NaN and uncommon identifiers. Do not require CUDA. |
| B5. Reuse a compatible heatmap linkage | An existing exact Euclidean tree on the identical retained-cell matrix. | Verify matching data, cell order, metric and lack of lossy PCA. Same tree and pixels; keep original clustering when those conditions fail. Do not make it depend on a new classification method. |

B1 is a promising early review even though it fails the requested strict
byte-identity definition: saved representative evidence shows gene writing
falling from 22.91 s to 2.68 s and a 108.66 s pipeline to 88.48 s on
SCPCL001108. B2's memory benefit is especially important for scaling.
Saved corrected process-tree measurements show median memory reductions
around 78–79% for the *combined* CPU stacks. Those numbers do not isolate
B2 and must not be attached to an independent B2 PR as if they did.

Fresh B4 edge verification found a byte-preservation defect: adjacent rows
with `-0.0` and `+0.0` compare numerically equal, so the repeated-row writer
can reuse `-0` for a row whose reference output is `0`. Require
representation-aware run detection (or equivalent formatted-value identity)
before calling this generally byte-preserving. The small reproducer is
`writer_edges.py`, with exact output in the result bundle.

## Wave 3: mathematically equivalent changes with finite-precision effects

A theoretically unchanged distance is not a guarantee of an unchanged
dendrogram. Ward linkage is a discrete sequence of minimum-cost decisions;
a small perturbation near a tie can change a merge and later labels.
Compare tree topology and decision margins, not just max matrix error.

| Candidate | Parent | Separate mathematical claim and observed deviation |
|---|---|---|
| C1. Condensed-distance Ward engine | B2 advisable for memory; independent of PCA removal. | Same Euclidean Ward objective; fastcluster distance-matrix versus vector implementation changes evaluation/rounding. Count both condensed matrix and working copy in the 8 GB budget (~31.6k cells). Show unchanged merges where observed, height error, tie cases, fallback behavior and memory tradeoff. Re-extract `350c1c0` onto its declared parent. |
| C2. Threaded distance fill | C1. | Compare condensed distance arrays bit for bit to `pdist`; source `44c15db` can be separated from its current repeated-bin ancestor. If identity holds, it is a transport/performance change on C1, not another precision change. |
| C3. Weighted repeated-bin collapse without PCA | C1, optionally C2. | For a run of length r, preserve squared distance with weight sqrt(r): sum r(x-y)^2. Floating-point sqrt/scaling and summation change rounding. Show topology changes, cut agreement, full-precision distance errors and decision margins. |
| C4. Bypass PCA when collapsed width fits the component cap | C3. | This is a separate behavior change. Mathematical rank sufficiency does not make a float32 randomized PCA numerically lossless. Reproduce the T989 mouse counterexample. Keep this out of `proposed/bit-identical`. |

The current `5768793` bundles C3 and C4. Split them. Existing T989 evidence
shows the final predictions and two-way labels unchanged, but different
linkage rows/heights; the pre-collapse float32 PCA versus collapsed path differs by
up to 3.6e-5 in heights. The saved row-based replay reports matching linkage structure after casting
the PCA input to float64, with ~1e-13 height differences. In the fresh
8-core replay, canonical leaf-set comparisons find zero changed rooted
clades and ARI=1 at k=2/6/50, despite different linkage-row hashes and
~5.3e-5 sorted-height differences. Do not equate reordered independent
merges/internal-node IDs with different topology. This explains the numerical
mechanism without treating matching final labels as proof of identity.
The saved smallest merge-height gap is ~5.4e-8: a generic “below the noise
floor” argument is insufficient for a dendrogram near those decisions.

For C3/C4 include paired dendrograms, a merge-height difference plot, ARI
at k=2/6/50 and counts of changed clades. Leaf permutations and different
internal-node numbering alone are not topology changes; compare actual
leaf sets. Include one known numerically changing dataset, one unchanged dataset,
and any observed topology-changing cases (explicitly report none if absent),
and the 10k/30k/full scaling results or explicitly censored runs.

## Wave 4: GPU execution and precision policy

Separate the device port from decisions about what scientific calculation
it performs. The `gpu-compat` name means compatibility of approximation
policy, **not output compatibility**.

1. Add backend dispatch, optional dependencies, CLI/AnnData integration
   and a CUDA smoke check in a plumbing commit.
2. Port individual operations: preprocessing/DLM; cluster medians/GMM;
   bin medians/baseline adjustment; segment sums. State storage and
   accumulation dtype, transfer/chunk policy and CPU oracle for each.
3. Add the GPU Ward engine with a CPU full-feature Ward oracle. Explain
   reciprocal-neighbour reducibility, tie policy, certified candidate
   bounds and FP64 fallback. Projection for candidate search only is
   different from clustering projected coordinates. Verify full tree
   structure on manageable data and cuts/heights at larger sizes.
4. Add Triton acceleration and memory refinements as subsequent
   performance commits; demonstrate the same chosen neighbours/tree as
   the unaccelerated GPU engine on ties and near-ties.
5. Make each precision policy change a separate opt-in commit: FP64
   segment accumulation; full-cell silhouette; full-feature step-4 Ward;
   full-tree heatmap ordering. Keep bounded raster/dendrogram rendering
   in a separate plotting commit because it changes displayed detail.

The current Ward implementation's claimed certification uses a dimension-
dependent FP32 slack and additional projection slack. Its audit packet
needs an explicit conservative error-bound derivation for the actual GEMM,
norm, cluster-size and projection arithmetic, including the effective
TF32/matmul setting. Equality against CPU trees on sampled data is empirical
validation, not a proof of this bound. Include adversarial cancellation,
dynamic ranges, ties and near-ties; document fallback behavior and the
precision assumptions under which the exactness claim holds.

FP64 segment sums are a numerical change around a measured small error,
but a threshold-adjacent downstream decision can amplify it. Full-feature
Ward is an **algorithmic geometry change**, not merely a rounding change:
it restores directions discarded by PCA and can substantially change
six-way clusters. Full-cell silhouette changes the estimator's sample
coverage. Full-tree heatmap ordering mostly changes visualization, and
must not be conflated with changed biological calls.

For a reviewer with mathematical/biological training, show the original
formula, revised formula, reason for the change, dtype/reduction details,
and measured error distribution. Link those errors to the relevant
threshold, reference-cluster sigma gap or Ward cost gap. Normalize error
by per-profile variability where appropriate; report absolute values too.
Use paired tree/profile plots to show whether numerical differences alter
which cells supply the normal reference.

The GPU precision series remains on your experimental fork `main` until its independent
evidence packets are complete. It can then enter `with-proposed-changes`
as an explicit opt-in backend. Every speed comparison must distinguish
CPU port speed from the effect of replacing approximations.

## Wave 5: statistical and biological method changes

These require their own review, even when described as “exact.” They must
not be prerequisites for CPU output or memory optimizations.

| Candidate | Independent decision being reviewed |
|---|---|
| Exact posterior-Gamma KS | Replace a statistic from two finite Monte Carlo samples with supremum distance between the posterior CDFs. Same posterior family, different finite-sample statistic/distribution; the original threshold calibration does not automatically transfer. Show breakpoint additions/removals, seed variability, CNA profile differences and call changes. Implement the CPU-capable option independently of CUDA. |
| Marker anchor | Change the definition of the normal reference, with documented fallbacks and eligibility. Show selected cluster, marker enrichment, sigma margins and call inversions; evaluate unseen studies and mouse/ineligible behavior explicitly. |
| Arm correlation | Change the final classifier independently of marker anchoring; evaluate all four anchor/classifier combinations. Show per-cell scores, reference-derived threshold and errors by sample. |
| Allele orientation | Independently review index/counting, phasing/HMM, then label-flip rule/integration. Show group imbalance evidence, coverage and significance/gating rationale, preserved default behavior and held-out harmful-flip rate. Counting acceleration should have its own CPU/counting oracle. |

Keep the research record, including failed approaches, but distinguish
training/development datasets from genuinely held-out evaluation. A
malignant label is not identical to an aneuploid label, particularly in
near-diploid disease; stratify interpretation rather than treating every
flip toward malignant labels as proof of better CNA inference. Plot WGS
or other orthogonal CNA evidence where it exists.

## Review packet and acceptance cadence

Each candidate gets: (1) a one-paragraph trigger/before/after explanation;
(2) a small diff against its declared parent; (3) the mathematical or
behavioral invariant and intentional deviations; (4) exact input/source
manifest; (5) a reproducible comparison script; (6) raw timing, memory and
full-precision equality/error results; (7) one figure when the scientific
result or dendrogram changes; (8) known failures and untested regimes.

For performance, run fresh processes in alternating order on a quiet
machine. Report cold start separately from warm kernels when relevant.
Use identical deterministic subsets and thread counts, retained-cell
counts, optional heatmap/output settings, and both stage and total times.
Report timeouts and failures, not just completed comparisons. The cursory
runs here prioritize candidates; repeat shortlisted comparisons before
claiming stable speedup. Do not sum individual speedups to predict an
integration result.

Wave 1 can proceed as soon as you approve this structure. Wave 2 follows
as focused output/storage reviews. Wave 3 should start with the dendrogram
counterexample and the finite-precision explanation. GPU and biological
method reviews can proceed later or alongside it as separately based
series; they do not block the immediate CPU value. Batch related review
questions rather than asking the maintainers to read the entire
experimental fork `main` diff.

## Benchmark execution and storage

Fresh-run results and source manifests are at
`/home/toresbe/cancer_research/benchmark_2026-10-09`. Inputs, source
snapshots, calculation scratch and output hashing use this SSD. Existing
archives were left in place; a compact `.tar.gz` evidence bundle is stored
on `/mnt/nas/cancer_research` after measurements complete. No raw archives
are duplicated on the SSD for these calculations.

The reproducible driver is `benchmarks/benchmark.py`; the serial
sweeps are `benchmarks/sweep.sh` and
`benchmarks/followup.sh`. It checks the imported source path,
preserves original-dtype hashes, monitors process-tree RSS, records
progress and stops at time/memory/SSD guards. It removes only its own
scratch output directories after hashing. Xenium subsets are nested,
seeded at 20261009, from the SSD HDF5 input. Large scaling TSVs are
formatted into `/dev/null`; **these are compute/formatting timings, not
persistent SSD output throughput or complete tabular byte checks**.
Kidney runs write real SSD outputs and hash them. Input loading and audit
hash/compression time are outside `pipeline_s`; the process elapsed time
includes them. CPU and GPU Python environments are recorded in results
and must be matched for definitive device-isolation comparisons.

Fresh measurements and figures are summarized in the companion
`benchmark-scaling.md`. Historical evidence is copied into a
separately labeled directory with source paths and SHA256 checksums;
none of those earlier timings is presented as a fresh measurement.

The follow-up `benchmark-cpu.md` records uncapped 10k/30k/full
170,057 runs. `parallel.py` admits concurrent jobs using
measured-memory estimates, reserves future growth and 11 GB of headroom,
and retains the 10 GB available-RAM and 2 GB SSD-space emergency guards.
Dedicated systemd cgroups measure user-plus-system CPU across all threads
and descendant workers at pipeline boundaries. Both elapsed time and CPU
core-seconds are reported; GPU CPU time excludes device work. This second
sweep consistently disables heatmap rendering, includes table formatting
and records original-precision output hashes. Do not compare its times
directly to the earlier heatmap-inclusive series. Completed points replace
timeouts only in the new figure; all original censored evidence is retained.
The visible report refreshes every minute while jobs run. On completion,
the scheduler writes a separate checksum-protected NAS evidence archive.

The expanded follow-up adds five lower-size and three upper-size points,
physical-core affinity, two-job admission, isolated diagnostic anchors and
client-independent user services. See `benchmark-expanded-plan.md` for
the exact queue, contention caveats and crash-recovery policy. Existing
completed results and the surviving 170k CPU calculation are preserved.

All baseline references to “main” in the measured snapshot manifest mean
upstream’s implementation at `ea1a15c`, not the future experimental fork
head. Contribution PRs target the upstream repository/main; their source
branches may live on your fork independently of its experimental main.

## Commit shape and integration history

Use one implementation commit per invariant or scientific decision, with
its focused verification beside it; a second evidence/documentation commit
is reasonable when the packet is substantial. Do not mix an operation's
CUDA port with a changed estimator, reference rule or threshold. Give each
commit a plain title such as “Accumulate segment sums in FP64” and a body
that states the original calculation, revised calculation, expected
deviation and observed effect. Record the original source SHA(s) in a
footer, even when the change is extracted from only part of a commit.

Keep independent PR branches based on the pinned upstream commit. For a
dependent series, the PR parent should contain only the named prerequisite,
and the review diff should exclude that prerequisite. After upstream
acceptance, rebase later candidate branches onto the accepted upstream
revision and rerun the relevant comparisons.

Integrate these small branches with explicit merge commits on
`with-proposed-changes` and the experimental fork `main`. Merge titles name
the candidate and its evidence version, so first-parent history reads as a
sequence of review decisions and branch history preserves the fine detail.
Keep development corrections as clearly named follow-up commits. Freeze
submitted review versions; if a mathematical policy changes, make a new
reviewable decision rather than silently replacing old evidence. The
manifest must distinguish upstream-accepted, proposed, experimental and
known-failing candidates at each measured integration tip.

## Executed branch construction (2026-10-10)

The local Git work has since been constructed from pinned upstream `main` at
`ea1a15c2457c0a69ee61ada07db57a0c9c726bf2`. The initial proposal now has
explicit merge commits for PRs #4, #6, #7, #8 and #9. The
`with-proposed-changes` branch has explicit merge commits for all six PRs,
including #5 Arrow output. Its formatting and dependency contract is called
out in the audit manifest. Semantic branches and the proposal-derived
assembly are recorded in [`fork-integration-audit.md`](fork-integration-audit.md)
and [`benchmark-manifest.json`](benchmark-manifest.json). Benchmark
reports remain tied to their pinned snapshots; the fresh T989 comparison is
recorded separately in [`mm10-order-validation.md`](mm10-order-validation.md).
