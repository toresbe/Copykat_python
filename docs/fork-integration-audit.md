# Fork integration and evidence audit

Date: 2026-10-10. This records the local assembly built from the pinned
upstream baseline. It supplements the earlier history and benchmark reports;
those reports remain measurements of their named snapshots, not this assembled
commit.

## Pinned proposal history

The proposal base is upstream `main` at
`ea1a15c2457c0a69ee61ada07db57a0c9c726bf2`. `proposed/bit-identical` contains
explicit `--no-ff` merges of PRs #4, #6, #7, #8, and #9. `with-proposed-changes`
contains explicit `--no-ff` merges of all six performance PRs, #4 through #9,
including #5. The merge commits preserve each original PR head as a second
parent; the exact SHAs and merge commits are in
[`benchmark-manifest.json`](benchmark-manifest.json).

PR #5 is deliberately visible as a text-output and dependency-policy change:
it uses Arrow to write the gene-level CNA table, requires PyArrow, and changes
the text representation/quoting contract. It is not classified as byte
preserving. PR #9 also raises the Matplotlib floor. The proposal manifests and
benchmark notes retain these constraints.

## Experimental branches and integration order

The branches below preserve source commits and dependencies. The assembled
`assembly/main` starts at `with-proposed-changes`, explicitly merges the DX
stack, storage/output, Ward, repeated-bin, PCA-bypass, GPU, precision,
statistical, marker/arm, and allele branches, then ports selected GPU and
anchor/allele changes onto the newer CPU implementation. The GPU-era snapshot
was not copied over the current metadata, reporting, typing, logging, or mouse
ordering work.

| Branch | Declared dependency | Original source and rationale | Evidence and qualification |
|---|---|---|---|
| `experiment/storage-output` | PR #5 Arrow writer | `6a43d8c`, `769aed3`, repeated-row hunks from `a0eaacf`; bounded memory/parallel formatting and reuse of repeated CNA rows. | Signed-zero run detection now compares floating row bytes. The assembled writer passed a byte comparison against full Arrow formatting for `-0.0`/`+0.0`; broad table and full-pipeline comparisons remain pending. |
| `experiment/ward` | storage/output | `350c1c0`, `44c15db`; condensed Euclidean distances plus Ward linkage, with threaded row blocks. Distances are mathematically unchanged; finite-precision ties and merge heights are still observable. | T989 collapse replay had unchanged rooted clades and calls but different final CNA hashes and merge heights; see `benchmark-scaling.md`. Current CPU route smoke checked on a small matrix. |
| `experiment/repeated-bin-collapse` | Ward | `5768793`; a run of `m` identical feature columns becomes one column scaled by `sqrt(m)`, preserving squared Euclidean distance algebraically. | Historical T989 comparison: 0 rooted clades replaced, ARI 1 at k=2/6/50, but CNA hashes differ and max sorted-height deviation was `5.27907e-05`. |
| `experiment/pca-bypass` | repeated-bin collapse | `5768793`; bypass float32 PCA when the collapsed representation is within the retained component rank. This avoids PCA truncation error but changes floating-point calculations. | Historical diagnostic replay attributes changed linkage heights to float32 PCA; not a claim of byte-identical heights. |
| `experiment/gpu-execution` | PCA bypass | GPU execution hunks from `a0eaacf` and `0460d50`; GPU Ward uses reciprocal-nearest-neighbour rounds and exact candidate certification/fallbacks. | Original GPU benchmark reports are pinned historical evidence. The assembly exposes GPU Ward, selected GMM fits, and baseline adjustment; remaining stages run on CPU. CUDA execution was not available in this environment. |
| `experiment/numerical-precision` | GPU execution | `a0eaacf`; separates `gpu-compat` from a full-feature `gpu` Ward geometry policy. | Historical full-policy comparisons show CNA/tree deviations. The assembled GPU path currently wires the Ward geometry policy only; FP64 segment sums, all-cell silhouette, and full-tree rendering are not ported here. |
| `experiment/gpu-ward-refinement` | numerical precision | `0460d50`; optional Triton candidate search and certified exact fallback for Ward. | Static syntax/import boundary checked; no CUDA/Triton runtime available. Device arithmetic, ties, and memory behavior need a fresh device run. |
| `experiment/statistical-decisions` | proposal CPU foundation; no GPU dependency | `a0eaacf`, `0460d50`; computes the supremum CDF difference of the adjacent Gamma posteriors at density crossings, avoiding Monte Carlo KS sampling noise. | Added direct checks for identical distributions, symmetry, deterministic breakpoint calls, and CLI wiring. The existing cutoff has not been recalibrated; exact KS remains opt-in. |
| `experiment/marker-anchor` | pipeline foundation | `d3f3831`; marker-enriched immune/endothelial cells choose the normal reference, with enrichment and sigma fallbacks. | Historical 43-study comparison: mean balanced-accuracy gain `+0.047` (`[+0.024,+0.078]`), inverted calls 29 to 5; annotation circularity and selection limits are documented in `accuracy-results.md`. |
| `experiment/allele-orientation` | marker/arm source branch and pipeline foundation | `de8a6e3`; phased B-allele imbalance is an orthogonal check on tumor/normal orientation. | Historical read-backed subset is documented in `docs/allele_orientation.md`; evidence is not independent of all prior cohorts. The assembled command is opt-in and has not been rerun against this merge. |

The DX stack (`dx/dependencies` → `dx/ruff` → `dx/typing` → `dx/logging`)
contains dependency floors, linting, typing, logging, reporting and metadata
work. Its merge also includes the mm10 genomic-order correction. The biological
validation used T989 with a matched scWGS/AneuFinder reference: gene Pearson
correlation improved from `0.348` to `0.490`; aneuploid calls changed from
5,141 to 4,308. This is one sample and a real inference change; details,
input hashes, settings, and limitations are in
[`mm10-order-validation.md`](mm10-order-validation.md).

## Audit package and benchmark evidence

Reproduction and evidence files are committed in this package:

- Benchmark drivers and utilities (in `benchmarks/`): `benchmark.py`,
  `serial.py`, `writer_edges.py`,
  `parallel.py`, `gpu_memory.py`, and the other
  `*.py` scripts listed in the manifest.
- Scaling and CPU-accounted results: `benchmark-scaling.md`,
  `benchmark-cpu.md`, and
  `benchmark-serial.md`.
- Output hashes, branch/source pins, run scripts, and saved dendrogram/CNA
  diagnostics: `benchmark-assets/` and the raw results referenced by the
  reports.
- Installed-data accuracy, paired findings and limitations:
  `accuracy-results.md` and `accuracy-assets/`.

At 80k cells, the pinned six-PR snapshot completed in `2081.32 s`; the pinned
upstream snapshot took `3337.18 s`. Its 120k six-PR run was recorded as running
at the last report refresh and 170,057 cells were queued; treat these as
incomplete until the run ledger is checked again. The six-PR fresh sweep did
not yet include completed 120k/170k results. Existing GPU and CPU experimental
timings are for their pinned older snapshots. They do not measure
`assembly/main` or isolate one optimization.

The accuracy report includes both gains and losses. For example, the paired
coverage-adjusted delta was `-0.267` for Wang2019_Brain/SF9259S and `-0.204`
for Li2019_Skin/p26, while some prior label inversions improved substantially.
It uses 121 completed aligned sample pairs from a selected, previously used
data collection; the labels are imperfect proxies, not independent DNA truth.
Mouse data and direct genomic-profile accuracy are out of scope for that cell
call panel.

Large raw evidence archives remain under `/mnt/nas/cancer_research/`; the repo
contains compact reports, scripts, and selected figures. Archive paths and
checksums are recorded in the source reports. No raw archive was copied into
the repository.

## Validation boundary for the assembled tree

Validated locally: Python compilation of the modified CPU/GPU modules; CPU
Ward smoke; exact-Gamma KS symmetry/equality/breakpoint checks; CLI parsing;
marker counting/reference-selection checks; mocked GPU dispatch through Ward,
GMM, and baseline adjustment; and byte equality of the repeated-row writer
with full Arrow formatting on a signed-zero fixture. `pytest` is not installed
in the configured environment, so the added pytest modules were exercised with
equivalent direct assertions.
The environment has neither Torch nor CuPy. The optional GPU path therefore
has only static validation and a clear dependency guard here; no assembled
CPU/GPU inference comparison is claimed. Historical GPU timings, labels,
dendrograms, and accuracy results remain valid only for their pinned source
snapshots. Any fresh integration-level numerical or output contract claims
must be measured against the assembled commit.
