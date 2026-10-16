# Checking this package against CopyKAT-R

`verify/verify.sh` runs CopyKAT-R, unmodified, and this package with the default
`r` profile on the same input. It then compares every intermediate result, step by
step, and prints whether the two runs made the same decisions. Every script here
is short and meant to be read:

| File | What it does |
|---|---|
| `verify.sh` | Runs both programs on each dataset, compares them, prints the verdict |
| `r/run_copykat.R` | Runs `copykat::copykat()` as published. `trace()` adds "save this variable" calls to the running copy; nothing in the package is patched |
| `run_port.sh` | Runs `copykat-py --profile r` with `--dump-stages` |
| `compare.py` | Compares the two runs' stages: one line per measurement |
| `verdict.py` | Summarises comparison tables: same decisions or not |
| `make_input.py` | Writes a dataset as `matrix.mtx`, `genes.txt`, `barcodes.txt` |

## Running it

You need R (4.x) with CopyKAT-R and its dependencies (`dlm`, `parallelDist`,
`MCMCpack`, `mixtools`, `transport`, `Matrix`), and this package installed
(`pip install -e .`). Then:

```bash
verify/make_input.py bundled data/bundled         # the 302-cell sample shipped with this package
verify/verify.sh data/bundled                     # compare with CopyKAT-R 1.2.5
```

To compare with CopyKAT-R 1.1.0 as well, install it into its own library and
point `R_LIBRARY` at it:

```bash
git clone https://github.com/navinlabcode/copykat && git -C copykat checkout d7d6569   # v1.1.0
mkdir -p ~/R-copykat-1.1.0 && R CMD INSTALL -l ~/R-copykat-1.1.0 copykat
R_VERSION=1.1.0 R_LIBRARY=~/R-copykat-1.1.0 verify/verify.sh data/bundled
```

The other settings are listed at the top of `verify.sh`. `R_NOISE=1` also runs R
with `n.cores = 4` and compares R with itself; `EXACT_MEANS=1` runs this package
with `--exact-segment-means`, a faster option described in `docs/profiles.md`.

## Reading the result

The verdict has one line per comparison:

```
| comparison | verdict | differing decisions | calls differing | breakpoints (A B shared) | max diff, bins |
| bundled: R-1.2.5-vs-port-r1.2.5 | same decisions | - | 0 of 302 | 294 294 294 | 3.74e-11 |
```

CopyKAT makes a handful of decisions, and everything else is arithmetic on top of
them: which cells pass the filters; which cells form the normal reference (step 4);
where the breakpoints are (step 5); which cells cluster together and how each is
called (steps 7 and 8). "same decisions" means all of these are identical. The last
column is the largest difference anywhere in the genomic-bin matrix. Differences of
about 1e-10 come from the smoothing in step 3, where R's `dlm` and this package
compute the same smoother in a different order.

The full table for each dataset, `OUT/<dataset>/R-<version>-vs-<run>.tsv`, has
the measurements for every step. Each run's folder also has the program's own
output files, and `stages/` with every intermediate result.

## Why R runs on one core

CopyKAT-R calls `set.seed(1234)`, so a run is reproducible, but only with
`n.cores = 1`. With more cores, `parallel::mclapply` gives each worker a randomly
seeded stream for the per-cell draws of step 5, so the values change from run to
run. Those changes are small, but they can move cells between the final clusters,
and the 1.2.5 final call has a threshold (a correlation of 0.6 makes every cell
diploid) that a small change can cross. On one simulated sample (`R_NOISE=1`), R
with one core called all 300 cells diploid and R with four cores called 30 of them
aneuploid. This package reproduces the one-core run.

## The stage format

Both sides write each intermediate value to its own folder under `stages/`:
numbers as raw little-endian float64 in `values.f64` (column-major, as R stores
matrices), text as one entry per line in `values.txt`, the dimensions in
`shape.txt`, and row, column or element names in `rownames.txt`, `colnames.txt` or
`names.txt`. `copykat_py._stages.load()` reads them.

| Stage | Contents | R variable |
|---|---|---|
| `smoothed` | genes x cells after step 3 | `norm.mat.smooth` |
| `relative/` | step 4: relative expression, clusters, normal cells, warning | `norm.mat.relat`, `CL`, `preN`, `WNS` |
| `segmentation_N/` | step 5, attempt N: breakpoints (1-based), segment values, KS cut | `CNA.MCMC()` result, `cut.cor` |
| `bins` | genomic bins x cells after step 6 | `convert.all.bins.hg20()$RNA.adj` |
| `final/` | steps 7-8: adjusted matrix, clusters, calls, consensus correlation | `mat.adj`, `hc.short`, `hc.umap`, `com.preN` |

## What this does not cover

- `cell.line = "yes"` and mouse (`mm10`) final calls: the `r` profile follows R in
  steps 1-6 there, but steps 7-8 use this package's own path.
- R on arm64 computes the KS statistic with 64-bit sums, so it can decide a
  breakpoint sitting exactly on the cut differently from R on x86-64. The `r`
  profile follows x86-64.
- When mixtools gives up on a fit and restarts from random starting values (a
  variance collapsing to zero), R's random stream moves on. This package logs a
  warning in that case; it has not been seen on real data.
