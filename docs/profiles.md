# Profiles: reproducing CopyKAT-R, or running fast

A profile chooses whose results a run reproduces. Most of this package's speed
work (the DLM smoother, multithreaded medians and distances, exact Ward linkage
engines, blocked I/O) leaves results unchanged and is used by every profile. The
profile decides only the places where a faster or different method would change
results.

| Profile | Use it for | Command line | Python |
|---|---|---|---|
| `r` (default) | Results that match CopyKAT-R 1.2.5, decision for decision | `copykat-py ...` | `copykat(...)` |
| `r` with R 1.1.0 rules | Matching CopyKAT-R 1.1.0 (and upstream CopyKAT-Python's final call) | `--r-version 1.1.0` | `r_version="1.1.0"` |
| `experimental` | Very large samples, when speed matters more than matching R | `--profile experimental` | `profile="experimental"` |

## What `r` reproduces, and how that is checked

With the `r` profile, a run makes the same decisions as CopyKAT-R with
`n.cores = 1`: the same cells pass the filters, the same cells form the normal
reference, the breakpoints are the same, the final clusters are the same and
every cell gets the same call. The matrices in between agree to about 1e-10.

`verify/verify.sh` shows this on any dataset: it runs CopyKAT-R unmodified, runs
this package, and compares every step ([`verify/README.md`](../verify/README.md)).
`tests/test_rcompat.py` checks the bundled 302-cell sample against R's saved
results on every test run, and checks the random number generator, the KS
statistic and the mixture-model fit against values printed by R.

Results so far (CopyKAT-R from github.com/navinlabcode/copykat on R 4.3, x86-64):

| Dataset | Cells | CopyKAT-R 1.2.5 | CopyKAT-R 1.1.0 |
|---|---|---|---|
| Bundled sample | 302 | same decisions; 0 calls differ | same decisions; 0 calls differ |
| Simulated, 60% tumour | 300 | same decisions | same decisions |
| Simulated, no tumour | 300 | same decisions | same decisions |
| Simulated, 10% tumour | 300 | same decisions | same decisions |
| Simulated, 90% tumour | 300 | same decisions | same decisions |
| Simulated, half depth | 300 | same decisions | same decisions |
| Simulated, 1/5 depth | 300 | same decisions | same decisions |
| Simulated, 60% tumour | 2,000 | same decisions | not run |

Every comparison in this table has every intermediate matrix within 1e-10 of R's.
On the 2,000-cell sample (4-core cloud machine), CopyKAT-R took 55 minutes with one
core; the `r` profile took 42 seconds and `experimental` 33 seconds. The `r`
profile's extra time is the serial replay of R's draws in step 5.

The simulated samples come from the seeded simulator of the synthetic-truth
tests (`tests/synthetic/synthetic_data.py` on the `claude/project-thread-60caw9`
branch): negative binomial counts for 8,000 real genes, with planted gains of
chr7 and chr20 and losses of chr10 and 3p in the tumour cells.

## CopyKAT-R does not reproduce itself with more than one core

CopyKAT-R sets a random seed, but with `n.cores > 1` the per-cell draws of step 5
come from randomly seeded worker streams, so each run differs a little. The final
call amplifies small differences: R 1.2.5 compares the median profiles of the most
and least normal clusters, and if they correlate at 0.6 or more it calls every
cell diploid.

| Simulated sample | R, 1 core vs R, 4 cores |
|---|---|
| 90% tumour | 30 of 300 calls differ (1 core: all diploid, correlation 0.70; 4 cores: 30 aneuploid, correlation -0.05) |
| 1/5 depth | 77 of 296 calls differ |

So "the R result" is only well defined for `n.cores = 1`, and that is what the
`r` profile reproduces. It does so by replaying R's own random draws in R's
order, which is serial. `--exact-segment-means` (`segment_values="closed_form"`)
instead uses the exact posterior mean that R's 1,000 draws estimate: it is much
faster for large samples and has no Monte Carlo noise, but it agrees with R only
as well as two multi-core R runs agree with each other (30 of 300 calls differed
on the 90%-tumour sample above).

## Where `r` and `experimental` differ

| Step | CopyKAT-R, and the `r` profile | `experimental` |
|---|---|---|
| 1 | A single cell with too few genes is kept (R removes them only when more than one qualifies) | Removed |
| 1-6 | Smoothed, segmented and binned values in float64 | float32 (half the memory) |
| 4 | Ward clustering on every gene; clusters numbered as `cutree` numbers them | PCA above 2,000 cells |
| 4 | Clusters need more than 10 cells (1.2.5) or 5 (1.1.0) | 5 |
| 4 | mixtools' EM: sigma from the previous iteration's residuals, signed stopping rule; `sd()` with n - 1 | Textbook EM; standard deviation with n |
| 4 | Silhouette over every cell; every cluster tied for the lowest sigma is normal | Silhouette over a sample above 3,000 cells; the first tied cluster |
| 4 | Per-cell GMM fallback adopted whenever it finds more than 2 normal cells, which clears the low-confidence warning | Adopted only if its profile fits better; the warning stays |
| 5 | R's random number generator from `set.seed(1234)`, R's KS statistic, 1,000 draws, R's draw order | NumPy's generator, SciPy's KS statistic (which decides ties at the cut differently), 500-1,000 draws |
| 5 | Per-cell values: R's 1,000 draws replayed | The exact posterior mean |
| 7-8 | `hclust(method = "ward.D")` on the full matrix, with `distance` | Ward (ward.D2), PCA when wide, always Euclidean |
| 7-8 | 1.2.5: k from 4 down to 2; a cluster's score is its number of normal cells; correlation gate at 0.6, then the nearer consensus by Wasserstein distance. 1.1.0: k = 2; the cluster with the larger normal fraction is diploid | k = 2; larger normal fraction (as 1.1.0) |

The R behaviour lives in `copykat_py/rcompat/` (one module per step) and the
profile switch in `copykat_py/profiles.py`. Options that CopyKAT-R does not have
(`--ks-method exact`, `--anchor markers`, `--final-call arm_correlation`) take
their step off the R path; the run's `runtime.json` records the profile actually
used.

## Not covered yet

- `cell_line="yes"` and mouse (`mm10`) final calls follow this package's own
  path in steps 7-8 under every profile.
- A `python-upstream` profile reproducing navinlabcode/Copykat_python. That needs
  the same kind of harness against upstream's code; until then, the differences
  between upstream and R are listed in `research/copykat-profiles-feasibility.md`
  in the project files.
