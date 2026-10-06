# R COMPATIBILITY reference mode

`copykat(..., r_compat=True)` makes CopyKAT-Py follow CopyKAT-R 1.2.5 wherever
the default pipeline differs, so the Python rewrite can be validated against R.
It is a **reference for validation, not a branch to build on and not a
recommended way to run the tool**: on some samples R's own call is wrong (see
"R is not always right" below).

Every deviation is marked with a comment containing **`R COMPATIBILITY`** in
`copykat_py/r_compat.py` and `copykat_py/copykat.py`:

```bash
grep -rn "R COMPATIBILITY" copykat_py/
```

The default code path is untouched (verified: byte-identical predictions and a
CNA matrix diff of 0.0 before/after this branch on Laughney 2020).

Scope: `genome="hg20"`, `cell_line="no"`. Other settings raise
`NotImplementedError`. Re-run the comparison on your own data with
`scripts/validate_r_compat.py`.

## What differs, and what r_compat changes

| # | Step | R (1.2.5) | Default CopyKAT-Py | Evidence |
|---|------|-----------|--------------------|----------|
| 1 | 4, 7, 8 | Clusters the full matrix, never PCA | PCA to 256 PCs above 2000 cells | Jerby-Arnon (2522 cells): ARI 0.30 vs R on the baseline clusters; different "normal" cluster; final call inverted. Disabling PCA: 2521/2522 cells agree with R |
| 2 | 4 | `baseline.norm.cl` min cluster size 10 | 5 | code |
| 3 | 4 | mixtools EM: sigma from residuals around the *old* means, absolute stop tolerance 1e-8 | textbook EM, relative tolerance | Lin cells 38/76/105/109 reproduced to 4 d.p.; otherwise different `baseline.GMM` anchor cells |
| 4 | 4 | `sd` with n-1 for EM start values | numpy n | code |
| 5 | 4 | silhouette on all cells; all clusters tied for min sigma are "normal" | subsampled above 3000 cells; first only | code |
| 6 | 4 | GMM fallback always adopted when > 2 diploid cells found; WNS reset to `""` | adopted only if GMM sigma < cluster sigma; WNS stays `unclassified.prediction` (labels `*.low.conf`) | su003: R's 5 GMM anchor cells invert the call; Python at `f6a0916^` reproduces R, current `main` does not |
| 7 | 7, 8 | `hclust(..., "ward.D")` (unsquared distances) | `ward` = ward.D2 | partitions identical to R's `cutree` when emulated via `ward` on `sqrt(d)`. Changes 0-0.3% of labels on the BCC/3CA samples, but **7.6% on Xenium** (ablation on R's own inputs: 92.4% agreement with `ward.D2` alone), and flips Lin via the 0.6 cliff of #12. NB R's *baseline* step uses ward.D2: the README's "R uses ward.D" is only true for steps 7-8 |
| 8 | 7, 8 | cut at the first k in 4,3,2 with all clusters > 10 cells | always k=2 | R source |
| 9 | 7, 8 | cluster score = count(preN in cluster) / length(preN) | divided by cluster size | R source |
| 10 | 7 | per-bin spread `sd` with n-1 | n | `mat.adj` matches R to 1e-15 |
| 11 | 8 | `distance` honoured in steps 7-8 | hard-coded euclidean (only plotting uses `distance`) | R source |
| 12 | 8 | if cor(diploid consensus, aneuploid consensus) >= 0.6: **all cells diploid**; otherwise each cluster goes to the nearer consensus by 1-D Wasserstein distance | max-score cluster diploid, min-score aneuploid | Lin: R's 0.612 -> all 1139 cells diploid |
| 13 | 8 | the `0.4 <= cor < 0.6 -> unclassified` branch is a no-op (`==` instead of `=` in R) | n/a | replicated as a no-op |
| 14 | known-normal | requires clusters > 10 cells, `preN` = list as supplied | > 5 cells, preN = those found | R source. **Smoke-tested only** (runs, sensible split on su003 with 100 known normals); *not* compared against R |

## Not reproduced (cannot be, or left alone)

* **Step 5 segmentation.** R uses Monte Carlo posterior draws (`MCMCpack`,
  `mclapply` RNG streams); Python uses the exact closed-form segment mean.
  Median per-cell correlation Python-vs-R is 0.957, the same as R-vs-R with
  different seeds (0.953). Breakpoint sets overlap ~90%. Gene-level matrices are
  therefore near-identical, never bit-identical.
* Plotting, R's cell-name mangling (`:` -> `.`, `X` prefix on numeric names),
  mixtools' "variance -> 0, restart" branch (never triggered).

## Validation results (R 1.2.5 vs this branch; seed fixed by copykat())

Cells with a different diploid/aneuploid call, R vs Python. "default" = `r_compat=False`.

| Dataset | cells | default | r_compat |
|---|---|---|---|
| Laughney 2020 (lung) | 993 | 0 | 0 |
| Gao 2021 (breast) | 1480 | 0 | 4 |
| BCC su008 | 7572 | 1 | 3 |
| BCC su003 | 1059 | 1034 (R inverted, see #6) | 10 |
| Jerby-Arnon 2021 | 2522 | 2521 (Python inverted, see #1) | 1 |
| Lin 2020 (pancreas) | 1139 | 845 | 293 (see below) |
| Xenium Atera 10k subset | 7739 (+2003 not.defined) | 633 (91.8% agree, PCA on); 582 with PCA off | 3659 vs R seed 1234, **0 vs R seed 1** (see below) |

Stage-level checks on R's own intermediates: smoothing 3e-8, clustering ARI 1.0,
bin conversion 1.5e-8, `baseline.GMM` baseline max difference 6e-16 (Lin),
steps 7-8 re-implementation 100% label agreement and `mat.adj` to 1e-15 (all six).

**Xenium Atera (10,015 cells sampled proportionally across the 29 graph clusters
in `analysis.tar.gz`, seed 0; `scripts/subsample_xenium.py`).** R cannot run on the
full 170k cells (quadratic `hclust`), so this is a subset. There is no
aneuploidy annotation, so only R-vs-Python agreement can be assessed.

* Default Python vs R (seed 1234), cells with a call in R: 91.8% agree; with PCA
  off 92.5%. Python calls more cells aneuploid (4196 / 4073 vs 3659). Only one of
  the 34 clusters changes its majority call (c10: R 37% aneuploid, Python 72%).
* PCA (#1) is a minor contributor here (0.7%). `ward.D2` instead of `ward.D`
  (#7) is the largest single factor: 7.6% of cells, on R's own step-7 inputs.
  `k=2` instead of the 4..2 rule (#8), and Python's label rule (#12), change the
  outcome much more because R's labelling depends on them (see the ablation
  caveat below).
* **R is bimodal on this subset.** Same data, three seeds: seed 1234 -> consensus
  correlation 0.315, 3659 aneuploid; seed 2 -> 0.474, 3606 aneuploid; seed 1 ->
  0.713, so #12 labels **all 7739 cells diploid**. R seeds 1234 and 2 agree on
  94.4% of cells; seed 1 agrees with either on only 52.7%.
* r_compat labels every cell diploid (correlation 0.914). That is **R's seed-1
  mode, reproduced at 100% agreement**, and it disagrees with R seed 1234 / 2
  (52.7% / 53.4%) for the same reason those two disagree with seed 1. Baseline
  (same 2664 anchor cells, clusters ARI 1.0) and R's steps 7-8 on R's own inputs
  (labels 100% identical, `mat.adj` to 1e-15) match exactly; the divergence
  between runs comes only from Monte Carlo noise in segmentation (per-cell
  correlation 0.958) being amplified by #12 and by the choice of min-score
  clusters at step 8 (R's seed-1234 "aneuploid" set is the union of two zero-score
  clusters, 3659 cells; compat's pair has 1965).

*Ablation caveat:* the `k=2 only` / `label rule only` rows of the ablation script
are not independent of each other (the label rules are tied to the number of
clusters; the Python rule leaves middle clusters unlabelled at k>2), so only the
single-factor linkage and score-denominator rows are clean.

**Lin is unreproducible even by R.** Its consensus correlation sits on the 0.6
cliff of #12. R with seed 1234 gives 0.612 (all diploid); R with seeds 1, 2, 3
gives 0.487 / 0.466 / 0.548 and 366-412 aneuploid cells. r_compat gives 0.578,
inside R's own range, so its 293 aneuploid calls are an equally valid draw.

## R is not always right

On Xenium there is no annotation to judge against, and R's two modes (all diploid vs ~3.6k aneuploid) differ by seed alone, so neither can be called R's answer.

These are cases where agreement with the dataset's cell-type annotation
(`Malignant`/`Tumor` = aneuploid) favours the Python default over R. The
annotations come from the original publications and are not independent of CNV
inference, so treat this as a proxy, not ground truth:

* BCC su003: R labels tumour cells diploid and immune cells aneuploid (#6).
* Lin 2020: R labels all cells diploid for seed 1234, including 844 annotated
  malignant cells (#12).

The opposite case: Jerby-Arnon, where PCA (#1) makes the Python default invert
the call and R is right.
