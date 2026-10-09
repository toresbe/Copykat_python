# Improving CopyKAT's tumour/normal calls — overnight study, 2026-10-07

## Summary

CopyKAT's step 4 picks its normal reference as the Ward cluster whose profile is flattest
(smallest GMM sigma). That effectively picks the **majority** population, and inverts every
call when the tumour dominates. Its steps 7–8 then split cells in two by Ward clustering,
which in immune-rich samples can follow cell type rather than copy number.

Two opt-in changes fix much of this:

- **Marker anchor:** use the largest immune (else endothelial) marker-defined population as the
  normal reference. These cell types are not malignant in solid tumours.
- **Arm-correlation call (F3):** call each cell by correlating its chromosome-arm profile with
  the tumour's arm-level consensus, against a null taken from the reference cells.

Results on data never used for development:

| test set | samples / studies | CopyKAT default | marker anchor + F3 | Δ per sample [95% CI] | inverted samples |
|---|---:|---:|---:|---|---|
| 3CA new studies (pre-registered) | 467 / 43 | 0.801 | 0.849 | +0.047 [+0.024, +0.078] | 29 → 5 |
| ↳ same, non-anchor cells only | | 0.776 | 0.807 | +0.031 [+0.011, +0.056] | 24 → 5 |
| 3CA sealed holdout (B6 anchor) | 261 / 21 | 0.810 | 0.854 | +0.045 [+0.020, +0.092] | 8 → 0 |
| ScPCA neuroblastoma | 21 scorable | 0.777 | 0.833 | +0.055 [+0.010, +0.103] | 1 → 0 |
| ScPCA Ewing sarcoma | 6 | 0.454 | 0.601 | +0.147 [−0.006, +0.385] | 2 → 1 |

The metric is mean balanced accuracy per sample (malignant vs not). For 3CA, CIs come from a
study-level cluster bootstrap. Per-study means improve more: +0.087 on the pre-registered
test, +0.109 on the holdout. 30 of 43 studies improved and 13 got worse.

Most of the gain comes from the anchor, i.e. from removing inversions. F3 adds a smaller, less
certain improvement on top: +0.023 [+0.010, +0.037] over CopyKAT's own final call with the
same anchor.

## Data

- **3CA** (Curated Cancer Cell Atlas): 90 solid-tumour studies downloaded; archives are in
  `../original_archives`, extracted to `../3ca`. UMI-count 10x data only (no Smart-seq TPM or
  TP10K). Samples need ≥ 500 cells and 5–95% malignant cells. Cells with an empty `cell_type`
  are not scored. This was a bug in the first evaluations; fixing it changes no conclusion.
- **Splits:**
  - development: the original 11 studies (77 samples), heavily inspected;
  - pre-registered test: all 43 new studies (467 samples), frozen rules evaluated once;
  - sealed holdout: a random half of the new studies by study (`split.json`, seed 20261007),
    used once, for the overnight refinements;
  - `dev2`: the other half, used for development after the pre-registered test.
- **ScPCA** (`../scpca`): only projects with OpenScPCA tumour labels are used.
  - Ewing SCPCP000015: tumour = `tumor*`.
  - Neuroblastoma SCPCP000004: tumour = `Neuroendocrine`, NBAtlas convention, decided before
    seeing results.
  - Converted by `scpca_to_3ca.py`. Samples need ≥ 200 cells after CopyKAT QC.

## Method of the harness

- `harness.py` runs CopyKAT (GPU backend, snapshot of commit 14aae8d in `code/`) once as-is,
  then once per step-4 cluster with that cluster forced as the reference.
- It captures the step-6 anchor-relative profiles per cell, as chromosome-arm and 10 Mb means.
- Any anchor rule × final-call rule can then be scored offline.
- Offline re-baselining (`offline.py`) approximates a rerun with a different reference
  (median per-cell agreement 99.1%). It was used for development only; final numbers use real
  forced runs.

## The rules

- **B0 (pre-registered):**
  - Anchor: the step-4 cluster with the highest fraction of "immune" cells (≥ 3 of PTPRC,
    LAPTM5, CORO1A, CD53, LCP1, CD52, ARHGDIB detected), if that fraction is ≥ 0.5; else
    CopyKAT's sigma cluster.
  - Final call F3: weighted correlation of each cell's arm profile with the mean of the
    top-10% arm-energy cells, called aneuploid above the 99th percentile of the anchor cells.
- **B6 (overnight refinement, in the patch):**
  - the immune population with the **most immune cells** among clusters ≥ max(20, 1%) of
    cells, which avoids tiny tumour–immune doublet clusters;
  - else the same rule with endothelial markers (PECAM1, VWF, CDH5, CLDN5, ESAM, EMCN, PLVAP);
  - else the cluster with ≥ 3× enriched mean immune-marker count (scale-free, which helps
    single-nucleus data);
  - else sigma.
  - On development data B6 beat B0 by +0.023 [+0.009, +0.041]. **On the holdout it tied:**
    +0.002 [−0.007, +0.018]. Treat B0 and B6 as equivalent. B6 is in the patch for its
    robustness to tiny clusters and single-nucleus data, not for a demonstrated accuracy gain.
- F3 is insensitive to its two parameters: threshold 97–99.5th percentile × consensus 5–20%
  all give 0.873–0.881 on development data.

## What did not work (development data)

- **Label-free orientation:** the idea was to use the log-scale asymmetry of losses vs gains.
  Gains were often larger than losses, so it was rejected before testing.
- **Union of immune clusters, or all immune cells, as the reference:** much worse (0.661–0.842).
  F3 needs a *homogeneous* reference, because its null distribution comes from the anchor.
- **Intersecting calls across several immune anchors:** collapsed to "all diploid".
- **Iterative consensus; consensus seeded from CopyKAT's split:** no gain, or worse
  (−0.022, with inversions going from 4 to 16).
- **Robust arm medians over 10 Mb chunks:** no gain. Chunk-level correlation: worse.
- **F1** (Ward on arm means) and **F2** (arm-energy threshold): worse than F3.

## Limitations

1. **Same-lineage normal cells:** relative to an immune reference, the tumour's apparent CNA
   profile contains part of its lineage expression programme. Normal epithelium in carcinomas
   (e.g. Ji 2020 skin cSCC) is then flagged too; 84% of "normal epithelial" cells are flagged
   on development data. CopyKAT's default flags them as well. Either the labels are wrong or
   the method confounds lineage; the data cannot tell which.
2. **Tumours with few CNAs** (e.g. Song 2022 prostate): no anchor choice helps; even the best
   possible anchor stays poor.
3. **Tumour-dominated samples without marker-defined normal cells** (neuroblastoma,
   neuroendocrine, brain) fall back to sigma. On the holdout, the sigma/enrichment path scores
   0.686, against 0.866 when a marker population is found. The path is reported as a
   confidence indicator. It separates good from bad samples better than CopyKAT's own
   low-confidence flag on the holdout, but not on development data, and neither flags most of
   the poor samples.
4. **Label circularity:** 3CA labels are author annotations, checked with inferCNA, so partly
   CNA-based. The marker anchor uses marker genes, as many annotations do. The "non-anchor
   cells only" metric removes the anchor's own cells from scoring. The gain survives (+0.031),
   but is smaller.
5. **Not for haematological malignancies** (the tumour cells are immune cells), by design.

## The patch

`anchor-markers-arm-correlation.patch` (+189/−4 lines; applies cleanly to commit 14aae8d,
not applied to the repo):

- new module `copykat_py/anchor.py`;
- `copykat(anchor="sigma"|"markers", final_call="clusters"|"arm_correlation")`;
- CLI flags `--anchor` and `--final-call`;
- the defaults are unchanged.

Validated by `validate_patch.py` on one sample per anchor path (immune, endothelial,
enrichment, sigma):

- default arguments give calls identical to the unpatched code;
- the new mode reproduces the evaluated B6+F3 calls exactly (100.0% agreement).

The runtime info records `anchor_path`. With `anchor="markers"`, only references from the
sigma/enrichment paths carry the `low.conf` label, and CopyKAT's GMM fallback is not used
(the evaluated rule never used it).

## Reproduce

```
PYTHONPATH=code:. python harness.py results [STUDY ...]   # forced-anchor runs (GPU)
python marker_scores.py endo                              # endothelial marker counts
python evaluate.py results eval.csv                       # pre-registered rules
python final_eval.py holdout final_holdout.csv            # final comparison (also: dev, ewing, nb)
python stats.py final_holdout.csv B6F3 default            # study-level cluster bootstrap
```

Key result files: `prereg_corrected.csv`, `prereg_original.csv`, `fair_new.csv`,
`final_{dev,holdout,ewing,nb}.csv`. Protocol: `FINAL_PROTOCOL.md`, `split.json`,
`holdout_opened_at.txt`.

## Appendix: final evaluation protocol (fixed before the holdout was opened)


Written 2026-10-07, before any result from the holdout studies (split.json "holdout",
21 studies) or the ScPCA sets was computed for these rules.

## Candidate (single, declared in advance)
**B6 + F3**
- Anchor B6: among step-4 Ward clusters with >= max(20 cells, 1% of cells) and >= 50% of cells
  "immune" (>= 3 of PTPRC, LAPTM5, CORO1A, CD53, LCP1, CD52, ARHGDIB detected), the one with
  the most immune cells. Else the same with endothelial cells (>= 3 of PECAM1, VWF, CDH5,
  CLDN5, ESAM, EMCN, PLVAP). Else the cluster with the highest mean immune-marker count if
  >= 3x the mean of all other cells, >= 1 on average and >= 20 cells. Else CopyKAT's sigma cluster.
- Final call F3 (unchanged from the pre-registered rule): weighted correlation of each cell's
  arm-level profile with the mean profile of the top-10% arm-energy cells; aneuploid if above
  the 99th percentile of the anchor cells' correlations.

## References
- default: CopyKAT GPU backend, unmodified.
- B0 + F3: the pre-registered rule (best immune-majority cluster by fraction, else sigma).
- B6 + ck: B6 anchor with CopyKAT's own steps 7-8 (decomposition only).

All rules use the real forced-run profiles for their anchor cluster (no offline approximation).

## Data
1. 3CA holdout: 21 studies in split.json "holdout" (primary).
2. ScPCA Ewing sarcoma (SCPCP000015, OpenScPCA "tumor*" labels) and neuroblastoma
   (SCPCP000004, OpenScPCA "Neuroendocrine" = tumour) — secondary, independent labels.
Samples need >= 200 cells after CopyKAT QC. Cells with empty labels are not scored.

## Metrics and tests
Balanced accuracy per sample, all cells and excluding the candidate's anchor cells (for
default vs candidate on identical cells). Mean per sample and per study; 95% CIs from a
study-level cluster bootstrap (10,000 resamples); counts of inverted (< 0.3) and poor (< 0.7).
Primary claims: B6+F3 vs default and B6+F3 vs B0+F3 on the 3CA holdout.
Results are reported as they come out, including if negative. No rule changes after opening.
