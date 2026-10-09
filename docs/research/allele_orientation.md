# Allele-based orientation of CopyKAT calls: evidence

Summary of the study behind `copykat_py.allele` (October 2026). Every rule was written down before it
was evaluated; the full chronological record, including all negative results, is in
[allele_design_log.md](allele_design_log.md) (pre-registrations) and
[allele_results_log.md](allele_results_log.md) (results). Usage: [../allele_orientation.md](../allele_orientation.md).

## Problem

CopyKAT's normal reference ("anchor") decides which side of its final split is called tumour. When
the anchor is wrong — typically tumour-dominated samples without marker-defined normal cells — the
split is right but the labels are swapped. The marker anchor (`--anchor markers`,
[anchor_study.md](anchor_study.md)) fixes most of these from expression, but not those without immune
or endothelial cells. Allele imbalance is an independent, cell-type-free signal.

## Data

25 samples from 6 public studies with tumour/normal labels (3CA) and public reads with cell barcodes:
Rao2020 (2 neuroendocrine), Lin2020 (6 pancreas), Song2019 (1 lung), Dong2020 (8 neuroblastoma),
Geistlinger2020 (3 ovarian), Ji2020 (5 skin cSCC). The first ones were chosen because CopyKAT's default
calls were inverted there, so the set is enriched for CopyKAT failures. CopyKAT runs and labels are those
of the anchor study; allele counts from STARsolo + cellsnp-lite (or original Cell Ranger BAMs).

## What failed (and why it matters)

| Version | Idea | Outcome |
|---|---|---|
| v1 | per-cell imbalance vs 50:50 | negative: imbalance shared by all cells (germline allele-specific expression) |
| v2 | per-cell, relative to the other cells | works where tumours carry large CNAs; one significant wrong flip (Rao PriNET) |
| v3 | + Eagle2 reference phasing | per-cell scores still confounded |
| v4 | + HLA excluded, "spread" check | failed a fresh safety test: 4 harmful flips in 16 call sets (Ji2020 skin; dendritic/Langerhans cells' allele-specific expression mimics imbalance per cell) |
| v3b | group-level phased test | safe (0 harmful flips in 25 samples x 2 call sets) but low power |
| **v5** | **phased-BAF HMM per group (shipped)** | see below |

Per-cell allele calls are therefore not offered: at 10x depth (tens to hundreds of informative UMIs per
cell) they are not safe. Group-level evidence pools hundreds of cells and is.

## v5 result (rule fixed before evaluation)

Flip CopyKAT's calls iff F(diploid group) - F(aneuploid group) >= 0.02, F = fraction of the covered
genome in HMM-detected imbalanced segments of the group's pseudobulk.

Research evaluation (immune-only marker anchor "B0" + F3): 0 harmful flips in 50 call sets, 9 actual
inversions fixed, 26 of 39 deliberately swapped call sets restored; B0F3 0.831 -> 0.895 mean balanced
accuracy; study-level bootstrap gain +0.064 [+0.000, +0.189].

**Shipped configuration** (branch code: `--anchor markers --final-call arm_correlation`, i.e. the
refined marker anchor "B6" with its enrichment fallback, plus `copykat_py.allele.orient`), same 25 samples:

| Configuration | Mean balanced accuracy | Samples < 0.5 |
|---|---|---|
| CopyKAT default | 0.597 | 9 |
| default + allele orientation | 0.849 | 2 |
| marker anchor + arm correlation | 0.871 | 1 |
| **marker anchor + arm correlation + allele orientation** | **0.907** | **0** |
| best possible anchor + arm correlation (ceiling, uses labels) | 0.914 | 0 |

Allele orientation: 0 harmful flips in 50 call sets; 8 actual wrong call sets fixed; 27 of 40
deliberately swapped (wrong) call sets restored. Misses: tumour-dominated neuroblastomas where both of
CopyKAT's groups carry the same imbalance (Dong T27, T44, T71). Abstains (F < 0.03 in both groups) on the
copy-number-quiet Ji2020 cSCC samples.

## GPU allele counter

Alignment-free counting (31-mer lookup on the GPU against SNP-spanning k-mers that are unique in the
genome; exon-junction k-mers included) replaces STARsolo + cellsnp-lite: 6-28x faster, 93% of their
UMIs, per-SNP allele fractions r = 0.98 (mean |diff| 0.016-0.021), and identical orientation decisions on
the 3 samples evaluated end to end (PriNET, Lin P06, Dong T19).

## Port verification

The branch code reproduces the research code exactly: orientation F values and decisions identical on
all 50 call sets; GPU counts byte-identical (FASTQ and SRA input); k-mer index byte-identical; phasing
identical (19,163/19,163 SNPs). End-to-end `copykat-py` run on Lin2020 P10: default 0.026 -> 0.974
with `--allele-counts/--allele-phase`; marker anchor + arm correlation 0.922, left unchanged.

## Caveats

- 25 samples from 6 studies, enriched for CopyKAT failures; means are not population estimates.
- The v5 design was fixed before running it, but its samples had been seen by earlier versions.
- Needs reads; GRCh38; 10x 3' and 5' chemistries tested.
