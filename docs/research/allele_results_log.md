# Allele orientation: results log

Chronological record of every evaluation, including negative results (verbatim). Summary: [allele_orientation.md](allele_orientation.md).


## v1 (pre-registered) - BCC06post: NEGATIVE
Goal B AUC 0.564, BA 0.501; goal A no gain. Cause: germline ASE/artefacts shared by all cells.

## v2 (developed on BCC06post)
| sample | allele AUC | allele BA | default | B0F3 | orient_B0F3 | orientsig_B0F3 | allele_anchor_F3 |
|---|---|---|---|---|---|---|---|
| BCC06post (dev, library cells) | 0.912 | 0.838 | 0.955* | 0.938* | 0.938* | - | - |
| BCC06 (test) | 0.513 | 0.520 | | | | | |
| Rao livMET (test) | 0.539 | 0.506 | 0.037 | 0.050 | 0.950 | 0.050 | 0.693 |
| Rao PriNET (test) | 0.428 | 0.493 | 0.043 | 0.676 | 0.324 | 0.324 | 0.373 |
| Lin P10 (test) | 0.554 | 0.508 | 0.026 | 0.149 | 0.851 | 0.851 | 0.256 |
| Song P4 (test) | 0.908 | 0.848 | 0.032 | 0.965 | 0.965 (orient_default 0.968, orientsig_default 0.968) | 0.965 | 0.965 |
| Dong T40 (test) | 0.999 | 0.998 | 0.001 | 0.995 | 0.995 (orient_default 0.999, orientsig_default 0.999) | 0.995 | 0.995 |
| Dong T27 (test) | 0.967 | 0.853 | 0.001 | 0.977 | 0.977 (orient_default 0.999, orientsig_default 0.999) | 0.977 | 0.977 |
(* CopyKAT run on the STARsolo library counts, not the harness.)
Goal A v2 (profile gate) on BCC06post and BCC06: no gain (Pearson 0.154 vs 0.200; 0.188 vs 0.210).

Reading: per-cell allele scores without phasing are unreliable at 10x depth. livMET's correct
flip was not significant (gated rule does not flip); PriNET's wrong flip WAS significant, i.e.
the unphased score carries systematic non-CNA structure (likely cell-type-specific ASE).
Diagnostic (post hoc): a plain sum of window z gives AUC 0.84 on BCC06 / 0.82 on BCC06post,
better than the PC1 score on BCC06; adopted as the v3 primary score.

## v3 / v3b development (all v2 samples; full 22-autosome panel)
| sample | v3 AUC sum/pc | default | orientsig_default | orientgrp_default (D an/dip, p) | B0F3 | orientgrp_B0F3 | allele_anchor_F3 | oracle |
|---|---|---|---|---|---|---|---|---|
| BCC06post | 0.812/0.939 | | | | | | | |
| BCC06 | 0.755/0.475 | | | | | | | |
| Rao PriNET | 0.576/0.650 | 0.043 | 0.957 | 0.957 (0.029/0.046, 0.021) | 0.676 | 0.676 | 0.758 | 0.915 |
| Rao livMET | 0.531/0.425 | 0.037 | 0.037 | 0.037 (0.017/0.027, 0.099) | 0.050 | 0.050 | 0.925 | 0.964 |
| Lin P10 | 0.793/0.582 | 0.026 | 0.974 | 0.974 (0.012/0.029, <0.001) | 0.149 | 0.851 | 0.922 | 0.922 |
| Song P4 | 0.852/0.825 | 0.032 | 0.968 | 0.032 (0.031/0.040, 0.149) | 0.965 | 0.965 | 0.965 | 0.965 |
| Dong T40 | 0.977/0.998 | 0.001 | 0.999 | 0.999 (0.017/0.054, <0.001) | 0.995 | 0.995 | 0.995 | 0.995 |
| Dong T27 | 0.722/0.733 | 0.001 | 0.999 | 0.001 (0.043/0.030, 0.641) | 0.977 | 0.977 | 0.987 | 0.987 |

## v3 fresh TEST set (pre-registered; primary: orientsig_default, orientgrp_default, allele_anchor_F3)
| sample | v3 AUC sum/pc | default | orientsig_default | orientgrp_default (D an/dip, p) | B0F3 | orientsig_B0F3 | orientgrp_B0F3 | allele_anchor_F3 | oracle |
|---|---|---|---|---|---|---|---|---|---|
| Lin P03 | 0.955/0.982 | 0.731 | 0.731 | 0.731 (0.068/0.039, 1.000) | 0.831 | 0.831 | 0.831 | 0.831 | 0.831 |
| Lin P06 | 0.819/0.967 | 0.977 | 0.977 | 0.977 (0.038/0.028, 0.834) | 0.955 | 0.955 | 0.955 | 0.808 | 0.967 |
| Lin P07 | 0.911/0.967 | 0.955 | 0.955 | 0.955 (0.055/0.031, 1.000) | 0.947 | 0.947 | 0.947 | 0.949 | 0.954 |
| Lin P08 | 0.981/1.000 | 0.993 | 0.993 | 0.993 (0.055/0.035, 0.970) | 0.970 | 0.970 | 0.970 | 0.971 | 0.971 |
| Lin P09 | 0.904/0.937 | 0.987 | 0.987 | 0.987 (0.058/0.021, 1.000) | 0.959 | 0.959 | 0.959 | 0.965 | 0.965 |
| Dong T19 | see log | 0.998 | 0.998 | 0.998 (0.055/0.037, 0.977) | 0.983 | 0.983 | 0.983 | 0.904 | 0.991 |
Suspended 2026-10-08 00:44 at the user's request; Dong T44 (interrupted) and T92 remain. Resume: RESUME.md.
| Dong T44 | 0.975/0.702 | 0.630 | 0.630 (ungated orient: 0.370) | 0.630 (0.075/0.073, 0.683) | 0.792 | 0.792 | 0.792 | 0.792 | 0.792 |
| Dong T92 | 0.841/0.532 | 0.988 | 0.988 | 0.988 (0.038/0.041, 0.497) | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 |
(Resumed 2026-10-08 01:48; all 8 test samples complete.)

## Verdict (2026-10-08)
v3 fresh test set (8 samples, pre-registered): none of the 8 happened to be inverted by CopyKAT,
so the test measures SAFETY, not benefit.
- Primary orientation rules (significance-gated per-cell rule; group-level phased test): 0 flips
  in 8 samples -> no harm. The ungated rule made 1 harmful flip (Dong T44: 0.630 -> 0.370), so the
  significance gate is necessary.
- allele_anchor_F3: mean 0.900 vs 0.927 for the marker anchor (B0F3); worse on Lin P06 (0.808 vs
  0.955) and Dong T19 (0.904 vs 0.983), never better by more than 0.006. Not a safe replacement
  for the marker anchor.
Benefit evidence comes from the inverted samples, all used for v3 development (but pre-registered
tests for v2):
- v2 pre-registered: fixed Song P4 (0.032->0.968), Dong T27 (0.001->0.999), Dong T40
  (0.001->0.999), Lin P10 via B0F3 (0.149->0.851); harmed Rao PriNET (0.676->0.324); livMET
  "fix" not significant.
- v3 gated rule (dev): fixes PriNET (0.043->0.957), Lin P10 (0.026->0.974), Song P4, Dong T27,
  Dong T40; livMET not fixed; no harmful flips on any dev or test sample.
Conclusion: allele information is a useful, low-risk ORIENTATION CHECK for CopyKAT's final
tumour/normal call (fixes inversions when the tumour has sizeable CNAs; abstains otherwise). It is
not a reliable per-cell classifier at 10x depth, and not a better normal anchor. Remaining gap: the
v3 rule's benefit needs confirmation on fresh, inverted samples (candidates: other public-read
3CA samples with default < 0.6).

## Fresh INVERTED test set (pre-registered 2026-10-08 03:40)
| sample | allele AUC sum/pc | default | orientsig_default | orientgrp_default | B0F3 | orientsig_B0F3 | orientgrp_B0F3 | allele_anchor_F3 | oracle |
|---|---|---|---|---|---|---|---|---|---|
| Geist T59 | 0.916/0.924 | 0.013 | **0.987** | **0.987** (p<0.001) | 0.962 | 0.962 | 0.962 | 0.966 | 0.966 |
| Dong T69 | 0.606/0.413 | 0.193 | **0.807** | 0.193 (p=0.32) | 0.748 | 0.748 | 0.748 | 0.748 | 0.766 |
| Ji P2 | 0.46/0.354 | 0.148 | 0.148 | 0.148 | 0.845 | **0.155 (harm)** | 0.845 | 0.145 | 0.848 |
Ji P2: tumour cells show no allelic imbalance (CNA-quiet cSCC); the significant B0F3 flip came from
the HLA window and two others (68% of the difference from 3 windows) -> cell-type ASE/artefact.
So the pre-registered orientsig: 2 fixes, 1 harm on fresh inverted samples.

## v4 (post hoc after Ji P2; HLA excluded + concentration check) on all 17 seen samples
8 fixes, 0 harms, mean change +0.208 (orientsig: 8 fixes, 1 harm). See results/explore_v4.csv.
Fresh v4 safety+power test running (DESIGN.md).

## v4 fresh SAFETY + POWER test (pre-registered 2026-10-08 07:10) - v4 FAILS SAFETY
8 fresh samples (Ji2020 P4/P6/P7/P10, Geistlinger T89/T90, Dong T71/T75), CopyKAT default and B0F3
calls, actual and deliberately swapped (results/eval_v4_*.csv, results/eval_grp_fresh.csv):
| rule | harmful flips (actual calls) | swapped (wrong) call sets restored |
|---|---|---|
| v4 (per-cell score, HLA excl., concentration check) | **4 of 16** (Ji P6, Ji P7, both call sets) | 6 of 16 |
| orientgrp (pre-registered v3b group test) | **0 of 16** | 5 of 16 (Geist T89, T90 both; Dong T71 default) |
In the Ji2020 cSCC samples the per-cell allele score is anti-correlated with the labels (AUC 0.45-0.47):
cells of the large dendritic / Langerhans compartment score high because the leave-one-out pooled
major allele at their cell-type-specific genes is set by their own germline allele-specific
expression. The per-cell score cannot separate that from copy number; the concentration check
does not catch it (spread over many genes).

## FINAL VERDICT on idea #2 (2026-10-08)
- Per-cell allele scores (v1-v4): NOT safe as a decision rule at 10x depth; rejected.
- Group-level phased test (orientgrp, pre-registered before every test sample): no harmful flip on any
  sample (25 samples x 2 call sets; 0/16 on the last fresh set). Fixes inversions when the tumour has
  substantial CNAs: Rao PriNET, Lin P10 (default and B0F3), Dong T40, Geist T59 (fresh, pre-registered),
  and restores swapped calls in Geist T89/T90, Dong T71. Abstains on CNA-quiet tumours (Ji2020 cSCC,
  Dong T69/T75, Rao livMET, Song P4, Dong T27). Recall ~ half of inversions; precision 100% so far.
- Deliverable: allele_check.py (standalone; cellsnp-lite output + Eagle2 phase + CopyKAT prediction ->
  oriented prediction + JSON report), implementing exactly the pre-registered orientgrp rule.
- Not adopted: allele-chosen anchor (worse than marker anchor), profile gating (goal A, no gain).

## v5 Numbat-style phased-BAF HMM (pre-registered 2026-10-08; evaluated once on the 25 harness samples)
Orientation (flip iff F_dip - F_an >= 0.02; results/v5_orientation.csv):
- harmful flips 0 of 50 actual call sets; 9 actual wrong call sets fixed (PriNET default, livMET default
  and B0F3, LinP10 default and B0F3, SongP4 default, DongT40 default, GeistT59 default, DongT69 default);
  26 of 39 deliberately swapped wrong call sets restored (orientgrp: 5 of 16 on the fresh set).
- Mean balanced accuracy over the 25: default 0.597 -> default+v5 0.849; B0F3 0.831 -> B0F3+v5 0.895
  (oracle F3 0.914). Samples below 0.5: default 9, default+v5 2, B0F3 2, B0F3+v5 0.
- Missed: DongT27 default (F 0.148 vs 0.160: both groups carry imbalance), DongT44/DongT71 (both groups
  imbalanced; tumour-dominated samples where CopyKAT's "diploid" group is mostly tumour too).
- Abstains correctly on CNA-quiet tumours: all Ji2020 cSCC samples have F < 0.03 in both groups.
Per-cell (secondary; results/v5_cells.csv): allele LLR alone 0.815, consensus 0.874 vs B0F3 0.831
(oracle 0.914). Large gains on DongT44 (0.792 -> 1.000), DongT71 (0.582 -> 0.942), LinP03 (0.831 -> 0.965),
LinP10; but harms where the tumour has almost no imbalance (JiP6 0.842 -> 0.606, JiP7 0.968 -> 0.862) and
on some neuroblastomas (DongT92 0.982 -> 0.817, DongT27 0.977 -> 0.819). The per-cell consensus is not
safe as specified; the orientation rule is.
Caveats: the 25 samples are enriched for CopyKAT failures (selected for that), from 6 studies; all had
been seen by earlier allele versions, but v5 was fully specified before running it on any of them.
Study-level bootstrap (6 studies, 10k resamples): B0F3+v5 - B0F3 = +0.064 [+0.000, +0.189];
default+v5 - default = +0.252 [+0.123, +0.524]. (Lower bound 0 for B0F3: v5 never hurts; gains come
from the Rao and Lin studies.)

## GPU allele counter (gpucount/; 2026-10-08)
Alignment-free: 31-mers spanning 4.27 M gene-body 1000G SNPs (REF/ALT), 223 M k-mers kept after a
GPU genome-wide uniqueness filter (index build 2 min, 2.6 GB). Reads -> barcode whitelist (exact) ->
all 31-mers -> GPU binary search -> per-read allele (conflicts dropped) -> UMI majority -> cellsnp-lite
format. SRA input decoded by 16 parallel fastq-dump workers (the single-threaded decoder was the
bottleneck); the GPU is mostly idle.
Validation vs STAR+cellsnp (PriNET): total UMIs ratio 1.02; per-cell depth r = 0.999; per-SNP alt
fraction r = 0.987 (depth >= 20, mean |diff| 0.021); per-SNP depth r = 0.59 (low-depth noise, marginal
read definitions).
Downstream v5 on GPU counts, 3 samples: identical orientation decisions (incl. swapped calls);
per-cell allele LLR BA 0.734 / 0.970 / 0.966 vs STAR 0.733 / 0.965 / 0.896 (PriNET / LinP06 / DongT19).
Time (allele counts from .sra): PriNET 58 s, LinP06 2.8 min, DongT19 7.1 min, vs ~5-6 / ~19 / ~45 min
for parallel fastq-dump + STARsolo + cellsnp-lite.
FASTQ.gz input (BCC06, 121 M read pairs, 10x 5'): 3.7 min vs ~32 min (STARsolo + cellsnp-lite);
86% of STAR's UMIs (SNPs outside gene bodies and reads spliced at the SNP are not indexed); per-cell
depth r = 0.975; per-SNP alt fraction r = 0.978 (mean |diff| 0.016). FASTQ parsing is still single-
threaded (133% CPU), so there is headroom.
v5 results files were regenerated after the validation runs overwrote them (identical numbers).
FASTQ path parallelised (2026-10-08): per file igzip | fqseq (C extractor of sequence lines at fixed
width) -> fixed-size chunk reads in a prefetch thread -> GPU (ASCII -> 2-bit on the GPU). BCC06,
121 M read pairs: 221 s -> 134 s (one parser process per file) -> 94 s (igzip|fqseq) -> 68 s
(prefetch thread), byte-identical output at every step. Dead ends, measured and reverted: a parser
pool fed by one reader (134 s, the reader copies the text several times), per-block shared memory
(175 s, page-fault cost), k parsers each decompressing the whole file (293 s, memory bandwidth).
Floor: igzip decompression of R2 ~39 s; the GPU side is now about as long.
Read recovery (BCC06, share of STAR+cellsnp UMIs): gene-body index 85.6% -> + exon-junction k-mers
(736k k-mers from 52k SNP-transcript junction windows; absent from the genome for both alleles) 88.0%
-> genome-wide SNPs (6.95 M SNPs, 361 M k-mers, 4.2 GB) + junctions 93.4%. Allele-fraction agreement
unchanged (r 0.978, mean |diff| 0.016). Gap breakdown before: 7.3% SNPs not indexed, 3.6% near exon
boundaries, 6.9% elsewhere (read errors / SNP near read ends), offset by +3.4% the GPU finds and STAR not.
Downstream v5 with the genome-wide+junction index: identical orientation decisions on PriNET and DongT19;
per-cell allele BA 0.688 / 0.959 (gene-body GPU 0.734 / 0.966; STAR 0.733 / 0.896). Time +10-30% (larger
index). Default index is now index_all_splice.

## Branch port and shipped configuration (2026-10-09)
Port of the research code to copykat_py.allele verified: v5 F values and flips identical on all 50 call
sets; GPU counts byte-identical (FASTQ and SRA); k-mer index byte-identical; phasing identical.
Shipped configuration = refined marker anchor (B6, with enrichment fallback) + arm correlation, as
committed, with and without copykat_py.allele.orient, on the 25 samples (no parameter changed):
default 0.597 (9 < 0.5); default + allele 0.849 (2); B6F3 0.871 (1); B6F3 + allele 0.907 (0); oracle 0.914.
Allele orientation: 0 harmful flips / 50; 8 wrong call sets fixed; 27 / 40 swapped call sets restored.
End-to-end copykat-py CLI on Lin2020 P10: default 0.026 -> 0.974 with --allele-counts; B6F3 0.922 unchanged.
