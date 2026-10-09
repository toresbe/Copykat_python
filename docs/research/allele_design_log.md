# Allele orientation: pre-registration log

Chronological record of the designs, written before each evaluation (verbatim). Summary: [allele_orientation.md](allele_orientation.md).


## Why
Expression-based copy-number (CopyKAT) sees a region's dosage only through expression,
which also changes with cell type. Heterozygous SNPs give an independent signal: in a
region where a tumour lost or gained one parental copy, the two alleles of a SNP are no
longer read ~50:50. Cell type cannot fake that. (Numbat/CaSpER use the same principle.)

## Data (public reads, DNA truth already in wgsbench)
- BCC06: SRR8315756 (su006 pre, CD45- "tumor" sort library: 1401 tumour, ~180 normal cells)
- BCC06post: SRR8315759 (su006 post tumor library: 320 tumour, ~370 normal cells)
- later if worthwhile: gastric lines (tumour-only), MM199 (no cell barcodes in SRA; bulk only)
Cell labels = GEO metadata clusters (Tumor_* = tumour; everything else = normal).

## Pipeline
STARsolo (GRCh38/GENCODE v44, 10x v2, whitelist = that library's GEO barcodes) ->
cellsnp-lite mode 1 on 1000 Genomes AF>=5% SNPs (minMAF 0, minCOUNT 1) -> per-cell
ref/alt UMI counts. No population phasing panel.

Heterozygous SNPs (no labels used): pooled over all cells, DP >= 8 and min(ref, alt) >= 2.

Phasing within a window without a panel = "cross-fitting": split cells randomly in two
halves (seed 0); in half A, call the majority allele of each SNP; measure the fraction of
half-B reads carrying the half-A majority allele (b). Swap halves, average. Under no
imbalance b ~ 0.5 in expectation (no winner's-curse bias); under imbalance b > 0.5.

Windows: non-overlapping 10 Mb genomic windows (fixed; not tuned).

## Goal A - profile accuracy vs DNA truth (same metrics as wgsbench/eval_bench.py)
Tumour pseudobulk = the labelled tumour cells (same as the CopyKAT runs being scored).
Per 100 kb truth bin, scores (bin inherits its 10 Mb window's b):
  A1 expression : CopyKAT py_gpu_ref pseudobulk e (existing)
  A2 allele     : b - 0.5 (unsigned; evaluated only for "any CNA" AUC)
  A3 combined   : e * max(b - 0.5, 0)   (parameter-free gate: shrinks balanced regions)
Reported: Pearson, AUC gains, AUC losses, max F1 (A1 vs A3); "any CNA" AUC (|e|, b, |e|*b).

## Goal B - tumour vs normal per cell (the original classification problem)
Unsupervised: a window is "imbalanced" if its cross-fit b over all cells is > 0.5 with
one-sided binomial p < 1e-3 (pooled half-B reads, both swaps). Per cell, the phase comes from
all OTHER cells' majority allele (leave-one-out). Per-cell LLR = sum over the cell's reads in
imbalanced windows of log binomial likelihood under p = window b vs p = 0.5 (window b is a
diluted underestimate of the tumour's own BAF, which makes the rule conservative, not tuned).
Call: tumour if LLR > 0, normal if LLR <= 0, undetermined if < 10 informative reads.
Reported: AUC of the per-cell score vs labels; balanced accuracy of the LLR call; and
the same for CopyKAT default and for marker anchor + F3 on the same cells.

## v1 outcome (BCC06post, pre-registered rule) - NEGATIVE
Goal B: AUC 0.564, balanced accuracy 0.501. Goal A: A3 no better than A1 (Pearson 0.181 vs 0.200).
Diagnosis: window imbalance is correlated between tumour and normal cells (r = 0.77): it is
mostly germline allele-specific expression (eQTL/imprinting) and genotyping artefacts (pooled
alt fraction 5% quantile = 0.07), which are present in every cell, not copy number. Windows at
b = 1.00 also made log(2(1-b)) = -inf.

## v2 (developed on BCC06post ONLY; BCC06 and other patients are the test)
Het SNPs: pooled DP >= 8, min(ref, alt) >= 2, and minor-allele fraction >= 0.1.
Key change: compare each cell with the other cells, not with 0.5. Germline ASE shifts every
cell equally; a tumour CNA shifts tumour cells only.
Goal B v2 (unsupervised): per SNP, leave-one-out pooled alt fraction q; major allele = side of
q > 0.5. Per cell x 10 Mb window: z = (k - E) / sqrt(V), k = major-allele reads, E = sum n q_maj,
V = sum n q(1-q). Windows with >= 200 pooled reads. Cell score = projection on the first
principal component of the z matrix (cells x windows, z clipped to +-5), oriented so that
the loading sum is positive (tumour cells carry the pooled major allele in excess). Call
tumour if score > 0. Rationale: ASE/bursting noise is independent across windows; a clonal
CNA makes the same cells deviate in many windows.
Goal A v2 (uses the tumour/normal labels, as the pseudobulk does): per SNP
D = alt fraction(tumour) - alt fraction(normal), cross-fit (sign from half A of the cells, value
from half B); window d = read-weighted mean signed D; gate = max(d, 0); A3v2 = 2 + (e-2) * gate.

## v2 test set and comparisons (fixed 2026-10-07 before any test sample was analysed)
Test samples: BCC06 (su006 pre tumour library), Rao2020_Neuroendocrine PriNET + livMET
(3CA labels; CopyKAT marker+F3 = 0.363 there, oracle 0.94), and further public-read 3CA samples
added below BEFORE they are analysed. Labels: 3CA cell_type, "Malignant" = tumour, empty = not scored.
Reported per sample, on the same cells: balanced accuracy of
  (1) allele v2 alone (score > 0),
  (2) CopyKAT default, (3) CopyKAT markers + F3,
  (4) "allele orientation": markers + F3 calls, flipped if the mean allele v2 score of its
      aneuploid-called cells is below that of its diploid-called cells.
Added before analysis (all chosen ONLY because CopyKAT default was inverted there, default < 0.6
with oracle > 0.85, and public reads with cell barcodes exist): Lin2020_Pancreas P10
(SRR12273043), Song2019_Lung P4_Tumor (SRR7586090, original Cell Ranger BAM),
Dong2020_Neuroendocrine Group1 Tumor_27 (SRR10156297) and Tumor_40 (SRR10156299).

## 3CA test samples: evaluation on the anchor-study harness outputs (fixed before analysis)
Cells, labels and CopyKAT runs = anchor_study/results/<sample>.npz (same cells as REPORT.md).
Per sample, balanced accuracy of:
  default            : CopyKAT unmodified
  B0F3               : immune (marker) anchor + F3 (the pre-registered patch rule)
  oracle_F3          : best step-4 anchor + F3 (ceiling, uses labels)
  allele             : allele v2 score > 0
  orient_default     : default calls, flipped if mean allele score(aneuploid) < mean(diploid)
  orient_B0F3        : same flip rule on B0F3 calls
  allele_anchor_F3   : anchor = step-4 cluster with the lowest mean allele v2 score; then F3
Cells absent from the allele data get allele score 0.

## v3 fresh test set (fixed 2026-10-07 BEFORE v3 was written and before these were analysed)
v2 failed on BCC06 (per-cell AUC 0.51) and livMET had ~no per-cell allele signal (83 informative
UMIs/cell). v3 will add reference-panel phasing. All v2 samples become v3 development data.
Fresh v3 test samples = every other sample of the same public-read studies that the anchor
study scored: Lin2020_Pancreas P03 (SRR12273036), P05 (..38), P06 (..39), P07 (..40),
P08 (..41), P09 (..42); Dong2020_Neuroendocrine Group1 Tumor_19 (SRR10156296), Tumor_44
(SRR10156300), Tumor_92 (SRR10156304). (MET06 excluded up front: 16 lane-split runs.)

## v3 method (fixed before any v3 result)
Het SNPs as v2, autosomes only, phased with Eagle2 v2.4.1 against the 1000G high-coverage
phased panel (biallelic SNPs at our SNP list; hg38 genetic map). Reads recoded as haplotype-A
counts and summed in 2 Mb blocks; each block is then treated exactly as a v2 SNP (leave-one-out
pooled orientation; per-cell z per 10 Mb window, windows >= 200 reads).
Primary cell score = sum of window z (score > 0 -> tumour); secondary = v2 PC1 projection.
Orientation rules on the 3CA harness outputs (compare3ca.py):
  orient_*      : as v2 (flip if mean score(aneuploid calls) < mean score(diploid calls))
  orientsig_*   : flip only if that difference is significant: one-sided Welch t-test p < 0.01
  allele_anchor_F3 : as v2 with the v3 score
Lin P05 dropped from the v3 test set (before analysis): 3CA->GEO barcode alignment r = 0.973 (< 0.995; others >= 0.998).

## v3b group-level orientation (group_orient.py; fixed after v3 dev on BCC06post/BCC06/Rao,
## before any v3 test sample was analysed)
For CopyKAT's aneuploid- and diploid-called groups: per 10 Mb window, phased haplotype-A fraction,
cross-fitted between random halves of the group (sign from one half, signed deviation from 0.5 in
the other, both ways); D = read-weighted mean over windows. Flip the calls if D(diploid group) >
D(aneuploid group) with bootstrap-over-windows p < 0.05 (2000 resamples, seed 0).
Reported as orientgrp_default and orientgrp_B0F3, alongside the v3 rules above.
Primary v3 endpoints on the fresh test set (decided now): orientsig_default and orientgrp_default
(fixing CopyKAT default without markers) and allele_anchor_F3; secondary: the *_B0F3 variants.

## Fresh INVERTED test set (fixed 2026-10-08 ~03:40, before any of these was analysed)
Purpose: confirm the v3 rules' benefit on samples CopyKAT inverts. Selection: every not-yet-used
3CA/dev sample with CopyKAT default < 0.6 whose public reads carry 10x cell barcodes and whose
3CA cell names give barcodes. Screened out: Olalekan2021 (12-nt Seq-Well barcodes), Laughney2020
(sample names not mappable to GEO), Kim2020/Steele2020/Biermann2022/Pal2021/Pelka2021 (no public
reads), EGA/dbGaP studies, Yuan2018 (microwell), Yost su005/su006 (multi-sort libraries, barcode
collisions). Included:
  Dong2020 Group1 Tumor_69 (SRR10156301; default 0.193, oracle 0.766)
  Geistlinger2020 T59 (original Cell Ranger BAM of SRR12244038; default 0.013, oracle 0.966)
  Ji2020_Skin P2 (original BAM P2_cSCC_scRNA of SRR11832840; default 0.30, oracle 0.78) - scored
    only on cells of that tumour-tissue library ("P2_Tumor_*"), as the normal-tissue library is
    a separate 10x run with colliding barcodes.
Same pre-registered endpoints as the v3 test (orientsig_default, orientgrp_default,
allele_anchor_F3; secondary *_B0F3), all v3 methods unchanged.

## Inverted test set outcome (2026-10-08 ~06:10), then v4
Geist T59: default 0.013 -> orientsig 0.987, orientgrp 0.987. Dong T69: 0.193 -> orientsig 0.807
(orientgrp no flip). Ji P2: no allelic imbalance in tumour cells (CNA-quiet); orientsig left default
alone (0.148, no fix) but FLIPPED the correct B0F3 call (0.845 -> 0.155), p < 0.001: the difference
came from the HLA region and two other windows (68% of it from 3 windows), i.e. cell-type ASE/
mapping artefacts in myeloid cells. Bug found and fixed on the way: "NA" chromosome strings in the
lifted hg19 SNP list were parsed as missing (Ji P2 initially had only 420 het SNPs).

## v4 rule (POST HOC, designed after Ji P2 on all 17 seen samples: 8 fixes, 0 harms)
v3 per-cell score with the HLA windows (10 Mb windows 6:2 and 6:3) excluded; flip CopyKAT's calls
only if (a) diploid-called cells score higher than aneuploid-called, one-sided Welch p < 0.01, AND
(b) the 3 windows with the largest diploid-minus-aneuploid mean z difference carry < 50% of the
total difference (genuine CNAs are spread over many windows; artefacts are concentrated).

## v4 fresh SAFETY + POWER test (fixed 2026-10-08 ~07:10, before any of these was analysed)
No fresh inverted samples with public reads remain (others are EGA/dbGaP/DUOS). So, on fresh
non-selected samples from public-read studies:
  Ji2020 P4, P6, P7, P10 (original hg19 Cell Ranger BAMs of the cSCC libraries; scored on
  "P<n>_Tumor_*" cells), Geistlinger2020 T89, T90 (original BAMs), Dong2020 Group1 Tumor_71,
  Tumor_75 (SRA with barcodes). (Ji P8: two colliding libraries; Geistlinger T77: name does not
  match GEO; excluded up front.)
Endpoints, for CopyKAT default and B0F3 calls separately:
  SAFETY  : number of harmful flips of the actual calls (loss > 0.3 in balanced accuracy)
  POWER   : with the calls deliberately swapped (1 <-> 0), fraction of samples where v4 swaps them
            back; reported together with whether the sample's tumour shows allelic signal at all.

## v5: Numbat-style phased-BAF HMM (design fixed 2026-10-08 BEFORE any v5 result on the 26 samples)
Development only on synthetic data and BCC06/BCC06post (not in the evaluation set).
Data: het SNPs (pooled DP>=8, min allele>=2, MAF>=0.1), autosomes, Eagle2 phase (h); per cell c and SNP
s: reads on haplotype A (a_cs) and total (n_cs).
Pseudobulk HMM, run separately on each CopyKAT group (aneuploid-called, diploid-called):
  states: 0 balanced (BAF 0.5); for theta in {0.6, 0.7, 0.8, 0.9}: A-major (BAF theta) and B-major
  (BAF 1-theta) -> 9 states.
  emission: beta-binomial(A_s | N_s, BAF, overdispersion rho); rho estimated once per group by maximum
  likelihood under state 0 on SNPs with N_s >= 10 (absorbs SNP-level allele-specific expression).
  transitions: between consecutive SNPs on a chromosome, CNA-state change p_cna = 1e-4 (to a uniformly
  chosen other state); within the same theta, A-major <-> B-major switch (phasing error / crossover)
  p_flip = 0.5 * (1 - exp(-2 * 0.01 * d_Mb)) + 0.005.
  decoding: forward-backward; P(imbalanced) per SNP = sum of the 8 non-balanced states.
  segments: maximal runs with P(imbalanced) > 0.5 spanning >= 5 Mb and >= 20 SNPs; per segment, SNP
  orientation = sign of P(A-major) - P(B-major); segment BAF theta_seg = pooled major-haplotype fraction.
Outputs per group g: F_g = fraction of the covered autosomal genome in segments.
Per-cell allele LLR: sum over segments (both groups' segments; where they overlap, the one with larger
  theta) of m log(2 theta_seg) + (n - m) log(2 (1 - theta_seg)), m = cell's reads on the segment's
  major haplotype (per-SNP orientation), n = cell's reads in the segment. Tumour if LLR > 0.
Endpoints (all 26 samples with harness outputs; CopyKAT default and B0F3 calls; actual and swapped):
  PRIMARY  orientation: flip CopyKAT's calls iff F_dip - F_an >= 0.02. Harmful flips (loss > 0.3) and
           restored wrong call sets (gain > 0.3), compared with the pre-registered orientgrp rule.
  SECONDARY per-cell: balanced accuracy of (a) allele LLR > 0, (b) consensus = allele call where
           |LLR| >= 2, else the B0F3 call; versus CopyKAT default, B0F3 and oracle F3.
Correction (before running): the evaluation set has 25 samples, not 26.
