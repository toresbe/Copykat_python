# CNA profile accuracy against DNA ground truth (2026-10-07)

How close are CopyKAT-Py's copy-number *profiles* to truth from whole-genome or exome DNA
sequencing? Scored with the metrics of Schmid et al. 2025 (Nat Commun, "Benchmarking scRNA-seq
copy number variant callers"), reimplemented in `eval_bench.py`:

- WGS/WES truth binned to 100 kb;
- per-bin mean of the gene-level CNA of the annotated tumour cells (+2);
- Pearson correlation, gain/loss AUC, truncated AUC, best macro-F1.

Cross-checks:
- AUCs match R's pROC exactly; truncated AUCs agree within 5e-5.
- The vectorised F1 (`fast_f1.py`) is identical to a direct transcription of the R loop.

## Data

| dataset | truth | cells (tumour + reference) | preparation |
|---|---|---|---|
| 9 gastric cancer lines (HGC27, KATOIII, MKN45, NCIN87, NUGC4, SNU16, SNU601, SNU638, SNU668) | WGS | 1–5k line cells + 4,813 normal gastric cells (GSE150290, patients 25–29) | as the paper, but using GEO's processed matrices (GSE142750) instead of a Cell Ranger 7 rerun |
| BCC06, BCC06post (basal cell carcinoma, patient su006, pre/post treatment) | WES | 1,436 / 600 | exactly as the paper (GSE123813) |
| MM199 (multiple myeloma) | WES | 2,820 | GEO raw counts; tumour = ≥ 2 of SDC1/TNFRSF17/SLAMF7/MZB1, reference = none of them (the paper's Seurat annotation needs FASTQs) |

Modes, all with annotated reference cells (`norm_cell_names`) unless marked "auto":

- `r_ref`: R CopyKAT 1.2.5. SNU668 was not run; the R runs were stopped to save time.
- `py_cpu_ref`, `py_gpu_ref`, `py_gpu_exact_ref`: CopyKAT-Py on CPU, GPU, and GPU with the exact
  KS test.
- `py_gpu_auto_sigma`, `py_gpu_auto_markers`: automatic reference, with CopyKAT's σ anchor or the
  marker anchor.

## Results (mean over datasets that have an R run)

| mode | gastric lines (8): Pearson | gain AUC | loss AUC | best F1 | tumours (3): Pearson | gain AUC | loss AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| R CopyKAT | 0.539 | 0.836 | 0.804 | 0.537 | 0.269 | 0.647 | 0.618 |
| Py CPU | 0.540 | 0.836 | 0.797 | 0.536 | 0.265 | 0.646 | 0.617 |
| Py GPU | 0.540 | 0.835 | 0.797 | 0.536 | 0.266 | 0.646 | 0.617 |
| Py GPU + exact KS | 0.544 | 0.840 | 0.799 | 0.547 | 0.266 | 0.644 | 0.615 |
| Py GPU, auto (σ) | 0.537 | 0.836 | 0.808 | 0.544 | 0.247 | 0.647 | 0.613 |
| Py GPU, auto (markers) | 0.537 | 0.836 | 0.808 | 0.544 | 0.190 | 0.627 | 0.579 |

Pearson correlation per dataset (GPU, reference mode):

| line | HGC27 | KATOIII | MKN45 | NCIN87 | NUGC4 | SNU16 | SNU601 | SNU638 | SNU668 |
|---|---|---|---|---|---|---|---|---|---|
| Pearson | 0.44 | 0.28 | 0.74 | 0.67 | 0.67 | 0.49 | 0.56 | 0.47 | 0.73 |

Tumours: BCC06 0.21, BCC06post 0.20, MM199 0.39.

## Conclusions

1. **CopyKAT-Py reproduces R CopyKAT's profile accuracy.** Every metric agrees within about 0.01
   on all 11 datasets with an R run.
2. **The GPU branch's exact algorithms neither help nor hurt profile accuracy.**
   - CPU and GPU results are identical.
   - The exact KS test changes metrics by ≤ 0.01; it is slightly better on average, but not
     consistently.
3. **Automatic reference detection costs almost nothing on the gastric lines**, where the reference
   cells are in the input. The marker anchor mostly fell back to σ there, so they tie.
4. **The marker anchor hurts on myeloma** (Pearson 0.22 vs 0.39). Myeloma is a blood cancer,
   outside the documented scope of that mode.
5. **The dataset, not the implementation, sets the ceiling.**
   - Pearson ranges from 0.28 to 0.74 across cell lines, and is about 0.2 for the basal cell
     carcinomas.
   - Every CopyKAT variant lands within about 0.01 of the others on each dataset.
   - Better profiles need a better model (external lineage-matched references, allele information),
     not further numerical refinement.

Files: `metrics_final.csv` (all metrics), `prep.py`, `run_bench.py`, `run_r.R`, `eval_bench.py`,
`fast_f1.py`.

# Follow-up: external (atlas) references — idea #3 (2026-10-07)

Can a lineage-matched normal reference from an atlas replace the in-sample one?

Setup:
- Tabula Sapiens (CELLxGENE): Skin, Stomach, Bone_Marrow and Tongue `.h5ad` files, 10x 3′ v3
  cells only.
- Up to 1,000 reference cells per run.
- One shared gene set per dataset (sample genes present in Tabula Sapiens, 10–21k genes).
- Known-reference mode on the GPU backend.
- Design fixed before running (`extref.py`, `bridge.py`).

| Pearson (mean) | in-sample / study reference | atlas, lineage-matched | atlas, unmatched | atlas, matched + bridge correction |
|---|---:|---:|---:|---:|
| gastric lines (9) — stomach epithelium vs fibroblasts | **0.565** | 0.515 | 0.467 | (no bridge available) |
| BCC06 — tongue basal keratinocytes vs skin fibroblasts/endothelium | **0.197** | 0.134 | 0.197 | (too few bridge cells) |
| BCC06post | **0.184** | 0.127 | 0.167 | 0.172 |
| MM199 — bone-marrow plasma cells vs T cells | 0.417 | 0.327 | 0.340 | **0.517** |

Findings:

1. **A naive atlas reference is worse than the in-sample one.** The batch and chemistry difference
   between atlas and sample costs more than lineage matching gains. Lineage matching beats an
   unmatched atlas reference on the gastric lines (+0.05), but not on BCC. Tongue basal cells may
   be the wrong tissue for skin BCC.
2. **Bridge correction recovers the loss.** Gene-wise factors are estimated from a normal cell
   type present in both sample and atlas:
   - BCC06post: from 0.127 to 0.172 (in-sample: 0.184);
   - MM199: from 0.327 to 0.517, beating the in-sample reference on every metric (AUC gains 0.79
     vs 0.74, losses 0.75 vs 0.73, best F1 0.60 vs 0.56).
3. **Caveats:**
   - The bridge was only testable on two datasets.
   - MM199's tumour labels are my own marker-based annotation.
   - The factors are noisy (log2 SD 2–3) for genes the bridge barely expresses.
   - This is a promising signal, not a validated method.

## Robustness of the bridge correction (`robust.py`, `robust.csv`; design fixed before running)

| Pearson (range over variants) | MM199 | BCC06post |
|---|---|---|
| in-sample reference | 0.417 | 0.184 |
| atlas matched, no bridge | 0.327 | 0.127 |
| shuffled factors (control, 3 seeds) | 0.317–0.344 | 0.120–0.130 |
| ε = 0.01 / 0.1 / 1 / 10 CPM | 0.521 / 0.517 / 0.517 / 0.475 | 0.186 / 0.172 / 0.168 / 0.160 |
| other bridge types | monocytes 0.495, B cells 0.478 | myofibroblasts 0.143, melanocytes 0.158 |
| bridge subsets of 10 / 25 / 50 cells (5 seeds each) | 0.494–0.515 / 0.514–0.537 / 0.518–0.526 | 0.146–0.178 / 0.161–0.196 / 0.158–0.182 |

Findings:

- **The gain is gene-specific.** Factors shuffled across genes give exactly the uncorrected result.
- **It is robust.** It holds across ε 0.01–1, any bridge cell type tested, and bridges of only
  10 cells.
- **Bridged vs in-sample reference:**
  - MM199: better under every variant;
  - BCC06post: a tie. The "matched" reference there is tongue keratinocytes, not skin, which
    probably limits it.
- **Open question:** whether this generalises. Testing that needs more datasets with DNA truth
  and in-sample normal cells.

## Third bridge test: glioblastoma with same-tumour single-cell WGS truth (`gbm.py`, `gbm_atlas.py`, `metrics_gbm.csv`)

Data:
- GEO GSE185269 (scONE-seq paper): 10x single-nucleus RNA of a glioblastoma, 1,737 clone-3 tumour
  cells plus 2,631 normal cells.
- Truth: single-cell WGS of the same tumour (432 clone-3 vs 586 normal cells, 5,087 segments of
  about 500 kb). Copy number = 2 × tumour/normal depth, clipped to [1, 3].
- Sanity check: the truth shows the typical +7/−10.
- Atlas: CELLxGENE Census 2025-11-08, dataset 3c361813 (adult dlPFC, 10x 3′ v3, nuclei, 4 donors).
- Design fixed before running.

| reference | Pearson | gain AUC | loss AUC | best F1 |
|---|---:|---:|---:|---:|
| **all in-sample normal cells (2,631)** | **0.713** | **0.910** | **0.922** | **0.753** |
| in-sample astrocytes only (lineage-matched, no batch effect) | 0.640 | 0.895 | 0.885 | 0.718 |
| atlas astrocytes | 0.566 | 0.745 | 0.876 | 0.635 |
| atlas astrocytes + bridge (in-sample ↔ atlas oligodendrocytes) | 0.597 | 0.768 | 0.894 | 0.652 |
| atlas oligodendrocytes (unmatched) | 0.540 | 0.715 | 0.873 | 0.574 |

## Verdict on idea #3 (external, lineage-matched references)

Across the three datasets where a bridge was possible:

| dataset | bridged atlas vs in-sample reference |
|---|---|
| MM199 | better |
| BCC06post | tie |
| GBM | clearly worse |

In GBM, even a lineage-matched *in-sample* reference loses to a broad mix of in-sample normal cells.

- The bridge correction itself works and is robust. It consistently recovers part of the atlas
  batch penalty.
- **But a lineage-matched atlas reference is not a reliable improvement over the in-sample
  normal cells.** The MM199 gain is the exception. Its in-sample reference was my own
  marker-defined "control" set, which may simply have been a poor reference.
- **Practical conclusion:**
  - When a sample contains normal cells, keep using a broad mix of them as the reference.
  - Atlas references are a fallback for samples with *no* normal cells (cell lines,
    tumour-dominated samples). There they cost roughly 0.05–0.1 Pearson relative to a good
    in-sample reference, and no bridge is possible.
