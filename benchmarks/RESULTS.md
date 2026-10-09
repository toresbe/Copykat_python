# GPU backend: benchmarks and validation

Branch `hardcore-optimization`. Question: how fast can CopyKAT-Py go, and
how precise can it be, if bit-identical output is not required and a GPU is
available?

**Short answer.** On the 11 labelled 3CA samples, the `gpu` backend runs the
full pipeline (including heatmaps and text output) **7.9× faster** than
`main` (765 s → 97 s). With the exact KS breakpoint test it is **11.3×**
(68 s). On the Xenium WTA breast set it is 4.6× at 20k cells and 6.8× at
50k cells, and the full 170k-cell set runs end to end in **96 s**. Classification accuracy is
unchanged on average. The exact algorithms remove several numerical
approximations, but they do not by themselves fix the method's
misclassified samples; the opt-in marker anchor does (mean 0.859 → 0.945 on
the same samples, see [Accuracy](#accuracy)).

Hardware: RTX 5080 (16 GB, consumer Blackwell: FP64 runs at 1/64 of FP32),
32-thread CPU, 125 GB RAM. CPU runs use `--n-cores 32`. Every timing was
taken with the machine otherwise idle (load average < 6 before each sample).

## Modes compared

| mode | what it is |
|---|---|
| `base` | `main` at d23282b, unchanged |
| `cpu` | this branch, `--backend cpu`. **Byte-identical output to `base`** (all 11 samples: same calls, max \|ΔCNA\| = 0) |
| `gpu-compat` | CUDA, but keeps the CPU path's approximations (PCA before step-4 clustering, subsampled silhouette, k-means heatmap ordering) |
| `gpu` | CUDA with exact algorithms in place of those approximations (table below) |
| `gpu+exactKS` | `gpu` plus `--ks-method exact` |
| `gpu+anchor` | `gpu` plus `--anchor markers --final-call arm_correlation`: marker-defined normal reference and arm-level correlation calls (accuracy only; see [Fixing the anchor](#fixing-the-anchor)) |

What `gpu` computes differently from `base`:

| Step | `base` | `gpu` |
|---|---|---|
| Step-4 Ward clustering, > 2,000 cells | randomized PCA to 128–256 components | full feature matrix; exact Ward tree |
| Silhouette check | stratified subsample | all cells |
| Segment means | differences of an FP32 running total | FP64 |
| Heatmap ordering, > 3,000 cells | MiniBatchKMeans blocks | full Ward dendrogram (the step-8 tree, reused) |
| Breakpoint KS test (`--ks-method exact` only) | KS between two 1,000-draw Monte Carlo samples of the posterior Gammas | exact KS distance between the Gammas |

## Speed: 11 labelled samples

Wall time per run (s), including heatmap and both text outputs:

| sample | cells | base | cpu | gpu-compat | gpu | gpu+exactKS | best speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bi2021_Kidney_P90 | 3726 | 68.3 | 48.6 | 42.2 | 7.4 | 5.2 | 13.3x |
| Chen2020_Head-and-Neck_P11 | 6465 | 56.9 | 34.5 | 27.3 | 9.5 | 6.5 | 8.7x |
| Choudhury2022_Brain_MSC6-BTI | 11102 | 119.9 | 77.3 | 45.8 | 16.4 | 10.5 | 11.4x |
| Dong2020_Prostate_patient5 | 7548 | 60.3 | 40.5 | 29.7 | 12.1 | 7.2 | 8.4x |
| Gao2021_Breast_DCIS1 | 1480 | 18.8 | 13.2 | 5.9 | 6.6 | 4.1 | 4.6x |
| Geistlinger2020_Ovarian_T59 | 9484 | 302.4 | 292.7 | 39.5 | 9.5 | 8.0 | 37.8x |
| Jerby-Arnon2021_Sarcoma_SyS14 | 2522 | 12.0 | 9.6 | 6.2 | 6.3 | 4.6 | 2.6x |
| Ji2020_Skin_P4 | 7956 | 66.9 | 58.0 | 31.4 | 9.7 | 7.9 | 8.5x |
| Laughney2020_Lung_RU681 | 982 | 14.9 | 14.4 | 5.6 | 5.7 | 3.9 | 3.8x |
| Lee2020_Colorectal_SMC09 | 2253 | 11.6 | 9.4 | 6.1 | 6.2 | 4.5 | 2.6x |
| Lin2020_Pancreas_P08 | 1138 | 33.1 | 32.5 | 7.0 | 7.1 | 5.4 | 6.1x |
| **total** | | **765** | **631** | **247** | **97** | **68** | **11.3x** |

A second `gpu` sweep took 86 s in total, with identical calls and CNA
matrices: the GPU path is deterministic. The ~10% difference between the two
sweeps is timing noise.

Time per stage, summed over the 11 samples (s):

| stage | base | cpu | gpu-compat | gpu | gpu+exactKS |
|---|---:|---:|---:|---:|---:|
| load + filter | 15.0 | 13.7 | 11.3 | 13.8 | 11.2 |
| DLM smoothing | 13.1 | 12.2 | 1.7 | 1.8 | 1.7 |
| baseline (step 4) | 377.2 | 350.1 | 16.0 | 15.6 | 14.5 |
| segmentation | 24.5 | 23.5 | 19.2 | 21.7 | 2.3 |
| bins | 9.2 | 8.7 | 3.0 | 3.5 | 3.6 |
| steps 7 + 8 | 45.3 | 33.7 | 5.5 | 5.9 | 5.3 |
| write outputs | 63.6 | 7.3 | 7.3 | 8.0 | 7.1 |
| heatmap | 215.6 | 180.0 | 172.0 | 13.5 | 11.3 |

What's left in a `gpu` run of a typical sample (~6–10 s): about 1 s of CUDA
context set-up, ~2 s of Monte Carlo breakpoint testing (mostly the numba JIT
compile; `--ks-method exact` removes it), ~1 s for heatmap rendering, and the
text output. Step 4's 253 s on Geistlinger was the low-confidence fallback,
which fits a 500-iteration GMM to one cell at a time. On the GPU every cell
is fitted in one kernel launch.

## Speed: scaling (Xenium WTA breast, unlabelled)

Random cell subsets of the 170,057-cell Xenium set. Text outputs are still
formatted but written to `/dev/null`: at 170k cells they total ~20 GB.

| input cells | after QC | base | gpu-compat | gpu | speedup (gpu) |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 16,143 | 57.6 s | 15.4 s | 12.6 s | 4.6x |
| 50,000 | 40,363 | 185.0 s | 29.5 s | 27.3 s | 6.8x |
| 170,057 | 137,069 | _see note_ | 91.0 s | 96.2 s | |

Note: the 170k `base` run was still in progress when this file was written;
see the end of this file.

At 170k the `gpu` run's largest steps are step 4 (46 s: exact Ward on
137k × 8.3k plus an exact all-pairs silhouette) and the steps-7/8 baseline
adjustment (10 s). Peak host RSS was 21 GB.

## Exact Ward linkage on the GPU

`copykat_py/gpu/ward.py` computes **exact** Ward trees (Euclidean, same
convention as SciPy/fastcluster) with O(n·d) memory, so the full feature
matrix can be clustered at any cell count:

- Ward is a *reducible* linkage, so all reciprocal-nearest-neighbour pairs
  can be merged at once without changing the tree. Each round is one
  batched nearest-neighbour search, and a row only needs re-searching if
  its neighbour was merged.
- Candidates come from an FP32 GEMM turned into **certified lower bounds**
  (minus a rounding-error bound). The best candidate is re-scored exactly
  in FP64 from coordinate differences. A fused Triton kernel then collects
  every column whose bound could still reach that cost, so ties and
  near-ties are resolved exactly, with the lowest index winning. Rows that
  cannot be certified retry with more candidates, then fall back to an
  exact full scan.
- For very wide inputs (≥ 30k cells), candidates can be searched in a 512-d
  principal projection. Projection never increases a distance, so the
  bounds stay valid and the tree stays exact.

On real step-4 input (Choudhury, smoothed expression, 9,571 genes), against
fastcluster on the same full-width data (threaded pdist + `fastcluster.linkage`):

| cells | fastcluster (32 threads) | GPU | speedup | ARI vs fastcluster at k = 2 / 6 / 50 |
|---:|---:|---:|---:|---|
| 2,000 | 13.1 s | 0.16 s | 80x | 1.000 / 1.000 / 1.000 |
| 5,000 | 51.6 s | 0.41 s | 125x | 1.000 / 1.000 / 1.000 |
| 11,154 | 181 s | 1.1 s | 162x | 1.000 / 1.000 / 1.000 |

Merge heights agree to 7.5e-9 relative. The real 137k × 159 steps-7/8 input
takes 3.3 s.

## Precision: what the CPU path's approximations cost

`benchmarks/precision.py` captures real intermediates from each sample and
compares each approximation with its exact counterpart:

| sample | ARI PCA vs exact, k=2 | k=6 | silhouette sub / exact | max FP32-cumsum error (log CNA) | breaks MC / exact | breaks changed by MC seed alone | MC vs exact |
|---|---:|---:|---|---:|---|---:|---:|
| Bi2021_Kidney_P90 | 0.994 | 0.780 | 0.165 / 0.166 | 5.1e-05 | 41 / 25 | 15 | 18 |
| Chen2020_Head-and-Neck_P11 | 0.617 | 0.490 | 0.185 / 0.186 | 1.0e-04 | 80 / 55 | 30 | 33 |
| Choudhury2022_Brain_MSC6-BTI | 0.977 | 0.854 | 0.222 / 0.223 | 2.1e-04 | 237 / 172 | 47 | 69 |
| Dong2020_Prostate_patient5 | 0.993 | 0.873 | 0.322 / 0.318 | 1.4e-04 | 165 / 130 | 29 | 35 |
| Gao2021_Breast_DCIS1 † | 1.000 | 0.700 | 0.224 / 0.224 | 2.1e-04 | 228 / 172 | 67 | 64 |
| Geistlinger2020_Ovarian_T59 | 0.940 | 0.699 | 0.156 / 0.156 | 1.6e-04 | 195 / 183 | 12 | 14 |
| Jerby-Arnon2021_Sarcoma_SyS14 | 1.000 | 0.300 | 0.153 / 0.153 | 1.2e-04 | 151 / 90 | 52 | 65 |
| Ji2020_Skin_P4 | 0.766 | 0.704 | 0.094 / 0.095 | 1.5e-04 | 163 / 125 | 47 | 46 |
| Laughney2020_Lung_RU681 † | 1.000 | 0.766 | 0.189 / 0.189 | 1.2e-04 | 216 / 180 | 31 | 38 |
| Lee2020_Colorectal_SMC09 | 1.000 | 0.450 | 0.190 / 0.190 | 1.3e-04 | 175 / 133 | 47 | 44 |
| Lin2020_Pancreas_P08 † | 1.000 | 0.925 | 0.085 / 0.085 | 1.1e-04 | 186 / 129 | 55 | 59 |

† ≤ 2,000 cells, where `base` clusters without PCA. Those PCA columns are
what PCA *would* change.

- **PCA before step-4 clustering is the largest approximation.** The
  2-cluster cut usually survives, but the 6-cluster partition (ARI
  0.30–0.93) is what chooses the normal anchor and the consensus profiles
  breakpoints are called on. The CPU path's 256-PC cap also discards real
  distance structure. Searching Choudhury's step-4 tree in a 256-d
  projection, the projected distances were too loose to certify nearest
  neighbours 118k times (over 92k queries), against 158 times at 512-d.
- **The silhouette subsample is harmless** (differences ≤ 0.004 against a
  0.15 threshold).
- **The FP32 running total** introduces up to 2e-4 error in log CNA, small
  next to the 0.25-SD noise floor. The GPU path is FP64.
- **The Monte Carlo KS test is the noisiest step.** Re-running it with
  only the random seed changed moves 10–30% of breakpoints. With 1,000
  draws per side its statistic has SD ≈ 0.02 and an upward bias of
  0.01–0.04 against a 0.1 cutoff. The exact KS distance between the two
  posterior Gammas is deterministic and noise-free, and calls fewer
  breakpoints, since the bias inflated the MC statistic.

## Accuracy

Balanced accuracy of the aneuploid call against the 3CA `Malignant` labels
(* = run flagged low confidence):

| sample | base | gpu-compat | gpu | gpu+exactKS | gpu+anchor |
|---|---:|---:|---:|---:|---:|
| Bi2021_Kidney_P90 | 0.997 | **0.004** | 0.996 | 0.996 | 0.993 |
| Chen2020_Head-and-Neck_P11 | 0.803 | 0.924* | 0.837 | **0.086** | 0.901 |
| Choudhury2022_Brain_MSC6-BTI | 0.847 | 0.847 | 0.847 | 0.848 | 0.917 |
| Dong2020_Prostate_patient5 | 0.991 | 0.971 | 0.974 | 0.990 | 0.924 |
| Gao2021_Breast_DCIS1 | 1.000 | 1.000 | 1.000 | 0.999 | 0.995 |
| Geistlinger2020_Ovarian_T59 | 0.071* | 0.034* | 0.030 | 0.031 | 0.949 |
| Jerby-Arnon2021_Sarcoma_SyS14 | **0.004** | 0.996 | 0.996 | 0.996 | 0.987* |
| Ji2020_Skin_P4 | 0.862* | 0.860* | 0.861* | 0.802* | 0.859 |
| Laughney2020_Lung_RU681 | 0.936 | 0.936 | 0.936 | 0.936 | 0.932 |
| Lee2020_Colorectal_SMC09 | 0.973 | 0.972 | 0.973 | 0.973 | 0.968 |
| Lin2020_Pancreas_P08 | 0.994* | 0.993* | 0.993* | 0.993* | 0.970 |
| **mean** | **0.771** | **0.776** | **0.859** | **0.786** | **0.945** |

`gpu+anchor` = `gpu` with `--anchor markers --final-call arm_correlation` (see
[below](#fixing-the-anchor)). The `gpu` column was re-run alongside it and
reproduced the earlier `gpu` numbers exactly.

The `gpu` mean is higher only because sarcoma flips from fully inverted to
correct. Without it, the means are equal (`base` 0.848, `gpu` 0.845). These
flips come from one fragile decision, not from numerical precision:

**Step 4 picks the normal anchor as the cluster whose consensus profile has
the smallest 3-component-GMM σ, and the margins are a few percent.**
Every label is then decided by overlap with that anchor. `diag_step4.py` shows:

| sample / mode | anchor σ | runner-up σ | anchor's malignant fraction |
|---|---:|---:|---:|
| Kidney, base | 0.02259 | 0.02327 | 0.001 (correct) |
| Kidney, gpu-compat | 0.02118 (a 242-cell tumour subcluster) | 0.02217 | **0.963** |
| Sarcoma, base | 0.04194 (tumour) | 0.04732 (the true normals) | **1.000** |
| Ovarian, every mode | 0.0699 | 0.0716 | **0.66** |

Anything that moves the step-4 partition can flip a sample: a different PCA
solver, exact instead of PCA clustering, or different breakpoints. That is
how `gpu-compat` inverts Kidney and `gpu+exactKS` inverts Chen. It also
explains why `base` and `gpu-compat`, which differ only in the randomized
PCA implementation, agree on just 81% of Xenium-50k calls. Exact
computation removes the arbitrariness of PCA and RNG seeds (`gpu` is
deterministic and seed-free apart from the MC test), but making the
classification robust needs a change to the anchor rule itself.

### Fixing the anchor

That change is now on the branch, opt-in
([docs/research/anchor_study.md](../docs/research/anchor_study.md)):

- `--anchor markers` takes as the normal reference the step-4 cluster holding
  the largest pan-immune (else pan-endothelial) marker-defined population,
  instead of the smallest-σ cluster.
- `--final-call arm_correlation` calls a cell aneuploid if its arm-level
  profile correlates with the consensus of the most deviant cells more
  strongly than 99% of the reference cells do.

On these 11 samples the mean rises from 0.859 to **0.945** (`gpu+anchor`),
mainly by un-inverting Geistlinger (0.030 → 0.949), with small losses on
samples that were already good (Dong prostate −0.050, Lin −0.023). These 11
samples were the development set of that study, so the in-sample gain
overstates it. The independent evidence is the pre-registered test on 43 new
3CA studies: +0.047 [+0.024, +0.078] mean balanced accuracy, inverted samples
29 → 5, confirmed on a sealed holdout (+0.045).

A further opt-in check uses allele imbalance from the reads
(`--allele-counts/--allele-phase`,
[docs/allele_orientation.md](../docs/allele_orientation.md)); it can only flip
the labels of CopyKAT's split. Three of these samples have public reads: it
flips Geistlinger's `gpu` calls (0.030 → 0.970) and leaves the other five call
sets unchanged. On 25 samples with public reads, the anchor options plus the
allele check gave 0.907 mean balanced accuracy (anchor options alone 0.871),
with no inverted sample left and no harmful flip
([docs/research/allele_orientation.md](../docs/research/allele_orientation.md)).
Scoring note: this table scores cells with an empty 3CA `cell_type` as normal;
the research documents exclude them, so their numbers differ slightly.

Synovial sarcoma is typically near-diploid (driven by the SS18-SSX fusion),
so "malignant" is not the same as "aneuploid" there. Its accuracy says
little either way.

## Reusable without the GPU

- **Repeated-row CSV writer** (`_write_csv_repeated_rows`). Segmented CNA
  matrices repeat each row across all bins or genes of a segment, so each
  distinct row is formatted once and spliced after each row's leading
  columns. Output is byte-identical (verified on all 11 samples). It cut
  output writing from 63.6 s to 7.3 s and made `cpu` 1.2× faster than
  `base` overall, with identical results.
- **Reusing the step-8 tree for the heatmap** when step 8 clustered the
  same matrix without PCA. This is identical output and saves one full
  Ward clustering.

## Reproducing

```bash
pip install -e ".[gpu]" cupy-cuda13x  # match your CUDA
PYTHON=python benchmarks/sweep.sh . runs/gpu --plot --kw backend_name=gpu
PYTHON=python benchmarks/sweep.sh . runs/gpu_anchor --kw backend_name=gpu \
    --kw anchor=markers --kw final_call=arm_correlation
python benchmarks/report.py runs base=base gpu=gpu
python benchmarks/precision.py precision.jsonl
python benchmarks/ward_bench.py step4.npy
```

Datasets come from `~/cancer_research` (`benchmarks/datasets.py`):
the README's 11 3CA samples, with `cell_type == "Malignant"` as ground
truth, and the Xenium WTA breast `cell_feature_matrix.h5`.
