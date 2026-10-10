# Installed-data accuracy comparison

The accuracy phase waits until the expanded performance sweep and all 64 serial repeats have finished and the serial evidence is archived. It then compares exactly two pinned implementations:

| Endpoint | Source | Inference |
| --- | --- | --- |
| Main baseline | upstream main `ea1a15c2457c0a69ee61ada07db57a0c9c726bf2` | Default CPU algorithm |
| GPU + anchoring | `54016687dada1f5832e26bde2f955fb6c0cf94a4` | GPU backend; B6 marker reference; F3 arm-correlation calls; Monte Carlo KS |

“Main” means the upstream scientific baseline used throughout this review. The requested future experimental role of the fork's main does not change this reference. Git archive snapshots isolate both endpoints from current working-tree edits. No branches or PRs are changed.

The candidate is the previously documented B6+F3 protocol in `~/cancer_research/anchor_study/FINAL_PROTOCOL.md`. Immune markers, with endothelial fallback, select a normal-reference population; the sigma rule remains a fallback where markers cannot establish one. F3 compares chromosome-arm profiles with the consensus of the highest-energy 10% of cells and uses the reference population's 99th correlation percentile. The published candidate's numerical precision and thresholds are retained. This endpoint comparison assesses the combined candidate; it cannot assign causal credit to GPU arithmetic, anchoring or the final-call rule individually.

## Frozen panel and labels

Metadata selection is frozen before any new inference: complete human samples with 500–20,000 cells, at least 30 author-labelled cells in each class, and a labelled malignant fraction of 5–95%. For 3CA, select up to two samples per study, taking the lowest and highest malignant fractions with lexical tie-breaking, including across split matrices. Include all eligible converted labelled ScPCA libraries and eligible README examples. Include up to two eligible healthy donors per raw-count Tabula Sapiens h5ad. This gives broad study coverage without letting studies with many samples dominate.

The frozen manifest selects **125 samples across 60 studies**: 11 README, 82 3CA, 22 converted ScPCA, and 10 healthy-donor controls. Sample sizes range from 514 to 14,495 cells. This schedules 250 endpoint attempts. Input validation may reject a selected sample; such failures remain in the denominator of scheduled comparisons and are reported explicitly.

Unknown annotations stay unscored. Mixed-tumour metadata labels are proxies for malignancy/aneuploidy, often derived partly from CNA or marker evidence; they are not independent DNA truth. Healthy controls assess false-positive calls. Previously used development and opened holdout studies are identified, and other studies' prior use is marked uncertain. This is a retrospective comparison, not a new sealed holdout. No thresholds or methods are tuned after seeing results.

The manifest records exclusions, including missing aligned metadata, unsuitable sizes/classes, non-UMI matrices, gene identifiers without usable symbols, duplicate panels, and haematological cancers where the marker premise does not apply. Unlabelled Xenium, mouse data and unmatched DNA references do not supply evaluable human cell-call labels. Raw-count values, axes and unique cell barcodes are validated during staging; invalid inputs and failed endpoints remain visible in the report.

## Parallel execution and evidence

Run up to three CPU baseline workers and one GPU candidate worker concurrently, with four distinct physical cores per worker and four requested threads. Admit jobs using conservative cell-count memory estimates, reserve 12 GB of RAM, check free GPU memory, and guard RAM/SSD space. The accuracy phase has no runtime cutoff. These checks protect resources; CPU time, wall time and performance comparisons are not collected. Intrinsic pipeline timing announcements are filtered from persistent logs and temporary runtime files are removed.

Both endpoints receive identical gene-by-cell raw-count arrays and the same seed, using the same installed Python environment. Labels never enter inference. Whole-study Matrix Market files are streamed once to selected sample caches on the SSD; caches are deleted only after both endpoint attempts finish. Large unused text outputs are suppressed identically, with inference calculations preserved. Existing source archives remain in their storage locations.

Report per-sample coverage, confusion counts, conditional balanced accuracy, and coverage-adjusted balanced recall. The latter averages correct malignant calls divided by all labelled malignant input cells and correct non-malignant calls divided by all labelled non-malignant input cells, penalizing abstentions. Repeat comparisons on cells defined by both methods, outside the candidate reference, and outside marker-positive cells. Show healthy specificity separately, subtype errors, reference contamination, and all failures. Aggregate within study before averaging across studies; paired study-bootstrap intervals describe study variation, not uncertainty in the annotations.

Scientific figures include study comparisons, matched chromosome-arm heatmaps and truncated dendrograms for a preset kidney example and the largest observed improvement/regression. These are descriptive examples. Compact arm summaries use FP64 without additional diagnostic downcasting; the candidate's inference precision is unchanged. Per-cell predictions, input hashes, source pins, parameters, original CNA numeric hashes and linkage matrices support auditability.

## Durable queue and outputs

- Coordinator: `copykat-bench-accuracy.service`, waits for the serial archive, survives client closure, adopts its own surviving workers on restart.
- Report refresher: `copykat-accuracy-report.service`, low priority, stops after archiving.
- SSD working directory: `/home/toresbe/cancer_research/accuracy_2026-10-10`.
- Frozen selection: `dataset_manifest.json`; protocol: `protocol.json`; results and schedule: `results/`.
- Live report: [accuracy-results.md](accuracy-results.md).
- Final NAS evidence: `/mnt/nas/cancer_research/accuracy_2026-10-10/accuracy-evidence.tar.gz`, with a SHA-256 sidecar. Existing final archives are preserved.

Harness checks cover sparse column order and duplicate summation, gzip/source digests, axis mismatches, rejection of noninteger counts, categorical metadata, unknown labels, abstention penalties, healthy specificity and subset masks. They do not start inference or benchmark performance.

## Concurrency update, 2026-10-10

At the user's request, the coordinator now admits up to **seven CPU and two GPU workers**, still with four requested threads per inference. CPU lanes cover logical CPUs 0–27; GPU host work uses 28–31. SMT sharing is permitted because this phase measures accuracy only. Existing workers retain their original affinities; each result records its actual environment. Memory/VRAM admission and emergency guards remain active. The protocol records both worker-policy epochs.

Matrix staging now runs in its own process, so parsing a large study cannot block dispatch of already cached samples. Source caches are written atomically, and restart adopts the surviving stager and inference workers. A partially staged valid sample is retained even if a later sample fails validation.
