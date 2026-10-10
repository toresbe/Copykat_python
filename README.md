# CopyKAT-Python (experimental fork of [Navin Lab's CopyKAT](https://github.com/navinlabcode/copykat_python))

This repo belongs to an unrelated coder who just thought it would be an interesting problem to optimize the code in his spare time. Importantly, *the output data differs from Navin Lab's and nothing has been examined by anyone with a clue about biology*, so please don't use this for anything.

CopyKAT-Python is a Python reimplementation of the [CopyKAT](https://github.com/navinlabcode/copykat) workflow for inferring large-scale copy number alterations (CNAs) from single-cell RNA-seq data. It reproduces the core CopyKAT strategy while improving scalability, usability, and integration with modern `AnnData`/`Scanpy` pipelines.

## Why CopyKAT-Python?

The original CopyKAT-R package is widely used for distinguishing aneuploid tumor cells from diploid normal cells using scRNA-seq data. Recurring practical limitations include:

- Long runtimes, with reports of >1 hour for ~8,000 cells
- Inability to handle very large datasets (hundreds of thousands to millions of cells) due to hierarchical clustering limits
  
**Highlights:**

- Identical core parameters as CopyKAT-R with convenient Python improvements
- Handles datasets from thousands to hundreds of thousands of cells with significantly faster speed
- fixed some known bugs of copykat-R to improve aneuploid prediction accuracy.
- Annotated CNA heatmaps with per-cell metadata sidebars (cell type, cluster labels, etc.)
- Pre-built Singularity container for reproducible deployment
- Validated across 11 human cancer samples and a 170k-cell Xenium whole-transcript dataset
  
---

## Installation

**From source:**

Installs `copykat-py` into your current environment

```bash
git clone https://github.com/navinlabcode/Copykat_python.git
cd Copykat_python
pip install -e .
```

**From `environment.yml` with conda:**

Creates a fresh conda environment for `copykat-py` with all required packages

```bash
git clone https://github.com/navinlabcode/Copykat_python.git
cd Copykat_python
conda env create -f environment.yml
conda activate copykit_py

## After activation, confirm the commands are available
copykat_matrix --help
copykat_anndata --help
```

**Singularity container** (recommended for HPC environments):

```bash
wget https://github.com/navinlabcode/Copykat_python/releases/download/sif/copykat_py.sif
singularity exec copykat_py.sif copykat-py --help
```

---

## How to Run

CopyKAT-Python supports two main entry points:

| Entry point | When to use |
| --- | --- |
| `copykat_matrix` / `copykat-py` | Input is a `.csv`, `.tsv`, or `.mtx` matrix file on disk in linux |
| `copykat_anndata()` | Input is an already-loaded `AnnData` object in python|

<img width="1183" height="566" alt="image" src="https://github.com/user-attachments/assets/2e8b1ae0-ef1c-4cc1-9e3d-f8cf6f9a88b5" />

### Terminal — `copykat_matrix` / `copykat-py`

**CSV or TSV matrix:**

```bash
copykat_matrix \
    --input sample_counts.csv \
    --sample-name sample1 \
    --genome hg20 \
    --n-cores 24 \
    --output-dir results/sample1
```

**10X matrix market input:**

```bash
copykat_matrix \
    --input filtered_feature_bc_matrix/matrix.mtx.gz \
    --genes filtered_feature_bc_matrix/features.tsv.gz \
    --barcodes filtered_feature_bc_matrix/barcodes.tsv.gz \
    --sample-name sample1 \
    --genome hg20 \
    --n-cores 24 \
    --output-dir results/sample1
```

Pass `--meta` (and optionally `--row-split`) to produce an annotated heatmap alongside the standard output. See [Annotated Heatmap with Metadata](#annotated-heatmap-with-metadata).

### Python API — `copykat()`

```python
import pandas as pd
from copykat_py import DistanceMetric, GeneIdType, Genome, copykat

counts = pd.read_csv("sample_counts.csv", index_col=0)

result = copykat(
    rawmat=counts,
    id_type=GeneIdType.SYMBOL,
    sam_name="sample1",
    genome=Genome.HG20,
    distance=DistanceMetric.EUCLIDEAN,
    n_cores=24,
)

print(result["prediction"].head())
```

`rawmat` can also be a dict with keys `matrix`, `genes`, and `barcodes` for sparse matrices.

### Python API — `copykat_anndata()`

```python
import anndata as ad
from copykat_py import DistanceMetric, Genome, copykat_anndata

adata = ad.read_h5ad("sample.h5ad")

result = copykat_anndata(
    adata=adata,
    selecting_meta=["CellType", "copykat_pred", "seurat_clusters"],
    row_split="CellType",
    sample_name="sample1",
    genome=Genome.HG20,
    distance=DistanceMetric.EUCLIDEAN,
    n_cores=24,
    output_dir="results/sample1_anndata",
)

print(result["prediction"]["copykat.pred"].value_counts())
```

Useful options: `layer` (use `adata.layers[...]`), `use_raw` (use `adata.raw`), `selecting_meta` (export obs columns for annotated heatmaps), `row_split` (column defining row groups).

### Typed option enums

The exported enums below describe fixed choices used by the Python APIs and their
result metadata. For input options, pass enum members or their string values; the
CLI and saved outputs continue to use those strings. Enum members make the available
choices discoverable in editors and avoid typos in Python calls.

| Enum | Choices | Used for |
| --- | --- | --- |
| `DistanceMetric` | `EUCLIDEAN`, `PEARSON`, `SPEARMAN` | Cell ordering in heatmaps |
| `Genome` | `HG20`, `MM10` | Reference genome assembly |
| `ExecutionBackend` | `CPU`, `GPU`, `GPU_COMPAT` | `copykat()` execution backend |
| `GeneIdType` | `SYMBOL`, `ENSEMBL` | Gene identifier type (`S` or `E`) |
| `CellLineMode` | `YES`, `NO` | Pure cell-line mode |
| `KSMethod` | `MONTE_CARLO`, `EXACT` | Segmentation breakpoint statistic |
| `AnchorStrategy` | `SIGMA`, `MARKERS` | Normal-reference selection strategy |
| `AnchorPath` | `SIGMA`, `IMMUNE`, `ENDOTHELIAL`, `ENRICHMENT` | Evidence used to select the reference cluster |
| `FinalCallStrategy` | `CLUSTERS`, `ARM_CORRELATION` | Final copy-number call strategy |
| `ReferenceMode` | `SYNTHETIC`, `KNOWN_NORMAL`, `AUTOMATIC` | Source of the normal reference population |
| `ReportFormat` | `TEXT`, `MARKDOWN`, `HTML`, `JSON` | Run-report output format |
| `PredictionLabel` | `DIPLOID`, `ANEUPLOID`, `DIPLOID_LOW_CONFIDENCE`, `ANEUPLOID_LOW_CONFIDENCE`, `NOT_DEFINED`, `UNKNOWN` | Prediction and missing-call states |
| `BaselineWarning` | `NONE`, `UNCLASSIFIED`, `CELL_LINE`, `KNOWN_NORMAL` | Baseline classification and run notes |
| `DataQualityStatus` | `OK`, `LOW` | Data-quality status used by baseline selection |

Pass enum members directly, for example `cell_line=CellLineMode.YES`,
`backend_name=ExecutionBackend.GPU`, or `final_call=FinalCallStrategy.ARM_CORRELATION`.
`ReportFormat` is accepted by `write_reports`; the analysis and report JSON files
continue to store the string values shown by each enum member.

### Output files

All entry points produce the same outputs as copykat-R:

- `*_copykat_CNA_results.txt`
- `*_copykat_prediction.txt`
- `*_copykat_heatmap.png`
- `copykat_run.log`

When metadata is supplied, an additional annotated heatmap PNG is produced. AnnData workflows also write `*_selected_obs_meta.csv`.

### Reports from completed runs

Each successful analysis also saves `*_copykat_runtime.json`. New runs record
parameters (including requested and effective `UP_DR`), software versions,
reference-cell counts, filtering counts, analysis notes, and step timings in this
file. Existing prediction and CNA outputs keep their formats.

Generate a report afterward, without repeating inference or loading the CNA matrix:

```bash
copykat-py-report \
    --run-dir results/sample1 \
    --formats txt,markdown,html,json
```

`--formats` accepts one or more comma-separated formats; the default is `html`.
Reports are written as `<sample>_copykat_report.txt`, `.md`, `.html`, or `.json` in
the run directory. Use `--output-dir` to write them elsewhere and `--sample-name`
to select a run when the directory contains multiple runtime files. Repeating the
command replaces only the selected report files.

All formats summarize the run, filtering, reference cells, prediction counts,
recorded warnings/notes, timings, and available output files. HTML embeds existing
standard and annotated heatmaps, so those figures remain visible when the HTML
file is shared. Markdown links to the heatmap files. Analytical output links still
require the original files; share those separately when needed. No additional
plots or reports are generated during inference.

PNG previews exceeding 12,000 pixels in either dimension or 40 million total
pixels are linked with a warning instead of embedded. This prevents malformed
plot layouts (such as excessively long categorical legends) from overwhelming
the report. The source plot is preserved; this check does not validate CNA calls.

Older runtime files are supported, with missing metadata explicitly marked as
unavailable. Reports also work without heatmaps or prediction tables (for example,
cell-line mode). Warnings summarize what the runtime metadata recorded, rather
than every third-party message in `copykat_run.log`. Reference details are counts,
not a list of cell barcodes. The `genes_after_annotation` count is after the
pipeline's gene exclusions, not a gene-identifier mapping success rate.

The same functionality is available in Python:

```python
from copykat_py.reporting import load_report, write_reports

report = load_report("results/sample1")
write_reports(report, ["markdown", "html"], "results/sample1")
```

### Progress messages

Progress messages are sent to the `copykat_py` logger. If your script or notebook has not configured logging, they are printed to stdout. Once logging is configured, they follow that configuration. For example, `logging.getLogger("copykat_py").setLevel(logging.WARNING)` shows only warnings, and `logging.basicConfig(level=logging.INFO)` routes progress through your own handlers.

---

## Annotated Heatmap with Metadata

Produce a CNA heatmap with per-cell metadata annotations (cell type, cluster labels, etc.). Rows are split into labelled groups by a chosen metadata column and ordered by hierarchical or K-means clustering within each group.

### CLI — standalone re-plot (`copykat-py-plot`)

Re-plot from an existing CNA results file without re-running the full analysis:

```bash
copykat-py-plot \
    --cna  sample_copykat_CNA_results.txt \
    --meta xenium_ft_full_meta_celltype_leiden.csv \
    --row-split inferred_CellType \
    --sample-name xenium_all_cells \
    --n-cores 40 \
    --output xenium_annotated_heatmap.png
```

| Flag | Default | Description |
| --- | --- | --- |
| `--cna` / `-c` | *(required)* | `*_copykat_CNA_results.txt` from a copykat-py run |
| `--meta` / `-m` | *(required)* | Annotation CSV — first column = cell name, rest = metadata |
| `--row-split` | second column | Column used to split and label row groups |

**Meta CSV format** — header row is auto-detected:

```csv
cell_name,leiden_cluster,inferred_CellType
aaaajgij-1,5,lumhr
aaaandia-1,5,lumhr
...
```

Cells present in the CNA results but absent from the CSV are labelled `"unknown"` and shown in grey. All remaining metadata columns are drawn as coloured annotation sidebars.

CopyKAT prediction annotations appear immediately to the right of the CNA
heatmap, followed by numeric measurements such as UMI counts, using
continuous viridis colors and a compact color scale. The continuous red–blue CNA
scale also remains on the right. Other categorical annotations stay on the left.
Small integer-coded categories (up to 20 distinct values)
and the row-split column remain categorical. Missing/nonfinite numeric values
are grey. To explicitly display an integer measurement with few distinct values
continuously, pass `--continuous-meta n_umi` to `copykat-py-plot`, or
`continuous_meta=["n_umi"]` to `plot_heatmap_annotated`. Row splitting still
requires a categorical column.

To show continuous measurements without dividing cells into categorical groups,
omit the group column from the metadata CSV and use `--no-row-split` (Python:
`row_split_col=""`). All cells are then clustered together.

CopyKAT prediction legends use readable labels: `diploid` → "Diploid",
`aneuploid` → "Aneuploid", and the `c1:`/`c2:` low-confidence states →
"Diploid (low confidence)" / "Aneuploid (low confidence)". `not.defined`
means "Not classified"; `unknown` means "Missing annotation". These are
display labels only; saved prediction and metadata values remain unchanged.
The standard metadata headings `copykat_pred_py` / `copykat_pred_R` display as
"CopyKAT Python" / "CopyKAT R", with borderless, titled legend groups.
`n_umi` displays as "UMI count per cell" (a count, not a percentage or read count).

Mouse (`mm10`) genes are ordered by chromosome and gene start position before
smoothing and segmentation. Mouse annotation `abspos` values are chromosome
offsets and cannot order genes within chromosomes. Runs generated before this
correction should be rerun from raw counts; sorting an existing CNA heatmap
does not correct the inference. The plot command detects mouse gene tables
automatically and displays chromosome codes 20/21 as X/Y.

### Python API — `plot_heatmap_annotated`

```python
import pandas as pd
from copykat_py.plotting import AnnotatedHeatmapOptions, plot_heatmap_annotated

cna = pd.read_csv("sample_copykat_CNA_results.txt", sep="\t")
plot_heatmap_annotated(
    mat           = cna.iloc[:, 3:].values.astype("float32"),
    cell_names    = cna.columns[3:].tolist(),
    chrom_info    = cna.iloc[:, 0].values,
    meta_csv      = "xenium_ft_full_meta_celltype_leiden.csv",
    options=AnnotatedHeatmapOptions(
        row_split_col="inferred_CellType",
        sample_name="xenium_all_cells",
        n_cores=40,
        output_path="xenium_annotated_heatmap.png",
    ),
)
```

---

## Benchmarking and Validation

### Experimental inference options

The proposal branches preserve the six performance PRs as explicit merges. The assembled fork also carries optional inference experiments:

- `--backend gpu-compat` routes Ward clustering, selected GMM fits, and baseline adjustment through the optional CUDA backend while retaining the CPU PCA policy. `--backend gpu` uses the full-feature GPU Ward policy. Other pipeline stages remain on CPU in this integration. Install `.[gpu]` and a CuPy build matching the local CUDA toolkit.
- `--ks-method exact` replaces Monte Carlo comparisons of sampled Gamma posteriors with an exact posterior-Gamma KS distance. This changes breakpoint decisions and remains opt-in; the original cutoff has not been recalibrated for it.
- `--anchor markers` selects an immune/endothelial marker-enriched normal reference when available. `--final-call arm_correlation` changes final human hg20 cell calls using arm-level profiles; both are opt-in and intended for solid tumors.
- `copykat-py-allele` provides a separate read-counting, phasing, and allele-orientation workflow. It needs external sequencing tools and data described in [the allele workflow guide](docs/allele_orientation.md).

The review package in `docs/navin-review-manifest.json`, `docs/fork-integration-audit.md`, and the linked benchmark reports records source commits, dependencies, numerical/output contracts, measured gains, regressions, and incomplete evidence. Existing timings remain evidence for their pinned snapshots, not this assembled commit.

### Validation for 11 datasets from [Cancer Cell Atlas (3CA)](https://www.weizmann.ac.il/sites/3CA/)

Both CopyKAT-R and CopyKAT-Python were tested on raw datasets (no QC filtering) using 24 cores. 
A total of 11 datasets with cell-type composition, aneuploid annotation, and UMAP embeddings from the metadata, and prepare per-sample count matrices in standard 10X MTX format for downstream CopyKAT-R vs CopyKAT-Py comparison.

<details>
<summary><b>3CA Benchmark Datasets</b></summary>

<br>

| # | Dataset | Cancer Type | Sample | n_cells | Tumor % (meta) | Ref |
|---|---------|-------------|--------|---------|----------------|-----|
| 1 | Gao2021_Breast | Breast cancer | DCIS1 | 1,480 | 74.4% | Gao et al. 2021 |
| 2 | Chen2020_Head-and-Neck | Nasopharyngeal carcinoma | P11 | 6,890 | 26.3% | Chen et al. 2020 |
| 3 | Laughney2020_Lung | Lung adenocarcinoma | RU681 | 993 | 77.0% | Laughney et al. 2020 |
| 4 | Bi2021_Kidney | Renal cell carcinoma (RCC) | P90 | 8,426 | 39.4% | Bi et al. 2021 |
| 5 | Dong2020_Prostate | Prostate cancer | patient #5 | 8,690 | 19.3% | Dong et al. 2020 |
| 6 | Jerby-Arnon2021_Sarcoma | Synovial sarcoma | SyS14 | 2,522 | 94.4% | Jerby-Arnon et al. 2021 |
| 7 | Choudhury2022_Brain | Meningioma | MSC6-BTI | 13,171 | 62.2% | Choudhury et al. 2022 |
| 8 | Lin2020_Pancreas | PDAC | P08 | 1,139 | 74.1% | Lin et al. 2020 |
| 9 | Lee2020_Colorectal | Colorectal cancer (CRC) | SMC09 | 2,272 | 77.9% | Lee et al. 2020 |
| 10 | Geistlinger2020_Ovarian | HGSOC | T59 | 12,659 | 25.1% | Geistlinger et al. 2020 |
| 11 | Ji2020_Skin | Cutaneous SCC | P4 | 7,956 | 53.0% | Ji et al. 2020 |

</details>

**Key Metrics Comparison**
<img width="2198" height="1874" alt="image" src="https://github.com/user-attachments/assets/764b4b2c-aac6-4e57-8630-5bc0104cbe3a" />

<img width="3125" height="1361" alt="image" src="https://github.com/user-attachments/assets/49e72ee4-6d6d-48b5-82d2-ea8738c01a99" />

#### Side-by-Side Comparison: CopyKAT-R vs CopyKAT-Python

<img width="1157" height="807" alt="image" src="https://github.com/user-attachments/assets/77fcaba2-8f5a-4b88-83fb-9245e38c881a" />

<img width="1157" height="807" alt="image" src="https://github.com/user-attachments/assets/2cefece9-6d78-492e-8430-596f4cf4eab1" />

---

### Large-Scale Testing: Xenium Atera Dataset

The full [FFPE Human Breast Cancer](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer) Xenium (Atera) dataset was subsetted to 50k, 100k, and full (~170k cells) to evaluate scalability.

**Runtime Comparison**
<img width="1014" height="677" alt="image" src="https://github.com/user-attachments/assets/f534aca8-2d7b-40c6-acb1-dafb1775e224" />

**CNV Heatmap with Annotation**

<img width="1150" height="874" alt="image" src="https://github.com/user-attachments/assets/6fdd0ee4-2398-4723-886e-17e1ab49b703" />


**CNV comparison**
<img width="1296" height="720" alt="image" src="https://github.com/user-attachments/assets/d21e99ce-2dcf-4d64-813f-2de59ef8641a" />

## Why Results May Differ from CopyKAT-R

From the above comparison of the final prediction, the Seurat cluster 4 was called diploid by CopyKAT-R but aneuploid by CopyKAT-Py.
The copykat-py call was confirmed correct through the corresponding H&E cell morphology in this case. 

The key difference is in the final prediction step (step 8), where both implementations perform hierarchical clustering on the adjusted CNA matrix and cut the tree at k=2. R's copykat explicitly uses method = "ward.D" in hclust(), while CopyKAT-Python uses scipy/fastcluster's "ward", which implements the mathematically correct ward.D2 criterion.
For cells cluster (like  Seurat cluster 4 here,) with subtle CNV profiles that sit near the boundary of the diploid/aneuploid split, the two linkage variants produce different dendrogram topologies, causing the binary label assignment to flip. 

CopyKAT-Python results may not be identical to CopyKAT-R due to differences in:

**High-confidence results** typically show:
- Clear chromosome-arm or whole-chromosome CNV patterns
- Consistent CNV profiles within clusters
- Strong separation between inferred diploid and aneuploid cells

**Lower-confidence results** may occur in samples with:
- Weak CNV signal or low sequencing depth
- Few normal reference cells
- Strong batch effects
- Near-diploid tumor genomes


**Disclaimer:** 
CopyKAT-Python is an independent reimplementation focused on scalability and usability, while faithfully reproducing the core CopyKAT analytical strategy.
Currently, CopyKat-Python is under internal testing.

- Gene annotation versions
- Filtering and preprocessing steps
- Numerical implementation details
- Smoothing and segmentation algorithms
- Clustering behavior (parDist + hcluster vs. PCA + fastcluster)
