# mm10 genomic-order validation

The mm10 annotation table stores `abspos` as a chromosome offset. Multiple
genes on the same chromosome therefore have the same `abspos`; sorting only by
that field can interleave genes and changes the genomic windows consumed by
smoothing and segmentation. The correction orders mouse genes by numeric
chromosome and gene start, while applying the same permutation to expression
rows. Human `hg20` ordering remains by `abspos`.

## Biological comparison

The validation used the T989 mouse tumor single-cell RNA count matrix and the
matched scWGS/AneuFinder pseudobulk gain/loss reference. Input files are in
`/home/toresbe/cancer_research/mm10_test/T989/`; the reference table is
`/home/toresbe/cancer_research/cnv_benchmark_data/zenodo_20260649/mouse_wgs_results_formated.csv`.
The matrix, gene, barcode, and reference SHA-256 values respectively are:

```text
113e3dc37062793b046ec559f44619603daba1362080cabab83d071c0d0a113e  matrix.mtx
905b2d3eaa90ddaba1e07a432665add975984d3c45b65940689f5d490aad9c  genes.tsv
624f261a6921922fb6f289cb088cdd3d7ecaaa8c3a8ef4e7758aa3d4eacadc78  barcodes.tsv
433f897f87cd38e0df056d697b51c1bf8e41ae0e865b1cb8fd620e5a8c3c4bdd  mouse_wgs_results_formated.csv
```

To isolate ordering, both analyses used the same current CPU implementation and
parameters (`id_type=S`, `cell_line=no`, `ngene_chr=5`, `min_gene_per_cell=200`,
`LOW_DR=0.05`, `UP_DR=0.1`, `win_size=25`, `KS_cut=0.1`, Euclidean distance,
8 cores, seed 1234, auto reference selection, and plotting disabled). The
control monkeypatched the annotation order to the previous stable `abspos`
sort; the comparison used chromosome/start order. Both retained 7,751 annotated
genes and 8,740 cells before later filtering. Outputs and runtime logs are kept
on NAS under `/mnt/nas/cancer_research/mm10_genomic_order_validation/`.

The WGS comparison joins each CopyKAT gene start to the overlapping reference
gain-minus-loss interval and correlates the mean CNA profile of CopyKAT-called
aneuploid cells against that reference. Both orders matched 5,760 of 5,929 CNA
genes to WGS intervals.

| Ordering | Aneuploid | Diploid | Not defined | Gene Pearson r | Gene Spearman r | Chromosome-mean Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable `abspos` control | 5,141 | 3,524 | 698 | 0.348 | 0.190 | 0.800 |
| Chromosome, gene start | 4,308 | 4,357 | 698 | 0.490 | 0.250 | 0.831 |

The corrected order improves concordance with the matched WGS reference in
these measurements. It also changes the diploid/aneuploid call counts
substantially, which is expected when smoothing and segmentation windows are
restored to genomic order. This is evidence from one biological sample, not a
general accuracy guarantee; the shift in cell calls should be considered when
interpreting older mm10 results. Such runs need to be rerun from raw counts.

The independent eT sample also completed on the corrected implementation
(364 low-confidence diploid, 330 low-confidence aneuploid, 2 not defined), but
no matched WGS reference was available for that sample, so it is not included
in the concordance metrics above.
