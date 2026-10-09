# Allele-based orientation of CopyKAT calls

CopyKAT's final two-way split is often right while its labels are swapped: when the normal
reference is wrong, the tumour cells are called "diploid". Expression alone cannot settle which
side is the tumour. Alleles can: a copy-number change makes one parental haplotype over-represented
across long stretches of the genome, in tumour cells only, and no cell type can imitate that.

`copykat_py.allele` checks CopyKAT's calls this way and flips them when the "diploid" group is the
one carrying the allelic imbalance. It needs the reads (FASTQ, SRA or a Cell Ranger BAM), not only
a count matrix. Evidence and limits: [research/allele_orientation.md](research/allele_orientation.md).

## What it does

For each of CopyKAT's two groups of cells, a hidden Markov model scans the group's pooled, phased
heterozygous SNPs for allelically imbalanced segments (>= 5 Mb, >= 20 SNPs). F = fraction of the
covered genome in such segments. If F(diploid) - F(aneuploid) >= 0.02, the calls are flipped.
Germline allele-specific expression affects single genes, not long phased segments, so it does not
trigger the check; tumours without copy-number changes give F near 0 in both groups and the calls
are left alone.

## Requirements

| Step | Tools |
|---|---|
| allele counts from reads (`count`) | CUDA GPU + PyTorch; `igzip` (ISA-L) for FASTQ.gz or `fastq-dump`/`vdb-dump` (sra-tools) for SRA; `gcc` |
| allele counts from a Cell Ranger BAM | [cellsnp-lite](https://github.com/single-cell-genetics/cellsnp-lite) instead of `count` |
| phasing (`panel`, `phase`) | [Eagle2](https://alkesgroup.broadinstitute.org/Eagle/) v2.4.1, `bcftools`, `bgzip` |
| reference data | GRCh38 FASTA, GENCODE GTF, a biallelic common-SNP list (e.g. cellsnp-lite's `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz`), the 1000 Genomes high-coverage phased panel (`1kGP_high_coverage_Illumina.chr{1..22}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz`, EBI FTP) |

All of these are optional: a plain CopyKAT-Py run never imports `copykat_py.allele`.

## One-off preparation

```bash
# k-mer index for the GPU counter: genome-wide SNPs + exon-junction k-mers (~5 min, 4.2 GB)
copykat-py-allele build-index --fasta GRCh38.fa --snps genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
    --gtf gencode.v44.primary_assembly.basic.annotation.gtf --out allele_index

# phasing panel: the 1000 Genomes panel reduced to the SNP list (biallelic SNPs, BCF per chromosome)
copykat-py-allele panel --panel-vcf-dir 1kGP_panel/ --snps genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
    --out phasing_panel
```

## Per sample

```bash
# 1. CopyKAT as usual; barcodes must match the read data (a '-1' suffix is ignored)
# 2. per-cell allele counts (GPU, no alignment): FASTQ.gz pair or .sra file
copykat-py-allele count --index allele_index --whitelist barcodes.txt --out allele_counts \
    --fastq S1_R1_001.fastq.gz S1_R2_001.fastq.gz        # or: --sra SRRxxxx.sra
#    (or, from a Cell Ranger BAM: cellsnp-lite -s possorted_genome_bam.bam -b barcodes.txt \
#        -R genome1K...hg38.vcf.gz -O allele_counts --minMAF 0 --minCOUNT 1 --gzip)
# 3. phase the sample's heterozygous SNPs
copykat-py-allele phase --counts allele_counts --panel phasing_panel \
    --genetic-map Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz --out phase.csv
# 4a. check an existing prediction ...
copykat-py-allele orient --prediction SAMPLE_copykat_prediction.txt --counts allele_counts --phase phase.csv
# 4b. ... or let copykat-py do it at the end of a run
copykat-py -i matrix.mtx --genes genes.txt --barcodes barcodes.txt --sample-name SAMPLE --backend gpu \
    --anchor markers --final-call arm_correlation --allele-counts allele_counts --allele-phase phase.csv
```

`orient` writes `SAMPLE_copykat_prediction.allele_oriented.txt` and `SAMPLE_copykat_allele_orientation.json`
(F per group, segments, cells used, flipped). With `copykat-py --allele-counts`, a flipped run
rewrites `SAMPLE_copykat_prediction.txt`, keeps the original as `..._prediction.before_allele.txt` and
writes `..._allele_orientation.json`; the heatmap keeps the pre-orientation labels.

## Speed (RTX 5080, 32 cores)

| Sample | Read pairs | Counting (GPU) | STARsolo + cellsnp-lite |
|---|---|---|---|
| BCC06, FASTQ.gz | 121 M | 1.3 min | ~32 min |
| Rao2020 PriNET, .sra | 45 M | 1.1 min | ~5-6 min |
| Dong2020 T19, .sra | 319 M | 7.9 min | ~45 min |

Phasing takes a few minutes per sample; the orientation check seconds to a minute.

## Limits

- Needs reads. Count matrices alone carry no allele information.
- Only orients: it never moves cells between CopyKAT's groups, so the split quality is CopyKAT's.
- Abstains on tumours without sizeable copy-number changes (both groups F ~ 0).
- Misses tumour-dominated samples where both of CopyKAT's groups are mostly tumour (both carry the
  same imbalance).
- Per-cell allele calls are deliberately not offered: at 10x depth they were not safe (cell-type
  allele-specific expression mimics imbalance in single cells).
- GRCh38 only (index, panel and genetic map). hg19 BAMs can be piled up with an hg19 SNP list and
  lifted over by rsID before phasing.
