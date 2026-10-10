"""copykat-py-allele: allele-based orientation of CopyKAT's tumour/normal calls.

  copykat-py-allele build-index --fasta GRCh38.fa --snps SNPS.vcf.gz --gtf genes.gtf --out INDEX
  copykat-py-allele count --index INDEX --whitelist BARCODES --out COUNTS (--fastq R1 R2 | --sra FILE)
  copykat-py-allele panel --panel-vcf-dir 1KGP_DIR --snps SNPS.vcf.gz --out PANEL
  copykat-py-allele phase --counts COUNTS --panel PANEL --genetic-map MAP --out phase.csv
  copykat-py-allele orient --prediction SAMPLE_copykat_prediction.txt --counts COUNTS --phase phase.csv

COUNTS is a cellsnp-lite-format directory: from `count`, or from cellsnp-lite on a Cell Ranger BAM.
"""

import argparse
import json
import sys


def _build_index(a):
    from copykat_py.allele.kmer_index import build_index

    n_keys, n_snps = build_index(
        a.fasta,
        a.snps,
        a.out,
        gtf=a.gtf,
        genome_wide=not a.gene_bodies_only,
        junctions=not a.no_junctions,
        device=a.device,
    )
    print(f"index: {n_keys} k-mers for {n_snps} SNPs in {a.out}")


def _count(a):
    from copykat_py.allele.gpu_count import count_alleles

    count_alleles(
        a.index,
        a.whitelist,
        a.out,
        fastq=a.fastq,
        sra=a.sra,
        umi_len=a.umi_len,
        workers=a.workers,
        igzip=a.igzip,
        fastq_dump=a.fastq_dump,
        device=a.device,
        log=lambda m: print(m, flush=True),
    )


def _panel(a):
    from copykat_py.allele.phase import build_panel

    build_panel(a.panel_vcf_dir, a.snps, a.out, bcftools=a.bcftools, jobs=a.jobs)


def _phase(a):
    from copykat_py.allele.phase import phase_sample

    res = phase_sample(
        a.counts, a.panel, a.genetic_map, a.out, eagle=a.eagle, bcftools=a.bcftools, bgzip=a.bgzip, threads=a.threads
    )
    print(f"{len(res)} heterozygous SNPs phased -> {a.out}")


def _orient(a):
    from copykat_py.allele.orient import orient_prediction_file

    _, report = orient_prediction_file(
        a.prediction, a.counts, a.phase, out_prefix=a.out_prefix, min_f_diff=a.min_f_diff
    )
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("segments_")}))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="copykat-py-allele", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-index", help="k-mer index for the GPU counter (once per genome/SNP list)")
    p.add_argument("--fasta", required=True, help="reference genome FASTA (GRCh38)")
    p.add_argument("--snps", required=True, help="biallelic SNP list VCF(.gz), e.g. 1000G AF >= 5%%")
    p.add_argument("--gtf", help="gene annotation GTF (needed for junction k-mers / --gene-bodies-only)")
    p.add_argument("--out", required=True)
    p.add_argument("--gene-bodies-only", action="store_true", help="index only SNPs in gene bodies (+-1 kb)")
    p.add_argument("--no-junctions", action="store_true", help="skip exon-junction k-mers")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_build_index)

    p = sub.add_parser("count", help="per-cell allele counts from 10x reads (GPU, no alignment)")
    p.add_argument("--index", required=True)
    p.add_argument("--whitelist", required=True, help="cell barcodes, one per line")
    p.add_argument("--out", required=True, help="output directory (cellsnp-lite format)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fastq", nargs=2, metavar=("R1", "R2"), help="gzipped FASTQ pair")
    src.add_argument("--sra", help=".sra file (with technical reads)")
    p.add_argument("--umi-len", type=int, default=10, help="UMI length (10x v2: 10, v3: 12)")
    p.add_argument("--workers", type=int, default=12, help="parallel fastq-dump decoders for --sra")
    p.add_argument("--igzip", help="path to igzip (ISA-L)")
    p.add_argument("--fastq-dump", help="path to fastq-dump (sra-tools)")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_count)

    p = sub.add_parser("panel", help="reduce the 1000 Genomes phased panel to the SNP list (once)")
    p.add_argument("--panel-vcf-dir", required=True, help="directory with the 1kGP high-coverage panel VCFs")
    p.add_argument("--snps", required=True, help="SNP list VCF(.gz) used for counting")
    p.add_argument("--out", required=True)
    p.add_argument("--bcftools")
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=_panel)

    p = sub.add_parser("phase", help="phase a sample's heterozygous SNPs with Eagle2")
    p.add_argument("--counts", required=True, help="cellsnp-lite-format directory")
    p.add_argument("--panel", required=True, help="directory from `panel`")
    p.add_argument("--genetic-map", required=True, help="Eagle2 tables/genetic_map_hg38_withX.txt.gz")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--eagle")
    p.add_argument("--bcftools")
    p.add_argument("--bgzip")
    p.add_argument("--threads", type=int, default=16)
    p.set_defaults(func=_phase)

    p = sub.add_parser("orient", help="check / flip a CopyKAT prediction with the phased-BAF HMM")
    p.add_argument("--prediction", required=True, help="CopyKAT *_copykat_prediction.txt")
    p.add_argument("--counts", required=True, help="cellsnp-lite-format directory")
    p.add_argument("--phase", required=True, help="CSV from `phase`")
    p.add_argument("--out-prefix", help="output prefix (default: next to the prediction file)")
    p.add_argument(
        "--min-f-diff",
        type=float,
        default=0.02,
        help="flip if F(diploid) - F(aneuploid) >= this (pre-registered: 0.02)",
    )
    p.set_defaults(func=_orient)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
