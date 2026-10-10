"""Phase a sample's heterozygous SNPs with Eagle2 against the 1000 Genomes reference panel (GRCh38).

One-off panel preparation (``build_panel``): the 1000 Genomes high-coverage phased panel
(1kGP_high_coverage_Illumina.chr{1..22}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz, from the EBI FTP
data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV) is reduced to
biallelic SNPs at the sites of the SNP list used for counting, as BCF.
Per sample (``phase_sample``): the pooled heterozygous SNPs are written as a single-sample VCF per
autosome and phased with Eagle2 (``--vcfRef``); output CSV chr, pos, ref, h (h = 1 if the ALT allele
is on haplotype A), in the counts' allele coding even where Eagle swapped REF/ALT.
External tools: Eagle2 (v2.4.1; its tables/genetic_map_hg38_withX.txt.gz), bcftools, bgzip.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from copykat_py.allele import counts as _counts

AUTOSOMES = [str(c) for c in range(1, 23)]
PANEL_NAME = "1kGP_high_coverage_Illumina.chr{c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"


def _run(cmd):
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def _tool(name, path=None):
    found = path or shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found; install it or pass its path")
    return found


def build_panel(panel_vcf_dir, snp_vcf, out_dir, bcftools=None, jobs=8):
    """Reduce the 1000 Genomes panel to biallelic SNPs at the SNP list's sites (chr-prefixed, GRCh38)."""
    bcftools = _tool("bcftools", bcftools)
    os.makedirs(out_dir, exist_ok=True)
    sites = os.path.join(out_dir, "sites.tsv.gz")
    if not os.path.exists(sites):
        _run(
            f'zcat {snp_vcf} | grep -v \'^#\' | awk \'BEGIN{{OFS="\\t"}} {{c=$1; sub(/^chr/,"",c); print "chr"c,$2}}\' '
            f"| gzip > {sites}"
        )
    pending = [c for c in AUTOSOMES if not os.path.exists(os.path.join(out_dir, f"panel_chr{c}.bcf.csi"))]
    for i in range(0, len(pending), jobs):
        procs = []
        for c in pending[i : i + jobs]:
            src = os.path.join(panel_vcf_dir, PANEL_NAME.format(c=c))
            out = os.path.join(out_dir, f"panel_chr{c}.bcf")
            procs.append(
                subprocess.Popen(
                    f"{bcftools} view -T {sites} -m2 -M2 -v snps {src} -Ob -o {out} && {bcftools} index -f {out}",
                    shell=True,
                    executable="/bin/bash",
                )
            )
        if any(p.wait() for p in procs):
            raise RuntimeError("bcftools failed while building the panel")


def phase_sample(
    counts_dir, panel_dir, genetic_map, out_csv, eagle=None, bcftools=None, bgzip=None, threads=16, work_dir=None
):
    """Phase the pooled heterozygous SNPs of one sample; writes out_csv and returns the table."""
    eagle, bcftools, bgzip = _tool("eagle", eagle), _tool("bcftools", bcftools), _tool("bgzip", bgzip)
    snps, AD, DP, _ = _counts.load_counts(counts_dir)
    het, _, _ = _counts.heterozygous(snps, AD, DP)
    het = het[het.chr.isin(AUTOSOMES)].drop_duplicates(["chr", "pos"])
    tmp = work_dir or tempfile.mkdtemp(prefix="copykat_phase_")
    os.makedirs(tmp, exist_ok=True)
    out = []
    for c in AUTOSOMES:
        h = het[het.chr == c].sort_values("pos")
        panel = os.path.join(panel_dir, f"panel_chr{c}.bcf")
        if len(h) < 2 or not os.path.exists(panel + ".csi"):
            continue
        vcf = os.path.join(tmp, f"target_chr{c}.vcf")
        with open(vcf, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write(f"##contig=<ID=chr{c}>\n")
            f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n")
            for r in h.itertuples():
                f.write(f"chr{c}\t{r.pos}\t.\t{r.ref}\t{r.alt}\t.\tPASS\t.\tGT\t0/1\n")
        _run(f"{bgzip} -f {vcf} && {bcftools} index -f {vcf}.gz")
        prefix = os.path.join(tmp, f"phased_chr{c}")
        _run(
            f"{eagle} --vcfRef {panel} --vcfTarget {vcf}.gz --geneticMapFile {genetic_map} --outPrefix {prefix} "
            f"--chrom chr{c} --numThreads {threads} --allowRefAltSwap > {prefix}.log 2>&1"
        )
        ph = pd.read_csv(
            f"{prefix}.vcf.gz", sep="\t", comment="#", header=None, usecols=[1, 3, 9], names=["pos", "ref", "gt"]
        )
        ph = ph[ph["gt"].isin(["0|1", "1|0"])].merge(h[["pos", "ref"]].rename(columns={"ref": "ref0"}), on="pos")
        hap = (ph["gt"] == "1|0").astype(int).to_numpy()
        hap = np.where(ph.ref == ph.ref0, hap, 1 - hap)  # Eagle may swap REF/ALT; keep the counts' coding
        out.append(pd.DataFrame({"chr": c, "pos": ph.pos, "ref": ph.ref0, "h": hap}))
    if not out:
        raise RuntimeError("no chromosome could be phased (panel missing or too few heterozygous SNPs)")
    res = pd.concat(out, ignore_index=True)
    res.to_csv(out_csv, index=False)
    if work_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return res
