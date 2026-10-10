"""Per-cell allele counts in cellsnp-lite's output format, heterozygous-SNP selection and phasing."""

import os

import numpy as np
import pandas as pd
from scipy import io, sparse

MIN_POOLED_DEPTH = 8  # heterozygous SNP: pooled UMIs over all cells ...
MIN_ALLELE_DEPTH = 2  # ... both alleles seen at least this often ...
MIN_MINOR_FRACTION = 0.1  # ... and the minor allele at least this fraction


def _mtx(directory, tag):
    path = os.path.join(directory, f"cellSNP.tag.{tag}.mtx")
    return io.mmread(path + ".gz" if os.path.exists(path + ".gz") else path).tocsr().astype(np.int64)


def barcode_key(cell):
    """Barcode without the 10x '-1' style suffix, used to match cells across tools."""
    return str(cell).split("-")[0]


def load_counts(directory):
    """Read a cellsnp-lite output directory.

    Returns (snps, AD, DP, cells): snps is a DataFrame with chr (no 'chr' prefix), pos, ref, alt;
    AD (ALT UMIs) and DP (REF+ALT UMIs) are SNP x cell CSR matrices; cells are the barcodes.
    """
    snps = pd.read_csv(
        os.path.join(directory, "cellSNP.base.vcf.gz"),
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4],
        names=["chr", "pos", "ref", "alt"],
        dtype={"chr": str},
        keep_default_na=False,
    )
    snps["chr"] = snps.chr.str.replace("chr", "", regex=False)
    cells = pd.read_csv(os.path.join(directory, "cellSNP.samples.tsv"), header=None)[0].astype(str).to_numpy()
    return snps, _mtx(directory, "AD"), _mtx(directory, "DP"), cells


def heterozygous(snps, AD, DP):
    """Keep SNPs that look heterozygous in the pooled cells (no labels are used)."""
    alt = np.asarray(AD.sum(axis=1)).ravel()
    dp = np.asarray(DP.sum(axis=1)).ravel()
    minor = np.minimum(alt, dp - alt)
    keep = (dp >= MIN_POOLED_DEPTH) & (minor >= MIN_ALLELE_DEPTH) & (minor >= MIN_MINOR_FRACTION * dp)
    return snps[keep].reset_index(drop=True), AD[keep], DP[keep]


def read_phase(path):
    """Phase table (CSV: chr, pos, h; h = 1 if the ALT allele is on haplotype A)."""
    ph = pd.read_csv(path, dtype={"chr": str})
    ph["chr"] = ph.chr.str.replace("chr", "", regex=False)
    return ph


def haplotype_counts(snps, AD, DP, phase):
    """Phased SNPs only, sorted by chromosome and position.

    Returns (snps[chr, pos], HA, DP) with HA = UMIs on haplotype A (SNP x cell CSR)."""
    hmap = pd.Series(phase.h.to_numpy(), index=phase.chr + ":" + phase.pos.astype(str))
    hmap = hmap[~hmap.index.duplicated()]
    h = (snps.chr + ":" + snps.pos.astype(str)).map(hmap)
    keep = h.notna().to_numpy()
    h = h[keep].to_numpy().astype(int)
    snps, AD, DP = snps[keep].reset_index(drop=True), AD[keep], DP[keep]
    HA = (sparse.diags(h) @ AD + sparse.diags(1 - h) @ (DP - AD)).tocsr()
    order = np.lexsort((snps.pos.to_numpy(), snps.chr.to_numpy()))
    return snps.iloc[order].reset_index(drop=True)[["chr", "pos"]], HA[order], DP[order].tocsr()
