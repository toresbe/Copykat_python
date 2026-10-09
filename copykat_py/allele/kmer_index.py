"""Build the k-mer -> (SNP, allele) index used by the GPU allele counter.

For each SNP of a biallelic SNP list (e.g. 1000 Genomes AF >= 5%, GRCh38), every 31-mer containing the
SNP is generated for the REF and the ALT allele (other positions = reference sequence) and stored in
canonical form (min of the k-mer and its reverse complement, 2 bits per base). A k-mer is kept if it
belongs to exactly one (SNP, allele) and, scanning the whole genome on the GPU, a REF k-mer occurs exactly
once (its own locus) and an ALT k-mer never.
With a GTF, exon-junction k-mers are added for SNPs near splice junctions: exons of each transcript are
concatenated and the 31-mers that contain the SNP and cross a junction are kept if they occur nowhere in
the genome and do not collide with another (SNP, allele); a key also present among the genomic k-mers
with a different value is dropped from both.
Output directory: keys.npy (int64, sorted), vals.npy (int32 = 2 * snp + allele; allele 0 REF, 1 ALT),
snps.tsv (chr, pos, ref, alt; row number = snp id). Genome-wide with junctions: ~361 M k-mers, 4.2 GB,
~5 min on one GPU.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

K = 31
GENOME_CHUNK = 50_000_000
_LUT = np.full(256, 4, np.uint8)            # A C G T (either case) -> 0..3, everything else -> 4 (N)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[_b + 32] = _i


def read_genome(fasta):
    """{chromosome without 'chr': uint8 codes} from a FASTA file."""
    seqs, name, buf = {}, None, []
    with open(fasta, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if name:
                    seqs[name] = _LUT[np.frombuffer(b"".join(buf), np.uint8)]
                name, buf = line[1:].split()[0].decode().replace("chr", ""), []
            else:
                buf.append(line.rstrip())
    seqs[name] = _LUT[np.frombuffer(b"".join(buf), np.uint8)]
    return seqs


def kmer_codes(windows, device):
    """(M, L) uint8 codes on `device` -> (M, L-K+1) int64 canonical k-mer codes, -1 where a k-mer has N."""
    w = windows.to(torch.int64)
    n = w.shape[1] - K + 1
    fwd = torch.zeros((w.shape[0], n), dtype=torch.int64, device=device)
    rev = torch.zeros((w.shape[0], n), dtype=torch.int64, device=device)
    bad = torch.zeros((w.shape[0], n), dtype=torch.bool, device=device)
    for i in range(K):
        col = w[:, i:i + n]
        bad |= col == 4
        c = col.clamp(max=3)
        fwd = (fwd << 2) | c
        rev = rev | ((3 - c) << (2 * i))
    can = torch.minimum(fwd, rev)
    can[bad] = -1
    return can


def genome_occurrences(keys, genome, device):
    """Occurrences of each (sorted) key among all k-mers of the genome, both strands."""
    keys_t = torch.from_numpy(keys).to(device)
    counts = torch.zeros(len(keys), dtype=torch.int32, device=device)
    for seq in genome.values():
        for st in range(0, len(seq), GENOME_CHUNK):
            s = torch.from_numpy(seq[st:min(st + GENOME_CHUNK + K - 1, len(seq))]).to(device)
            q = kmer_codes(s[None, :], device)[0]
            q = q[q >= 0]
            i = torch.searchsorted(keys_t, q).clamp(max=len(keys) - 1)
            hit = keys_t[i] == q
            counts += torch.bincount(i[hit], minlength=len(keys)).to(torch.int32)
            del q, i, hit
    return counts.cpu().numpy()


def _unique_keys(keys, vals):
    """Sort by key and drop keys that occur more than once."""
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    single = np.r_[True, keys[1:] != keys[:-1]] & np.r_[keys[1:] != keys[:-1], True]
    return keys[single], vals[single]


def _first_base_codes(series):
    return _LUT[series.str.encode("ascii").map(lambda b: b[0]).to_numpy().astype(np.uint8)]


def select_snps(genome, snp_vcf, gtf=None, gene_flank=1000):
    """Biallelic single-base SNPs whose REF matches the genome; with a GTF, only those in gene bodies
    (+- gene_flank); without one, genome-wide."""
    s = pd.read_csv(snp_vcf, sep="\t", comment="#", header=None, usecols=[0, 1, 3, 4],
                    names=["chr", "pos", "ref", "alt"], dtype={"chr": str})
    s["chr"] = s.chr.str.replace("chr", "", regex=False)
    s = s[(s.ref.str.len() == 1) & (s.alt.str.len() == 1)]
    keep = np.ones(len(s), bool)
    if gtf is not None:
        g = pd.read_csv(gtf, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4], names=["chr", "type", "start", "end"])
        g = g[g.type == "gene"]
        g["chr"] = g.chr.str.replace("chr", "", regex=False)
        keep[:] = False
        for c, gc in g.groupby("chr"):
            m = (s.chr == c).to_numpy()
            if not m.any() or c not in genome:
                continue
            iv = np.zeros(len(genome[c]) + 1, np.int32)
            np.add.at(iv, np.clip(gc.start.to_numpy() - 1 - gene_flank, 0, len(iv) - 1), 1)
            np.add.at(iv, np.clip(gc.end.to_numpy() + gene_flank, 0, len(iv) - 1), -1)
            cov = np.cumsum(iv) > 0
            keep[m] = cov[np.clip(s.pos.to_numpy()[m] - 1, 0, len(cov) - 1)]
    s = s[keep & s.chr.isin(list(genome)).to_numpy()].reset_index(drop=True)
    s = s[np.array([p <= len(genome[c]) for c, p in zip(s.chr, s.pos)])].reset_index(drop=True)
    ref = np.array([genome[c][p - 1] for c, p in zip(s.chr, s.pos)])
    return s[ref == _first_base_codes(s.ref)].reset_index(drop=True)


def snp_kmers(genome, snps, device):
    """Genomic k-mers of every SNP (REF and ALT), filtered for uniqueness as described above."""
    keys, vals = [], []
    for c, sc in snps.groupby("chr"):
        seq = genome[c]
        for b0 in range(0, len(sc), 200_000):
            part = sc.iloc[b0:b0 + 200_000]
            idx = (part.pos.to_numpy() - 1)[:, None] + np.arange(-(K - 1), K)[None, :]
            inside = (idx >= 0).all(1) & (idx < len(seq)).all(1)
            win = seq[np.clip(idx, 0, len(seq) - 1)]
            for allele in (0, 1):
                w = win.copy()
                if allele == 1:
                    w[:, K - 1] = _first_base_codes(part.alt)
                codes = kmer_codes(torch.from_numpy(w).to(device), device).cpu().numpy()
                v = (2 * part.index.to_numpy()[:, None] + allele) * np.ones((1, K), np.int64)
                m = (codes >= 0) & inside[:, None]
                keys.append(codes[m])
                vals.append(v[m].astype(np.int32))
    keys, vals = _unique_keys(np.concatenate(keys), np.concatenate(vals))
    counts = genome_occurrences(keys, genome, device)
    good = np.where((vals & 1) == 0, counts == 1, counts == 0)
    return keys[good], vals[good]


def junction_kmers(genome, gtf, snps, device):
    """Exon-junction k-mers containing a SNP (REF and ALT); absent from the genome and unambiguous."""
    g = pd.read_csv(gtf, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 8],
                    names=["chr", "type", "start", "end", "attr"])
    ex = g[g.type == "exon"].copy()
    ex["chr"] = ex.chr.str.replace("chr", "", regex=False)
    ex["tx"] = ex.attr.str.extract(r'transcript_id "([^"]+)"')[0]
    ex = ex[ex.chr.isin(snps.chr.unique())]
    by_chr = {c: (s.pos.to_numpy() - 1, s.index.to_numpy(), s.alt.to_numpy()) for c, s in snps.groupby("chr")}
    wins, snp_ids, alts = [], [], []
    for (c, _), e in ex.groupby(["chr", "tx"], sort=False):
        if len(e) < 2:
            continue
        e = e.sort_values("start")
        tpos = np.concatenate([np.arange(s - 1, t) for s, t in zip(e.start, e.end)])   # 0-based genomic
        jump = np.flatnonzero(np.diff(tpos) != 1)          # a junction follows transcript index jump[i]
        p, sid, alt = by_chr[c]
        lo, hi = np.searchsorted(p, tpos[0]), np.searchsorted(p, tpos[-1], side="right")
        if hi <= lo:
            continue
        ti = np.searchsorted(tpos, p[lo:hi])
        inexon = (ti < len(tpos)) & (tpos[np.minimum(ti, len(tpos) - 1)] == p[lo:hi])
        ti, s_ids, s_alt = ti[inexon], sid[lo:hi][inexon], alt[lo:hi][inexon]
        if len(ti) == 0:
            continue
        d = np.min(np.abs(ti[:, None] - (jump[None, :] + 0.5)), axis=1)
        ok = (d < K - 1) & (ti >= K - 1) & (ti + K - 1 < len(tpos))
        for t_i, s_i, a in zip(ti[ok], s_ids[ok], s_alt[ok]):
            wins.append(tpos[t_i - (K - 1): t_i + K])
            snp_ids.append(s_i)
            alts.append(a)
    W, snp_ids, alts = np.array(wins), np.array(snp_ids), np.array(alts)
    chrs = snps.chr.to_numpy()[snp_ids]
    seq = np.empty(W.shape, np.uint8)
    for c in np.unique(chrs):
        m = chrs == c
        seq[m] = genome[c][W[m]]
    contig = np.diff(W, axis=1) == 1
    crosses = np.stack([~contig[:, j:j + K - 1].all(1) for j in range(K)], 1)
    keys, vals = [], []
    for allele in (0, 1):
        s = seq.copy()
        if allele == 1:
            s[:, K - 1] = _LUT[np.frombuffer("".join(alts).encode(), np.uint8)]
        codes = kmer_codes(torch.from_numpy(s).to(device), device).cpu().numpy()
        v = np.repeat((2 * snp_ids + allele)[:, None], K, 1)
        m = crosses & (codes >= 0)
        keys.append(codes[m])
        vals.append(v[m].astype(np.int32))
    kv = np.unique(np.stack([np.concatenate(keys), np.concatenate(vals).astype(np.int64)], 1), axis=0)
    keys, vals = kv[:, 0], kv[:, 1].astype(np.int32)        # same junction in several transcripts -> once
    single = np.r_[True, keys[1:] != keys[:-1]] & np.r_[keys[1:] != keys[:-1], True]
    keys, vals = keys[single], vals[single]
    absent = genome_occurrences(keys, genome, device) == 0
    return keys[absent], vals[absent]


def merge(gk, gv, jk, jv):
    """Union of genomic and junction k-mers; a key present in both with different values is dropped."""
    allk, allv = np.concatenate([gk, jk]), np.concatenate([gv, jv])
    o = np.argsort(allk, kind="stable")
    allk, allv = allk[o], allv[o]
    dup = np.r_[allk[1:] == allk[:-1], False]
    same = dup & np.r_[allv[1:] == allv[:-1], False]
    conflict = np.zeros(len(allk), bool)
    cidx = np.flatnonzero(dup & ~same)
    conflict[cidx] = True
    conflict[cidx + 1] = True
    keep = ~conflict & ~np.r_[False, same[:-1]]
    return allk[keep], allv[keep]


def build_index(fasta, snp_vcf, out_dir, gtf=None, genome_wide=True, junctions=True, device="cuda", log=print):
    """Build the counter index in out_dir (see module docstring). gtf is needed for junction k-mers and
    for gene-body-only SNP selection (genome_wide=False)."""
    t0 = time.time()
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    genome = read_genome(fasta)
    snps = select_snps(genome, snp_vcf, gtf=None if genome_wide else gtf)
    log(f"{len(snps)} SNPs with REF matching the genome ({time.time() - t0:.0f}s)")
    keys, vals = snp_kmers(genome, snps, device)
    log(f"{len(keys)} genomic k-mers ({time.time() - t0:.0f}s)")
    if junctions and gtf is not None:
        jk, jv = junction_kmers(genome, gtf, snps, device)
        keys, vals = merge(keys, vals, jk, jv)
        log(f"{len(jk)} junction k-mers; {len(keys)} k-mers in total ({time.time() - t0:.0f}s)")
    np.save(os.path.join(out_dir, "keys.npy"), keys)
    np.save(os.path.join(out_dir, "vals.npy"), vals)
    snps.to_csv(os.path.join(out_dir, "snps.tsv"), sep="\t", index=False)
    return len(keys), len(snps)
