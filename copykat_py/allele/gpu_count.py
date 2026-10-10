"""GPU allele counter: 10x reads -> per-cell REF/ALT UMI counts at known SNPs, without alignment.

Every 31-mer of each cDNA read is looked up (GPU binary search) in an index from
copykat_py.allele.kmer_index. A read supports (SNP, allele) if any of its k-mers hits it; a read
supporting both alleles of a SNP is dropped for that SNP. Reads count only if their cell barcode
(exact match, first 16 bases of the barcode read) is in the whitelist. Per (cell, SNP, UMI) the
majority allele is taken (ties dropped); counts are UMIs. Output is cellsnp-lite's format
(cellSNP.base.vcf.gz, cellSNP.tag.AD.mtx, cellSNP.tag.DP.mtx, cellSNP.samples.tsv).

Input
  FASTQ.gz pair: per file `igzip -dc | fqseq L` (fqseq.c keeps only sequence lines at a fixed width,
    compiled on first use), read in fixed-size chunks by a prefetch thread; reads paired by position.
  SRA (.sra file): decoded by parallel fastq-dump processes over disjoint spot ranges.
The barcode read (24-30 nt) and cDNA read (>= 50 nt) are recognised by length. Validated against
STARsolo + cellsnp-lite on 4 samples: identical downstream orientation decisions; 93% of their UMIs;
per-SNP allele fractions r = 0.98 (docs/research/allele_orientation.md).
"""

import gzip
import os
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
import torch

K = 31
CB_LEN = 16
_ASCII = np.full(256, 4, np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _ASCII[_b] = _i
_FQSEQ_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fqseq.c")


def _tool(name, path=None):
    found = path or shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found; install it or pass its path")
    return found


def fqseq_binary():
    """Path of the compiled fqseq helper (built with gcc into ~/.cache/copykat_py on first use)."""
    cache = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "copykat_py")
    exe = os.path.join(cache, "fqseq")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(_FQSEQ_SRC):
        os.makedirs(cache, exist_ok=True)
        subprocess.run([_tool("gcc"), "-O3", "-o", exe, _FQSEQ_SRC], check=True)
    return exe


def _encode(seqs, device):
    """(n, L) codes 0-3 (4 = N) -> (n,) int64 forward 2-bit code; -1 if any N."""
    s = seqs.to(torch.int64)
    code = torch.zeros(s.shape[0], dtype=torch.int64, device=device)
    for i in range(s.shape[1]):
        code = (code << 2) | s[:, i].clamp(max=3)
    code[(s == 4).any(1)] = -1
    return code


def _kmers(seqs, device):
    """(n, L) codes -> (n, L-K+1) canonical k-mer codes, -1 where the k-mer contains N."""
    w = seqs.to(torch.int64)
    n = w.shape[1] - K + 1
    fwd = torch.zeros((w.shape[0], n), dtype=torch.int64, device=device)
    rev = torch.zeros_like(fwd)
    bad = torch.zeros((w.shape[0], n), dtype=torch.bool, device=device)
    for i in range(K):
        col = w[:, i : i + n]
        bad |= col == 4
        c = col.clamp(max=3)
        fwd = (fwd << 2) | c
        rev |= (3 - c) << (2 * i)
    can = torch.minimum(fwd, rev)
    can[bad] = -1
    return can


def _parse_spots(stream, lens, bi, ci, block):
    """(barcode, cDNA) ASCII arrays from a FASTQ stream of whole spots (len(lens) records each)."""
    lines_per_spot = 4 * len(lens)
    rest = b""
    while True:
        chunk = stream.read(block)
        buf = rest + chunk if chunk else rest
        if not buf:
            return
        a = np.frombuffer(buf, np.uint8)
        nl = np.flatnonzero(a == 10)
        n = len(nl) // lines_per_spot
        if n:
            nl_all = np.r_[-1, nl]
            out = []
            for ri in (bi, ci):
                first = nl_all[np.arange(n) * lines_per_spot + 4 * ri + 1] + 1
                out.append(a[first[:, None] + np.arange(lens[ri])[None, :]])
            yield out[0], out[1]
            rest = buf[nl_all[n * lines_per_spot] + 1 :]
        else:
            rest = buf
        if not chunk:
            return


def _sra_worker(fastq_dump, sra, n0, n1, lens, bi, ci, q):
    p = subprocess.Popen(
        [fastq_dump, "--split-spot", "-Z", "-N", str(n0), "-X", str(n1), sra],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1 << 20,
    )
    for item in _parse_spots(p.stdout, lens, bi, ci, block=1 << 26):
        q.put(item)
    p.wait()
    q.put(None)


def _read_roles(lens):
    bi = next(i for i, L in enumerate(lens) if 24 <= L <= 30)
    ci = next(i for i, L in enumerate(lens) if L >= 50)
    return bi, ci


def _sra_spots(sra, workers, fastq_dump, vdb_dump, log):
    head = subprocess.run([fastq_dump, "--split-spot", "-Z", "-X", "1", sra], capture_output=True).stdout
    lens = [len(x) for x in head.split(b"\n")[1::4]]
    info = subprocess.run([vdb_dump, "--info", sra], capture_output=True, text=True).stdout
    nspots = int(next(x for x in info.splitlines() if x.startswith("SEQ")).split(":")[1].replace(",", ""))
    bi, ci = _read_roles(lens)
    log(f"spot layout {lens}: barcode read {bi}, cDNA read {ci}; {nspots} spots")
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=4 * workers)
    edges = np.linspace(1, nspots + 1, workers + 1).astype(int)
    procs = [
        ctx.Process(target=_sra_worker, args=(fastq_dump, sra, edges[i], edges[i + 1] - 1, lens, bi, ci, q))
        for i in range(workers)
    ]
    for p in procs:
        p.start()
    done = 0
    while done < len(procs):
        item = q.get()
        if item is None:
            done += 1
        else:
            yield item
    for p in procs:
        p.join()


def _fastq_spots(r1, r2, igzip, log, chunk=1 << 20):
    with gzip.open(r1) as f1, gzip.open(r2) as f2:
        f1.readline()
        f2.readline()
        lens = [len(f1.readline().rstrip()), len(f2.readline().rstrip())]
    bi, ci = _read_roles(lens)
    log(f"read lengths {lens}: barcode read {bi}, cDNA read {ci}")
    fqseq = fqseq_binary()
    procs = [
        subprocess.Popen(
            f"{igzip} -dc {f} | {fqseq} {L}",
            shell=True,
            stdout=subprocess.PIPE,
            executable="/bin/bash",
            bufsize=1 << 24,
        )
        for f, L in zip((r1, r2), lens, strict=False)
    ]
    q = queue.Queue(maxsize=3)

    def reader():  # keeps the decompressors busy while the GPU works (pipe reads release the GIL)
        while True:
            raw = [p.stdout.read(chunk * L) for p, L in zip(procs, lens, strict=False)]
            n = min(len(r) // L for r, L in zip(raw, lens, strict=False))
            if n:
                arrs = [np.frombuffer(r, np.uint8)[: n * L].reshape(n, L) for r, L in zip(raw, lens, strict=False)]
                q.put((arrs[bi], arrs[ci]))
            if n < chunk:
                q.put(None)
                return

    threading.Thread(target=reader, daemon=True).start()
    while (item := q.get()) is not None:
        yield item
    for p in procs:
        p.wait()


def count_alleles(
    index_dir,
    whitelist,
    out_dir,
    fastq=None,
    sra=None,
    umi_len=10,
    workers=12,
    igzip=None,
    fastq_dump=None,
    vdb_dump=None,
    device="cuda",
    log=print,
):
    """Count per-cell REF/ALT UMIs at the index's SNPs. Give either fastq=(R1, R2) or sra=path.
    whitelist: file with one cell barcode per line (a '-1' style suffix is ignored for matching and
    kept in the output's cellSNP.samples.tsv)."""
    if (fastq is None) == (sra is None):
        raise ValueError("give exactly one of fastq=(R1, R2) or sra=path")
    t0 = time.time()
    device = torch.device(device)
    keys = torch.from_numpy(np.load(os.path.join(index_dir, "keys.npy"))).to(device)
    vals = torch.from_numpy(np.load(os.path.join(index_dir, "vals.npy"))).to(device)
    snps = np.loadtxt(os.path.join(index_dir, "snps.tsv"), dtype=str, skiprows=1, ndmin=2)
    snp_bits = int(np.ceil(np.log2(2 * len(snps) + 1)))  # vals = 2 * snp + allele < 2 ** snp_bits
    with open(whitelist) as stream:
        wl_lines = [x.strip() for x in stream if x.strip()]
    wl = np.array([x.split("-")[0] for x in wl_lines])
    lut = torch.from_numpy(_ASCII).to(device)
    wl_codes = _encode(
        lut[torch.from_numpy(np.frombuffer("".join(wl).encode(), np.uint8).reshape(len(wl), CB_LEN)).to(device).long()],
        device,
    )
    wl_sorted, wl_order = torch.sort(wl_codes)
    log(f"index {len(keys)} k-mers / {len(snps)} SNPs, {len(wl)} cells ({time.time() - t0:.0f}s)")
    if sra is not None:
        fd = _tool("fastq-dump", fastq_dump)
        spots = _sra_spots(
            sra, workers, fd, _tool("vdb-dump", vdb_dump or os.path.join(os.path.dirname(fd), "vdb-dump")), log
        )
    else:
        spots = _fastq_spots(fastq[0], fastq[1], _tool("igzip", igzip), log)
    records, n_spots, n_cell, n_hit = [], 0, 0, 0
    for bc, cd in spots:
        n_spots += len(bc)
        bc_t = lut[torch.from_numpy(bc).to(device).long()]
        cb = _encode(bc_t[:, :CB_LEN], device)
        j = torch.searchsorted(wl_sorted, cb).clamp(max=len(wl_sorted) - 1)
        ok = wl_sorted[j] == cb
        if not ok.any():
            continue
        cell = wl_order[j[ok]]
        umi = _encode(bc_t[ok][:, CB_LEN : CB_LEN + umi_len], device)
        cdt = lut[torch.from_numpy(cd).to(device)[ok].long()]
        n_cell += int(ok.sum())
        for s0 in range(0, len(cdt), 1 << 20):
            q = _kmers(cdt[s0 : s0 + (1 << 20)], device)
            i = torch.searchsorted(keys, q).clamp(max=len(keys) - 1)
            hit = (keys[i] == q) & (q >= 0)
            r, _ = torch.nonzero(hit, as_tuple=True)
            v = vals[i[hit]].to(torch.int64)
            key = torch.unique(((r + s0) << snp_bits) | v)  # unique (read, snp, allele)
            rd, sa = key >> snp_bits, key & ((1 << snp_bits) - 1)
            snp, allele = sa >> 1, sa & 1
            _, cnt = torch.unique_consecutive((rd << snp_bits) | snp, return_counts=True)
            single = torch.repeat_interleave(cnt == 1, cnt)  # drop reads with both alleles
            rd, snp, allele = rd[single], snp[single], allele[single]
            n_hit += len(rd)
            records.append(torch.stack([cell[rd], snp, umi[rd], allele], 1).cpu())
        log(
            f"  {n_spots / 1e6:.0f} M spots, {n_cell / 1e6:.0f} M in cells, {n_hit / 1e6:.1f} M read-SNP hits "
            f"({time.time() - t0:.0f}s)"
        )
    R = torch.cat(records).to(device)
    for col in (2, 1, 0):  # sort by (cell, snp, umi)
        R = R[torch.argsort(R[:, col], stable=True)]
    new = torch.ones(len(R), dtype=torch.bool, device=device)
    new[1:] = (R[1:, :3] != R[:-1, :3]).any(1)
    gid = torch.cumsum(new, 0) - 1
    ng = int(gid[-1]) + 1
    alt = torch.zeros(ng, dtype=torch.int64, device=device).index_add_(0, gid, R[:, 3])
    tot = torch.zeros(ng, dtype=torch.int64, device=device).index_add_(0, gid, torch.ones_like(R[:, 3]))
    first = torch.nonzero(new, as_tuple=True)[0]
    decided = 2 * alt != tot  # UMI majority allele; ties dropped
    cell_u, snp_u = R[first, 0][decided], R[first, 1][decided]
    is_alt = (2 * alt > tot)[decided].to(torch.int64)
    pair, inv = torch.unique(snp_u * len(wl) + cell_u, return_inverse=True)
    AD = torch.zeros(len(pair), dtype=torch.int64, device=device).index_add_(0, inv, is_alt)
    DP = torch.bincount(inv, minlength=len(pair))
    pair, AD, DP = pair.cpu().numpy(), AD.cpu().numpy(), DP.cpu().numpy()
    s_idx, c_idx = pair // len(wl), pair % len(wl)
    used, s_row = np.unique(s_idx, return_inverse=True)
    _write_cellsnp(out_dir, snps[used], s_row, c_idx, AD, DP, wl_lines)
    log(f"done: {n_spots / 1e6:.0f} M spots, {len(used)} SNPs covered, {int(DP.sum())} UMIs ({time.time() - t0:.0f}s)")


def _write_cellsnp(out_dir, snp_rows, s_row, c_idx, AD, DP, cells):
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, "cellSNP.base.vcf.gz"), "wt") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, r, a in snp_rows:
            f.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\n")
    for tag, m in (("AD", AD), ("DP", DP)):
        nz = m > 0
        with open(os.path.join(out_dir, f"cellSNP.tag.{tag}.mtx"), "w") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n%\n")
            f.write(f"{len(snp_rows)}\t{len(cells)}\t{int(nz.sum())}\n")
            np.savetxt(f, np.column_stack([s_row[nz] + 1, c_idx[nz] + 1, m[nz]]), fmt="%d", delimiter="\t")
    with open(os.path.join(out_dir, "cellSNP.samples.tsv"), "w") as f:
        f.write("\n".join(cells) + "\n")
