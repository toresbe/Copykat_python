"""Phased B-allele-frequency HMM over a group of cells (Numbat-style, allele evidence only).

The pseudobulk of a group of cells is scanned SNP by SNP along each chromosome. Hidden states:
balanced (BAF 0.5), and for each imbalance level theta in THETAS, haplotype A major (BAF theta) or
haplotype B major (BAF 1 - theta). Emissions are beta-binomial with an overdispersion fitted under
the balanced state (it absorbs SNP-level allele-specific expression). Switching between A- and
B-major at the same theta models phasing errors and grows with distance; any other state change is
rare. Segments are runs with P(imbalanced) > 0.5 long enough to be a copy-number change rather
than a single gene's allele-specific expression.
"""

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp

THETAS = np.array([0.6, 0.7, 0.8, 0.9])
P_CNA = 1e-4  # probability of a state change between consecutive SNPs
MIN_SEG_BP = 5_000_000
MIN_SEG_SNPS = 20
# state 0 balanced; 1 + 2k: A-major at THETAS[k]; 2 + 2k: B-major at THETAS[k]
BAF = np.concatenate([[0.5], np.ravel(np.column_stack([THETAS, 1 - THETAS]))])
N_STATES = len(BAF)


def p_flip(d_bp):
    """Probability that the phase (A- vs B-major) switches over d_bp base pairs."""
    return 0.5 * (1 - np.exp(-2 * 0.01 * d_bp / 1e6)) + 0.005


def log_transition(d_bp):
    T = np.full((N_STATES, N_STATES), P_CNA / (N_STATES - 1))
    np.fill_diagonal(T, 1 - P_CNA)
    pf = p_flip(d_bp)
    for k in range(len(THETAS)):
        a, b = 1 + 2 * k, 2 + 2 * k
        for i, j in ((a, b), (b, a)):
            T[i, j] = pf
            T[i, i] = 1 - pf - P_CNA * (N_STATES - 2) / (N_STATES - 1)
    T /= T.sum(1, keepdims=True)
    return np.log(T)


def bb_logpmf(k, n, p, rho):
    """Beta-binomial log pmf with mean p and overdispersion rho; k, n: (S,), p: (K,) -> (S, K)."""
    s = 1 / rho - 1
    a, b = p * s, (1 - p) * s
    k, n = k[:, None], n[:, None]
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)


def fit_rho(k, n):
    """Maximum-likelihood overdispersion under BAF 0.5 (grid search) on SNPs with >= 10 reads."""
    m = n >= 10
    if m.sum() < 50:
        return 0.01
    grid = np.logspace(-4, np.log10(0.5), 60)
    ll = [bb_logpmf(k[m], n[m], np.array([0.5]), r).sum() for r in grid]
    return float(grid[int(np.argmax(ll))])


def forward_backward(log_emission, pos):
    """Posterior state probabilities (S, N_STATES) for one chromosome."""
    S = len(pos)
    la = np.empty((S, N_STATES))
    lb = np.zeros((S, N_STATES))
    la[0] = np.log(1 / N_STATES) + log_emission[0]
    T = [log_transition(max(pos[i] - pos[i - 1], 1)) for i in range(1, S)]
    for i in range(1, S):
        la[i] = logsumexp(la[i - 1][:, None] + T[i - 1], axis=0) + log_emission[i]
    for i in range(S - 2, -1, -1):
        lb[i] = logsumexp(T[i] + (log_emission[i + 1] + lb[i + 1])[None, :], axis=1)
    lp = la + lb
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def segments(snps, k, n):
    """Imbalanced segments of one pseudobulk.

    snps: DataFrame chr, pos (sorted within chromosome); k, n: haplotype-A and total UMIs per SNP.
    Returns (segment table [chr, start, end, snps, theta], fraction of the covered genome in segments).
    """
    rho = fit_rho(k, n)
    segs, covered = [], 0
    for c, idx in snps.groupby("chr", sort=False).indices.items():
        idx = idx[np.argsort(snps.pos.to_numpy()[idx])]
        pos = snps.pos.to_numpy()[idx]
        if len(idx) < 2:
            continue
        covered += pos[-1] - pos[0]
        post = forward_backward(bb_logpmf(k[idx], n[idx], BAF, rho), pos)
        on = 1 - post[:, 0] > 0.5
        a_major = post[:, 1::2].sum(1) >= post[:, 2::2].sum(1)
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2], strict=False):
            if pos[s1 - 1] - pos[s0] >= MIN_SEG_BP and s1 - s0 >= MIN_SEG_SNPS:
                j = idx[s0:s1]
                major = np.where(a_major[s0:s1], k[j], n[j] - k[j]).sum()
                segs.append(
                    {
                        "chr": c,
                        "start": int(pos[s0]),
                        "end": int(pos[s1 - 1]),
                        "snps": int(s1 - s0),
                        "theta": float(major / max(n[j].sum(), 1)),
                    }
                )
    seg = pd.DataFrame(segs, columns=["chr", "start", "end", "snps", "theta"])
    fraction = float((seg.end - seg.start).sum() / covered) if covered else 0.0
    return seg, fraction
