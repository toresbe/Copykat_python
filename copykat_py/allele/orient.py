"""Orient CopyKAT's tumour/normal calls with allele evidence.

For each of CopyKAT's two groups, the phased-BAF HMM (copykat_py.allele.hmm) is run on the group's
pseudobulk; F = fraction of the covered genome in allelically imbalanced segments. Copy-number
changes put the tumour's F well above the normal cells' (germline allele-specific expression does
not form segments). The calls are flipped if F(diploid group) - F(aneuploid group) >= MIN_F_DIFF.
Pre-registered and evaluated on 25 samples: no harmful flip in 50 call sets, 9 inversions fixed
(docs/research/allele_orientation.md). The check abstains when neither group shows imbalance.
"""

import json

import numpy as np
import pandas as pd

from copykat_py.allele import counts as _counts
from copykat_py.allele import hmm as _hmm

MIN_F_DIFF = 0.02
MIN_GROUP_CELLS = 10


def _swap_labels(pred):
    swap = {"aneuploid": "diploid", "diploid": "aneuploid"}
    return pred.str.replace(r"aneuploid|diploid", lambda m: swap[m.group(0)], regex=True)


def orient_prediction(prediction, counts_dir, phase_csv, min_f_diff=MIN_F_DIFF):
    """Check (and if needed flip) a CopyKAT prediction table.

    prediction: DataFrame with columns cell.names, copykat.pred (CopyKAT's prediction.txt).
    counts_dir: cellsnp-lite-format allele counts for the same cells (barcodes may differ by a
        '-1' style suffix). phase_csv: phased heterozygous SNPs (chr, pos, h), see allele.phase.
    Returns (oriented prediction DataFrame, report dict). The report holds F per group, the number
    of segments, the cells with allele data in each group, and whether the calls were flipped.
    """
    snps, AD, DP, cells = _counts.load_counts(counts_dir)
    snps, AD, DP = _counts.heterozygous(snps, AD, DP)
    snps, HA, DP = _counts.haplotype_counts(snps, AD, DP, _counts.read_phase(phase_csv))
    col = pd.Series(np.arange(len(cells)), index=[_counts.barcode_key(c) for c in cells])
    col = col[~col.index.duplicated()]
    pred = prediction.iloc[:, 1].astype(str)
    idx = prediction.iloc[:, 0].map(_counts.barcode_key).map(col)
    has = idx.notna().to_numpy()
    groups = {"aneuploid": (pred.str.contains("aneuploid").to_numpy() & has),
              "diploid": (pred.str.contains("diploid").to_numpy() & ~pred.str.contains("aneuploid").to_numpy() & has)}
    report = {"snps_phased": int(len(snps)), "flipped": False}
    for g, m in groups.items():
        cols = idx[m].astype(int).to_numpy()
        report[f"cells_{g}"] = int(len(cols))
        if len(cols) < MIN_GROUP_CELLS:
            report["note"] = f"fewer than {MIN_GROUP_CELLS} {g} cells with allele data; calls left as they are"
            return prediction.copy(), report
        k = np.asarray(HA[:, cols].sum(axis=1)).ravel()
        n = np.asarray(DP[:, cols].sum(axis=1)).ravel()
        seg, frac = _hmm.segments(snps, k, n)
        report[f"F_{g}"] = frac
        report[f"segments_{g}"] = seg.to_dict(orient="records")
    report["flipped"] = bool(report["F_diploid"] - report["F_aneuploid"] >= min_f_diff)
    out = prediction.copy()
    if report["flipped"]:
        out.iloc[:, 1] = _swap_labels(pred).to_numpy()
    return out, report


def orient_prediction_file(prediction_txt, counts_dir, phase_csv, out_prefix=None, min_f_diff=MIN_F_DIFF):
    """File wrapper: writes <out_prefix>prediction.allele_oriented.txt and <out_prefix>allele_orientation.json
    (out_prefix defaults to the prediction file's path without 'prediction.txt')."""
    pred = pd.read_csv(prediction_txt, sep="\t")
    out, report = orient_prediction(pred, counts_dir, phase_csv, min_f_diff)
    prefix = out_prefix if out_prefix is not None else prediction_txt[: -len("prediction.txt")] \
        if prediction_txt.endswith("prediction.txt") else prediction_txt + "."
    out.to_csv(f"{prefix}prediction.allele_oriented.txt", sep="\t", index=False)
    with open(f"{prefix}allele_orientation.json", "w") as f:
        json.dump(report, f, indent=1)
    return out, report
