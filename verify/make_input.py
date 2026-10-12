#!/usr/bin/env python
"""Write a dataset in the layout verify/verify.sh reads: matrix.mtx, genes.txt, barcodes.txt.

Usage:
  verify/make_input.py bundled OUTPUT_DIR          the 302-cell sample shipped with this package
  verify/make_input.py 10x INPUT_DIR OUTPUT_DIR    a Cell Ranger folder (matrix.mtx[.gz],
                                                   features.tsv[.gz] or genes.tsv, barcodes.tsv[.gz])
  verify/make_input.py table INPUT_FILE OUTPUT_DIR a genes x cells text table (tab- or
                                                   comma-separated, gene names in the first column)

Counts are written unchanged as a genes x cells Matrix Market file; genes are named
by symbol, which is what both runs are told (id.type = "S").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parents[1] / "copykat_py" / "data"


def write(out: Path, counts: sp.spmatrix, genes: list[str], barcodes: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(out / "matrix.mtx", sp.csc_matrix(counts))
    (out / "genes.txt").write_text("".join(f"{g}\n" for g in genes))
    (out / "barcodes.txt").write_text("".join(f"{b}\n" for b in barcodes))
    print(f"{out}: {counts.shape[0]} genes x {counts.shape[1]} cells")


def first_existing(folder: Path, names: list[str]) -> Path:
    for name in names:
        if (folder / name).exists():
            return folder / name
    raise SystemExit(f"none of {names} in {folder}")


def main() -> None:
    kind = sys.argv[1]
    if kind == "bundled":
        counts = scipy.io.mmread(DATA_DIR / "exp_rawdata_sparse.mtx")
        genes = (DATA_DIR / "exp_rawdata_genes.txt").read_text().split()
        barcodes = (DATA_DIR / "exp_rawdata_barcodes.txt").read_text().split()
        write(Path(sys.argv[2]), counts, genes, barcodes)
    elif kind == "10x":
        folder = Path(sys.argv[2])
        counts = scipy.io.mmread(first_existing(folder, ["matrix.mtx.gz", "matrix.mtx"]))
        features = pd.read_csv(
            first_existing(folder, ["features.tsv.gz", "genes.tsv.gz", "features.tsv", "genes.tsv"]),
            sep="\t",
            header=None,
        )
        barcodes = pd.read_csv(first_existing(folder, ["barcodes.tsv.gz", "barcodes.tsv"]), sep="\t", header=None)
        symbols = features[1] if features.shape[1] > 1 else features[0]
        write(Path(sys.argv[3]), counts, symbols.astype(str).tolist(), barcodes[0].astype(str).tolist())
    elif kind == "table":
        path = Path(sys.argv[2])
        table = pd.read_csv(path, sep=None, engine="python", index_col=0)
        counts = sp.csc_matrix(table.to_numpy(dtype=np.float64))
        write(Path(sys.argv[3]), counts, table.index.astype(str).tolist(), table.columns.astype(str).tolist())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
