"""Fetch / prepare the input data for the benchmarks.

    python benchmarks/fetch_data.py check
    python benchmarks/fetch_data.py xenium [--url URL | --file PATH]
    python benchmarks/fetch_data.py 3ca SRC_DIR

The 10x and 3CA hosts are not scriptable without a browser session as far as
we could verify, so this script automates what it can and tells you exactly
what to download by hand for the rest. See benchmarks/README.md ("Data").

check   Report which benchmark inputs are present under COPYKAT_BENCH_DATA.
xenium  Install the Xenium cell_feature_matrix.h5. Give --url (a direct link to
        either cell_feature_matrix.h5 or an *_outs.zip bundle, copied from the
        dataset page) or --file (an already-downloaded .h5/.zip).
3ca     Convert downloaded 3CA study folders into the 10x-style per-sample
        layout (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz, metadata.csv)
        for the 11 samples used in the README validation.
"""

import argparse
import gzip
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import bench_config as cfg

XENIUM_PAGE = "https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer"
CELL_FEATURE = "cell_feature_matrix.h5"

# README validation set: (3CA study, sample label in the study's Cells*.csv).
README_SAMPLES = [
    ("Gao2021_Breast", "DCIS1"),
    ("Chen2020_Head-and-Neck", "P11"),
    ("Laughney2020_Lung", "RU681"),
    ("Bi2021_Kidney", "P90"),
    ("Dong2020_Prostate", "patient5"),
    ("Jerby-Arnon2021_Sarcoma", "SyS14"),
    ("Choudhury2022_Brain", "MSC6-BTI"),
    ("Lin2020_Pancreas", "P08"),
    ("Lee2020_Colorectal", "SMC09"),
    ("Geistlinger2020_Ovarian", "T59"),
    ("Ji2020_Skin", "P4"),
]


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _same_sample(value, wanted):
    value, wanted = _norm(value), _norm(wanted)
    return value == wanted or value == wanted.replace("patient", "")


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(request) as response:
        if done and response.status != 206:
            done = 0
        with open(tmp, "ab" if done else "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
    tmp.replace(dest)
    return dest


def _install_xenium(path):
    path = Path(path)
    cfg.XENIUM_H5.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.endswith(CELL_FEATURE)]
            if not members:
                sys.exit(f"{path} contains no {CELL_FEATURE}")
            with z.open(members[0]) as src, open(cfg.XENIUM_H5, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    else:
        shutil.copyfile(path, cfg.XENIUM_H5)
    print(f"installed {cfg.XENIUM_H5}")


def cmd_xenium(args):
    if args.file:
        return _install_xenium(args.file)
    if not args.url:
        sys.exit(
            "Pass --url or --file. Open\n"
            f"  {XENIUM_PAGE}\n"
            "choose the cell-feature matrix (HDF5) or the outs bundle on that page, then either\n"
            "copy its download link into --url or download it and pass --file."
        )
    target = cfg.XENIUM_H5.parent / Path(args.url.split("?")[0]).name
    print(f"downloading {args.url}")
    _install_xenium(download(args.url, target))
    if target.suffix == ".zip":
        target.unlink()


def _write_sample(study, sample, mtx, cells_csv, genes_txt, dest_root):
    import numpy as np
    import pandas as pd
    from scipy import io, sparse

    meta = pd.read_csv(cells_csv, dtype=str)
    scol = next((c for c in ("sample", "patient") if c in meta), None)
    if scol is None or not {"cell_name", "cell_type"} <= set(meta):
        return False
    rows = np.flatnonzero([_same_sample(v, sample) for v in meta[scol]])
    if not len(rows):
        return False
    genes = pd.read_csv(genes_txt, header=None, dtype=str).iloc[:, 0].to_numpy()
    matrix = sparse.csc_matrix(io.mmread(str(mtx)))
    if matrix.shape != (len(genes), len(meta)):
        sys.exit(f"{mtx}: matrix shape {matrix.shape} does not match {len(genes)} genes x {len(meta)} cells")
    sub = meta.iloc[rows]
    out = dest_root / f"{study}_{_norm(sample) if sample.startswith('patient') else sample}"
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "matrix.mtx.gz", "wb") as f:
        io.mmwrite(f, matrix[:, rows], field="integer" if np.issubdtype(matrix.dtype, np.integer) else "real")
    pd.DataFrame({"id": genes, "name": genes, "type": "Gene Expression"}).to_csv(
        out / "features.tsv.gz", sep="\t", header=False, index=False
    )
    sub[["cell_name"]].to_csv(out / "barcodes.tsv.gz", sep="\t", header=False, index=False)
    sub[["cell_name", "cell_type"]].to_csv(out / "metadata.csv", index=False)
    print(f"wrote {out} ({len(rows)} cells)")
    return True


def cmd_3ca(args):
    src = Path(args.src)
    failures = []
    for study, sample in README_SAMPLES:
        found = False
        dirs = [p for p in src.rglob("*") if p.is_dir() and study.lower() in p.name.lower()] or [src]
        for d in dirs:
            for mtx in sorted(d.rglob("*UMI*.mtx")):
                cells = sorted(mtx.parent.glob("[Cc]ells*.csv"))
                genes = sorted(mtx.parent.glob("[Gg]enes*.txt")) or sorted(mtx.parent.parent.glob("[Gg]enes*.txt"))
                if cells and genes and study.lower() in "/".join(mtx.parts).lower():
                    found = _write_sample(study, sample, mtx, cells[0], genes[0], cfg.SAMPLES_DIR) or found
            if found:
                break
        if not found:
            failures.append(f"{study} / {sample}")
    if failures:
        print("not found under", src, "-", "; ".join(failures), file=sys.stderr)
        sys.exit(1)


def cmd_check(_args):
    ok = True
    present = cfg.XENIUM_H5.exists()
    ok &= present
    print(f"[{'x' if present else ' '}] Xenium matrix: {cfg.XENIUM_H5}")
    for study, sample in README_SAMPLES:
        d = cfg.SAMPLES_DIR / f"{study}_{_norm(sample) if sample.startswith('patient') else sample}"
        present = (d / "matrix.mtx.gz").exists() and (d / "metadata.csv").exists()
        ok &= present
        print(f"[{'x' if present else ' '}] README sample: {d}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    x = sub.add_parser("xenium")
    x.add_argument("--url")
    x.add_argument("--file")
    x.set_defaults(fn=cmd_xenium)
    c = sub.add_parser("3ca")
    c.add_argument("src")
    c.set_defaults(fn=cmd_3ca)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
