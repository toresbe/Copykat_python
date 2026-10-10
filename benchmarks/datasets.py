"""Benchmark datasets: the README's 11 labelled tumour samples (+ a large Xenium set).

Each sample directory holds 10x-style matrix.mtx.gz / features.tsv.gz /
barcodes.tsv.gz plus metadata.csv whose ``cell_type`` column marks
``Malignant`` cells, which serves as ground truth for aneuploid calls.
Parsed matrices are cached as CSC .npz next to the benchmark cache dir.
"""

import bench_config as cfg
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

SAMPLES_DIR = cfg.SAMPLES_DIR
CACHE_DIR = cfg.CACHE_DIR


XENIUM_H5 = cfg.XENIUM_H5


def _load_xenium(n_cells=None, seed=0):
    """Xenium WTA breast cancer (170k cells); optional random cell subset. No labels."""
    import h5py

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "xenium_full.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        mat = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        genes, barcodes = z["genes"], z["barcodes"]
    else:
        with h5py.File(XENIUM_H5, "r") as f:
            g = f["matrix"]
            shape = tuple(g["shape"][:])
            mat = sparse.csc_matrix((g["data"][:], g["indices"][:].astype(np.int32), g["indptr"][:]), shape=shape)
            ftype = g["features/feature_type"][:].astype(str)
            genes = g["features/name"][:].astype(str).astype(object)
            barcodes = g["barcodes"][:].astype(str).astype(object)
        keep = ftype == "Gene Expression"
        mat = sparse.csc_matrix(mat.tocsr()[keep])
        genes = genes[keep]
        np.savez(
            cache,
            data=mat.data,
            indices=mat.indices,
            indptr=mat.indptr,
            shape=mat.shape,
            genes=genes,
            barcodes=barcodes,
        )
    if n_cells is not None and n_cells < mat.shape[1]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(mat.shape[1], size=n_cells, replace=False))
        mat = mat[:, idx]
        barcodes = barcodes[idx]
    return {
        "matrix": mat,
        "genes": np.asarray(genes, dtype=object),
        "barcodes": np.asarray(barcodes, dtype=object),
    }, None


def sample_names():
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if (p / "matrix.mtx.gz").exists())


def load_sample(name):
    """Return (rawmat dict for copykat(), truth Series indexed by barcode: True = malignant).

    ``xenium`` / ``xenium<N>k`` load the unlabelled Xenium set (truth None).
    """
    if name.startswith("xenium"):
        k = name[len("xenium") :]
        return _load_xenium(int(k.rstrip("k")) * 1000 if k else None)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}.npz"
    d = SAMPLES_DIR / name
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        mat = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        genes, barcodes = z["genes"], z["barcodes"]
    else:
        mat = sparse.csc_matrix(mmread(str(d / "matrix.mtx.gz")))
        feats = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None, dtype=str)
        genes = feats.iloc[:, 1 if feats.shape[1] > 1 else 0].to_numpy(dtype=object)
        barcodes = (
            pd.read_csv(d / "barcodes.tsv.gz", sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy(dtype=object)
        )
        np.savez(
            cache,
            data=mat.data,
            indices=mat.indices,
            indptr=mat.indptr,
            shape=mat.shape,
            genes=genes,
            barcodes=barcodes,
        )
    meta = pd.read_csv(d / "metadata.csv", dtype={"cell_name": str})
    truth = pd.Series((meta["cell_type"] == "Malignant").to_numpy(), index=meta["cell_name"].astype(str))
    return {
        "matrix": mat,
        "genes": np.asarray(genes, dtype=object),
        "barcodes": np.asarray(barcodes, dtype=object),
    }, truth


def score(pred, truth):
    """Classification metrics of copykat predictions against malignant labels."""
    from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score

    pred = pred.set_index("cell.names")["copykat.pred"].astype(str)
    common = pred.index.intersection(truth.index)
    p = pred.loc[common]
    defined = ~p.str.contains("not.defined")
    p, t = p[defined], truth.loc[common][defined]
    y = p.str.contains("aneuploid").to_numpy()
    t = t.to_numpy()
    return {
        "n_scored": len(t),
        "low_conf": bool(p.str.contains("low.conf").any()),
        "accuracy": float((y == t).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(t, y)),
        "f1_aneuploid": float(f1_score(t, y)),
        "ari": float(adjusted_rand_score(t, y)),
    }
