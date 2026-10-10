"""Frozen study-balanced selection and bounded-memory raw-count staging."""

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path("/home/toresbe/cancer_research/accuracy_2026-10-10")
DATA = Path("/home/toresbe/cancer_research")
UNKNOWN = {"", "unknown", "unassigned", "undetermined", "nan", "na", "n/a", "?", "not assigned", "not available"}
HEMATOLOGICAL = re.compile(r"leukem|lymphom|myelom|hematolog", re.I)


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def uid(study, sample, path):
    pretty = re.sub("[^A-Za-z0-9_.-]", "_", study + "_" + str(sample))[:75]
    return pretty + "_" + hashlib.sha256((str(path) + "/" + str(sample)).encode()).hexdigest()[:10]


def labels(values):
    a = np.asarray(pd.Series(values).fillna("").astype(str).str.strip(), dtype=str)
    known = np.array([x.lower() not in UNKNOWN for x in a])
    return np.where(known, (np.char.lower(a) == "malignant").astype(np.int8), -1), a


def col(group, name):
    node = group[name]
    if hasattr(node, "keys"):
        categories = np.asarray(
            node["categories"].asstr()[:] if node["categories"].dtype.kind in "OS" else node["categories"][:]
        )
        codes = node["codes"][:]
        return np.array([str(categories[i]) if i >= 0 else "" for i in codes], dtype=str)
    return np.asarray(node.asstr()[:] if node.dtype.kind in "OS" else node[:]).astype(str)


def index_col(group):
    name = group.attrs.get("_index", "_index")
    return name.decode() if isinstance(name, bytes) else name


def discover():
    """Metadata-only selection. Never chooses samples based on CopyKAT outcomes."""
    import h5py

    split = json.loads((DATA / "anchor_study/split.json").read_text())
    cases = []
    excluded = []
    seen = set()
    dev_studies = set(split.get("dev2", []))

    def eligible(truth):
        pos = int((truth == 1).sum())
        neg = int((truth == 0).sum())
        return min(pos, neg) >= 30 and 0.05 <= pos / (pos + neg) <= 0.95

    def add(case, barcodes):
        signature = (case["study"], hashlib.sha256("\n".join(sorted(map(str, barcodes))).encode()).hexdigest())
        if signature in seen:
            excluded.append({"study": case["study"], "sample": case["sample"], "reason": "duplicate study/cell panel"})
        else:
            seen.add(signature)
            cases.append(case)

    for d in sorted((DATA / "copykat_readme_data/samples").iterdir()):
        if not (d / "matrix.mtx.gz").exists():
            continue
        meta = pd.read_csv(d / "metadata.csv", dtype=str)
        truth, _ = labels(meta.cell_type)
        study = d.name.rsplit("_", 1)[0]
        dev_studies.add(study)
        if 500 <= len(meta) <= 20000 and eligible(truth):
            case = {
                "id": uid(study, d.name, d),
                "study": study,
                "sample": d.name,
                "family": "README",
                "cohort": "previous development",
                "kind": "readme",
                "path": str(d),
                "n_cells": len(meta),
                "positive": int((truth == 1).sum()),
                "negative": int((truth == 0).sum()),
            }
            add(case, meta.cell_name)
        else:
            excluded.append({"study": study, "sample": d.name, "reason": "outside frozen cell/class criteria"})
    for root in [DATA / "3ca", DATA / "scpca_3ca"]:
        for mtx in sorted(root.rglob("*.mtx")):
            if "UMI" not in mtx.name:
                excluded.append({"path": str(mtx), "reason": "not explicitly UMI-count matrix"})
                continue
            dirs = [p for p in mtx.parts if p.startswith("Data_")]
            if not dirs:
                continue
            study = dirs[-1][5:]
            if HEMATOLOGICAL.search(study):
                excluded.append({"study": study, "reason": "marker anchor not applicable to haematological malignancy"})
                continue
            cell_files = sorted(mtx.parent.glob("[Cc]ells*.csv"))
            gene_files = sorted(mtx.parent.glob("[Gg]enes*.txt")) or sorted(mtx.parent.parent.glob("[Gg]enes*.txt"))
            if not cell_files or not gene_files:
                excluded.append({"study": study, "path": str(mtx), "reason": "no aligned cell metadata or genes"})
                continue
            meta = pd.read_csv(cell_files[0], dtype=str)
            scol = "sample" if "sample" in meta else ("patient" if "patient" in meta else None)
            if scol is None or not {"cell_name", "cell_type"} <= set(meta):
                excluded.append({"study": study, "reason": "no sample/cell_type metadata"})
                continue
            candidates = []
            for sample, g in meta.groupby(scol, sort=True):
                truth, _ = labels(g.cell_type)
                if 500 <= len(g) <= 20000 and eligible(truth):
                    candidates.append((float((truth == 1).sum() / max(1, (truth >= 0).sum())), str(sample), g))
                else:
                    excluded.append(
                        {
                            "study": study,
                            "sample": str(sample),
                            "n_cells": len(g),
                            "reason": "outside frozen cell/class criteria",
                        }
                    )
            candidates.sort(key=lambda x: (x[0], x[1]))
            selected = (
                candidates
                if root.name == "scpca_3ca"
                else ([candidates[0], candidates[-1]] if len(candidates) > 1 else candidates)
            )
            selected_ids = {x[1] for x in selected}
            for _, sample, _g in candidates:
                if sample not in selected_ids:
                    excluded.append(
                        {"study": study, "sample": sample, "reason": "two-per-study cap; fraction-extreme selection"}
                    )
            for fraction, sample, g in selected:
                truth, _ = labels(g.cell_type)
                cohort = (
                    "ScPCA previously evaluated"
                    if root.name == "scpca_3ca"
                    else (
                        "previous development"
                        if study in dev_studies
                        else "previous holdout (already opened)"
                        if study in split.get("holdout", [])
                        else "other installed study; prior use uncertain"
                    )
                )
                case = {
                    "id": uid(study, sample, mtx),
                    "study": study,
                    "sample": sample,
                    "family": root.name,
                    "cohort": cohort,
                    "kind": "mtx",
                    "path": str(mtx),
                    "cells_path": str(cell_files[0]),
                    "genes_path": str(gene_files[0]),
                    "sample_col": scol,
                    "n_cells": len(g),
                    "positive": int((truth == 1).sum()),
                    "negative": int((truth == 0).sum()),
                    "malignant_fraction": fraction,
                }
                add(case, g.cell_name)
    # Enforce the cap across split matrix files, not just within one matrix.
    for study in sorted({c["study"] for c in cases if c["family"] == "3ca"}):
        study_cases = sorted(
            [c for c in cases if c["family"] == "3ca" and c["study"] == study],
            key=lambda c: (c["malignant_fraction"], c["sample"], c["path"]),
        )
        keep = {c["id"] for c in (study_cases if len(study_cases) <= 2 else [study_cases[0], study_cases[-1]])}
        for c in study_cases:
            if c["id"] not in keep:
                cases.remove(c)
                excluded.append(
                    {"study": study, "sample": c["sample"], "reason": "two-per-study cap across split matrices"}
                )
    for p in sorted((DATA / "tabula_sapiens").glob("*.h5ad")):
        with h5py.File(p) as f:
            x = f["raw/X"] if "raw/X" in f else f["X"]
            var = f["raw/var"] if "raw/var" in f else f["var"]
            if "feature_name" not in var or x.attrs.get("encoding-type", "") != "csr_matrix":
                excluded.append({"path": str(p), "reason": "healthy raw-count/symbol CSR layout unavailable"})
                continue
            # Validate all selected values during staging; this check rejects normalized data early.
            v = x["data"][: min(100000, len(x["data"]))]
            if not np.all(np.isfinite(v) & (v >= 0) & np.isclose(v, np.rint(v), atol=1e-6, rtol=0)):
                excluded.append({"path": str(p), "reason": "healthy matrix not integer raw counts"})
                continue
            obs = f["obs"]
            donor = "donor_id" if "donor_id" in obs else None
            if donor is None:
                excluded.append({"path": str(p), "reason": "no donor identifiers"})
                continue
            ids = col(obs, donor)
            barcodes = col(obs, index_col(obs))
            disease = col(obs, "disease") if "disease" in obs else np.repeat("normal", len(ids))
            valid = np.isin(np.char.lower(disease), ["normal", "healthy"])
            options = [(int(((ids == d) & valid).sum()), d) for d in np.unique(ids)]
            options = sorted((n, d) for n, d in options if 500 <= n <= 20000)
            chosen = options if len(options) <= 2 else [options[0], options[-1]]
            for n, d in chosen:
                mask = (ids == d) & valid
                case = {
                    "id": uid(p.stem, d, p),
                    "study": p.stem,
                    "sample": d,
                    "family": "healthy controls",
                    "cohort": "normal-only control; prior use uncertain",
                    "kind": "h5ad",
                    "path": str(p),
                    "donor_col": donor,
                    "n_cells": n,
                    "positive": 0,
                    "negative": n,
                }
                add(case, barcodes[mask])
            if not chosen:
                excluded.append({"path": str(p), "reason": "no eligible healthy donor"})
    manifest = {"cases": cases, "excluded": excluded, "protocol": json.loads((ROOT / "protocol.json").read_text())}
    save(ROOT / "dataset_manifest.json", manifest)
    return cases


def open_counts(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def stream_mtx(path, selected_columns, expected_shape):
    """One bounded-memory ASCII pass, selecting columns before assembling CSC."""
    selected = np.asarray(selected_columns, dtype=np.int64)
    h = hashlib.sha256()
    rr = []
    cc = []
    vv = []
    with open_counts(path) as stream:
        first = stream.readline()
        h.update(first)
        if not first.lower().startswith(b"%%matrixmarket matrix coordinate") or b"symmetric" in first.lower():
            raise ValueError("Unsupported Matrix Market encoding")
        while True:
            line = stream.readline()
            h.update(line)
            if not line.startswith(b"%"):
                break
        shape = list(map(int, line.split()))
        if tuple(shape[:2]) != tuple(expected_shape):
            raise ValueError(f"Matrix axes {shape[:2]} vs genes/cells {expected_shape}")
        mapping = np.full(shape[1], -1, dtype=np.int64)
        mapping[selected] = np.arange(len(selected))
        count = 0
        while True:
            block = []
            for _ in range(100000):
                line = stream.readline()
                if not line:
                    break
                h.update(line)
                if line.strip() and not line.startswith(b"%"):
                    block.append(line)
            if not block:
                break
            a = np.fromstring(b"".join(block).decode("ascii"), sep=" ", dtype=np.float64)
            if len(a) != 3 * len(block):
                raise ValueError("Malformed matrix entry")
            a = a.reshape(-1, 3)
            count += len(a)
            if not np.all(np.isfinite(a)) or np.any(a < 0) or not np.allclose(a, np.rint(a), rtol=0, atol=1e-6):
                raise ValueError("Noninteger/nonpositive raw-count entry")
            rows = a[:, 0].astype(np.int64) - 1
            cols = a[:, 1].astype(np.int64) - 1
            if np.any(rows < 0) | np.any(rows >= shape[0]) | np.any(cols < 0) | np.any(cols >= shape[1]):
                raise ValueError("Out-of-range matrix coordinate")
            out = mapping[cols]
            keep = out >= 0
            rr.append(rows[keep].astype(np.int32))
            cc.append(out[keep].astype(np.int32))
            vv.append(a[keep, 2].astype(np.int64))
        if count != shape[2]:
            raise ValueError(f"Entry count {count} vs declared {shape[2]}")
    matrix = sparse.coo_matrix(
        (
            np.concatenate(vv) if vv else np.array([], dtype=np.int64),
            (
                np.concatenate(rr) if rr else np.array([], dtype=np.int32),
                np.concatenate(cc) if cc else np.array([], dtype=np.int32),
            ),
        ),
        shape=(shape[0], len(selected)),
    ).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, h.hexdigest()


def save_input(case, matrix, genes, cells, truth, ctypes, source_info):
    genes = np.asarray(genes, dtype=str)
    cells = np.asarray(cells, dtype=str)
    if len(np.unique(cells)) != len(cells):
        raise ValueError("Duplicate cell names")
    if np.mean(np.char.startswith(genes, "ENSG")) > 0.5:
        raise ValueError("Ensembl-only genes without supplied symbols; marker input not evaluable")
    if matrix.shape != (len(genes), len(cells)):
        raise ValueError("Staged matrix/metadata mismatch")
    p = ROOT / "cache" / (case["id"] + ".npz")
    temporary = p.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=matrix.shape,
            genes=genes,
            cells=cells,
            truth=truth,
            cell_types=np.asarray(ctypes, dtype=str),
        )
    temporary.replace(p)
    h = hashlib.sha256()
    for arr in [matrix.data, matrix.indices, matrix.indptr, genes, cells, np.asarray(truth)]:
        h.update(np.ascontiguousarray(arr).tobytes())
    save(
        ROOT / "results" / (case["id"] + ".input.json"),
        {"case": case, "input_sha256": h.hexdigest(), "shape": list(matrix.shape), "source": source_info},
    )
    return p


def stage_group(cases):
    """Cases share a study matrix; stream it once for both selected samples."""
    import h5py

    first = cases[0]
    kind = first["kind"]
    if kind == "mtx":
        meta = pd.read_csv(first["cells_path"], dtype=str)
        genes = pd.read_csv(first["genes_path"], header=None, dtype=str).iloc[:, 0].to_numpy()
        selection = [np.flatnonzero(meta[first["sample_col"]].astype(str).to_numpy() == c["sample"]) for c in cases]
        joined = np.concatenate(selection)
        matrix, digest = stream_mtx(first["path"], joined, (len(genes), len(meta)))
        offset = 0
        for case, indices in zip(cases, selection, strict=False):
            g = meta.iloc[indices]
            truth, ctypes = labels(g.cell_type)
            save_input(
                case,
                matrix[:, offset : offset + len(indices)],
                genes,
                g.cell_name,
                truth,
                ctypes,
                {
                    "matrix_path": first["path"],
                    "matrix_uncompressed_sha256": digest,
                    "cells_path": first["cells_path"],
                    "genes_path": first["genes_path"],
                },
            )
            offset += len(indices)
    elif kind == "readme":
        case = first
        d = Path(case["path"])
        meta = pd.read_csv(d / "metadata.csv", dtype=str)
        genes = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None, dtype=str).iloc[:, 1].to_numpy()
        cells = pd.read_csv(d / "barcodes.tsv.gz", sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()
        matrix, digest = stream_mtx(d / "matrix.mtx.gz", np.arange(len(cells)), (len(genes), len(cells)))
        aligned = meta.set_index("cell_name").reindex(cells)
        truth, ctypes = labels(aligned.cell_type)
        save_input(
            case,
            matrix,
            genes,
            cells,
            truth,
            ctypes,
            {"matrix_path": str(d / "matrix.mtx.gz"), "matrix_uncompressed_sha256": digest},
        )
    else:
        with h5py.File(first["path"]) as f:
            x = f["raw/X"] if "raw/X" in f else f["X"]
            var = f["raw/var"] if "raw/var" in f else f["var"]
            obs = f["obs"]
            genes = col(var, "feature_name")
            barcodes = col(obs, index_col(obs))
            donors = col(obs, first["donor_col"])
            disease = col(obs, "disease") if "disease" in obs else np.repeat("normal", len(donors))
            for case in cases:
                selected = np.flatnonzero(
                    (donors == case["sample"]) & np.isin(np.char.lower(disease), ["normal", "healthy"])
                )
                indptr = x["indptr"][:]
                rows = []
                vals = []
                ptr = [0]
                for i in selected:
                    lo, hi = int(indptr[i]), int(indptr[i + 1])
                    rows.append(x["indices"][lo:hi])
                    v = x["data"][lo:hi]
                    if not np.all(np.isfinite(v) & (v >= 0) & np.isclose(v, np.rint(v), rtol=0, atol=1e-6)):
                        raise ValueError("Healthy values are not raw counts")
                    vals.append(np.rint(v).astype(np.int64))
                    ptr.append(ptr[-1] + hi - lo)
                # CSR cells x genes transposed into CSC genes x cells.
                matrix = sparse.csc_matrix(
                    (np.concatenate(vals), np.concatenate(rows), np.asarray(ptr)), shape=(len(genes), len(selected))
                )
                save_input(
                    case,
                    matrix,
                    genes,
                    barcodes[selected],
                    np.zeros(len(selected), dtype=np.int8),
                    np.repeat("healthy", len(selected)),
                    {
                        "h5ad_path": case["path"],
                        "counts_group": x.name,
                        "genes_group": var.name,
                        "source_size_bytes": Path(case["path"]).stat().st_size,
                    },
                )
