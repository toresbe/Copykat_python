"""Accuracy endpoints only: no benchmark timers or CPU-time instrumentation."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from accuracy_data import ROOT, save
from scipy import sparse


def metrics(truth, calls, mask=None):
    truth = np.asarray(truth)
    calls = np.asarray(calls)
    labelled = truth >= 0
    if mask is not None:
        labelled &= np.asarray(mask, dtype=bool)
    defined = labelled & (calls >= 0)
    pos = labelled & (truth == 1)
    neg = labelled & (truth == 0)
    tp = int((pos & (calls == 1)).sum())
    tn = int((neg & (calls == 0)).sum())
    fp = int((neg & (calls == 1)).sum())
    fn = int((pos & (calls == 0)).sum())
    p = int(pos.sum())
    n = int(neg.sum())
    scored = int(defined.sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "n_labelled": p + n,
        "n_defined": scored,
        "coverage": scored / (p + n) if p + n else None,
        "positive": p,
        "negative": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "abstained_positive": p - tp - fn,
        "abstained_negative": n - tn - fp,
        "accuracy_defined": (tp + tn) / scored if scored else None,
        "balanced_accuracy_defined": (sens + spec) / 2 if sens is not None and spec is not None else None,
        "sensitivity_defined": sens,
        "specificity_defined": spec,
        "f1_aneuploid": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "coverage_adjusted_balanced_recall": (tp / p + tn / n) / 2 if p and n else None,
        "coverage_adjusted_specificity": tn / n if n else None,
    }


class DiagnosticOutput:
    """Keep biological logs, discard intrinsic pipeline timing announcements."""

    def __init__(self, target):
        self.target = target
        self.buffer = ""

    def write(self, value):
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if "runtime" not in line.lower() and "elapsed time" not in line.lower():
                self.target.write(line + "\n")
        return len(value)

    def flush(self):
        self.target.flush()


def run(case_id, variant):
    case = next(c for c in json.loads((ROOT / "dataset_manifest.json").read_text())["cases"] if c["id"] == case_id)
    name = case_id + "__" + variant
    result = {
        "case": case,
        "variant": variant,
        "performance_measurements": False,
        "source_sha": json.loads((ROOT / "protocol.json").read_text())["reference"][variant],
        "input": json.loads((ROOT / "results" / (case_id + ".input.json")).read_text()),
        "options": {"genome": "hg20", "id_type": "S", "n_cores": 4, "plot_genes": False},
    }
    work = ROOT / "scratch" / name
    work.mkdir(exist_ok=True)
    z = np.load(ROOT / "cache" / (case_id + ".npz"), allow_pickle=False)
    matrix = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    genes = z["genes"]
    cells = z["cells"]
    truth = z["truth"]
    ctypes = z["cell_types"]
    raw = {"matrix": matrix, "genes": genes.astype(object), "barcodes": cells.astype(object)}
    sys.path.insert(0, str(ROOT / "snapshots" / variant))
    ck = importlib.import_module("copykat_py.copykat")
    assert str(ck.__file__).startswith(str(ROOT / "snapshots" / variant))
    spec = importlib.util.spec_from_file_location("accuracy_anchor", ROOT / "snapshots/gpu_anchor/copykat_py/anchor.py")
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)
    immune = anchor.count_markers(raw, anchor.IMMUNE_MARKERS).reindex(cells).to_numpy()
    endothelial = anchor.count_markers(raw, anchor.ENDOTHELIAL_MARKERS).reindex(cells).to_numpy()
    state = {}
    for function in ["baseline_norm_cl", "baseline_gmm"]:
        if hasattr(ck, function):
            original = getattr(ck, function)

            def wrapped(*args, _original=original, **kwargs):
                r = _original(*args, **kwargs)
                state["reference_cells"] = list(r.get("preN", []))
                state["anchor_path"] = r.get("anchor_path", "sigma/default")
                return r

            setattr(ck, function, wrapped)
    # Only suppress unused large text serialization, identically for both modes.
    if hasattr(ck, "_write_cna_csv"):
        ck._write_cna_csv = lambda *args, **kwargs: None
    original_csv = pd.DataFrame.to_csv

    def csv(frame, path=None, *args, **kwargs):
        if isinstance(path, str | Path) and any(
            token in str(path) for token in ["CNA_raw_results_gene_by_cell", "CNA_results.txt"]
        ):
            return None
        return original_csv(frame, path, *args, **kwargs)

    pd.DataFrame.to_csv = csv
    options = {"genome": "hg20", "id_type": "S", "n_cores": 4, "plot_genes": False}
    if variant == "gpu_anchor":
        options.update(backend_name="gpu", anchor="markers", final_call="arm_correlation", ks_method="mc")
    result["options"] = options
    os.chdir(work)
    np.random.seed(20261010)  # noqa: NPY002 - preserve the frozen benchmark's legacy RNG sequence.
    previous_stdout = sys.stdout
    sys.stdout = DiagnosticOutput(previous_stdout)
    try:
        r = ck.copykat(raw, sam_name="accuracy", **options)
        pred = r["prediction"].set_index("cell.names")["copykat.pred"].astype(str)
        if pred.index.duplicated().any():
            raise ValueError("Duplicated output barcodes")
        labels = pred.reindex(cells).fillna("missing").str.lower()
        calls = np.where(
            labels.str.contains("aneuploid", regex=False),
            1,
            np.where(labels.str.contains("diploid", regex=False), 0, -1),
        ).astype(np.int8)
        references = np.isin(cells, np.asarray(state.get("reference_cells", []), dtype=str))
        result.update(
            status="ok",
            metrics=metrics(truth, calls),
            anchor_path=r.get("runtime", {}).get("anchor_path", state.get("anchor_path")),
            anchor_cells=int(references.sum()),
            anchor_labelled_cells=int((references & (truth >= 0)).sum()),
            anchor_malignant_fraction=float((truth[references & (truth >= 0)] == 1).mean())
            if (references & (truth >= 0)).any()
            else None,
            non_anchor_metrics=metrics(truth, calls, ~references),
            non_marker_metrics=metrics(truth, calls, (immune < 3) & (endothelial < 3)),
            low_confidence_calls=int(labels.str.contains("low.conf", regex=False).sum()),
            call_counts=labels.value_counts().to_dict(),
        )
        np.savez_compressed(
            ROOT / "results" / (name + ".cells.npz"),
            cells=cells,
            truth=truth,
            cell_types=ctypes,
            calls=calls,
            labels=labels.to_numpy(dtype=str),
            reference=references,
            immune_counts=immune,
            endothelial_counts=endothelial,
        )
        cna = r["CNAmat"]
        out_cells = np.asarray(cna.columns[3:], dtype=str)
        arm_ids = anchor._arm_ids(cna.iloc[:, 0].to_numpy(), cna.iloc[:, 1].to_numpy())
        ids, inverse, sizes = np.unique(arm_ids, return_inverse=True, return_counts=True)
        agg = sparse.csr_matrix(
            (1 / sizes[inverse], (inverse, np.arange(len(arm_ids)))), shape=(len(ids), len(arm_ids))
        )
        arms = np.empty((len(ids), len(out_cells)), dtype=np.float64)
        h = hashlib.sha256()
        for i, cell in enumerate(out_cells):
            values = np.ascontiguousarray(cna[cell].to_numpy())
            h.update(str(cell).encode() + b"\0")
            h.update(values.tobytes())
            arms[:, i] = agg @ values.astype(np.float64)
        clustering = r["hclustering"]
        np.savez_compressed(
            ROOT / "results" / (name + ".diagnostics.npz"),
            cells=out_cells,
            arms=arms,
            arm_ids=ids,
            arm_bin_counts=sizes,
            Z=np.asarray(clustering["Z"], dtype=np.float64),
        )
        result.update(
            cna_numeric_sha256=h.hexdigest(),
            cna_shape=list(cna.shape),
            prediction_sha256=hashlib.sha256(calls.tobytes()).hexdigest(),
            environment={
                "python": sys.version,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "packages": {
                    k: getattr(sys.modules[k], "__version__", None)
                    for k in ["numpy", "scipy", "pandas", "numba", "torch", "cupy"]
                    if k in sys.modules
                },
            },
        )
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    finally:
        sys.stdout.flush()
        sys.stdout = previous_stdout
        pd.DataFrame.to_csv = original_csv
        save(ROOT / "results" / (name + ".json"), result)
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"name": name, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--variant", required=True, choices=["main", "gpu_anchor"])
    a = p.parse_args()
    run(a.case, a.variant)
