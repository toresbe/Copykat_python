"""Run copykat() on one benchmark sample and print a JSON result line.

Usage: python run_sample.py SAMPLE OUTDIR [--n-cores N] [--plot] [--kw key=value ...]
The copykat_py package is imported from PYTHONPATH, so the same script can
benchmark any checkout. Saves the final CNA matrix (float32 .npy), the
prediction table and runtime.json into OUTDIR.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_sample, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("outdir")
    ap.add_argument("--n-cores", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--keep-outputs", action="store_true")
    ap.add_argument("--null-outputs", action="store_true",
                    help="send the two large CNA text files to /dev/null (still formatted and written)")
    ap.add_argument("--kw", action="append", default=[], help="extra copykat() kwargs, key=value (value parsed as JSON if possible)")
    args = ap.parse_args()

    kwargs = {}
    for item in args.kw:
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kwargs[k] = v

    rawmat, truth = load_sample(args.sample)
    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)
    if args.null_outputs:
        for suffix in ("CNA_raw_results_gene_by_cell.txt", "CNA_results.txt"):
            fn = f"{args.sample}_copykat_{suffix}"
            if os.path.lexists(fn):
                os.remove(fn)
            os.symlink(os.devnull, fn)

    import numpy as np
    from copykat_py.copykat import copykat
    import copykat_py

    load_before = os.getloadavg()[0]
    t0 = time.perf_counter()
    res = copykat(rawmat, sam_name=args.sample, n_cores=args.n_cores, plot_genes=args.plot, **kwargs)
    wall = time.perf_counter() - t0

    cna = res["CNAmat"]
    # float16 keeps the comparison data small; the text outputs are deleted
    # (they can be many GB) unless --keep-outputs is given.
    np.save("cna.npy", cna.iloc[:, 3:].to_numpy(dtype=np.float16))
    np.save("cna_cells.npy", np.asarray(cna.columns[3:], dtype=object), allow_pickle=True)
    res["prediction"].to_csv("prediction.tsv", sep="\t", index=False)
    if not args.keep_outputs:
        for fn in os.listdir("."):
            if fn.endswith((".txt", ".pkl")):
                os.remove(fn)
    out = {
        "sample": args.sample,
        "code": os.path.dirname(copykat_py.__file__),
        "kwargs": kwargs,
        "wall_seconds": wall,
        "loadavg_before": load_before,
        "loadavg_after": os.getloadavg()[0],
        "steps": {s["step"]: s["seconds"] for s in res["runtime"]["steps"]},
        "metrics": score(res["prediction"], truth) if truth is not None else None,
    }
    with open("bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("BENCH_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
