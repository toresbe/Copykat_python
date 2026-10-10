"""Print a table from one or more results.jsonl files produced by sweep.sh."""

import json
import sys


def load(path):
    rows = {}
    with open(path) as stream:
        for line in stream:
            if line.startswith("FAILED"):
                rows[line.split()[1]] = None
                continue
            d = json.loads(line)
            rows[d["sample"]] = d
    return rows


def main(paths):
    for path in paths:
        print(f"== {path}")
        tot = 0.0
        for name, d in load(path).items():
            if d is None:
                print(f"{name[:30]:30s} FAILED")
                continue
            m = d["metrics"]
            tot += d["wall_seconds"]
            big = {k: round(v, 1) for k, v in d["steps"].items() if v >= 0.5}
            if m is None:
                print(f"{name[:30]:30s} wall={d['wall_seconds']:7.1f}s (unlabelled) {big}")
                continue
            print(
                f"{name[:30]:30s} wall={d['wall_seconds']:7.1f}s "
                f"acc={m['accuracy']:.3f} bacc={m['balanced_accuracy']:.3f} "
                f"f1={m['f1_aneuploid']:.3f} ari={m['ari']:.3f} lowconf={int(m['low_conf'])} {big}"
            )
        print(f"total wall {tot:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
