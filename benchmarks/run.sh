#!/usr/bin/env bash
# Benchmark the CURRENT checkout (or any git ref) on one sample.
#
#   benchmarks/run.sh check                       # which inputs are present
#   benchmarks/run.sh list                        # runnable samples
#   benchmarks/run.sh SAMPLE [benchmark.py args]  # e.g. Bi2021_Kidney_P90, xenium10000
#
# Environment (all optional; see benchmarks/README.md):
#   COPYKAT_BENCH_PYTHON   interpreter to use         (default: python3 on PATH)
#   COPYKAT_BENCH_DATA     input data directory       (default: ~/copykat_bench/data)
#   COPYKAT_BENCH_ROOT     results/snapshots location (default: ~/copykat_bench/benchmark)
#   BENCH_REF              git ref to benchmark       (default: HEAD, committed state only)
#   BENCH_CORES            cores to request           (default: all, via nproc)
#   BENCH_NAME             result name                (default: SAMPLE_<variant>)
#
# The ref is exported with `git archive` into $COPYKAT_BENCH_ROOT/snapshots/<variant>,
# so uncommitted changes are NOT benchmarked; commit or stash-apply first.
# Extra arguments go to benchmark.py, e.g. --backend gpu --plot --timeout 600.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
py=${COPYKAT_BENCH_PYTHON:-python3}
export COPYKAT_BENCH_PYTHON="$py"

cfg () { PYTHONPATH="$here" "$py" -c "import bench_config as c; print($1)"; }

usage () { sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  check) exec "$py" "$here/fetch_data.py" check ;;
  list)
    PYTHONPATH="$here" "$py" -c "
import datasets, bench_config as c
names = datasets.sample_names() if c.SAMPLES_DIR.exists() else []
print('\n'.join(names) or 'no README samples found under ' + str(c.SAMPLES_DIR))
print('xenium10000, xenium30000, xenium<N> (N cells)' if c.XENIUM_H5.exists() else 'no Xenium matrix at ' + str(c.XENIUM_H5))"
    exit 0 ;;
esac

sample=$1; shift

"$py" -c "import numpy, scipy, pandas, sklearn" 2>/dev/null || {
  echo "error: $py is missing numpy/scipy/pandas/scikit-learn; set COPYKAT_BENCH_PYTHON to an environment with the project installed" >&2
  exit 1
}

ref=${BENCH_REF:-HEAD}
sha=$(git -C "$repo" rev-parse --short "$ref")
variant=${BENCH_VARIANT:-ref-$sha}
root=$(cfg c.ROOT)
snap="$root/snapshots/$variant"
mkdir -p "$root/results" "$root/scratch"

if [ ! -d "$snap" ]; then
  echo "exporting $ref ($sha) to $snap"
  mkdir -p "$snap"
  git -C "$repo" archive "$ref" | tar -x -C "$snap"
fi

name=${BENCH_NAME:-${sample}_${variant}}
cores=${BENCH_CORES:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}

echo "benchmarking $sample with $variant on $cores cores"
"$py" "$here/benchmark.py" run --variant "$variant" --sample "$sample" --name "$name" --cores "$cores" "$@"
echo "results: $root/results/$name.json"
