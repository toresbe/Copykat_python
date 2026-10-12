#!/usr/bin/env bash
# Run CopyKAT-R and this package on the same inputs and compare every step.
#
# Usage: verify/verify.sh DATASET_DIR [DATASET_DIR ...]
#
# Each DATASET_DIR holds matrix.mtx (genes x cells counts), genes.txt and
# barcodes.txt; verify/make_input.py writes them from other formats.
#
# Settings, as environment variables:
#   OUT          results folder (default: verify-results)
#   R_VERSION    CopyKAT-R release to compare with: 1.2.5 (default) or 1.1.0
#   R_LIBRARY    R library that holds that release, if not R's default library
#   N_CORES      cores for this package (default 4); its results do not depend on it
#   EXACT_MEANS  1: run this package with --exact-segment-means (exact posterior
#                means instead of R's replayed draws), to see what that shortcut changes
#   R_NOISE      1: also run R with n.cores = 4, to show how much R differs from itself
#
# R always runs with n.cores = 1: with more cores R draws step 5's per-cell values
# from randomly seeded streams, so two R runs differ (see verify/README.md).
#
# For each dataset, OUT/<dataset>/ gets one folder per run (each with a stages/
# folder of intermediate results and the program's own output files) and a
# comparison table. The last command prints the verdict for every dataset.
set -euo pipefail

here=$(dirname "$(realpath "$0")")
OUT=${OUT:-verify-results}
R_VERSION=${R_VERSION:-1.2.5}
N_CORES=${N_CORES:-4}
port_args=(--r-version "$R_VERSION")
port_label=port-r$R_VERSION
if [ "${EXACT_MEANS:-0}" = 1 ]; then
    port_args+=(--exact-segment-means)
    port_label=$port_label-exact-means
fi

tables=()
for dataset in "$@"; do
    name=$(basename "$dataset")
    r_run=$OUT/$name/R-$R_VERSION
    port_run=$OUT/$name/$port_label
    mkdir -p "$OUT/$name"

    # 1. CopyKAT-R, unmodified (skipped if this run already finished).
    if [ ! -f "$r_run/sessionInfo.txt" ]; then
        echo "== $name: CopyKAT-R $R_VERSION"
        Rscript "$here/r/run_copykat.R" "$dataset" "$r_run" 1 ${R_LIBRARY:-} > "$r_run.log" 2>&1
    fi
    grep -q "^copykat $R_VERSION\$" "$r_run/sessionInfo.txt" || {
        echo "$r_run was made by another CopyKAT-R release:" "$(head -1 "$r_run/sessionInfo.txt")"
        exit 1
    }

    # 2. This package with the r profile.
    echo "== $name: copykat-py --profile r ${port_args[*]}"
    PORT_EXTRA_ARGS="${port_args[*]}" "$here/run_port.sh" "$dataset" "$port_run" r "$N_CORES"

    # 3. Step-by-step comparison.
    python "$here/compare.py" "$r_run" "$port_run" > "$OUT/$name/R-$R_VERSION-vs-$port_label.tsv"
    tables+=("$OUT/$name/R-$R_VERSION-vs-$port_label.tsv")

    # Optional: R against itself with 4 cores, for scale.
    if [ "${R_NOISE:-0}" = 1 ]; then
        r_noise=$OUT/$name/R-$R_VERSION-4cores
        if [ ! -f "$r_noise/sessionInfo.txt" ]; then
            echo "== $name: CopyKAT-R $R_VERSION with n.cores = 4"
            Rscript "$here/r/run_copykat.R" "$dataset" "$r_noise" 4 ${R_LIBRARY:-} > "$r_noise.log" 2>&1
        fi
        python "$here/compare.py" "$r_run" "$r_noise" > "$OUT/$name/R-$R_VERSION-vs-R-$R_VERSION-4cores.tsv"
        tables+=("$OUT/$name/R-$R_VERSION-vs-R-$R_VERSION-4cores.tsv")
    fi
done

python "$here/verdict.py" "${tables[@]}"
