#!/usr/bin/env bash
# Run this Python port on one dataset and save the intermediate result of each step.
#
# Usage: verify/run_port.sh INPUT_DIR OUTPUT_DIR PROFILE N_CORES
#
# INPUT_DIR holds matrix.mtx, genes.txt and barcodes.txt (the same files R reads).
set -euo pipefail
input=$(realpath "$1")
out=$(realpath -m "$2")
profile=$3
n_cores=$4

mkdir -p "$out"
copykat-py \
    --input "$input/matrix.mtx" --genes "$input/genes.txt" --barcodes "$input/barcodes.txt" \
    --sample-name sample --genome hg20 --n-cores "$n_cores" --no-plot-genes \
    --profile "$profile" ${PORT_EXTRA_ARGS:-} \
    --dump-stages "$out/stages" --output-dir "$out" \
    > "$out/run.log" 2>&1
