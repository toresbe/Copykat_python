#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
python benchmarks/benchmark.py run --variant ward_only --sample T989 --name t989_ward --timeout 180
python benchmarks/benchmark.py run --variant collapse --sample T989 --name t989_collapse --timeout 180
python benchmarks/benchmark.py run --variant cpu_stack --sample xenium170057 --name xenium170k_cpu_stack --plot --null-outputs --timeout 180
python benchmarks/benchmark.py smoothing --variant main --sample SCPCL001108 --name smoothing_main_8 --reps 3 --timeout 90
python benchmarks/benchmark.py smoothing --variant optimistic --sample SCPCL001108 --name smoothing_optimistic_8 --reps 3 --timeout 90
