#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
python benchmarks/benchmark.py run --variant gpu --sample xenium30000 --name xenium30k_gpu --backend gpu --plot --null-outputs --timeout 240
python benchmarks/benchmark.py run --variant gpu --sample xenium170057 --name xenium170k_gpu --backend gpu --plot --null-outputs --timeout 300
python benchmarks/benchmark.py run --variant optimistic --sample Bi2021_Kidney_P90 --name kidney_optimistic --plot --timeout 180
python benchmarks/benchmark.py run --variant gpu --sample Bi2021_Kidney_P90 --name kidney_gpu --backend gpu --plot --timeout 180
python benchmarks/benchmark.py run --variant gpu --sample Bi2021_Kidney_P90 --name kidney_compat --backend gpu-compat --plot --timeout 180
python benchmarks/benchmark.py run --variant gpu --sample Bi2021_Kidney_P90 --name kidney_exactks --backend gpu --ks-method exact --plot --timeout 180
python benchmarks/benchmark.py run --variant main --sample xenium10000 --name xenium10k_main --plot --null-outputs --timeout 180
python benchmarks/benchmark.py run --variant optimistic --sample xenium10000 --name xenium10k_optimistic --plot --null-outputs --timeout 180
python benchmarks/benchmark.py run --variant cpu_stack --sample xenium10000 --name xenium10k_cpu_stack --plot --null-outputs --timeout 180
python benchmarks/benchmark.py run --variant cpu_stack --sample xenium30000 --name xenium30k_cpu_stack --plot --null-outputs --timeout 180
python benchmarks/benchmark.py run --variant optimistic --sample xenium30000 --name xenium30k_optimistic --plot --null-outputs --timeout 240
python benchmarks/benchmark.py run --variant main --sample xenium30000 --name xenium30k_main --plot --null-outputs --timeout 240
