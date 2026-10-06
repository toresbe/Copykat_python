#!/usr/bin/env bash
# Usage: sweep.sh CODE_ROOT OUT_ROOT [extra run_sample.py args...]
# Runs every benchmark sample sequentially against the copykat_py under
# CODE_ROOT. SAMPLES (space-separated) restricts the set; samples that already
# have a result in OUT_ROOT/results.jsonl are skipped. Before each sample it
# waits until the 1-minute load average is below MAX_LOAD (default 6), so
# other jobs on the machine do not skew the timings.
set -u
code=$1; out=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python}
mkdir -p "$out"
samples=${SAMPLES:-$(PYTHONPATH=$here $py -c "from datasets import sample_names; print(' '.join(sample_names()))")}
wait_quiet () {
  while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge "${MAX_LOAD:-6}" ]; do sleep 20; done
}
for s in $samples; do
  wait_quiet
  if [ -f "$out/results.jsonl" ] && grep -q "\"sample\": \"$s\"" "$out/results.jsonl"; then continue; fi
  PYTHONPATH=$code $py -u "$here/run_sample.py" "$s" "$out/$s" "$@" > "$out/$s.log" 2>&1
  grep -h BENCH_JSON "$out/$s.log" | sed 's/^BENCH_JSON //' >> "$out/results.jsonl" || echo "FAILED $s" >> "$out/results.jsonl"
done
