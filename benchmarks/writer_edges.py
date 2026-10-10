"""Check signed-zero byte preservation in the repeated-row writer."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/home/toresbe/cancer_research/benchmark_2026-10-09")
code = r"""
import sys
sys.path.insert(0,sys.argv[1])
import numpy as np, pandas as pd
from copykat_py.copykat import _write_cna_csv
values = np.zeros((100, 2), dtype=np.float64)
values[0, 0] = -0.0
lead = pd.DataFrame({'gene': ['g'+str(i) for i in range(100)]})
_write_cna_csv(sys.argv[2], lead, values, ['cellA','cellB'], round_floats=False, quote_strings=False)
"""
paths = []
for variant in ["cpu_stack", "gpu"]:
    path = ROOT / "results" / ("writer_signed_zero_" + variant + ".tsv")
    env = dict(os.environ, MPLCONFIGDIR=str(ROOT / "scratch/matplotlib"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        ["/home/toresbe/envs/copykat_py_main/bin/python", "-c", code, str(ROOT / "snapshots" / variant), str(path)],
        env=env,
        check=True,
    )
    paths.append(path)
a, b = (p.read_bytes() for p in paths)
result = {
    "case": "adjacent numerically equal rows with different IEEE signed zero bits",
    "byte_identical": a == b,
    "reference_row1": a.splitlines()[2].decode(),
    "repeated_writer_row1": b.splitlines()[2].decode(),
    "source_reference": "ff63f19",
    "source_repeated_writer": "14aae8d",
}
(ROOT / "results/writer_signed_zero.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
