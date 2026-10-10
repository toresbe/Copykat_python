"""Reconstruct benchmark source snapshots without modifying Git refs.

Usage: python benchmarks/prepare.py /SSD/review-directory
Requires existing local source commit objects. Use the same root as the
benchmark driver (COPYKAT_REVIEW_ROOT) and configure its Python runtimes.
"""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1])
for directory in ["snapshots", "results", "scratch"]:
    (root / directory).mkdir(parents=True, exist_ok=True)
refs = {
    "main": "ea1a15c",
    "optimistic": "ea1a15c",
    "cpu_stack": "ff63f19",
    "ward_only": "350c1c0",
    "collapse": "5768793",
    "gpu": "14aae8d",
}
manifest = {}
for name, ref in refs.items():
    target = root / "snapshots" / name
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing source snapshot: {target}")
    target.mkdir()
    archive = subprocess.check_output(["git", "archive", ref], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        t.extractall(target, filter="data")
    manifest[name] = {
        "base": subprocess.check_output(["git", "rev-parse", ref], cwd=repo, text=True).strip(),
        "patches": [],
    }
for ref in ["24475ca", "1cf6437", "516d719", "bb4abc5", "2215462"]:
    diff = subprocess.check_output(["git", "diff", ref + "^", ref], cwd=repo)
    subprocess.run(
        ["git", "apply", "--whitespace=error", "-"], cwd=root / "snapshots/optimistic", input=diff, check=True
    )
    manifest["optimistic"]["patches"].append(
        subprocess.check_output(["git", "rev-parse", ref], cwd=repo, text=True).strip()
    )
(root / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Snapshots prepared. No Git refs or index changed.")
