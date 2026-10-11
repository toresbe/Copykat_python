# Synthetic-truth tests

End-to-end runs of `copykat()` on simulated scRNA-seq counts where the copy
number of every cell is known. `synthetic_data.py` draws negative binomial
counts for 8,000 real genes and 300 cells (180 tumour, 120 normal in two cell
types with their own expression programs). Every test runs once for hg20 and
once for mm10. The human tumour carries gains of chr7 and chr20 and losses of
chr10 and 3p; the mouse tumour gains chr15 and chr11 and loses chr4 and
proximal chr14. Everything is seeded, so a run is exactly reproducible on one
machine.

| File | What it checks |
| --- | --- |
| `test_recovery.py` | Planted events are found with the right sign and dose; a pure-diploid population gets no confident aneuploid calls; half depth and 10% / 90% purity still classify correctly. |
| `test_perturbation.py` | A copy change injected on chr5 (x0.5 to x3) moves the output in that direction, monotonically, without moving other chromosomes or calls; a focal gain stays focal. |
| `test_contamination.py` | Tumour cells mixed into a supplied normal reference (0–75%): how much tumour signal the reference absorbs, and where the calls break. |
| `test_invariance.py` | Repeat runs give identical output; shuffled cells and genes agree to float32 rounding. |
| `test_subsampling.py` | Dropping 20% or 50% of cells keeps the calls and profiles of the remaining cells. |

Run them with `pytest tests/synthetic` (about four minutes on four cores, both genomes; set
`SYNTHETIC_N_CORES` to change the core count). Set `SYNTHETIC_METRICS_JSON` to
a path to save every metric the tests assert on; `report.py` turns one or two
of those files into a Markdown comparison and a figure, which is what the
`Synthetic-truth tests` workflow posts to its job summary on each pull request.
