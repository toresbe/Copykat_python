"""The r profile against values printed by R (R 4.x, CopyKAT-R 1.2.5 and 1.1.0, mixtools 2.0.0)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse
from scipy.stats import ks_2samp

from copykat_py import copykat
from copykat_py._types import PredictionLabel
from copykat_py.rcompat import final_call
from copykat_py.rcompat.baseline import cutree
from copykat_py.rcompat.mixtools import normalmix_em
from copykat_py.rcompat.rng import RRandom
from copykat_py.rcompat.rstats import ks_statistic

DATA = Path(__file__).resolve().parents[1] / "copykat_py" / "data"
REFERENCE = Path(__file__).parent / "data" / "r_reference"


def test_rng_matches_r_draw_for_draw() -> None:
    # set.seed(1234); runif(4); rnorm(4); rgamma(4, 30.5, 26); rgamma(4, 0.4)
    rng = RRandom(1234)
    np.testing.assert_array_equal(
        rng.runif(4), [0.11370341130532324, 0.62229940481483936, 0.60927473288029432, 0.62337944167666137]
    )
    np.testing.assert_array_equal(
        rng.rnorm(4), [1.084441176683056, -2.3456977026293493, 0.42912468881104998, 0.50605589215757396]
    )
    np.testing.assert_array_equal(
        rng.rgamma(4, 30.5, 26), [1.0359461960303935, 1.3703113484944156, 1.0264433414366074, 1.0555090830329403]
    )
    np.testing.assert_array_equal(
        rng.rgamma(4, 0.4),
        [0.00045091157060938929, 1.4075176249736396, 0.00063169978913666085, 0.2583622536703013],
    )


def test_ks_statistic_matches_r() -> None:
    # set.seed(42); x <- rgamma(1000, 120.3, 26); y <- rgamma(1000, 125.1, 26); ks.test(x, y)$statistic
    rng = RRandom(42)
    x, y = rng.rgamma(1000, 120.3, 26), rng.rgamma(1000, 125.1, 26)
    assert ks_statistic(x, y) == 0.19700000000000001


def test_ks_statistic_on_the_cut_is_not_a_breakpoint() -> None:
    """A KS distance of exactly 100/1000 is 0.1 in R, so ``> 0.1`` is FALSE; SciPy's is a bit above."""
    y = [0.0]
    x = list(1000.0 + np.arange(101))
    for k in range(899):  # alternate one y, one x: the distance stays at 100/1000
        y.append(5000.0 + k)
        x.append(5000.5 + k)
    y += list(5909.0 + np.arange(100))
    x_arr, y_arr = np.array(x), np.array(y)
    assert ks_statistic(x_arr, y_arr) == 0.1  # R: 0.10000000000000001, and d > 0.1 is FALSE
    assert ks_2samp(x_arr, y_arr).statistic > 0.1


def test_normalmix_em_matches_mixtools() -> None:
    # set.seed(7); x <- c(rnorm(300, 0, 0.1), rnorm(150, 0.25, 0.1), rnorm(50, -0.3, 0.1))
    # normalmixEM(x, lambda = rep(1, 3)/3, mu = c(-0.2, 0, 0.2), sigma = max(0.05, 0.5 * sd(x)),
    #             arbvar = FALSE, ECM = FALSE, maxit = 5000)
    rng = RRandom(7)
    x = np.concatenate([rng.rnorm(300, 0, 0.1), rng.rnorm(150, 0.25, 0.1), rng.rnorm(50, -0.3, 0.1)])
    mu, lam, sigma, restarted = normalmix_em(x, [max(0.05, 0.5 * np.std(x, ddof=1))], maxit=5000)
    np.testing.assert_allclose(mu[0], [-0.29758454664725165, 0.0084599959724264151, 0.25435052643415368], rtol=1e-12)
    np.testing.assert_allclose(lam[0], [0.097572037088399108, 0.61407973619795675, 0.28834822671364413], rtol=1e-12)
    np.testing.assert_allclose(sigma[0], 0.097128798057431562, rtol=1e-12)
    assert not restarted[0]


def test_cutree_numbers_clusters_by_first_cell() -> None:
    from scipy.cluster.hierarchy import linkage

    points = np.array([[10.0], [0.0], [10.1], [20.0], [0.1], [20.1]])
    labels = cutree(linkage(points, "ward"), 3)
    np.testing.assert_array_equal(labels, [1, 2, 1, 3, 2, 3])  # R: cutree(hclust(dist(points), "ward.D2"), 3)


def test_ward_d_heights_are_on_r_scale() -> None:
    # hclust(dist(c(0, 1, 3, 7)), "ward.D")$height: 1, 3, 7.5
    mat = np.array([[0.0, 1.0, 3.0, 7.0]])
    Z = final_call.ward_d(mat, final_call.DistanceMetric.EUCLIDEAN)
    np.testing.assert_allclose(Z[:, 2], [1.0, 3.0, 7.5])


def test_r110_tied_scores_recycle_cluster_ids_as_r_does() -> None:
    """R 1.1.0: ``hc.umap == which(cl.ID == max(cl.ID))`` compares element-wise against c(1, 2)."""
    labels = np.array([1, 2, 2, 1, 1, 2])
    tied = np.array([1, 2])
    np.testing.assert_array_equal(final_call._recycled_equal(labels, tied), [True, True, False, False, True, True])


def _bundled_sample() -> dict:
    matrix = scipy.sparse.csc_matrix(scipy.io.mmread(DATA / "exp_rawdata_sparse.mtx"))
    genes = np.array((DATA / "exp_rawdata_genes.txt").read_text().split())
    barcodes = np.array((DATA / "exp_rawdata_barcodes.txt").read_text().split())
    return {"matrix": matrix, "genes": genes, "barcodes": barcodes}


@pytest.mark.parametrize("r_version", ["1.2.5", "1.1.0"])
def test_r_profile_reproduces_copykat_r_on_the_bundled_sample(tmp_path, monkeypatch, r_version) -> None:
    """Same normal cells, breakpoints, final clusters and calls as CopyKAT-R (n.cores = 1)."""
    monkeypatch.chdir(tmp_path)
    reference = REFERENCE / f"exp_sample_r{r_version}"
    stages = tmp_path / "stages"
    result = copykat(
        _bundled_sample(),
        sam_name="sample",
        plot_genes=False,
        n_cores=2,
        r_version=r_version,
        dump_stages=str(stages),
    )
    from copykat_py._stages import load

    dumped = load(stages)
    breaks = dumped["segmentation_1"]["breaks"]["values"].ravel().astype(int)
    np.testing.assert_array_equal(breaks, np.loadtxt(reference / "breakpoints.txt", dtype=int))
    assert sorted(dumped["relative"]["preN"]["values"].ravel()) == (reference / "normal_cells.txt").read_text().split()

    expected = pd.read_csv(reference / "calls.tsv", sep="\t", index_col="cell")
    prediction = result["prediction"].set_index("cell.names")["copykat.pred"]
    assert (prediction.reindex(expected.index) == expected["prediction"]).all()
    labels = pd.Series(
        dumped["final"]["step8_clusters"]["values"].ravel(), index=dumped["final"]["step8_clusters"]["names"]
    )
    pd.testing.assert_series_equal(
        labels.reindex(expected.index).astype(int), expected["step8_cluster"], check_names=False
    )
    assert set(expected["prediction"]) <= {PredictionLabel.DIPLOID, PredictionLabel.ANEUPLOID}
