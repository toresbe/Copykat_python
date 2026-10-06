"""R COMPATIBILITY reference mode: reproduce CopyKAT-R (v1.2.5) predictions.

PURPOSE
-------
This module is a *validation reference*, not a recommended way to run
CopyKAT-Py.  Calling ``copykat(..., r_compat=True)`` swaps every place where
the default Python pipeline deliberately or accidentally deviates from the R
package for the R behaviour, so that the two implementations can be compared
(or the rewrite can be validated) without the deviations getting in the way.

Every deviation is marked with a comment containing the string
"R COMPATIBILITY" (grep for it, here and in ``copykat.py``).  Each such
comment states (a) what R does, (b) what default CopyKAT-Py does instead, and
(c) the evidence for it.

WHAT IS MATCHED, AND HOW WELL (R 1.2.5 vs. ``main`` @ ea1a15c, measured on
six datasets; see ``scripts/validate_r_compat.py``)
-------------------------------------------------------------------------
* Steps 1-3 (filtering, annotation, Freeman-Tukey, DLM smoothing), the
  clustering engine and step 6 (bin conversion) are already numerically
  identical to R in the default pipeline (max abs diff ~1e-8).  No change.
* Step 4 (baseline) and steps 7-8 (final prediction) are what this module
  re-implements, together with a handful of switches in ``copykat.py``.

WHAT IS *NOT* MATCHED (cannot be, or is deliberately left alone)
----------------------------------------------------------------
R COMPATIBILITY (not reproduced): step 5 segmentation.  R draws Monte Carlo
posterior samples (MCMCpack::MCpoissongamma, mclapply RNG streams) for both the
KS breakpoint test and the segment means; default CopyKAT-Py uses the exact
closed-form segment mean and its own RNG for the KS test.  The two agree to
within R's *own* run-to-run noise (median per-cell correlation 0.957
Python-vs-R vs. 0.953 R-vs-R with different seeds, su003), but the gene-level
matrices are therefore near-identical, never bit-identical.  Breakpoint sets
overlap ~90% (borderline KS calls).  Predicted labels were identical in
practice on every sample tested.

R COMPATIBILITY (not reproduced): plotting (R uses heatmap.3 and its own
``hclust``); output cell-name mangling (R's ``data.frame`` turns ``:`` into ``.``
and prefixes ``X`` to numeric names); the "known-normal" mode's error
behaviour on degenerate input; mixtools' "variance going to zero -> restart"
branch in the EM (never triggered in testing).

SUPPORTED CONFIGURATIONS
------------------------
``genome="hg20"`` and ``cell_line="no"`` only, with ``distance`` in
{"euclidean", "pearson", "spearman"}.  mm10 and cell-line mode raise
``NotImplementedError`` rather than silently differing from R.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import f as f_dist
from scipy.stats import rankdata, wasserstein_distance
from sklearn.metrics import silhouette_score


def check_supported(genome, cell_line):
    """R COMPATIBILITY: restrict r_compat to the code paths that were validated.

    The mm10 branch and cell-line mode of R copykat() were not compared against
    the Python port, so r_compat refuses them instead of pretending.
    """
    if str(genome).strip().lower() != "hg20":
        raise NotImplementedError("r_compat=True is only validated for genome='hg20'")
    if cell_line == "yes":
        raise NotImplementedError("r_compat=True is only validated for cell_line='no'")


# --------------------------------------------------------------------------
# Hierarchical clustering exactly as R does it
# --------------------------------------------------------------------------
def hclust_ward_d(data, distance="euclidean"):
    """Return a scipy linkage equivalent to R ``hclust(d, method="ward.D")``.

    R COMPATIBILITY: R's steps 7 and 8 (the prediction steps) call
    ``hclust(parDist(t(mat), method="euclidean"), method="ward.D")``.
    ``ward.D`` applies the Ward update to the *unsquared* distances, whereas
    ``scipy.linkage(method="ward")`` / ``fastcluster`` (what default CopyKAT-Py
    uses everywhere) implement ``ward.D2``, i.e. Ward on *squared* distances.
    The two give different dendrograms.

    The identity used here: ``ward.D`` on ``d``  ==  ``ward.D2`` on ``sqrt(d)``
    (ward.D2 squares its input before applying Lance-Williams).  So we feed
    ``sqrt(d)`` to scipy's "ward".  The *topology* (hence every ``cutree``
    partition) is identical to R's; the merge *heights* in the returned
    linkage are those of ward.D2-on-sqrt(d), NOT R's ``hcc$height`` values.
    Verified: partitions identical to R's ``cutree`` (ARI 1.0) on all six
    test datasets.

    NOTE: R's *baseline* step (``baseline.norm.cl``) uses ``ward.D2``; the
    README's statement that R uses ward.D throughout is therefore inaccurate.
    That step needs no change (see ``baseline_norm_cl_r``).

    ``data`` is cells x features.  For ``distance`` other than "euclidean" R
    uses ``as.dist(1 - cor(t(data), method=distance))`` (still ward.D).
    """
    if distance == "euclidean":
        d = pdist(data, metric="euclidean")
    elif distance in ("pearson", "spearman"):
        x = data
        if distance == "spearman":
            x = np.apply_along_axis(rankdata, 1, x)
        c = np.corrcoef(x)
        iu = np.triu_indices(c.shape[0], k=1)
        d = 1.0 - c[iu]
        d = np.clip(d, 0.0, None)
    else:
        raise ValueError(f"unsupported distance for r_compat: {distance!r}")
    return linkage(np.sqrt(d), method="ward")


def cut_like_r_prediction(Z, n_cells_per_cluster_min=10):
    """Choose the cut of the final tree the way R copykat 1.2.5 does.

    R COMPATIBILITY: R tries ``k`` in 4, 3, 2 and keeps the first ``k`` whose
    *smallest* cluster has more than 10 cells
    (``for (k in 4:2) { ... if (min_size > 10) {final_k <- k; break} }``).
    Default CopyKAT-Py always cuts at k=2 (as does the README's description of
    R, which does not match the 1.2.5 code).  With k=3 or 4, R then labels the
    resulting clusters with the Wasserstein rule in ``predict_like_r``.

    If no k in 4..2 satisfies the size condition R ends with ``final_k <- NULL``
    and ``cutree(hcc, NULL)`` fails; we raise ValueError to mirror that failure.
    """
    for k in (4, 3, 2):
        cl = fcluster(Z, k, criterion="maxclust")
        if np.bincount(cl)[1:].min() > n_cells_per_cluster_min:
            return cl
    raise ValueError("R COMPATIBILITY: no k in 4..2 yields all clusters > 10 cells (R errors here too)")


# --------------------------------------------------------------------------
# mixtools::normalmixEM, as called by copykat (ECM=FALSE, arbvar=FALSE)
# --------------------------------------------------------------------------
def normalmix_em_mixtools(x, sigma_init, max_iter, epsilon=1e-8):
    """3-component shared-variance Gaussian EM, bug-for-bug as mixtools does it.

    R COMPATIBILITY: R's baseline code calls
    ``mixtools::normalmixEM(x, lambda=rep(1,3)/3, mu=c(-0.2,0,0.2), sigma=sg,
    arbvar=FALSE, ECM=FALSE, maxit=...)``.  Default CopyKAT-Py's
    ``_fit_gmm_3component`` is a textbook EM that differs from mixtools in two
    ways, both of which change which cells ``baseline.GMM`` picks as "diploid":

    1. mixtools (with ``ECM=FALSE``) updates the shared sigma from squared
       residuals computed in the *previous E-step*, i.e. around the *old*
       means, not the freshly updated means.  Textbook EM uses the new means.
       The two follow different trajectories and, on flat likelihoods, stop at
       different points (mixtools often never converges within maxit).
    2. mixtools stops when ``loglik_new - loglik_old <= epsilon`` with an
       absolute epsilon of 1e-8 (signed difference) and ``maxit`` iterations;
       the Python version used a relative tolerance.

    Evidence: with these two changes the fitted mu/lambda/sigma match R to four
    decimals on every cell checked (Lin 2020 cells 38, 76, 105, 109); without
    them, Python picked anchor cells {12,15,85,105,109} where R picked
    {12,15,38,76,85}.

    Returns (means[3], weights[3], sigma).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    mu = np.array([-0.2, 0.0, 0.2])
    lam = np.full(3, 1.0 / 3.0)
    sigma = float(sigma_init)

    def e_step(mu, sigma, lam):
        res = (x[:, None] - mu[None, :]) ** 2
        dens = np.exp(-0.5 * res / sigma ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        w = dens * lam[None, :]
        tot = w.sum(axis=1)
        tot[tot <= 0] = np.finfo(np.float64).tiny
        return w / tot[:, None], res, float(np.log(tot).sum())

    post, res, ll = e_step(mu, sigma, lam)
    diff = epsilon + 1.0
    it = 0
    while diff > epsilon and it < max_iter:
        lam = post.mean(axis=0)
        mu = (post * x[:, None]).sum(axis=0) / (n * lam)
        # R COMPATIBILITY: `res` is still the residual matrix from the previous
        # E-step (around the OLD mu) -- this is the mixtools ECM=FALSE quirk.
        sigma = float(np.sqrt((post * res).sum() / n))
        post, res, new_ll = e_step(mu, sigma, lam)
        diff = new_ll - ll
        ll = new_ll
        it += 1
    return mu, lam, sigma


# --------------------------------------------------------------------------
# Step 4: baseline
# --------------------------------------------------------------------------
def baseline_norm_cl_r(norm_mat_smooth, cell_names, min_cells=10):
    """Re-implementation of R ``baseline.norm.cl`` (cluster-based baseline).

    R COMPATIBILITY, differences from default ``baseline_norm_cl``:

    * ``min_cells=10`` (R's copykat() passes 10).  Default Python passes 5, so a
      cluster of 6-10 cells is accepted by Python and rejected by R (R then
      lowers k).
    * No PCA reduction.  Default Python clusters on a 256-component PCA when
      there are more than ``FULL_CLUSTER_MAX_CELLS`` (2000) cells; R always
      clusters on the full gene x cell matrix.  On Jerby-Arnon 2021 (2522 cells)
      this alone changes the 6-cluster partition (ARI 0.30 vs R), selects a
      different "normal" cluster and inverts the final diploid/aneuploid call.
    * The per-cluster GMM uses the mixtools-faithful EM (maxit=5000) and
      ``sd`` with n-1 (R) instead of n (numpy default).
    * Silhouette is computed on all cells; default Python subsamples above
      3000 cells.
    * When several clusters tie for the minimum sigma R takes *all* of them
      (``ct %in% which(SDM == min(SDM))``); default Python takes the first.

    The hierarchical clustering itself is ``ward.D2`` in R's baseline step, which
    is what scipy/fastcluster "ward" computes, so it is unchanged.
    """
    n_genes, n_cells = norm_mat_smooth.shape
    data_t = norm_mat_smooth.T
    Z = linkage(pdist(data_t, metric="euclidean"), method="ward")  # ward.D2, as in R

    km = 6
    ct = fcluster(Z, km, criterion="maxclust")
    while not np.all(np.bincount(ct)[1:] > min_cells):
        km -= 1
        ct = fcluster(Z, km, criterion="maxclust")
        if km == 2:
            break

    ids = np.unique(ct)
    sdm = []
    for i in ids:
        data_c = np.median(norm_mat_smooth[:, ct == i], axis=1)
        sx = max(0.05, 0.5 * np.std(data_c, ddof=1))  # R COMPATIBILITY: R's sd() uses n-1
        _, _, sig = normalmix_em_mixtools(data_c, sx, 5000)
        sdm.append(sig)
    sdm = np.asarray(sdm)

    labels2 = fcluster(Z, 2, criterion="maxclust")
    wn = silhouette_score(data_t, labels2, metric="euclidean")  # R COMPATIBILITY: no subsampling
    f_stat = sdm.max() ** 2 / sdm.min() ** 2
    pdt = f_dist.sf(f_stat, n_genes, n_genes)

    if wn <= 0.15 or not np.all(np.bincount(ct)[1:] > min_cells) or pdt > 0.05:
        wns = "unclassified.prediction"
        print("  low confidence in classification")
    else:
        wns = ""

    normal_ids = ids[sdm == sdm.min()]  # R COMPATIBILITY: all tied clusters, not just the first
    mask = np.isin(ct, normal_ids)
    basel = np.median(norm_mat_smooth[:, mask], axis=1)
    pre_n = np.asarray(cell_names, dtype=object)[mask].tolist()
    return {"basel": basel, "WNS": wns, "preN": pre_n, "cl": ct}


def baseline_gmm_r(norm_mat_smooth, cell_names, re_before, max_normal=5, mu_cut=0.05, nfraq_cut=0.99):
    """Re-implementation of R ``baseline.GMM`` (cell-by-cell GMM fallback).

    R COMPATIBILITY: R's loop fits each cell in column order with the
    mixtools-faithful EM (maxit=500) and stops after ``max_normal`` cells are
    called diploid.  If more than 2 cells were found it returns a *mean*
    baseline over them and ``WNS=""``; otherwise it returns ``RE.before``
    unchanged.  The difference from default Python is in the EM (see
    ``normalmix_em_mixtools``) and in ``sd`` with n-1 for the initial sigma.
    """
    n_cells = norm_mat_smooth.shape[1]
    found = []
    for m in range(n_cells):
        sam = norm_mat_smooth[:, m]
        sg = max(0.05, 0.5 * np.std(sam, ddof=1))
        mu, lam, _ = normalmix_em_mixtools(sam, sg, 500)
        neutral = np.abs(mu) <= mu_cut
        if neutral.sum() >= 1 and lam[neutral].sum() > nfraq_cut:
            found.append(m)
        if len(found) >= max_normal:
            break
    if len(found) > 2:
        basel = norm_mat_smooth[:, found].mean(axis=1)
        pre_n = np.asarray(cell_names, dtype=object)[found].tolist()
        # R COMPATIBILITY: WNS becomes "" here.  Default Python keeps
        # "unclassified.prediction" (and therefore labels results *.low.conf)
        # even when the GMM fallback is accepted.
        return {"basel": basel, "WNS": "", "preN": pre_n, "cl": re_before["cl"]}
    return re_before


# --------------------------------------------------------------------------
# Steps 7-8: baseline re-adjustment and final prediction
# --------------------------------------------------------------------------
def _cluster_scores(cl, cell_names, pre_n_names, n_pre_n):
    """R COMPATIBILITY: cluster 'diploid-ness' score is |cluster ∩ preN| / |preN|.

    R divides by ``length(preN)`` (a constant), i.e. ranks clusters by the raw
    *count* of preN cells they contain.  Default Python divides by the cluster
    size (the *fraction* of the cluster that is preN), which ranks differently
    when clusters differ in size.
    """
    names = np.asarray(cell_names, dtype=object)
    scores = []
    for c in np.unique(cl):
        in_c = np.isin(names[cl == c], list(pre_n_names)).sum()
        scores.append(in_c / n_pre_n)
    return np.asarray(scores)


def adjust_baseline_r(uber, diploid_mask):
    """R COMPATIBILITY: noise-suppression step of R step 7.

    Same algorithm as the default code, except the per-bin spread ``cf.h`` is
    ``sd`` with n-1 (R) instead of ``np.std`` with n.  Matches R's ``mat.adj`` to
    1e-15 on all test data.
    """
    rat = uber - uber[:, diploid_mask].mean(axis=1, keepdims=True)
    rat = rat - rat.mean(axis=0, keepdims=True)
    norm = rat[:, diploid_mask]
    cf_h = norm.std(axis=1, ddof=1)
    base = norm.mean(axis=1)
    keep = np.abs(rat - base[:, None]) <= 0.25 * cf_h[:, None]
    adj = np.where(keep, rat.mean(axis=0, keepdims=True), rat)
    return adj - adj.mean(axis=0, keepdims=True)


def predict_like_r(uber, results_com, cell_names, pre_n_names, n_pre_n, wns, distance="euclidean"):
    """Steps 7 and 8 of R copykat() (hg20, non-cell-line).

    Returns dict with ``mat_adj``, ``com_preN`` (array of label strings aligned
    to ``cell_names``), ``labels_final``, ``Z_final`` and ``labels_step7``.

    R COMPATIBILITY summary of what differs from the default Python code path:

    1. Linkage is ward.D (``hclust_ward_d``), not ward.D2.
    2. The cut is k in 4,3,2 with min cluster size > 10 (``cut_like_r_prediction``),
       not k=2 -- in both step 7 and step 8.
    3. Cluster score is divided by |preN| not by cluster size (``_cluster_scores``).
    4. ``sd`` uses n-1 in the noise suppression (``adjust_baseline_r``).
    5. The ``distance`` argument is honoured in steps 7-8 (default Python
       hard-codes euclidean here and only uses ``distance`` for plotting).
    6. Final labelling follows R's correlation/Wasserstein rule (below), instead
       of "max score -> diploid, min score -> aneuploid".
    """
    names = np.asarray(cell_names, dtype=object)

    # ---- step 7: first pass, only to define the diploid reference ----------
    z7 = hclust_ward_d(uber.T, distance)
    cl7 = cut_like_r_prediction(z7)
    s7 = _cluster_scores(cl7, names, pre_n_names, n_pre_n)
    dip_ids = np.unique(cl7)[s7 == s7.max()]  # R: which(cl.ID %in% max(cl.ID)); ties keep all
    mat_adj = adjust_baseline_r(uber, np.isin(cl7, dip_ids))

    # ---- step 8: final prediction on the adjusted matrix --------------------
    z8 = hclust_ward_d(mat_adj.T, distance)
    cl8 = cut_like_r_prediction(z8)
    s8 = _cluster_scores(cl8, names, pre_n_names, n_pre_n)
    ids = np.unique(cl8)
    cl_dip = ids[s8 == s8.max()]
    cl_aneu = ids[s8 == s8.min()]

    def consensus(mask):
        # R COMPATIBILITY: the consensus profiles are medians of the *gene-level*
        # centred segmentation output (``results.com``), not of the bin-level
        # adjusted matrix.
        return np.median(results_com[:, mask], axis=1)

    cons_dip = consensus(np.isin(cl8, cl_dip))
    cons_aneu = consensus(np.isin(cl8, cl_aneu))
    r = np.corrcoef(cons_dip, cons_aneu)[0, 1]

    com = np.empty(len(names), dtype=object)
    if r >= 0.6:
        # R COMPATIBILITY: if the diploid and aneuploid consensus profiles
        # correlate >= 0.6, R declares EVERY cell diploid.  Default Python has
        # no such rule.  This cliff is why a tiny change (e.g. ward.D vs
        # ward.D2) can flip a whole sample: on Lin 2020 R's correlation is
        # 0.612 -> all 1139 cells called diploid, 844 of them annotated
        # malignant; ward.D2 alone pushes it below 0.6 and the split appears.
        com[:] = "diploid"
    else:
        for c in ids:
            cons = consensus(cl8 == c)
            # R COMPATIBILITY: each cluster goes to whichever consensus is
            # closer in 1-D Wasserstein distance (transport::wasserstein1d,
            # p=1 == scipy.stats.wasserstein_distance).
            if wasserstein_distance(cons, cons_dip) < wasserstein_distance(cons, cons_aneu):
                com[cl8 == c] = "diploid"
            else:
                com[cl8 == c] = "aneuploid"

    # R COMPATIBILITY: R also contains
    #   if (cor(cd, ca) >= 0.4 & cor(cd, ca) < 0.6) { WNS == "unclassified.prediction" }
    # which is a *comparison* (==), not an assignment (=): a no-op.  We replicate
    # it by doing nothing, so correlations in [0.4, 0.6) do NOT trigger low.conf.
    if wns == "unclassified.prediction":
        com = np.where(com == "diploid", "c1:diploid:low.conf", com)
        com = np.where(com == "aneuploid", "c2:aneuploid:low.conf", com)

    return {
        "mat_adj": mat_adj,
        "com_preN": com,
        "labels_final": cl8,
        "Z_final": z8,  # R COMPATIBILITY: heights are ward.D2-on-sqrt(d), not R's hcc$height
        "labels_step7": cl7,
        "consensus_correlation": float(r),
    }
