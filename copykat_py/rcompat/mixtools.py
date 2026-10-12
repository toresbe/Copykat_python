"""``mixtools::normalmixEM`` as CopyKAT-R calls it: three components, one shared sigma.

CopyKAT calls ``normalmixEM(x, lambda = rep(1, 3)/3, mu = c(-0.2, 0, 0.2), sigma = s,
arbvar = FALSE, ECM = FALSE, maxit = ...)``. This is a port of that path of mixtools
2.0.0 (``R/normalmixEM.R`` and ``src/normpost.c``). It differs from a textbook EM in
three ways, all kept here:

* the shared sigma is updated from the squared residuals around the means of the
  *previous* iteration (``normpost`` computes them before the means move);
* iteration stops as soon as the log-likelihood rises by at most ``epsilon``,
  including when it falls (a signed difference, not an absolute one);
* the log-likelihood is computed relative to the closest component, as ``normpost``
  does, so the stopping iteration is the same as R's.

mixtools restarts from random starting values when sigma falls below 1e-8. CopyKAT's
inputs never get there in practice; this port reports it (``restarted``) instead of
drawing new starting values.
"""

import math

import numpy as np
from numba import njit, prange

_HALF_LOG_2PI = 0.91893853320467274178


@njit(cache=True)
def _normpost(x, mu, sigma, lam, res2, post):  # type: ignore[no-untyped-def]
    """``normpost`` from mixtools ``src/normpost.c``; returns the log-likelihood."""
    n = x.shape[0]
    k = mu.shape[0]
    ratio = np.empty(k)
    log_ratio = np.empty(k)
    work = np.empty(k)
    loglik = -float(n) * _HALF_LOG_2PI
    for j in range(k):
        ratio[j] = lam[j] / sigma[j]
        log_ratio[j] = math.log(ratio[j])
    for i in range(n):
        xi = x[i]
        minj = 0
        smallest = 0.0
        for j in range(k):
            r = xi - mu[j]
            r = r * r
            res2[i, j] = r
            r = r / (2.0 * sigma[j] * sigma[j])
            work[j] = r
            if j == 0 or r < smallest:
                minj = j
                smallest = r
        rowsum = 1.0
        for j in range(k):
            if j == minj:
                work[j] = 1.0
            else:
                work[j] = (ratio[j] / ratio[minj]) * math.exp(smallest - work[j])
                rowsum += work[j]
        for j in range(k):
            post[i, j] = work[j] / rowsum
        loglik += math.log(rowsum) - smallest + log_ratio[minj]
    return loglik


@njit(cache=True)
def _fit(x, mu0, sigma0, maxit, epsilon, out_mu, out_lambda, out_sigma):  # type: ignore[no-untyped-def]
    """One normalmixEM fit; returns 1 if mixtools would have restarted, else 0."""
    n = x.shape[0]
    k = mu0.shape[0]
    lam = np.full(k, 1.0 / k)
    mu = mu0.copy()
    sigma = np.full(k, sigma0)
    res2 = np.empty((n, k))
    post = np.empty((n, k))
    obsloglik = _normpost(x, mu, sigma, lam, res2, post)
    diff = epsilon + 1.0
    iteration = 0
    restarted = 0
    while diff > epsilon and iteration < maxit:
        # lambda <- colMeans(postprobs)
        for j in range(k):
            s = 0.0
            for i in range(n):
                s += post[i, j]
            lam[j] = s / n
        # mu[w] <- sum(postprobs[, w] * x) / (n * lambda[w])
        for j in range(k):
            s = 0.0
            for i in range(n):
                s += post[i, j] * x[i]
            mu[j] = s / (n * lam[j])
        # one shared sigma from res2, the residuals around the previous means
        s = 0.0
        for j in range(k):
            for i in range(n):
                s += post[i, j] * res2[i, j]
        lam_sum = 0.0
        for j in range(k):
            lam_sum += lam[j]
        shared = math.sqrt(s / (n * lam_sum))
        for j in range(k):
            sigma[j] = shared
        if shared < 1e-8:
            restarted = 1
            break
        newobsloglik = _normpost(x, mu, sigma, lam, res2, post)
        diff = newobsloglik - obsloglik
        obsloglik = newobsloglik
        iteration += 1
    for j in range(k):
        out_mu[j] = mu[j]
        out_lambda[j] = lam[j]
    out_sigma[0] = sigma[0]
    return restarted


@njit(parallel=True, cache=True)
def _fit_columns(X, mu0, sigma0, maxit, epsilon, out_mu, out_lambda, out_sigma, out_restarted):  # type: ignore[no-untyped-def]
    for c in prange(X.shape[1]):
        x = np.ascontiguousarray(X[:, c])
        out_restarted[c] = _fit(x, mu0, sigma0[c], maxit, epsilon, out_mu[c], out_lambda[c], out_sigma[c : c + 1])


def normalmix_em(
    columns: np.ndarray, sigma_init: np.ndarray, maxit: int, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit every column of ``columns`` (genes x profiles) as CopyKAT-R does.

    Returns means (profiles x 3), lambdas (profiles x 3), sigmas (profiles,) and a
    boolean array marking fits where mixtools would have restarted.
    """
    X = np.asarray(columns, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n_profiles = X.shape[1]
    mu = np.empty((n_profiles, 3))
    lam = np.empty((n_profiles, 3))
    sigma = np.empty(n_profiles)
    restarted = np.zeros(n_profiles, dtype=np.int64)
    _fit_columns(
        X,
        np.array([-0.2, 0.0, 0.2]),
        np.asarray(sigma_init, dtype=np.float64),
        int(maxit),
        float(epsilon),
        mu,
        lam,
        sigma,
        restarted,
    )
    return mu, lam, sigma, restarted.astype(bool)
