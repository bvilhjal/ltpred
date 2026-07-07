"""Gibbs sampler for the truncated multivariate normal distribution.

This is the port of LTFHPlus's ``rtmvnorm.gibbs`` (R) plus its Rcpp inner loop
(``rtmvnorm_gibbs_cpp``). Given a covariance matrix and per-coordinate lower/upper
truncation bounds, it draws from the truncated MVN by sweeping one coordinate at
a time: each coordinate is resampled from its *conditional* normal --
mean ``P[:, j] . x`` and standard deviation ``sd[j]`` -- restricted to
``(lower[j], upper[j])`` by inverse-CDF sampling
(``x_j = mu_j + sd_j * Phi^-1(U(Phi(a), Phi(b)))``; Kotecha & Djuric 1999).

``P`` and ``sd`` are the conditional-regression coefficients and residual SDs:
``P[:, j] = Sigma[-j,-j]^-1 Sigma[-j, j]`` (with a 0 in slot ``j``) and
``sd[j]^2 = Sigma[j,j] - P[:,j] . Sigma[:,j]``. They depend only on ``Sigma``, so
they are precomputed once and reused across every sweep.

In LT-FH++ the coordinates are ordered ``g`` (genetic liability), ``o`` (proband
full liability), then one per relative; ``out`` picks which posterior samples to
return (0 = genetic, 1 = full). Coordinates whose bounds coincide are ``fixed``
(a case's liability pinned at its age-of-onset threshold -- the ADuLT model) and
are held constant rather than resampled.
"""

from __future__ import annotations

import numpy as np

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf

__all__ = ["rtmvnorm_gibbs", "gibbs_params", "gibbs_estimate_batched"]

# U(Fa, Fb) draws are clamped this far off {0, 1} before the inverse-CDF step so
# a boundary draw (np.random.random() can return exactly 0.0) cannot map to an
# infinite liability. Statistically negligible; the R/Rcpp code omits it only
# because R's runif never returns its endpoints.
_U_EPS = 1e-15


def gibbs_params(covmat):
    """Precompute the sweep's conditional-regression matrix ``P`` and SDs ``sd``.

    ``P[:, j] = Sigma[-j,-j]^-1 Sigma[-j, j]`` (0 in slot ``j``) and
    ``sd[j] = sqrt(Sigma[j,j] - P[:,j] . Sigma[:,j])``. Returned separately so a
    caller sampling the same family repeatedly (the convergence loop in
    :mod:`ltpred.estimate`) pays this ``O(d^4)`` cost only once."""
    cov = np.ascontiguousarray(covmat, dtype=np.float64)
    d = cov.shape[0]
    if cov.shape != (d, d):
        raise ValueError("covmat must be square")
    P = np.zeros((d, d), dtype=np.float64)
    sd = np.empty(d, dtype=np.float64)
    idx = np.arange(d)
    for j in range(d):
        rest = idx[idx != j]
        # conditional regression of coord j on all the others
        pj = np.linalg.solve(cov[np.ix_(rest, rest)], cov[rest, j])
        P[rest, j] = pj
        var_j = cov[j, j] - pj @ cov[rest, j]
        sd[j] = np.sqrt(max(var_j, 0.0))
    return P, sd


@_jit
def _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, n_sim, burn_in, res):
    """Inner Gibbs loop (the Rcpp ``rtmvnorm_gibbs_cpp`` port).

    Runs ``burn_in + n_sim`` sweeps, writing the post-burn-in draws of the
    requested coordinates into ``res``. ``x`` is the working state (updated in
    place); ``to_return[j] >= 0`` gives the output column for coordinate ``j``,
    or -1 to drop it. Everything here is scalar/loop so it compiles under
    ``numba.njit`` and matches the pure-Python path bit-for-bit given the same
    RNG draws."""
    d = sd.shape[0]
    for k in range(-burn_in, n_sim):
        for j in range(d):
            if not fixed[j]:
                mu_j = 0.0
                for i in range(d):
                    mu_j += P[i, j] * x[i]
                sd_j = sd[j]
                fa = _norm_cdf((lower[j] - mu_j) / sd_j)
                fb = _norm_cdf((upper[j] - mu_j) / sd_j)
                u = fa + np.random.random() * (fb - fa)
                if u < _U_EPS:
                    u = _U_EPS
                elif u > 1.0 - _U_EPS:
                    u = 1.0 - _U_EPS
                x[j] = mu_j + sd_j * _norm_ppf(u)
            if k >= 0 and to_return[j] >= 0:
                res[k, to_return[j]] = x[j]
    return res


@_jit_parallel
def _gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                            batch_size, n_batch, seeds, total_sum, batch_mean):
    """Sample many families in parallel, accumulating means online (no sample store).

    All families share the conditional-regression factorisation ``(P, sd)`` (they
    have the same covariance structure); only their truncation bounds differ. Each
    ``prange`` iteration runs one family's ``burn_in + n_sim`` sweeps and writes its
    running sum ``total_sum[f]`` and its ``n_batch`` batch means ``batch_mean[f]``
    (for the batch-means Monte-Carlo SE) for the coordinates in ``out_idx``. Because
    each family seeds its own RNG (``seeds[f]``) at the top of the iteration, results
    are deterministic regardless of how ``prange`` maps families to threads. Not
    keeping the full ``(n_sim, ncols)`` sample array is what lets thousands of
    families run in compiled, parallel code with flat memory."""
    F = lowers.shape[0]
    d = sd.shape[0]
    ncols = out_idx.shape[0]

    for f in prange(F):
        lower = lowers[f]
        upper = uppers[f]
        if seeds[f] >= 0:
            np.random.seed(seeds[f])

        # per-family working state (thread-local)
        x = np.empty(d)
        fixed = np.empty(d, dtype=np.bool_)
        for j in range(d):
            fixed[j] = (upper[j] - lower[j]) < 1e-8
            p0 = (_norm_cdf(lower[j] / sd0[j]) + _norm_cdf(upper[j] / sd0[j])) * 0.5
            xj = _norm_ppf(p0) * sd0[j]
            x[j] = xj if np.isfinite(xj) else 0.0

        tot = np.zeros(ncols)
        batch_sum = np.zeros(ncols)
        bidx = 0
        in_batch = 0

        for k in range(-burn_in, n_sim):
            for j in range(d):
                if not fixed[j]:
                    mu_j = 0.0
                    for i in range(d):
                        mu_j += P[i, j] * x[i]
                    sd_j = sd[j]
                    fa = _norm_cdf((lower[j] - mu_j) / sd_j)
                    fb = _norm_cdf((upper[j] - mu_j) / sd_j)
                    u = fa + np.random.random() * (fb - fa)
                    if u < _U_EPS:
                        u = _U_EPS
                    elif u > 1.0 - _U_EPS:
                        u = 1.0 - _U_EPS
                    x[j] = mu_j + sd_j * _norm_ppf(u)
            if k >= 0:
                for c in range(ncols):
                    v = x[out_idx[c]]
                    tot[c] += v
                    if bidx < n_batch:
                        batch_sum[c] += v
                if bidx < n_batch:
                    in_batch += 1
                    if in_batch == batch_size:
                        for c in range(ncols):
                            batch_mean[f, c, bidx] = batch_sum[c] / batch_size
                            batch_sum[c] = 0.0
                        bidx += 1
                        in_batch = 0

        for c in range(ncols):
            total_sum[f, c] = tot[c]


def gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                           batch_size, n_batch, seeds):
    """Thin wrapper over the parallel kernel; returns ``(total_sum, batch_mean)``.

    ``total_sum[f, c]`` is the sum of ``n_sim`` post-burn-in draws (divide by
    ``n_sim`` for the posterior mean) and ``batch_mean[f, c, :]`` holds the
    ``n_batch`` batch means used to form the Monte-Carlo standard error."""
    F = lowers.shape[0]
    ncols = out_idx.shape[0]
    total_sum = np.zeros((F, ncols), dtype=np.float64)
    batch_mean = np.zeros((F, ncols, n_batch), dtype=np.float64)
    _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                            np.ascontiguousarray(sd0),
                            np.ascontiguousarray(lowers, dtype=np.float64),
                            np.ascontiguousarray(uppers, dtype=np.float64),
                            np.asarray(out_idx, dtype=np.int64),
                            int(n_sim), int(burn_in), int(batch_size),
                            int(n_batch), np.asarray(seeds, dtype=np.int64),
                            total_sum, batch_mean)
    return total_sum, batch_mean


@_jit
def _seed_rng(seed):
    """Seed the RNG the kernel draws from (Numba's generator under njit, NumPy's
    global generator in the pure-Python fallback). Isolated so the with/without
    -numba reproducibility switch stays in one jitted spot."""
    np.random.seed(seed)


def rtmvnorm_gibbs(covmat, lower=-np.inf, upper=np.inf, *, fixed=None,
                   out=(0,), n_sim=100_000, burn_in=1000, seed=None,
                   params=None):
    """Draw truncated-MVN samples of the coordinates listed in ``out``.

    Parameters
    ----------
    covmat : (d, d) array
        Symmetric covariance of the (untruncated) multivariate normal.
    lower, upper : float or (d,) array
        Per-coordinate truncation bounds; scalars are broadcast. ``upper`` must
        be >= ``lower`` (they are swapped with a warning otherwise, as in R).
    fixed : (d,) bool array, optional
        Coordinates to hold constant instead of resampling. Defaults to
        ``upper - lower < 1e-8`` (a pinned point mass, e.g. an age-of-onset case).
    out : sequence of int
        Zero-based coordinate indices to return (0 = genetic, 1 = full in the
        LT-FH++ ordering). Duplicates dropped, order sorted.
    n_sim, burn_in : int
        Post-burn-in draws to keep and sweeps to discard first.
    seed : int, optional
        Seeds the sampler for reproducibility.
    params : (P, sd), optional
        Precomputed :func:`gibbs_params` output; recomputed from ``covmat`` when
        omitted.

    Returns
    -------
    (n_sim, len(out)) ndarray
        Samples, columns in the sorted order of ``out``.
    """
    cov = np.ascontiguousarray(covmat, dtype=np.float64)
    d = cov.shape[0]

    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (d,)).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (d,)).copy()
    swap = upper < lower
    if np.any(swap):
        lower[swap], upper[swap] = upper[swap], lower[swap].copy()

    if fixed is None:
        fixed = (upper - lower) < 1e-8
    fixed = np.broadcast_to(np.asarray(fixed, dtype=bool), (d,)).copy()

    out = sorted(set(int(o) for o in out if 0 <= int(o) < d))
    if not out:
        out = [0]
    to_return = np.full(d, -1, dtype=np.int64)
    for col, o in enumerate(out):
        to_return[o] = col

    if params is None:
        params = gibbs_params(cov)
    P, sd = params

    # start each coordinate at the median of its *marginal* truncated normal
    # (LTFHPlus's init); fixed coords collapse to their pinned value.
    sd0 = np.sqrt(np.diag(cov))
    p0 = (_vec_norm_cdf(lower / sd0) + _vec_norm_cdf(upper / sd0)) / 2.0
    x = _vec_norm_ppf(p0) * sd0
    x = np.ascontiguousarray(np.where(np.isfinite(x), x, 0.0), dtype=np.float64)

    if seed is not None:
        _seed_rng(int(seed))

    res = np.empty((int(n_sim), len(out)), dtype=np.float64)
    _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, int(n_sim),
                 int(burn_in), res)
    return res


def _vec_norm_cdf(x):
    from ._mathfun import norm_cdf
    return norm_cdf(x)


def _vec_norm_ppf(p):
    from ._mathfun import norm_ppf
    return norm_ppf(p)
