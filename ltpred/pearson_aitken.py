"""Pearson-Aitken selection formula -- deterministic liability estimation (PA-FGRS).

An analytical alternative to the Gibbs sampler, following Krebs et al. 2024
(*Am. J. Hum. Genet.*, "Genetic liability estimated from large-scale family
data ...", PA-FGRS). Instead of sampling the truncated multivariate normal, it
folds each observed relative in one at a time using the classical Pearson-Aitken
selection theorem: if a jointly-Gaussian vector's component ``i`` has its marginal
moved from ``N(m_i, v_i)`` to a selected mean/variance ``(m*, v*)``, every other
component ``j`` updates in closed form,

    mean_j  += (Sigma_ji / v_i) (m* - m_i)
    cov_jk  += (Sigma_ji Sigma_ik / v_i^2) (v* - v_i).

Processing the observed liabilities sequentially (each treated as a truncated
normal, so ``(m*, v*)`` are its truncated moments) and reading off the target
genetic-liability component gives ``E[l_g | family]`` with **no Monte-Carlo
error**. For a single truncation this is exact; for several it is the standard
sequential-selection approximation, and it is orders of magnitude faster than
Gibbs.

The PA-FGRS extension for **age-censored controls** (:func:`_tnorm_mixture`)
models an as-yet-unaffected relative as a mixture of a true control and a
not-yet-onset future case, weighted by how far their individual cumulative
incidence ``K_i`` lags the population lifetime prevalence ``K_pop``.
"""

from __future__ import annotations

import math

import numpy as np

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf

__all__ = ["pa_algorithm", "pa_estimate_batched", "tnorm_moments",
           "tnorm_mixture_conditional"]

_INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2*pi)


@_jit
def _norm_pdf(x):
    """Standard-normal density; ``0`` at ``+/-inf`` (``exp(-inf) -> 0``)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@_jit
def _tnorm_mean(mu, sd, lower, upper):
    """Mean of ``N(mu, sd^2)`` truncated to ``(lower, upper)``.

    Returns ``mu`` for an infinite interval and ``lower`` for a point mass
    (``lower == upper``). Infinite bounds are handled through ``_norm_cdf`` /
    ``_norm_pdf`` limits (no ``inf * 0`` NaNs)."""
    if lower == -math.inf and upper == math.inf:
        return mu
    if lower == upper:
        return lower
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    return mu - sd * (_norm_pdf(b) - _norm_pdf(a)) / (_norm_cdf(b) - _norm_cdf(a))


@_jit
def _tnorm_var(mu, sd, lower, upper):
    """Variance of ``N(mu, sd^2)`` truncated to ``(lower, upper)``.

    ``sd^2`` for an infinite interval, ``0`` for a point mass. The ``a*phi(a)`` /
    ``b*phi(b)`` terms are forced to their ``0`` limit at infinite bounds."""
    if lower == -math.inf and upper == math.inf:
        return sd * sd
    if lower == upper:
        return 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    pa = _norm_pdf(a)
    pb = _norm_pdf(b)
    z = _norm_cdf(b) - _norm_cdf(a)
    term_a = 0.0 if math.isinf(a) else a * pa
    term_b = 0.0 if math.isinf(b) else b * pb
    ratio = (pa - pb) / z
    return sd * sd * (1.0 + (term_a - term_b) / z - ratio * ratio)


@_jit
def _tnorm_mixture(mu, var, lower, upper, K_i, K_pop):
    """Selected moments of one observation, with the censored-control mixture.

    Returns ``(mean, var)`` after conditioning. With ``K_i``/``K_pop`` NaN this is
    just the truncated-normal moments on ``(lower, upper)`` (the plain PA / LT-FH
    behaviour). When both are given and the individual is not a fully observed case
    (finite ``upper`` or a point mass), the result is a two-component mixture: a
    genuine control on ``(lower, upper)`` with weight ``mixture_prob``, and a
    not-yet-onset future case on ``(upper, inf)`` with the complement -- the
    PA-FGRS age-censoring correction (supp. eqs. S3-S5)."""
    sd = math.sqrt(var)
    use_mix = (not math.isnan(K_pop)) and (not math.isnan(K_i)) and \
        (upper != math.inf or lower == upper)
    if use_mix:
        thr_pop = _norm_ppf(1.0 - K_pop)
        cdf_pop = _norm_cdf((thr_pop - mu) / sd)
        mixture_prob = cdf_pop / (cdf_pop + (1.0 - cdf_pop) * (K_pop - K_i) / K_pop)
    else:
        mixture_prob = 1.0

    m0 = _tnorm_mean(mu, sd, lower, upper)
    v0 = _tnorm_var(mu, sd, lower, upper)
    if upper == math.inf or lower == upper:  # observed case -> no upper component
        m1 = 0.0
        v1 = 0.0
    else:
        m1 = _tnorm_mean(mu, sd, upper, math.inf)
        v1 = _tnorm_var(mu, sd, upper, math.inf)

    new_mean = mixture_prob * m0 + (1.0 - mixture_prob) * m1
    new_var = mixture_prob * (m0 * m0 + v0) + (1.0 - mixture_prob) * (m1 * m1 + v1) \
        - new_mean * new_mean
    return new_mean, new_var


@_jit
def _pa_update(cov, mu, i, nm, nv):
    """Rank-1 Pearson-Aitken update after conditioning coordinate ``i``.

    Only the **active leading block** ``0..i-1`` is touched: once ``i`` has been
    folded in (last-to-first), coordinates ``> i`` are never read again, so the
    columns/rows above ``i`` need no update. This halves-to-thirds the work of a
    full-matrix rank-1 update and avoids snapshotting ``cov[:, i]`` (column ``i`` is
    not written here, so it can be read directly). Exactly equivalent to updating
    the whole matrix for the target's final ``(mean, var)``."""
    v_i = cov[i, i]
    dm = (nm - mu[i]) / v_i
    fac = (nv - v_i) / (v_i * v_i)
    for j in range(i):
        mu[j] += cov[j, i] * dm
    for j in range(i):
        cji_fac = cov[j, i] * fac
        for k in range(i):
            cov[j, k] += cji_fac * cov[k, i]


@_jit
def _pa_family(cov, lower, upper, K_i, K_pop):
    """One family's PA sweep **with** the censored-control mixture; target row 0.

    Mutates ``cov`` in place, folding observations ``d-1, ..., 1``. Returns
    ``(est, var)`` = the target's posterior mean and variance."""
    d = cov.shape[0]
    mu = np.zeros(d)
    for i in range(d - 1, 0, -1):
        nm, nv = _tnorm_mixture(mu[i], cov[i, i], lower[i], upper[i], K_i[i], K_pop[i])
        _pa_update(cov, mu, i, nm, nv)
    return mu[0], cov[0, 0]


@_jit
def _pa_family_nomix(cov, lower, upper):
    """PA sweep **without** the mixture -- plain truncated-normal moments.

    The default fast path: skips all ``K_i``/``K_pop`` handling (no NaN arrays, no
    per-coordinate mixture branch). Same active-block update as :func:`_pa_family`."""
    d = cov.shape[0]
    mu = np.zeros(d)
    for i in range(d - 1, 0, -1):
        sd_i = math.sqrt(cov[i, i])
        nm = _tnorm_mean(mu[i], sd_i, lower[i], upper[i])
        nv = _tnorm_var(mu[i], sd_i, lower[i], upper[i])
        _pa_update(cov, mu, i, nm, nv)
    return mu[0], cov[0, 0]


@_jit_parallel
def _pa_batched(cov, lowers, uppers, K_is, K_pops, est, var):
    """Mixture PA over many same-structure families in parallel."""
    F = lowers.shape[0]
    for f in prange(F):
        c = cov.copy()
        e, v = _pa_family(c, lowers[f], uppers[f], K_is[f], K_pops[f])
        est[f] = e
        var[f] = v


@_jit_parallel
def _pa_batched_nomix(cov, lowers, uppers, est, var):
    """No-mixture PA over many same-structure families in parallel (no K arrays)."""
    F = lowers.shape[0]
    for f in prange(F):
        c = cov.copy()
        e, v = _pa_family_nomix(c, lowers[f], uppers[f])
        est[f] = e
        var[f] = v


def tnorm_moments(mu=0.0, var=1.0, lower=-np.inf, upper=np.inf):
    """Public scalar helper: mean and variance of a truncated normal ``(mean, var)``."""
    sd = math.sqrt(var)
    return _tnorm_mean(mu, sd, lower, upper), _tnorm_var(mu, sd, lower, upper)


def tnorm_mixture_conditional(mu, var, lower, upper, K_i=np.nan, K_pop=np.nan):
    """Public scalar helper for the censored-control mixture; returns ``(mean, var)``."""
    K_i = np.nan if K_i is None else K_i
    K_pop = np.nan if K_pop is None else K_pop
    return _tnorm_mixture(mu, var, lower, upper, K_i, K_pop)


def pa_algorithm(covmat, lower, upper, target=0, K_i=None, K_pop=None):
    """Pearson-Aitken estimate of the target liability for a single family.

    ``covmat`` is the ``(d, d)`` liability covariance; ``lower``/``upper`` are the
    per-row truncation bounds (the target row should be ``(-inf, inf)``). ``target``
    is the row to estimate (0 = the genetic liability ``g`` in the usual ordering).
    ``K_i``/``K_pop`` (per row, ``nan`` where unused) switch on the censored-control
    mixture. Returns ``(est, var)`` -- the target's posterior mean and variance."""
    cov = np.array(covmat, dtype=np.float64, copy=True)
    d = cov.shape[0]
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    lo, hi = lower[order], upper[order]
    if K_i is None and K_pop is None:          # no-mixture fast path
        return _pa_family_nomix(cov, lo, hi)
    K_i = np.full(d, np.nan) if K_i is None else np.asarray(K_i, dtype=np.float64)
    K_pop = np.full(d, np.nan) if K_pop is None else np.asarray(K_pop, dtype=np.float64)
    return _pa_family(cov, lo, hi, K_i[order], K_pop[order])


def pa_estimate_batched(covmat, lowers, uppers, target=0, K_is=None, K_pops=None):
    """Vectorised :func:`pa_algorithm` over families sharing one covariance.

    ``lowers``/``uppers`` are ``(F, d)`` per-family bounds; ``covmat`` is shared.
    Reorders once so the target is row 0, then runs the parallel kernel. When no
    ``K_is``/``K_pops`` are given, dispatches to the no-mixture kernel, which never
    allocates the ``(F, d)`` mixture arrays. Returns ``(est, var)`` of length ``F``."""
    cov = np.array(covmat, dtype=np.float64, copy=True)
    d = cov.shape[0]
    lowers = np.ascontiguousarray(lowers, dtype=np.float64)
    uppers = np.ascontiguousarray(uppers, dtype=np.float64)
    F = lowers.shape[0]
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    lo, hi = np.ascontiguousarray(lowers[:, order]), np.ascontiguousarray(uppers[:, order])
    est = np.empty(F)
    var = np.empty(F)
    if K_is is None and K_pops is None:        # no-mixture fast path (no K arrays)
        _pa_batched_nomix(cov, lo, hi, est, var)
        return est, var
    K_is = np.full((F, d), np.nan) if K_is is None else np.ascontiguousarray(K_is, dtype=np.float64)
    K_pops = np.full((F, d), np.nan) if K_pops is None else np.ascontiguousarray(K_pops, dtype=np.float64)
    _pa_batched(cov, lo, hi, np.ascontiguousarray(K_is[:, order]),
                np.ascontiguousarray(K_pops[:, order]), est, var)
    return est, var
