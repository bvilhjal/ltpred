"""Pearson-Aitken selection formula -- deterministic liability inference.

An analytical alternative to the Gibbs sampler. Pearson-Aitken is the inference
engine: the supplied liability bounds and inclusion of relatives determine whether
the fitted model is classic LT-FH, onset-pinned LT-FH++, family-free ADuLT, or
base PA-FGRS or an age-dependent PA-FGRS-style interval variant. Following
Dybdahl Krebs et al. 2024
(*Am. J. Hum. Genet.*, "Genetic liability estimated from large-scale family
data ..."), it
folds each observed relative in one at a time using the classical Pearson-Aitken
selection theorem: if a jointly-Gaussian vector's component ``i`` has its marginal
moved from ``N(m_i, v_i)`` to a selected mean/variance ``(m*, v*)``, every other
component ``j`` updates in closed form,

    mean_j  += (Sigma_ji / v_i) (m* - m_i)
    cov_jk  += (Sigma_ji Sigma_ik / v_i^2) (v* - v_i).

Processing the observed liabilities sequentially (each treated as a truncated
normal, so ``(m*, v*)`` are its truncated moments) and reading off the target
genetic-liability component gives a deterministic sequential-moment approximation
to ``E[l_g | family]`` and its conditional variance, with **no Monte-Carlo
error**. Zero Monte-Carlo error does not mean zero approximation error. For a
single truncation the moments are exact; for several they are the standard
sequential-selection approximation, which is orders of magnitude faster than Gibbs.

The PA-FGRS component for **age-censored controls** (:func:`_tnorm_mixture`)
models an as-yet-unaffected relative as a mixture of a true control and a
not-yet-onset future case, weighted by how far their individual cumulative
incidence ``K_i`` lags the population lifetime prevalence ``K_pop``. Base PA-FGRS
uses a lifetime-threshold interval for observed cases; the age-specific case
intervals emitted by :func:`ltpred.thresholds.pa_thresholds` are a separate,
age-dependent PA-FGRS-style variant.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf
from .gibbs import as_bounds
from ._validation import validate_bounds, validate_mixture_inputs

__all__ = ["pa_algorithm", "pa_estimate_batched"]

_SQRT_2 = 1.4142135623730951
_LOG_SQRT_2PI = 0.9189385332046727
_FAR_TAIL_START = 16.0
_FAR_TAIL_QUAD_LIMIT = 40.0
_GL8_NODES = np.array([
    -0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975363,
])
_GL8_WEIGHTS = np.array([
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
])


@_jit
def _log_norm_cdf(x):
    """Log standard-normal CDF without rounding a tail probability to zero."""
    if x == -math.inf:
        return -math.inf
    if x < -37.0:
        # Phi(x) = phi(x) / -x * (1 - x^-2 + 3x^-4 - ...).  ``erfc``
        # underflows beyond about -38, while this expansion remains finite.
        inv_x2 = 1.0 / (x * x)
        term = 1.0
        series = 1.0
        for k in range(1, 9):
            term *= -(2.0 * k - 1.0) * inv_x2
            series += term
        return -0.5 * x * x - _LOG_SQRT_2PI - math.log(-x) + math.log(series)
    return math.log(0.5 * math.erfc(-x / _SQRT_2))


@_jit
def _narrow_std_tnorm_moments(a, b):
    """Moments on a narrow finite interval, evaluated on ``[-1, 1]``.

    Centering and rescaling make the variance ``half_width**2 * Var(y)``;
    unlike the closed-form tail expression, this never subtracts quantities of
    order ``a**2`` to recover a result of order ``(b-a)**2``. Eight-point
    Gauss-Legendre integration is ample for the smooth density over this path's
    maximum standardized width of ``1e-3``.
    """
    half_width = 0.5 * (b - a)
    center = a + half_width

    # Scale weights by their maximum log-density on [-1, 1]. This changes no
    # moments and prevents overflow for intervals far into either tail.
    mode_y = -center / half_width
    mode_y = min(1.0, max(-1.0, mode_y))
    peak = -center * half_width * mode_y - 0.5 * half_width * half_width * mode_y * mode_y

    mass = 0.0
    first = 0.0
    second = 0.0
    for i in range(8):
        y = _GL8_NODES[i]
        log_density = -center * half_width * y - 0.5 * half_width * half_width * y * y
        weight = _GL8_WEIGHTS[i] * math.exp(log_density - peak)
        mass += weight
        first += weight * y
        second += weight * y * y

    mean_y = first / mass
    var_y = second / mass - mean_y * mean_y
    return center + half_width * mean_y, half_width * half_width * var_y


@_jit
def _far_right_std_tnorm_moments(a, b):
    """Moments on ``(a, b)`` when ``a`` is far into the right tail.

    With ``y = a * (x - a)``, the unnormalised density is proportional to
    ``exp(-y - y**2 / (2*a**2))``.  Its mass is concentrated at order-one
    ``y`` even when ``a`` is enormous, so recovering ``Var(y)`` does not
    subtract quantities of order ``a**2``.  Composite eight-point
    Gauss-Legendre integration over unit-width panels is effectively exact at
    double precision; mass beyond 40 is below machine precision.
    """
    if b == math.inf:
        limit = _FAR_TAIL_QUAD_LIMIT
    else:
        limit = a * (b - a)
        if limit > _FAR_TAIL_QUAD_LIMIT:
            limit = _FAR_TAIL_QUAD_LIMIT

    n_panels = max(1, int(math.ceil(limit)))
    panel_width = limit / n_panels
    mass = 0.0
    first = 0.0
    for panel in range(n_panels):
        left = panel * panel_width
        half_width = 0.5 * panel_width
        center = left + half_width
        for i in range(8):
            y = center + half_width * _GL8_NODES[i]
            density = math.exp(-y - 0.5 * (y / a) * (y / a))
            weight = half_width * _GL8_WEIGHTS[i] * density
            mass += weight
            first += weight * y

    mean_y = first / mass
    centered_second = 0.0
    for panel in range(n_panels):
        left = panel * panel_width
        half_width = 0.5 * panel_width
        center = left + half_width
        for i in range(8):
            y = center + half_width * _GL8_NODES[i]
            density = math.exp(-y - 0.5 * (y / a) * (y / a))
            weight = half_width * _GL8_WEIGHTS[i] * density
            centered_second += weight * (y - mean_y) * (y - mean_y)

    var_y = centered_second / mass
    return a + mean_y / a, var_y / a / a


@_jit
def _std_tnorm_moments(a, b):
    """Stable moments of ``N(0, 1)`` truncated to ``(a, b)``.

    Positive intervals are reflected into the lower tail, where ``Phi`` retains
    the small probability that ``1 - Phi`` would round away. Interval mass is
    then evaluated as a log-space difference using ``expm1``. This covers both
    finite tail intervals and one-sided truncation with the same calculation.
    """
    # The usual closed form obtains the variance by subtracting terms of order
    # ``a**2``.  Beyond the moderate tail that destroys all useful digits even
    # though the log interval mass remains accurate.  Reflect left tails and
    # evaluate both cases in a local, order-one coordinate instead.
    if a >= _FAR_TAIL_START:
        return _far_right_std_tnorm_moments(a, b)
    if b <= -_FAR_TAIL_START:
        mean, var = _far_right_std_tnorm_moments(-b, -a)
        return -mean, var

    if a != -math.inf and b != math.inf and b - a <= 1e-3:
        return _narrow_std_tnorm_moments(a, b)

    sign = 1.0
    if a > 0.0:
        a, b = -b, -a
        sign = -1.0

    if b <= 0.0:
        log_cdf_b = _log_norm_cdf(b)
        if a == -math.inf:
            log_z = log_cdf_b
        else:
            log_cdf_a = _log_norm_cdf(a)
            log_z = log_cdf_b + math.log(-math.expm1(log_cdf_a - log_cdf_b))
    else:
        # An interval crossing zero has no tail cancellation.
        log_z = math.log(_norm_cdf(b) - _norm_cdf(a))

    ratio_a = 0.0 if a == -math.inf else \
        math.exp(-0.5 * a * a - _LOG_SQRT_2PI - log_z)
    ratio_b = 0.0 if b == math.inf else \
        math.exp(-0.5 * b * b - _LOG_SQRT_2PI - log_z)
    mean = ratio_a - ratio_b
    term_a = 0.0 if a == -math.inf else a * ratio_a
    term_b = 0.0 if b == math.inf else b * ratio_b
    var = 1.0 + term_a - term_b - mean * mean
    # Roundoff can only make this infinitesimally negative for a very narrow
    # interval; a truncated-normal variance is never negative.
    return sign * mean, max(0.0, var)


@_jit
def _tnorm_moments_loc(mu, sd, lower, upper):
    """Mean and variance of ``N(mu, sd^2)`` truncated to ``(lower, upper)``.

    Returns ``(mu, sd^2)`` for an infinite interval and ``(lower, 0)`` for a
    point mass. One call into :func:`_std_tnorm_moments` — the sweep used to
    evaluate the mean and variance kernels separately on the same interval."""
    if lower == -math.inf and upper == math.inf:
        return mu, sd * sd
    if lower == upper:
        return lower, 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    mean, var = _std_tnorm_moments(a, b)
    return mu + sd * mean, sd * sd * var


@_jit
def _tnorm_mean(mu, sd, lower, upper):
    """Mean of ``N(mu, sd^2)`` truncated to ``(lower, upper)``."""
    mean, _ = _tnorm_moments_loc(mu, sd, lower, upper)
    return mean


@_jit
def _tnorm_var(mu, sd, lower, upper):
    """Variance of ``N(mu, sd^2)`` truncated to ``(lower, upper)``."""
    _, var = _tnorm_moments_loc(mu, sd, lower, upper)
    return var


@_jit
def _tnorm_mixture(mu, var, lower, upper, K_i, K_pop):
    """Selected moments of one observation, with the censored-control mixture.

    Returns ``(mean, var)`` after conditioning. With ``K_i``/``K_pop`` NaN this is
    just the truncated-normal moments on ``(lower, upper)`` (the plain PA / LT-FH
    behaviour). When both are given and the individual is not a fully observed case
    (finite ``upper`` or a point mass), the result is the PA-FGRS age-censoring
    mixture (Dybdahl Krebs et al. 2024, supp. eqs. S3-S5): a genuine control on
    ``(lower, thr_pop)`` with weight ``mixture_prob``, and a not-yet-onset future
    case on ``(thr_pop, inf)`` with the complement.

    The split point is the *lifetime* threshold ``thr_pop = Phi^-1(1 - K_pop)`` --
    the quantity the model defines the two components by -- **not** the passed
    ``upper``. Age enters solely through the mixture weight via ``K_i``. The passed
    ``upper`` only flags a censored control (finite) versus an observed case
    (``+inf``); its exact value is irrelevant in mixture mode, so feeding either the
    lifetime bound ``thr_pop`` or an age-specific bound ``Phi^-1(1 - K_i)`` (as
    :func:`ltpred.thresholds.pa_thresholds` emits for the no-mixture interval path)
    yields the same censored-control moments. Sourcing the split from ``upper``
    instead would double-correct an age-specific bound -- the censoring gets
    encoded twice."""
    sd = math.sqrt(var)
    use_mix = (not math.isnan(K_pop)) and (not math.isnan(K_i)) and \
        (upper != math.inf or lower == upper)
    if use_mix:
        split = _norm_ppf(1.0 - K_pop)          # lifetime threshold thr_pop
        cdf_pop = _norm_cdf((split - mu) / sd)
        # denom = P(control) + P(eventual case not yet onset); both terms >= 0, so
        # denom == 0 requires cdf_pop underflowing to 0 (extreme conditional mean)
        # *and* K_i == K_pop (a legal input: no future cases remain). That 0/0 has a
        # defensible limit -- with no future cases a censored control is certainly a
        # genuine lifetime control -- so pin mixture_prob to 1.0 instead of NaN.
        denom = cdf_pop + (1.0 - cdf_pop) * (K_pop - K_i) / K_pop
        mixture_prob = cdf_pop / denom if denom > 0.0 else 1.0
    else:
        split = upper                            # plain truncated normal on (lower, upper)
        mixture_prob = 1.0

    m0, v0 = _tnorm_moments_loc(mu, sd, lower, split)
    # ``split == inf``: plain-mode observed case (split = upper = +inf).
    # ``lower == split``: the lower bound coincides with the split point
    # (lifetime threshold for mixture mode, upper bound for plain mode) —
    # the lower component has zero width, so nothing to mix.
    if split == math.inf or lower == split:
        m1 = 0.0
        v1 = 0.0
    else:
        m1, v1 = _tnorm_moments_loc(mu, sd, split, math.inf)

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

    Mutates ``cov`` in place, folding observations ``d-1, ..., 1``. Then
    applies the target's own interval to the updated ``N(mu[0], cov[0,0])``.
    Unbounded targets (``g``) are a no-op there; ``out="full"`` therefore
    conditions on the proband's own status, matching Gibbs. Returns
    ``(est, var)`` as sequential-moment approximations to the target's posterior
    mean and conditional variance."""
    d = cov.shape[0]
    mu = np.zeros(d)
    for i in range(d - 1, 0, -1):
        nm, nv = _tnorm_mixture(mu[i], cov[i, i], lower[i], upper[i], K_i[i], K_pop[i])
        _pa_update(cov, mu, i, nm, nv)
    return _tnorm_mixture(mu[0], cov[0, 0], lower[0], upper[0], K_i[0], K_pop[0])


@_jit
def _pa_family_nomix(cov, lower, upper):
    """PA sweep **without** the mixture -- plain truncated-normal moments.

    The default fast path: skips all ``K_i``/``K_pop`` handling (no NaN arrays, no
    per-coordinate mixture branch). Same active-block update as :func:`_pa_family`,
    including the final target-interval update."""
    d = cov.shape[0]
    mu = np.zeros(d)
    for i in range(d - 1, 0, -1):
        nm, nv = _tnorm_moments_loc(mu[i], math.sqrt(cov[i, i]), lower[i], upper[i])
        _pa_update(cov, mu, i, nm, nv)
    return _tnorm_moments_loc(mu[0], math.sqrt(cov[0, 0]), lower[0], upper[0])


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


def _validate_pa_covmat(covmat):
    """Return a supported positive-semidefinite covariance as float64.

    The PA counterpart of the Gibbs gate (:func:`ltpred.gibbs._validate_covmat`),
    with the same scale-relative symmetry tolerance.  PA does not require strict
    positive-definiteness, because its sequential rank-1 updates never invert the
    supplied matrix, but it still requires a genuine covariance and positive
    marginal variances for every coordinate that may be truncated."""
    cov = np.asarray(covmat, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covmat must be square")
    if cov.shape[0] == 0:
        raise ValueError("covmat must contain at least one coordinate")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covmat must contain only finite values")
    matrix_scale = float(np.max(np.abs(cov))) if cov.size else 1.0
    symmetry_tolerance = 1e-10 * matrix_scale
    asymmetry = float(np.max(np.abs(cov - cov.T))) if cov.size else 0.0
    if asymmetry > symmetry_tolerance:
        raise ValueError(
            "covmat must be symmetric (maximum asymmetry "
            f"{asymmetry:.3g}, relative tolerance {symmetry_tolerance:.3g})")
    diag = np.diag(cov)
    if np.any(diag <= 0.0):
        raise ValueError("covmat diagonal variances must be strictly positive")
    min_eigenvalue = float(np.linalg.eigvalsh(cov)[0])
    psd_tolerance = 1e-10 * max(matrix_scale, 1.0)
    if min_eigenvalue < -psd_tolerance:
        raise ValueError(
            "covmat must be positive-semidefinite (minimum eigenvalue "
            f"{min_eigenvalue:.3g}, tolerance {-psd_tolerance:.3g})")
    return cov


def pa_algorithm(covmat: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                 target: int = 0, K_i: ArrayLike | None = None,
                 K_pop: ArrayLike | None = None) -> tuple[float, float]:
    """Pearson-Aitken estimate of the target liability for a single family.

    ``covmat`` is the ``(d, d)`` liability covariance; ``lower``/``upper`` are the
    per-row truncation bounds. An unbounded target row (``(-inf, inf)``, the
    usual ``g``) is a no-op after the relative fold; a bounded target
    (``out="full"``) is conditioned on last. ``target``
    is the row to estimate (0 = the genetic liability ``g`` in the usual ordering).
    ``K_i``/``K_pop`` (per row, ``nan`` where unused) switch on the censored-control
    mixture. Returns ``(est, var)`` -- sequential-moment approximations to the
    target's posterior mean and conditional variance (exact for one truncation)."""
    cov = _validate_pa_covmat(covmat)
    d = cov.shape[0]
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    lo, hi = lower[order], upper[order]
    validate_bounds(lo, hi, context="pa_algorithm bounds")
    if K_i is None and K_pop is None:          # no-mixture fast path
        return _pa_family_nomix(cov, lo, hi)
    K_i, K_pop = validate_mixture_inputs(
        K_i, K_pop, expected_shape=(d,), lower=lower, upper=upper,
        context="pa_algorithm mixture inputs")
    K_i = as_bounds(K_i)
    K_pop = as_bounds(K_pop)
    return _pa_family(cov, lo, hi, K_i[order], K_pop[order])


def pa_estimate_batched(covmat: ArrayLike, lowers: ArrayLike, uppers: ArrayLike,
                        target: int = 0, K_is: ArrayLike | None = None,
                        K_pops: ArrayLike | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`pa_algorithm` over families sharing one covariance.

    ``lowers``/``uppers`` are ``(F, d)`` per-family bounds; ``covmat`` is shared.
    Reorders once so the target is row 0, then runs the parallel kernel. When no
    ``K_is``/``K_pops`` are given, dispatches to the no-mixture kernel, which never
    allocates the ``(F, d)`` mixture arrays. Returns the PA sequential-moment
    approximations ``(est, var)`` of length ``F``."""
    cov = _validate_pa_covmat(covmat)
    d = cov.shape[0]
    lowers = as_bounds(lowers)             # keeps float32 to halve memory
    uppers = as_bounds(uppers)
    F = lowers.shape[0]
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    lo, hi = np.ascontiguousarray(lowers[:, order]), np.ascontiguousarray(uppers[:, order])
    validate_bounds(lo, hi, context="pa_estimate_batched bounds")
    est = np.empty(F)
    var = np.empty(F)
    if K_is is None and K_pops is None:        # no-mixture fast path (no K arrays)
        _pa_batched_nomix(cov, lo, hi, est, var)
        return est, var
    K_is, K_pops = validate_mixture_inputs(
        K_is, K_pops, expected_shape=(F, d), lower=lowers, upper=uppers,
        context="batched PA mixture inputs")
    K_is = as_bounds(K_is)
    K_pops = as_bounds(K_pops)
    _pa_batched(cov, lo, hi, np.ascontiguousarray(K_is[:, order]),
                np.ascontiguousarray(K_pops[:, order]), est, var)
    return est, var
