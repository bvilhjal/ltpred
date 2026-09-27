"""Pearson-Aitken (PA) selection -- deterministic liability inference.

The analytical alternative to the Gibbs sampler. The supplied bounds and
relatives, not the engine, name the model (classic LT-FH, onset-pinned
LT-FH++, ADuLT, base PA-FGRS or an age-dependent PA-FGRS-style interval
variant). Reference: methods report, "Computing the truncated mean"
(Algorithms P and M) and "Reducing the inference problem"; Dybdahl Krebs et
al. 2024 (*Am. J. Hum. Genet.*). If coordinate ``i`` of a Gaussian vector is
selected from ``N(m_i, v_i)`` to moments ``(m*, v*)``, all others update in
closed form (`_pa_update`, applied to the leading block ``0..i-1`` only):

    m_j      <- m_j      + (Sigma_ji / v_i) (m* - m_i)
    Sigma_jk <- Sigma_jk + (Sigma_ji Sigma_ik / v_i^2) (v* - v_i)

**Algorithm P** (no ``K_i``/``K_pop``; `pa_algorithm`, `pa_estimate_batched`):

- P1. Permute the target to row 0; other rows keep the caller's order
  (`ltpred.estimate` sorts roles by name, so fold order is a property of
  the role set).
- P2. `_pa_reduced_nomix`: marginalise absent non-target rows ``(-inf, inf)``
  (take the principal sub-block), condition all pins jointly
  (`_condition_pins`: ``m_X = Sigma_XP Sigma_PP^-1 p``,
  ``V_X = Sigma_XX - Sigma_XP Sigma_PP^-1 Sigma_PX``) and centre the remaining
  bounds by ``m_X``. Families are grouped by the byte string of row states
  (0 absent, 1 interval, 2 pin), so ``V_X`` is formed once per observation
  mask. A singular pin block is reduced to independent pins after a support
  check; a zero conditional variance marks a deterministic row whose bound
  must contain its value. Incompatible observations raise ``ValueError``.
- P3. `_pa_family_nomix`: fold the remaining intervals last-to-first (row
  ``d-1`` down to 1, original relative order), replacing each marginal by its
  truncated-normal moments and applying the update.
- P4. Apply the target's own interval last: a no-op for an unbounded ``g``;
  ``out="full"`` conditions on the proband's own status.
- P5. Return the target's updated mean and variance ``(est, var)``.

Pins followed by at most one interval (the target's own included) give exact
posterior moments. With several intervals, each selected law is replaced by
the Gaussian with its two moments: a sequential two-moment approximation that
depends on fold order. Zero Monte-Carlo error is not zero approximation error.

**Algorithm M** (censored-control mixture, `_pa_family` via `_tnorm_mixture`,
PA-only): a control observed to its current age is a lifetime control on
``(lower, T_pop)`` or a not-yet-onset case on ``(T_pop, inf)``, split at the
lifetime threshold ``T_pop = -Phi^-1(K_pop)``. Its selected moments are the
two-component moments with weight
``pi = Phi_below / (Phi_below + (1 - Phi_below) (K_pop - K_i) / K_pop)``,
``Phi_below`` the current conditional mass below ``T_pop``; this replaces P3's
moments. ``(K_pop - K_i)/K_pop`` assumes onset timing independent of liability
among eventual cases. Age enters only through ``K_i``; the finite ``upper``
merely flags censoring. P2's reduction is skipped: pins, absent rows (no-op
updates) and intervals are all folded sequentially in row order, and a row
whose conditional variance has collapsed is skipped as known. P2's
compatibility checks still apply (`_check_pin_support`), so both paths accept
and reject the same exact observations.

Truncated moments never form ``1 - Phi``: `_std_tnorm_moments` reflects
positive intervals into the lower tail and differences log CDFs with
``expm1`` (`_log_norm_cdf` uses an asymptotic series below -37). Intervals
wholly beyond +-16 standard deviations, and finite ones narrower than
``1e-3``, use Gauss-Legendre quadrature in a local coordinate, so a small
variance is never recovered by cancelling order-``a**2`` terms.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import qr

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf
from .covariance import _PSD_CERTIFIED
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
    point mass. One call into `_std_tnorm_moments` gives both moments."""
    if lower == -math.inf and upper == math.inf:
        return mu, sd * sd
    if lower == upper:
        return lower, 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    mean, var = _std_tnorm_moments(a, b)
    return mu + sd * mean, sd * sd * var


@_jit
def _tnorm_mixture(mu, var, lower, upper, K_i, K_pop):
    """Selected moments of one observation, with the censored-control mixture.

    Returns ``(mean, var)`` after conditioning ``N(mu, var)``. With ``K_i``/``K_pop``
    NaN this is just the truncated-normal moments on ``(lower, upper)`` (the plain
    PA / LT-FH behaviour). When both are given and the individual is not a fully
    observed case (finite ``upper`` or a point mass), the result is Algorithm M,
    the PA-FGRS age-censoring mixture (Dybdahl Krebs et al. 2024, supp. eqs.
    S3-S5): a genuine control on ``(lower, thr_pop)`` with weight
    ``mixture_prob = c / (c + (1 - c) (K_pop - K_i) / K_pop)``, where
    ``c = Phi((thr_pop - mu) / sd)``, and a not-yet-onset future case on
    ``(thr_pop, inf)`` with the complement. A zero denominator (``c`` underflows
    and ``K_i == K_pop``) gives weight 1. The result is the two-component mean
    and variance.

    The split point is the *lifetime* threshold ``thr_pop = Phi^-1(1 - K_pop)`` --
    the quantity the model defines the two components by -- **not** the passed
    ``upper``. Age enters solely through the mixture weight via ``K_i``. The passed
    ``upper`` only flags a censored control (finite) versus an observed case
    (``+inf``); its exact value is irrelevant in mixture mode, so feeding either the
    lifetime bound ``thr_pop`` or an age-specific bound ``Phi^-1(1 - K_i)`` (as
    `ltpred.thresholds.pa_thresholds` emits for the no-mixture interval path)
    yields the same censored-control moments. Sourcing the split from ``upper``
    instead would double-correct an age-specific bound -- the censoring gets
    encoded twice."""
    sd = math.sqrt(var)
    use_mix = (not math.isnan(K_pop)) and (not math.isnan(K_i)) and \
        (upper != math.inf or lower == upper)
    if use_mix:
        split = -_norm_ppf(K_pop)               # lifetime threshold thr_pop
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

    Algorithm M: no P2 reduction. Rows whose conditional variance has
    collapsed below ``_COLLAPSED_VARIANCE`` times their prior variance are
    skipped as known; `_check_pin_support` has already rejected incompatible
    exact observations with P2's rules. Mutates
    ``cov`` in place, folding every row ``d-1, ..., 1`` (pins as point masses,
    absent rows as no-op updates) with `_tnorm_mixture`. Then
    applies the target's own interval to the updated ``N(mu[0], cov[0,0])``.
    Unbounded targets (``g``) are a no-op there; ``out="full"`` therefore
    conditions on the proband's own status, matching Gibbs. Returns
    ``(est, var)`` as sequential-moment approximations to the target's posterior
    mean and conditional variance."""
    d = cov.shape[0]
    mu = np.zeros(d)
    # A row whose conditional variance has collapsed (for instance a second pin
    # on a coordinate the earlier pins determine) is known: folding it would
    # divide by ~0 and it carries no further information. Its compatibility
    # with the bounds is checked before the kernel (`_check_pin_support`).
    floor = np.empty(d)
    for i in range(d):
        floor[i] = _COLLAPSED_VARIANCE * cov[i, i]
    for i in range(d - 1, 0, -1):
        if cov[i, i] <= floor[i]:
            continue
        nm, nv = _tnorm_mixture(mu[i], cov[i, i], lower[i], upper[i], K_i[i], K_pop[i])
        _pa_update(cov, mu, i, nm, nv)
    if cov[0, 0] <= floor[0]:
        return mu[0], 0.0
    return _tnorm_mixture(mu[0], cov[0, 0], lower[0], upper[0], K_i[0], K_pop[0])


@_jit
def _pa_family_nomix(cov, lower, upper):
    """PA sweep **without** the mixture -- Algorithm P steps P3-P5.

    Normally receives the P2-reduced, centred problem. Skips absent rows and
    all ``K_i``/``K_pop`` handling. A row with zero variance is deterministic:
    it is skipped if its bounds contain its mean, otherwise ``ValueError``.
    Same active-block update as `_pa_family`, including the final
    target-interval update."""
    d = cov.shape[0]
    mu = np.zeros(d)
    for i in range(d - 1, 0, -1):
        if lower[i] == -math.inf and upper[i] == math.inf:
            continue
        if cov[i, i] <= 0.0:
            if lower[i] <= mu[i] <= upper[i]:
                continue
            raise ValueError("observation excludes a deterministic liability")
        nm, nv = _tnorm_moments_loc(mu[i], math.sqrt(cov[i, i]), lower[i], upper[i])
        _pa_update(cov, mu, i, nm, nv)
    if cov[0, 0] <= 0.0:
        if lower[0] <= mu[0] <= upper[0]:
            return mu[0], 0.0
        raise ValueError("observation excludes a deterministic liability")
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


#: Relative conditional variance below which the mixture kernel treats a row as
#: known (its standard deviation is below 1e-6 of the prior one).
_COLLAPSED_VARIANCE = 1e-12


def _check_pin_support(cov, lowers, uppers, K_is):
    """Apply Algorithm P's step-P2 compatibility checks on the mixture path.

    Algorithm M folds pins sequentially instead of conditioning on them
    jointly, so it would otherwise accept pins that contradict each other or
    the bounds of a row they determine. Group families by pin mask, condition
    on the pins with `_condition_pins` (which rejects pins outside the
    covariance support) and require every row the pins determine to lie within
    its bounds. A censored-control mixture row may exceed its ``upper`` (a
    future case), so only its ``lower`` bound is checked."""
    pins = lowers == uppers
    if not np.any(pins):
        return
    mixture = ~np.isnan(np.asarray(K_is, dtype=np.float64)) & (uppers != np.inf)
    effective_upper = np.where(mixture, np.inf, uppers)
    keys = pins.view(np.dtype((np.void, pins.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    for k, row in enumerate(first):
        pinned = np.flatnonzero(pins[row])
        if not len(pinned):
            continue
        rows = np.flatnonzero(inverse == k)
        retained = np.flatnonzero(~pins[row])
        means, conditional = _condition_pins(
            cov, retained, pinned, lowers[np.ix_(rows, pinned)])
        deterministic = np.diag(conditional) == 0.0
        if not np.any(deterministic):
            continue
        values = means[:, deterministic]
        columns = retained[deterministic]
        tolerance = 128 * np.finfo(float).eps * len(cov) * np.maximum(1.0, np.abs(values))
        if (np.any(values < lowers[np.ix_(rows, columns)] - tolerance)
                or np.any(values > effective_upper[np.ix_(rows, columns)] + tolerance)):
            raise ValueError("observation excludes a deterministic liability after pin conditioning")


def _condition_pins(cov, retained, pinned, values):
    """Gaussian reduction shared by a batch with one pin/observation mask.

    ``values`` is ``(F, n_pins)``. Computes ``m_X = Sigma_XP Sigma_PP^-1 p`` per
    family and the shared ``V_X = Sigma_XX - Sigma_XP Sigma_PP^-1 Sigma_PX`` for
    the ``retained`` rows, by solves on the pin correlation block.
    Cholesky preserves small positive eigenvalues. A numerically singular
    pin correlation block uses its spectral support at a dimension-scaled
    machine-precision tolerance; incompatible pins are rejected.
    For non-PD input only, augmented correlation blocks use the same support
    tolerance to remove roundoff remnants. Positive residuals of a PD input
    are retained even below that tolerance.
    Returns the per-family means and one shared conditional covariance.
    """
    scale = np.sqrt(np.diag(cov)[pinned])
    block = cov[np.ix_(pinned, pinned)] / np.outer(scale, scale)
    np.fill_diagonal(block, 1.0)
    standardized = values / scale
    if len(pinned) > 1:
        # A single pin's block is exactly [[1.0]]: its Cholesky factor is 1 and
        # the weights below never read it, so factor only multi-pin batches.
        try:
            chol = np.linalg.cholesky(block)
        except np.linalg.LinAlgError:
            eigenvalues, vectors = np.linalg.eigh(block)
            tolerance = 64 * np.finfo(float).eps * len(pinned) * max(1.0, np.max(eigenvalues))
            active = eigenvalues > tolerance
            basis = vectors[:, active]
            projection = standardized @ basis @ basis.T
            support_tolerance = 128 * np.finfo(float).eps * len(pinned) * np.maximum(
                1.0, np.max(np.abs(standardized), axis=1))
            if np.any(np.max(np.abs(standardized - projection), axis=1) > support_tolerance):
                raise ValueError("pinned observations are incompatible with covariance support")
            # Condition on independent original coordinates. Unlike subtracting
            # a spectral pseudoinverse product, identical pins now cancel exactly.
            _, _, pivot = qr(basis.T, mode="economic", pivoting=True)
            chosen = pivot[:np.count_nonzero(active)]
            pinned, scale = pinned[chosen], scale[chosen]
            standardized = standardized[:, chosen]
            block = block[np.ix_(chosen, chosen)]
            chol = np.linalg.cholesky(block)
    if len(pinned) == 1:
        # np.ix_ on one row costs more than the row fancy-index it builds.
        cross = cov[pinned[0], retained][None, :] / scale[0]
    else:
        cross = cov[np.ix_(pinned, retained)] / scale[:, None]
    weights = (cross.T if len(pinned) == 1 else
               np.linalg.solve(chol.T, np.linalg.solve(chol, cross)).T)
    means = standardized @ weights.T
    conditional = cov[np.ix_(retained, retained)] - weights @ cross
    conditional = (conditional + conditional.T) * 0.5
    diag = np.diag(conditional).copy()
    tolerance = 128 * np.finfo(float).eps * max(1.0, np.max(np.diag(cov))) * len(cov)
    if np.any(diag < -tolerance):
        raise ValueError("pin conditioning produced a negative conditional variance")
    full_pd = None
    for j in np.flatnonzero(diag <= tolerance):
        # A tiny Schur remainder alone does not imply determinism. Check the
        # original augmented correlation block: a positive Cholesky residual
        # retains even a genuinely tiny conditional variance. Only a singular
        # augmented block establishes a numerical support constraint.
        original_var = cov[retained[j], retained[j]]
        augmented = np.empty((len(pinned) + 1, len(pinned) + 1))
        augmented[:-1, :-1] = block
        augmented[-1, :-1] = augmented[:-1, -1] = cross[:, j] / np.sqrt(original_var)
        augmented[-1, -1] = 1.0
        try:
            joint_chol = np.linalg.cholesky(augmented)
        except np.linalg.LinAlgError:
            deterministic = True
        else:
            if full_pd is None:
                full_scale = np.sqrt(np.diag(cov))
                full_correlation = cov / np.outer(full_scale, full_scale)
                np.fill_diagonal(full_correlation, 1.0)
                try:
                    np.linalg.cholesky(full_correlation)
                    full_pd = True
                except np.linalg.LinAlgError:
                    full_pd = False
            deterministic = False
            if not full_pd:
                # A rank-one covariance formed in floating point can leave a
                # spurious positive Cholesky pivot of order machine epsilon.
                eigenvalues = np.linalg.eigvalsh(augmented)
                support_tolerance = (64 * np.finfo(float).eps * len(augmented)
                                     * max(1.0, eigenvalues[-1]))
                deterministic = eigenvalues[0] <= support_tolerance
        if deterministic:
            conditional[j, :] = conditional[:, j] = 0.0
        else:
            conditional[j, j] = original_var * joint_chol[-1, -1] ** 2
    return means, conditional


def _pa_reduced_nomix(cov, lowers, uppers):
    """Algorithm P step P2, then P3-P5 on the reduced problem; target row 0.

    Marginalizes absent non-target rows and conditions pins jointly. Families
    share reductions by their observation mask (row states 0 absent, 1
    interval, 2 pin, compared as byte strings), so different onset values
    reuse the same Gaussian algebra. Remaining interval order is unchanged.
    A pinned or deterministic target returns its value with zero variance.
    Batches with no pin and no absent non-target row go straight to the
    kernel. Mixture calls keep the sequential semantics and do not enter here.
    """
    F, d = lowers.shape
    est, var = np.empty(F), np.empty(F)
    if F == 0:
        return est, var
    if F == 1:
        # Register-style single-family call: the shared routing below would
        # allocate five (1, d) temporaries and a uint8 state row before
        # reaching the same kernel. Only the routing decision is inlined; a
        # pin or an absent relative falls through to the general machinery.
        lo0, hi0 = lowers[0], uppers[0]
        if not np.any(lo0 == hi0) and not np.any(
                (lo0[1:] == -np.inf) & (hi0[1:] == np.inf)):
            est[0], var[0] = _pa_family_nomix(cov.copy(), lo0, hi0)
            return est, var
    pins = lowers == uppers
    absent = (lowers == -np.inf) & (uppers == np.inf)
    if not np.any(pins) and not np.any(absent[:, 1:]):
        if F == 1:
            est[0], var[0] = _pa_family_nomix(cov.copy(), lowers[0], uppers[0])
        else:
            _pa_batched_nomix(cov, lowers, uppers, est, var)
        return est, var
    state = (~absent).astype(np.uint8) + pins.astype(np.uint8)
    if np.all(state == state[0]):
        groups = [(state[0], np.arange(F))]
    else:
        # A state is a byte string over {0, 1, 2}. Comparing the whole string
        # avoids NumPy's per-column structured comparisons for axis=0. Keep
        # every byte: long families must not collide in a fixed-width key.
        keys = state.view(np.dtype((np.void, d))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        masks = state[first]
        order = np.argsort(inverse, kind="stable")
        boundaries = np.r_[0, np.cumsum(np.bincount(inverse))]
        groups = ((mask, order[boundaries[k]:boundaries[k + 1]])
                  for k, mask in enumerate(masks))
    for mask, rows in groups:
        pinned = np.flatnonzero(mask == 2)
        keep = mask == 1
        keep[0] = mask[0] != 2
        retained = np.flatnonzero(keep)
        # Own just two float64 work arrays. Center them in place below; even
        # float32 bounds used float64 centering in the original reduction.
        lo = np.asarray(lowers[np.ix_(rows, retained)], dtype=np.float64, order="C")
        hi = np.asarray(uppers[np.ix_(rows, retained)], dtype=np.float64, order="C")
        if len(pinned):
            means, conditional = _condition_pins(
                cov, retained, pinned, lowers[np.ix_(rows, pinned)])
        else:
            means = None
            conditional = cov[np.ix_(retained, retained)]
        deterministic = np.diag(conditional) == 0.0
        if np.any(deterministic):
            values = means[:, deterministic]
            tolerance = 128 * np.finfo(float).eps * d * np.maximum(1.0, np.abs(values))
            if (np.any(values < lo[:, deterministic] - tolerance)
                    or np.any(values > hi[:, deterministic] + tolerance)):
                raise ValueError("observation excludes a deterministic liability after pin conditioning")
        if mask[0] == 2:
            est[rows], var[rows] = lowers[rows, 0], 0.0
            continue
        if deterministic[0]:
            est[rows], var[rows] = means[:, 0], 0.0
            continue
        active = ~deterministic
        subcov = np.ascontiguousarray(conditional[np.ix_(active, active)])
        if means is not None:
            lo -= means
            hi -= means
        if np.any(deterministic):
            lo = np.ascontiguousarray(lo[:, active])
            hi = np.ascontiguousarray(hi[:, active])
        e, v = np.empty(len(rows)), np.empty(len(rows))
        if len(rows) == 1:
            e[0], v[0] = _pa_family_nomix(subcov, lo[0], hi[0])
        else:
            _pa_batched_nomix(subcov, lo, hi, e, v)
        est[rows], var[rows] = (e if means is None else means[:, 0] + e), v
    return est, var


def _validate_pa_covmat(covmat, _psd_certified=None):
    """Return a supported positive-semidefinite covariance as float64.

    The PA counterpart of the Gibbs gate (`ltpred.gibbs._validate_covmat`),
    with the same scale-relative symmetry tolerance. PA does not require strict
    positive-definiteness; pin conditioning checks singular support and the
    interval fold uses rank-1 updates. It still requires a genuine covariance and positive
    marginal variances for every coordinate that may be truncated.

    ``_psd_certified`` (the module-private ``_PSD_CERTIFIED`` sentinel) skips
    only the ``eigvalsh``: it is for callers whose array
    `ltpred.covariance.correct_positive_definite` has just returned, which
    guarantees a minimum eigenvalue above ``1e-8`` — strictly stronger than
    this gate's ``-1e-10``-relative tolerance on the very same array."""
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
    if _psd_certified is _PSD_CERTIFIED:
        return cov
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
    usual ``g``) is a no-op after the relative fold; a target interval
    (``out="full"``) is folded last. Without a mixture, all pins are conditioned
    jointly before intervals and unobserved non-target rows are marginalized. ``target``
    is the row to estimate (0 = the genetic liability ``g`` in the usual ordering).
    ``K_i``/``K_pop`` (per row, ``nan`` where unused) switch on the censored-control
    mixture (Algorithm M, which skips the pin reduction). Returns ``(est, var)`` -- sequential-moment approximations to the
    target's posterior mean and conditional variance. Without a mixture the
    result is exact after pins when at most one interval remains."""
    cov = _validate_pa_covmat(covmat)
    d = cov.shape[0]
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # `lower[order]` below is a fancy index by a length-d permutation: a longer
    # bounds vector would have its tail silently dropped (a confident wrong
    # estimate) and a shorter one would raise a bare IndexError.
    if lower.shape != (d,) or upper.shape != (d,):
        raise ValueError(
            f"lower and upper must each have shape ({d},) to match the "
            f"covariance; got {lower.shape} and {upper.shape}")
    if not 0 <= target < d:
        raise ValueError(f"target must be in [0, {d}); got {target}")
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    lo, hi = lower[order], upper[order]
    validate_bounds(lo, hi, context="pa_algorithm bounds")
    if K_i is None and K_pop is None:
        est, var = _pa_reduced_nomix(cov, lo[None, :], hi[None, :])
        return float(est[0]), float(var[0])
    K_i, K_pop = validate_mixture_inputs(
        K_i, K_pop, expected_shape=(d,), lower=lower, upper=upper,
        context="pa_algorithm mixture inputs")
    K_i = as_bounds(K_i)
    K_pop = as_bounds(K_pop)
    _check_pin_support(cov, lo[None, :], hi[None, :], K_i[order][None, :])
    return _pa_family(cov, lo, hi, K_i[order], K_pop[order])


def pa_estimate_batched(covmat: ArrayLike, lowers: ArrayLike, uppers: ArrayLike,
                        target: int = 0, K_is: ArrayLike | None = None,
                        K_pops: ArrayLike | None = None, *,
                        _psd_certified: object = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `pa_algorithm` over families sharing one covariance.

    ``lowers``/``uppers`` are ``(F, d)`` per-family bounds; ``covmat`` is shared.
    Reorders once so the target is row 0. Without ``K_is``/``K_pops`` it runs
    Algorithm P, sharing the pin reduction by observation mask and never
    allocating the ``(F, d)`` mixture arrays; otherwise the parallel Algorithm M
    kernel. Returns the PA sequential-moment
    approximations ``(est, var)`` of length ``F``."""
    cov = _validate_pa_covmat(covmat, _psd_certified=_psd_certified)
    d = cov.shape[0]
    lowers = as_bounds(lowers)             # keeps float32 to halve memory
    uppers = as_bounds(uppers)
    F = lowers.shape[0]
    # Same fancy-index hazard as pa_algorithm, one dimension up: extra columns
    # are dropped without complaint, short ones raise a bare IndexError.
    if lowers.ndim != 2 or lowers.shape[1] != d:
        raise ValueError(
            f"lowers must have shape (F, {d}) to match the covariance; got "
            f"{lowers.shape}")
    if lowers.shape != uppers.shape:
        raise ValueError(
            f"lowers and uppers must have the same shape; got {lowers.shape} "
            f"and {uppers.shape}")
    if not 0 <= target < d:
        raise ValueError(f"target must be in [0, {d}); got {target}")
    order = np.concatenate(([target], np.delete(np.arange(d), target)))
    cov = np.ascontiguousarray(cov[np.ix_(order, order)])
    # as_bounds already made contiguous inputs. The usual target is zero;
    # an identity permutation would duplicate both full cohort arrays.
    lo = lowers if target == 0 else np.ascontiguousarray(lowers[:, order])
    hi = uppers if target == 0 else np.ascontiguousarray(uppers[:, order])
    validate_bounds(lo, hi, context="pa_estimate_batched bounds")
    if K_is is None and K_pops is None:        # no-mixture fast path (no K arrays)
        return _pa_reduced_nomix(cov, lo, hi)
    est = np.empty(F)
    var = np.empty(F)
    K_is, K_pops = validate_mixture_inputs(
        K_is, K_pops, expected_shape=(F, d), lower=lowers, upper=uppers,
        context="batched PA mixture inputs")
    K_is = as_bounds(K_is)
    K_pops = as_bounds(K_pops)
    ki = K_is if target == 0 else np.ascontiguousarray(K_is[:, order])
    kp = K_pops if target == 0 else np.ascontiguousarray(K_pops[:, order])
    _check_pin_support(cov, lo, hi, ki)
    _pa_batched(cov, lo, hi, ki, kp, est, var)
    return est, var
