"""Deterministic Pearson--Aitken truncated-Gaussian moments.

Conditioning coordinate ``i`` updates the remaining means by
``Sigma_ji (m* - m_i) / v_i`` and covariances by a rank-one update. The
no-mixture path marginalises absent rows, conditions pins jointly by
observation mask, then folds intervals last-to-first and the target last.
Pins plus at most one interval are exact; multiple intervals use a sequential
two-moment approximation and can depend on fold order.

The PA-only censoring mixture splits each eligible control at its lifetime
threshold, with weights from ``K_i``/``K_pop``. Stable truncated moments use
log-CDF differences and local Gauss--Legendre quadrature for narrow or extreme
intervals. See Algorithms P and M in ``docs/algorithm.md``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import qr

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf
from .covariance import _PSD_CERTIFIED
from .gibbs import _as_bounds
from ._validation import validate_bounds, validate_mixture_inputs

__all__ = ["pa_algorithm", "pa_estimate_batched"]

_LOG_SQRT_2PI = 0.9189385332046727
_MIN_NORMAL = float(np.finfo(np.float64).tiny)
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
    return math.log(_norm_cdf(x))


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

    # One pass suffices in the local coordinate: y, E[y] and Var(y) are all
    # O(1) (E[y^2]/Var(y) ~ 2 for the limiting exponential), so
    # E[y^2] - E[y]^2 loses no digits -- the a^2-scale subtraction this
    # transform exists to avoid would only reappear in x coordinates.
    n_panels = max(1, int(math.ceil(limit)))
    panel_width = limit / n_panels
    half_width = 0.5 * panel_width
    mass = 0.0
    first = 0.0
    second = 0.0
    for panel in range(n_panels):
        center = panel * panel_width + half_width
        for i in range(8):
            y = center + half_width * _GL8_NODES[i]
            density = math.exp(-y - 0.5 * (y / a) * (y / a))
            weight = half_width * _GL8_WEIGHTS[i] * density
            mass += weight
            first += weight * y
            second += weight * y * y

    mean_y = first / mass
    var_y = second / mass - mean_y * mean_y
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
def _narrow_mixture_component(mu, sd, lower, width):
    """Log mass and moments of a short control interval, in physical units.

    Integrate on [0, 1] using its supplied width, which survives even when
    subtracting a remote mean makes the standardised endpoints coincide.
    Called only when the log density varies by at most about one unit.
    """
    a, step = (lower - mu) / sd, width / sd
    mass, first, second = 0.0, 0.0, 0.0
    for i in range(8):
        y = 0.5 * (1.0 + _GL8_NODES[i])
        weight = _GL8_WEIGHTS[i] * math.exp(-a * step * y - 0.5 * (step * y)**2)
        mass += weight
        first += weight * y
        second += weight * y * y
    mean_y = first / mass
    log_mass = math.log(step) - 0.5 * a*a - _LOG_SQRT_2PI + math.log(0.5 * mass)
    return log_mass, lower + width * mean_y, width*width * (second / mass - mean_y**2)


@_jit
def _tnorm_mixture(mu, var, lower, upper, K_i, K_pop):
    """Return truncated moments, optionally mixing censored controls.

    With finite ``K_i``/``K_pop``, an eligible control is split at the
    lifetime threshold ``-Phi^-1(K_pop)`` into a genuine control and a future
    case. Component weights use incidence remaining after ``K_i``; the passed
    finite ``upper`` only flags censoring and is not applied a second time.
    NaN pairs use ordinary truncated-normal moments.
    """
    sd = math.sqrt(var)
    use_mix = (not math.isnan(K_pop)) and (not math.isnan(K_i)) and \
        (upper != math.inf or lower == upper)
    if not use_mix:
        # Plain mode is exactly the truncated normal on (lower, upper); the
        # mixture algebra below would compute a second component on
        # (upper, inf) only to multiply it by zero.
        return _tnorm_moments_loc(mu, sd, lower, upper)
    split = -_norm_ppf(K_pop)               # lifetime threshold thr_pop
    if lower != upper and lower >= split:
        return _tnorm_moments_loc(mu, sd, lower, math.inf)
    # Use survival differences for a finite lower bound in the right tail:
    # subtracting ordinary CDFs there can erase the entire control component.
    # The ordinary unbounded-control path retains its original arithmetic.
    a = -math.inf if lower == upper else (lower - mu) / sd
    b = (split - mu) / sd
    width = split - lower
    narrow = (lower != upper and math.isfinite(lower)
              and width / sd <= 1e-3 and abs(a * (width / sd)) <= 1.0)
    log_control, m0, v0 = 0.0, 0.0, 0.0
    if narrow:
        log_control, m0, v0 = _narrow_mixture_component(mu, sd, lower, width)
    if K_i == K_pop:
        return (m0, v0) if narrow else _tnorm_moments_loc(mu, sd, lower, split)
    if narrow:
        control = math.exp(log_control)
    elif a > 0.0:
        control = _norm_cdf(-a) - _norm_cdf(-b)
    else:
        control = _norm_cdf(b) - _norm_cdf(a)
    remaining = (K_pop - K_i) / K_pop
    future = _norm_cdf(-b) * remaining
    denom = control + future
    if denom < _MIN_NORMAL:
        # Both masses may underflow while their ratio remains ordinary.
        # Reflect before taking a log-CDF difference, then normalise relative
        # to the larger log mass instead of exponentiating either raw mass.
        if not narrow:
            if a > 0.0:
                log_hi, log_lo = _log_norm_cdf(-a), _log_norm_cdf(-b)
            else:
                log_hi, log_lo = _log_norm_cdf(b), _log_norm_cdf(a)
            log_control = log_hi + math.log(-math.expm1(log_lo - log_hi))
        log_future = _log_norm_cdf(-b) + math.log(remaining)
        scale = max(log_control, log_future)
        control, future = math.exp(log_control - scale), math.exp(log_future - scale)
        denom = control + future
    p, q = control / denom, future / denom
    if p == 0.0:
        return _tnorm_moments_loc(mu, sd, split, math.inf)
    if q == 0.0:
        return (m0, v0) if narrow else _tnorm_moments_loc(mu, sd, lower, split)
    if not narrow:
        m0, v0 = _tnorm_moments_loc(mu, sd, lower, split)
    m1, v1 = _tnorm_moments_loc(mu, sd, split, math.inf)
    new_mean = p * m0 + q * m1
    # Total variance in centred form; E[X^2] - E[X]^2 loses small posterior
    # variances when either component has a large conditional mean.
    new_var = p * v0 + q * v1 + p * q * (m0 - m1) ** 2
    return new_mean, new_var


@_jit
def _pa_update(cov, mu, i, nm, nv):
    """Apply one rank-one PA update to the active leading block only."""
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
    """Run the sequential mixture fold for one family, target first."""
    d = cov.shape[0]
    mu = np.zeros(d)
    # Pins can determine a row exactly; their support is checked before this
    # kernel (`_check_pin_support`). A small positive variance still carries
    # information and must not suppress the row's observation.
    for i in range(d - 1, 0, -1):
        if lower[i] == -math.inf and upper[i] == math.inf:
            # An absent row carries no bound and no mixture pair, so its
            # infinite interval has unchanged moments; folding it would only
            # add rounding noise.
            continue
        if cov[i, i] <= 0.0:
            continue
        nm, nv = _tnorm_mixture(mu[i], cov[i, i], lower[i], upper[i], K_i[i], K_pop[i])
        _pa_update(cov, mu, i, nm, nv)
    if cov[0, 0] <= 0.0:
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


@_jit
def _pa_batched_nomix_serial(cov, lowers, uppers, est, var):
    """Serial twin of `_pa_batched_nomix` for small batches (see
    `_PARALLEL_MIN_FAMILIES`). Same loop, no ``prange``: each family folds its
    own copy of ``cov``, so the two kernels give identical results."""
    F = lowers.shape[0]
    for f in range(F):
        c = cov.copy()
        e, v = _pa_family_nomix(c, lowers[f], uppers[f])
        est[f] = e
        var[f] = v


@_jit
def _pa_batched_serial(cov, lowers, uppers, K_is, K_pops, est, var):
    """Serial twin of `_pa_batched` for small batches (see
    `_PARALLEL_MIN_FAMILIES`)."""
    F = lowers.shape[0]
    for f in range(F):
        c = cov.copy()
        e, v = _pa_family(c, lowers[f], uppers[f], K_is[f], K_pops[f])
        est[f] = e
        var[f] = v


#: Batches below this many families run the serial kernel twins. Launching a
#: parallel kernel costs tens of microseconds that a small batch never repays,
#: and a mask-grouped call launches one kernel per observation-mask group, so
#: chunks and register-size batches otherwise pay it dozens of times over.
#: The crossover was measured near 128 families (2026-09-28, ltpred314,
#: four Numba threads); both kernels return identical values.
_PARALLEL_MIN_FAMILIES = 128


def _pa_kernel_nomix(cov, lowers, uppers, est, var):
    """`_pa_batched_nomix`, serial below `_PARALLEL_MIN_FAMILIES` families."""
    if lowers.shape[0] < _PARALLEL_MIN_FAMILIES:
        _pa_batched_nomix_serial(cov, lowers, uppers, est, var)
    else:
        _pa_batched_nomix(cov, lowers, uppers, est, var)


def _pa_kernel_mixture(cov, lowers, uppers, K_is, K_pops, est, var):
    """`_pa_batched`, serial below `_PARALLEL_MIN_FAMILIES` families."""
    if lowers.shape[0] < _PARALLEL_MIN_FAMILIES:
        _pa_batched_serial(cov, lowers, uppers, K_is, K_pops, est, var)
    else:
        _pa_batched(cov, lowers, uppers, K_is, K_pops, est, var)


def _pin_tolerance(lowers, values, n):
    """Compatibility tolerance for a pin-determined value against its bounds.

    ``128 * eps * n * max(1, |values|)`` with ``eps`` taken from the bounds'
    stored dtype: float32 bounds (the batched APIs' memory-saving dtype) carry
    their own storage rounding, and a coherent float64 configuration whose
    pin and bound round to opposite sides must not read as a contradiction.
    Pin-versus-pin contradictions use distinct user-supplied values and keep
    the tight float64 tolerance inside `_condition_pins`."""
    eps = (np.finfo(lowers.dtype).eps if lowers.dtype == np.float32
           else np.finfo(float).eps)
    return 128 * eps * n * np.maximum(1.0, np.abs(values))


def _check_pin_support(cov, lowers, uppers, K_is):
    """Apply Algorithm P's step-P2 compatibility checks on the mixture path.

    Algorithm M folds pins sequentially instead of conditioning on them
    jointly, so it would otherwise accept pins that contradict each other or
    the bounds of a row they determine. Group families by pin mask, condition
    on the pins with `_condition_pins` (which rejects pins outside the
    covariance support) and require every row the pins determine to lie within
    its bounds. A censored-control mixture row may exceed its ``upper`` (a
    future case), so only its ``lower`` bound is checked. The bounds are
    compared at their stored precision (`_pin_tolerance`): storage rounding at
    float32 is absorbed, not read as a contradiction."""
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
        tolerance = _pin_tolerance(lowers, values, len(cov))
        if (np.any(values < lowers[np.ix_(rows, columns)] - tolerance)
                or np.any(values > effective_upper[np.ix_(rows, columns)] + tolerance)):
            raise ValueError("observation excludes a deterministic liability after pin conditioning")


def _condition_pins(cov, retained, pinned, values):
    """Condition one observation-mask group on its exact pins.

    Uses Cholesky solves for ordinary blocks and spectral support for singular
    ones. Incompatible pins or negative conditional variances raise. Returns
    per-family means and the shared conditional covariance, preserving genuine
    small positive variances while removing numerical determinism.
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
    """Marginalise absent rows and share pin conditioning by observation mask.

    Reduced families then use the no-mixture fold. Deterministic targets return
    immediately; pin-free, fully observed batches skip grouping altogether.
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
        _pa_kernel_nomix(cov, lowers, uppers, est, var)
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
        # Own just two float64 work arrays, even for float32 bounds; center
        # them in place below.
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
            tolerance = _pin_tolerance(lowers, values, d)
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
            _pa_kernel_nomix(subcov, lo, hi, e, v)
        est[rows], var[rows] = (e if means is None else means[:, 0] + e), v
    return est, var


def _validate_pa_covmat(covmat, _psd_certified=None):
    """Validate a finite, symmetric PSD covariance with positive variances.

    Singular support is checked during conditioning; strict PD is not required.
    The private certificate only skips a redundant eigenvalue check.
    """
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


def _check_interval_rank(cov, lowers, uppers):
    """Refuse unsupported dependent intervals on singular Gaussian support.

    Pins are handled by exact conditioning. After that, positive-variance
    interval coordinates must have a nonsingular joint covariance. A sequence
    of Gaussian moment projections cannot preserve their linear support: it
    could otherwise return a posterior even when their bounds are incompatible.
    Singular targets or unobserved rows alone impose no such restriction.
    High-level callers with a certified strictly-PD covariance skip this gate.
    """
    if not len(lowers):
        return
    scale = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(scale, scale)
    rank_tol = 64 * np.finfo(float).eps
    eigenvalues = np.linalg.eigvalsh(correlation)
    # Cholesky can succeed on exactly dependent rows after floating-point
    # scaling; numerical rank must use the same relative gate as the blocks.
    if eigenvalues[0] > rank_tol * len(cov) * eigenvalues[-1]:
        return
    pins = lowers == uppers
    intervals = ~pins & (np.isfinite(lowers) | np.isfinite(uppers))
    states = pins.astype(np.uint8) + 2 * intervals.astype(np.uint8)
    for state in np.unique(states, axis=0):
        observed = np.flatnonzero(state == 2)
        if len(observed) < 2:
            continue
        pinned = np.flatnonzero(state == 1)
        if len(pinned):
            _, block = _condition_pins(cov, observed, pinned,
                                       np.zeros((1, len(pinned))))
        else:
            block = cov[np.ix_(observed, observed)]
        positive = np.diag(block) > 0
        block = block[np.ix_(positive, positive)]
        if len(block) < 2:
            continue
        sd = np.sqrt(np.diag(block))
        eigenvalues = np.linalg.eigvalsh(block / np.outer(sd, sd))
        if eigenvalues[0] <= rank_tol * len(block) * eigenvalues[-1]:
            raise ValueError(
                "PA does not support linearly dependent interval observations "
                "on singular covariance support; combine redundant bounds or "
                "supply an independently specified nonsingular observation model")


def pa_algorithm(covmat: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                 target: int = 0, K_i: ArrayLike | None = None,
                 K_pop: ArrayLike | None = None) -> tuple[float, float]:
    """Return PA posterior mean and variance for one family.

    ``target`` selects the covariance row to estimate. Without a mixture, pins
    are conditioned jointly and absent non-target rows marginalised; active
    ``K_i``/``K_pop`` pairs select the censoring mixture. All-NaN pairs leave
    the plain path unchanged. Without a mixture, the result is exact after
    pins with at most one remaining interval. Singular support with dependent
    interval observations is rejected.
    """
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
    _check_interval_rank(cov, lo[None, :], hi[None, :])
    if K_i is None and K_pop is None:
        est, var = _pa_reduced_nomix(cov, lo[None, :], hi[None, :])
        return float(est[0]), float(var[0])
    K_i, K_pop = validate_mixture_inputs(
        K_i, K_pop, expected_shape=(d,), lower=lower, upper=upper,
        context="pa_algorithm mixture inputs")
    if np.all(np.isnan(K_i)):
        est, var = _pa_reduced_nomix(cov, lo[None, :], hi[None, :])
        return float(est[0]), float(var[0])
    K_i = _as_bounds(K_i)
    K_pop = _as_bounds(K_pop)
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
    allocating the ``(F, d)`` mixture arrays. With arrays, only families with
    active pairs use Algorithm M; all-NaN families still use Algorithm P.
    Returns the PA sequential-moment approximations ``(est, var)`` of length
    ``F``."""
    cov = _validate_pa_covmat(covmat, _psd_certified=_psd_certified)
    d = cov.shape[0]
    lowers = _as_bounds(lowers)             # keeps float32 to halve memory
    uppers = _as_bounds(uppers)
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
    # _as_bounds already made contiguous inputs. The usual target is zero;
    # an identity permutation would duplicate both full cohort arrays.
    lo = lowers if target == 0 else np.ascontiguousarray(lowers[:, order])
    hi = uppers if target == 0 else np.ascontiguousarray(uppers[:, order])
    validate_bounds(lo, hi, context="pa_estimate_batched bounds")
    if _psd_certified is not _PSD_CERTIFIED:
        _check_interval_rank(cov, lo, hi)
    if K_is is None and K_pops is None:        # no-mixture fast path (no K arrays)
        return _pa_reduced_nomix(cov, lo, hi)
    K_is, K_pops = validate_mixture_inputs(
        K_is, K_pops, expected_shape=(F, d), lower=lowers, upper=uppers,
        context="batched PA mixture inputs")
    active = np.any(~np.isnan(K_is), axis=1)
    if not np.any(active):
        return _pa_reduced_nomix(cov, lo, hi)
    K_is = _as_bounds(K_is)
    K_pops = _as_bounds(K_pops)
    ki = K_is if target == 0 else np.ascontiguousarray(K_is[:, order])
    kp = K_pops if target == 0 else np.ascontiguousarray(K_pops[:, order])
    if np.all(active):
        est, var = np.empty(F), np.empty(F)
        _check_pin_support(cov, lo, hi, ki)
        _pa_kernel_mixture(cov, lo, hi, ki, kp, est, var)
        return est, var
    # Mixture participation belongs to each family, not the enclosing cohort.
    # Inactive rows retain joint pin conditioning and its exact special cases.
    est, var = np.empty(F), np.empty(F)
    est[~active], var[~active] = _pa_reduced_nomix(cov, lo[~active], hi[~active])
    mix_lo, mix_hi, mix_ki = lo[active], hi[active], ki[active]
    mix_est, mix_var = np.empty(np.count_nonzero(active)), np.empty(np.count_nonzero(active))
    _check_pin_support(cov, mix_lo, mix_hi, mix_ki)
    _pa_kernel_mixture(cov, mix_lo, mix_hi, mix_ki, kp[active], mix_est, mix_var)
    est[active], var[active] = mix_est, mix_var
    return est, var
