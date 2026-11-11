"""Truncated multivariate-normal Gibbs sampling.

For ``Q = Sigma^-1``, `gibbs_params` forms conditional means
``P[i,j] = -Q[i,j]/Q[j,j]`` and standard deviations ``sqrt(1/Q[j,j])``.
Each free coordinate is sampled from its truncated normal conditional;
coordinates spanning less than ``1e-8`` are fixed. Strongly coupled blocks
also receive joint Gaussian updates along whitened directions.

Batched estimation integrates out each family's unbounded coordinates and
streams Rao--Blackwellised sums and batch means without retaining draws.
Per-family seeds make results independent of scheduling and chunking.
Persistent chains used by the fitters draw uniforms from a thread-local
NumPy generator. See Algorithm G in ``docs/algorithm.md``.
"""

from __future__ import annotations

import operator
import threading
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf
from ._validation import validate_bounds

__all__ = ["rtmvnorm_gibbs", "gibbs_params"]

# Numba's ``np.random`` state is tied to worker threads, so seeding the calling
# thread cannot make a ``prange`` kernel scheduler-independent. For
# `_gibbs_advance_chain`, generate float64 uniforms here instead and pass them into a
# random-free kernel (the estimator kernels reseed per family instead). The
# RNG is thread-local so concurrent seeded fits cannot overwrite one another.
_advance_rng_state = threading.local()
_MAX_ADVANCE_UNIFORMS = 1 << 20       # 8 MiB of float64 temporary storage
_MAX_SEED = (1 << 32) - 1
# Coordinates whose bounds span less than this are held fixed (a pinned point
# mass, e.g. an onset-pinned case in LT-FH++) rather than resampled.
_FIXED_TOL = 1e-8
# A survival probability below this leaves fewer than 2**24 representable
# doubles between the bounds, so inverse-CDF interpolation of the two survival
# values collapses to a handful of distinct draws (bounds [38.45, 38.46] drew
# 20000 identical values). Hand such intervals to the tail-ratio route instead.
_TAIL_QUANT_FLOOR = 2.0 ** -1050
# A coordinate conditional SD less than one quarter of its marginal SD signals
# a narrow Gaussian direction. Supplement coordinate Gibbs in this regime;
# this is a sampler choice, not a convergence certificate.
_DIRECTIONAL_VARIANCE_RATIO = 16.0


@_jit
def _far_right_std_tnorm_quantile(a, b, u):
    """Sample a far-right truncated-normal quantile without CDF underflow.

    Uses the log-scale Gaussian tail ratio and a rationalised Rayleigh inverse.
    ``u`` is rescaled to the interval's tail mass and the result is clamped to
    ``[a,b]``. The omitted Mills-ratio correction vanishes as ``a`` grows.
    """
    span = b - a
    if span <= 0.0:
        return a
    if np.isinf(span):
        share = 1.0
    else:
        share = -np.expm1(-a * span - 0.5 * span * span)
    p = u * share
    if p > 1.0 - 1e-16:
        p = 1.0 - 1e-16
    # Rationalised sqrt(a^2 + d) - a == d / (sqrt(a^2 + d) + a) with
    # d = -2 log(1-p): no cancellation when d << a^2 (a >= 38 here).
    d = -2.0 * np.log1p(-p)
    t = d / (np.sqrt(a * a + d) + a)
    x = a + t
    if x < a:
        return a
    if x > b:
        return b
    return x


@_jit
def _std_tnorm_quantile(a, b, u):
    """Quantile of a standard normal truncated to ``[a,b]``.

    Positive tails use survival probabilities and symmetry; extreme or
    subnormal tails use `_far_right_std_tnorm_quantile`. The final clamp only
    removes inverse-approximation roundoff.
    """
    if a >= 0.0:
        sa = _norm_cdf(-a)
        if sa < _TAIL_QUANT_FLOOR:          # too few representable survivals
            return _far_right_std_tnorm_quantile(a, b, u)
        sb = _norm_cdf(-b)
        q = (1.0 - u) * sa + u * sb
        x = -_norm_ppf(q)
        if not np.isfinite(x):
            return _far_right_std_tnorm_quantile(a, b, u)
    else:
        fb = _norm_cdf(b)
        if fb < _TAIL_QUANT_FLOOR:          # far left tail: reflect to the right
            return -_far_right_std_tnorm_quantile(-b, -a, 1.0 - u)
        fa = _norm_cdf(a)
        p = (1.0 - u) * fa + u * fb
        x = _norm_ppf(p)
        if not np.isfinite(x):
            return -_far_right_std_tnorm_quantile(-b, -a, 1.0 - u)

    if x < a:
        return a
    if x > b:
        return b
    return x


@_jit
def _gibbs_conditional_draw(P, sd, x, j, lower_j, upper_j, u):
    """One coordinate update: conditional mean -> truncated-normal draw -> clamp.

    The shared body of every sweep kernel: the conditional mean
    ``mu_j = P[:, j] . x`` with conditional SD ``sd_j``, inverse-CDF sampling of
    the conditional normal restricted to ``(lower_j, upper_j)`` (Kotecha &
    Djuric 1999), then a clamp that only guards the last-bit error of the
    inverse approximation -- it never moves a draw across a truncation boundary.
    ``u`` is the coordinate's uniform deviate: drawn from the kernel RNG in the
    seeded samplers, caller-supplied in the random-free advance kernels."""
    d = sd.shape[0]
    mu_j = 0.0
    for i in range(d):
        mu_j += P[i, j] * x[i]
    sd_j = sd[j]
    a = (lower_j - mu_j) / sd_j
    b = (upper_j - mu_j) / sd_j
    z = _std_tnorm_quantile(a, b, u)
    xj = mu_j + sd_j * z
    if xj < lower_j:
        return lower_j
    if xj > upper_j:
        return upper_j
    return xj


@_jit
def _init_chain(lower, upper, sd0):
    """Initial chain state: each coordinate's marginal truncated median (step G3).

    LTFHPlus's init -- the median of the *marginal* ``N(0, sd0[j]^2)`` truncated
    to ``[lower[j], upper[j]]``. Each value lies in its own interval, so the
    point is feasible for the rectangle and fixed coordinates start at their
    pinned value; a non-finite quantile falls back to 0. Works on one chain's
    ``(d,)`` arrays, so the fit paths can share it with a unit ``sd0``."""
    d = sd0.shape[0]
    x = np.empty(d, dtype=np.float64)
    for j in range(d):
        xj = _std_tnorm_quantile(lower[j] / sd0[j],
                                 upper[j] / sd0[j], 0.5) * sd0[j]
        x[j] = xj if np.isfinite(xj) else 0.0
    return x


def _validate_covmat(covmat):
    """Return ``covmat`` as float64 after validating the Gibbs covariance."""
    cov = np.ascontiguousarray(covmat, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covmat must be square")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covmat must contain only finite values")
    matrix_scale = float(np.max(np.abs(cov))) if cov.size else 1.0
    symmetry_tolerance = 1e-10 * matrix_scale
    asymmetry = float(np.max(np.abs(cov - cov.T))) if cov.size else 0.0
    if asymmetry > symmetry_tolerance:
        raise ValueError(
            "covmat must be symmetric (maximum asymmetry "
            f"{asymmetry:.3g}, relative tolerance {symmetry_tolerance:.3g})")
    d = cov.shape[0]
    eigvals = np.linalg.eigvalsh(cov) if d else np.array([1.0])
    min_eig = float(np.min(eigvals))
    scale = float(np.max(np.abs(eigvals)))
    tolerance = 1e-12 * scale
    if min_eig <= tolerance:
        raise ValueError(
            "covmat must be positive-definite (minimum eigenvalue "
            f"{min_eig:.3g}, relative tolerance {tolerance:.3g}); singular, "
            "indefinite, or numerically rank-deficient covariance cannot define "
            "Gibbs conditionals -- see correct_positive_definite")
    return cov


def gibbs_params(covmat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Precompute conditional-regression coefficients and standard deviations.

    For precision ``Q = Sigma^-1``, returns ``P[i,j] = -Q[i,j]/Q[j,j]`` with
    zero diagonal and ``sd[j] = sqrt(1/Q[j,j])``. ``covmat`` must be finite,
    symmetric and strictly positive definite. One matrix inverse is reused
    across all coordinates and convergence rounds.
    """
    cov = _validate_covmat(covmat)
    Q = np.linalg.inv(cov)
    qdiag = np.diag(Q).copy()
    P = -Q / qdiag[np.newaxis, :]              # P[i,j] = -Q[i,j] / Q[j,j]
    np.fill_diagonal(P, 0.0)
    sd = np.sqrt(1.0 / qdiag)
    return np.ascontiguousarray(P), np.ascontiguousarray(sd)


def _validate_params(params, d):
    """Return precomputed ``(P, sd)`` from `gibbs_params`, checked against ``d``.

    ``params`` is documented as the output of `gibbs_params` for *this*
    covariance, and nothing else constrains it: a pair built from a
    different-size matrix reaches the jitted sweep, which indexes the bounds
    and the initial chain with the arrays' own length and reads out of bounds
    -- a crash (observed as aborts/segfaults), not an exception.
    """
    try:
        P, sd = params
    except (TypeError, ValueError):
        raise ValueError(
            "params must be a (P, sd) pair from gibbs_params") from None
    P = np.asarray(P)
    sd = np.asarray(sd)
    if P.ndim != 2 or P.shape != (d, d) or sd.ndim != 1 or sd.shape != (d,):
        raise ValueError(
            f"params must match covmat: P of shape ({d}, {d}) and sd of shape "
            f"({d},); got shapes {P.shape} and {sd.shape}. params must be the "
            "gibbs_params output of this same covmat")
    return (np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(sd, dtype=np.float64))


def _unbounded_patterns(lowers, uppers):
    """Group families by their own set of ``(-inf, inf)`` coordinates.

    The collapse set of step G2 is each family's own unbounded set, so a
    family's draws depend only on its bounds and seed, never on which other
    families share the call (a round, a chunk, the rest of the cohort).
    Returns ``[(mask, rows), ...]`` with ``mask`` a ``(d,)`` boolean array and
    ``rows`` the families that share it, in first-appearance order."""
    lo = np.asarray(lowers)
    hi = np.asarray(uppers)
    unbounded = np.ascontiguousarray(~np.isfinite(lo) & ~np.isfinite(hi))
    if unbounded.shape[0] == 0 or (unbounded == unbounded[0]).all():
        return [(unbounded[0] if unbounded.shape[0] else unbounded.any(axis=0),
                 np.arange(unbounded.shape[0]))]
    keys = unbounded.view(np.dtype((np.void, unbounded.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Sort once instead of rescanning all families for every distinct mask.
    # Stable slices preserve the input row order within each group.
    order = np.argsort(inverse, kind="stable")
    boundaries = np.r_[0, np.cumsum(np.bincount(inverse))]
    return [(unbounded[row], order[boundaries[k]:boundaries[k + 1]])
            for k, row in sorted(enumerate(first), key=lambda item: item[1])]


def _blup_from_keep(cov, keep_idx, coll_idx):
    """Conditional mean map and residual variance of collapsed coordinates.

    For jointly Gaussian ``(z, y)`` with ``y`` the kept (possibly truncated)
    block, ``E[z | y] = W y`` with ``W = Sigma_zy Sigma_yy^-1`` and
    ``Var(z | y) = cond_var`` (diagonal of ``Sigma_zz - W Sigma_yz``, clipped
    at 0; constant in ``y``). Truncating ``y`` does not change either map.
    """
    cov = np.ascontiguousarray(cov, dtype=np.float64)
    Syy = cov[np.ix_(keep_idx, keep_idx)]
    Szy = cov[np.ix_(coll_idx, keep_idx)]
    Szz = cov[np.ix_(coll_idx, coll_idx)]
    W = np.linalg.solve(Syy.T, Szy.T).T
    resid = Szz - W @ Szy.T
    cond_var = np.maximum(np.diag(resid), 0.0)
    return (np.ascontiguousarray(W, dtype=np.float64),
            np.ascontiguousarray(cond_var, dtype=np.float64))


@_jit
def _mixing_directions(P, sd, sd0, fixed):
    """Whitened directions on the free coordinates, only for strong coupling.

    Given precision Q and fixed coordinates p, diagonalize Q_ff = U D U'.
    Columns v = U / sqrt(D), padded with zeros on p, obey v'Qv = 1.
    Along x + t*v the conditional law is N(-v'Qx, 1), truncated by the
    rectangle. Store v and Qv as rows. Exact pins therefore never move, and
    their nonzero values enter the conditional mean through (Qv)_p.
    Ordinary covariance blocks return before any matrix factorization.
    """
    d = len(sd)
    empty = np.empty((0, d))
    ratio = 0.0
    for j in range(d):
        ratio = max(ratio, (sd0[j] / sd[j])**2)
    if ratio <= _DIRECTIONAL_VARIANCE_RATIO:
        return empty, empty
    free = np.flatnonzero(~fixed)
    n = len(free)
    if n < 2:
        return empty, empty
    precision = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a, b = free[i], free[j]
            precision[i, j] = (1.0 if a == b else -P[a, b]) / sd[b]**2
    precision = (precision + precision.T) * .5
    eigenvalues, vectors = np.linalg.eigh(precision)
    if eigenvalues[0] <= 0.0:
        raise ValueError("Gibbs joint-update precision must be positive definite")
    # Conditioning on pins can remove the original coupling entirely.
    ratio = 0.0
    for i in range(n):
        variance = 0.0
        for j in range(n):
            variance += vectors[i, j]**2 / eigenvalues[j]
        ratio = max(ratio, precision[i, i] * variance)
    if ratio <= _DIRECTIONAL_VARIANCE_RATIO:
        return empty, empty
    directions = np.zeros((n, d))
    dual = np.zeros((n, d))
    for k in range(n):
        root = np.sqrt(eigenvalues[k])
        for i in range(n):
            directions[k, free[i]] = vectors[i, k] / root
            dual[k, free[i]] = vectors[i, k] * root
        for j in range(d):
            if fixed[j]:
                for i in range(n):
                    dual[k, j] -= P[j, free[i]] / sd[free[i]]**2 * directions[k, free[i]]
    return directions, dual


@_jit
def _directional_sweep(directions, dual, lower, upper, x):
    """Additional Gaussian conditional updates along joint free directions.

    Each move preserves the same TMVN law as a coordinate update. Intersect
    every original bound on the line before drawing its scalar displacement.
    """
    for k in range(len(directions)):
        lo, hi, mean = -np.inf, np.inf, 0.0
        for j in range(len(x)):
            v = directions[k, j]
            mean -= dual[k, j] * x[j]
            if v > 0.0:
                lo = max(lo, (lower[j] - x[j]) / v)
                hi = min(hi, (upper[j] - x[j]) / v)
            elif v < 0.0:
                lo = max(lo, (upper[j] - x[j]) / v)
                hi = min(hi, (lower[j] - x[j]) / v)
        step = mean + _std_tnorm_quantile(lo - mean, hi - mean, np.random.random())
        for j in range(len(x)):
            if directions[k, j] != 0.0:
                x[j] = min(upper[j], max(lower[j], x[j] + directions[k, j] * step))


@_jit
def _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, n_sim, burn_in, res,
                 directions, dual):
    """Inner Gibbs loop (the Rcpp ``rtmvnorm_gibbs_cpp`` port).

    Runs ``burn_in + n_sim`` sweeps, writing the post-burn-in draws of the
    requested coordinates into ``res``. ``x`` is the working state (updated in
    place); ``to_return[j] >= 0`` gives the output column for coordinate ``j``,
    or -1 to drop it. Everything here is scalar/loop so it compiles under
    ``numba.njit`` when Numba is installed and otherwise uses the serial Python
    fallback. Both paths match given the same RNG draws."""
    d = sd.shape[0]
    for k in range(-burn_in, n_sim):
        for j in range(d):
            if not fixed[j]:
                x[j] = _gibbs_conditional_draw(P, sd, x, j, lower[j], upper[j],
                                               np.random.random())
        if len(directions):
            _directional_sweep(directions, dual, lower, upper, x)
        for j in range(d):
            if k >= 0 and to_return[j] >= 0:
                res[k, to_return[j]] = x[j]
    return res


@_jit_parallel
def _gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                            batch_size, n_batch, seeds, total_sum, total_sumsq,
                            bm_sum, bm_sumsq):
    """Sample many families, accumulating summaries online (steps G3--G5).

    All families share ``(P, sd)``; only their bounds differ. Each ``prange``
    iteration reseeds the kernel RNG with ``seeds[f]`` (unless ``-1``), starts a
    fresh chain at `_init_chain`, runs ``burn_in + n_sim`` sweeps and writes per
    output coordinate: ``total_sum[f]`` and ``total_sumsq[f]`` over the ``n_sim``
    retained draws (posterior mean and variance of the TMVN itself), and
    ``bm_sum[f]`` / ``bm_sumsq[f]``, the sum and sum of squares of the means of
    the first ``n_batch`` consecutive batches of ``batch_size`` draws (the
    batch-means Monte-Carlo SE). Draws past ``n_batch * batch_size`` enter the
    totals only. State is ``O(d + ncols)`` per family, plus ``O(d**2)`` for
    strongly coupled direction coefficients; the family loop is parallel
    under Numba and serial in the pure-Python fallback."""
    F = lowers.shape[0]
    d = sd.shape[0]
    ncols = out_idx.shape[0]

    for f in prange(F):
        lower = lowers[f]
        upper = uppers[f]
        if seeds[f] >= 0:
            np.random.seed(seeds[f])

        # per-family working state (thread-local)
        fixed = np.empty(d, dtype=np.bool_)
        for j in range(d):
            fixed[j] = (upper[j] - lower[j]) < _FIXED_TOL
        x = _init_chain(lower, upper, sd0)
        directions, dual = _mixing_directions(P, sd, sd0, fixed)

        tot = np.zeros(ncols)
        tot_sq = np.zeros(ncols)      # sum of x^2 -> posterior variance
        batch_sum = np.zeros(ncols)
        s1 = np.zeros(ncols)          # sum of batch means Y_k
        s2 = np.zeros(ncols)          # sum of Y_k^2
        bidx = 0
        in_batch = 0

        for k in range(-burn_in, n_sim):
            for j in range(d):
                if not fixed[j]:
                    x[j] = _gibbs_conditional_draw(P, sd, x, j, lower[j],
                                                   upper[j],
                                                   np.random.random())
            if len(directions):
                _directional_sweep(directions, dual, lower, upper, x)
            if k >= 0:
                for c in range(ncols):
                    v = x[out_idx[c]]
                    tot[c] += v
                    tot_sq[c] += v * v
                    if bidx < n_batch:
                        batch_sum[c] += v
                if bidx < n_batch:
                    in_batch += 1
                    if in_batch == batch_size:
                        for c in range(ncols):
                            y = batch_sum[c] / batch_size
                            s1[c] += y
                            s2[c] += y * y
                            batch_sum[c] = 0.0
                        bidx += 1
                        in_batch = 0

        for c in range(ncols):
            total_sum[f, c] = tot[c]
            total_sumsq[f, c] = tot_sq[c]
            bm_sum[f, c] = s1[c]
            bm_sumsq[f, c] = s2[c]


@_jit_parallel
def _gibbs_estimate_batched_collapsed(
        P, sd, sd0, lowers, uppers, out_kind, out_local, W, cond_var,
        n_sim, burn_in, batch_size, n_batch, seeds,
        total_sum, total_sumsq, bm_sum, bm_sumsq):
    """Like ``_gibbs_estimate_batched`` on the kept block; BLUP the rest (G2).

    ``P``, ``sd``, ``sd0``, bounds and ``x`` cover only the kept coordinates.
    ``out_kind[c] == 0`` reads kept coordinate ``out_local[c]``.
    ``out_kind[c] == 1`` accumulates ``W[out_local[c]] . x`` and adds
    ``cond_var[out_local[c]]`` to the sum of squares so the streamed
    posterior variance is ``Var(E[z|y]) + Var(z|y)``. Batch means are formed
    from the streamed value, i.e. from ``E[z|y]`` for a collapsed output.
    """
    F = lowers.shape[0]
    d = sd.shape[0]
    ncols = out_kind.shape[0]

    for f in prange(F):
        lower = lowers[f]
        upper = uppers[f]
        if seeds[f] >= 0:
            np.random.seed(seeds[f])

        fixed = np.empty(d, dtype=np.bool_)
        for j in range(d):
            fixed[j] = (upper[j] - lower[j]) < _FIXED_TOL
        x = _init_chain(lower, upper, sd0)
        directions, dual = _mixing_directions(P, sd, sd0, fixed)

        tot = np.zeros(ncols)
        tot_sq = np.zeros(ncols)
        batch_sum = np.zeros(ncols)
        s1 = np.zeros(ncols)
        s2 = np.zeros(ncols)
        bidx = 0
        in_batch = 0

        for k in range(-burn_in, n_sim):
            for j in range(d):
                if not fixed[j]:
                    x[j] = _gibbs_conditional_draw(P, sd, x, j, lower[j],
                                                   upper[j],
                                                   np.random.random())
            if len(directions):
                _directional_sweep(directions, dual, lower, upper, x)
            if k >= 0:
                for c in range(ncols):
                    loc = out_local[c]
                    if out_kind[c] == 0:
                        v = x[loc]
                        extra = 0.0
                    else:
                        v = 0.0
                        for i in range(d):
                            v += W[loc, i] * x[i]
                        extra = cond_var[loc]
                    tot[c] += v
                    tot_sq[c] += v * v + extra
                    if bidx < n_batch:
                        batch_sum[c] += v
                if bidx < n_batch:
                    in_batch += 1
                    if in_batch == batch_size:
                        for c in range(ncols):
                            y = batch_sum[c] / batch_size
                            s1[c] += y
                            s2[c] += y * y
                            batch_sum[c] = 0.0
                        bidx += 1
                        in_batch = 0

        for c in range(ncols):
            total_sum[f, c] = tot[c]
            total_sumsq[f, c] = tot_sq[c]
            bm_sum[f, c] = s1[c]
            bm_sumsq[f, c] = s2[c]


def _as_bounds(a):
    """Contiguous float array for a kernel: **keep float32** (to halve the memory
    of large per-family bound arrays), otherwise coerce to float64. The kernels'
    internal state and accumulators stay float64 regardless, so only the big
    ``(F, d)`` inputs shrink; the covariance / conditional-regression factors are
    unaffected."""
    a = np.ascontiguousarray(a)
    return a if a.dtype == np.float32 else np.ascontiguousarray(a, dtype=np.float64)


def _collapsed_round(P, sd, sd0, cov, unbounded, lowers, uppers, out_idx, n_sim,
                     burn_in, batch_size, n_batch, seeds, _cache=None):
    """One round of step G2 for families sharing the unbounded set ``unbounded``.

    Samples only the kept block and streams ``E[z | y]`` for collapsed outputs
    (see `_gibbs_round`); with nothing to collapse it runs the full
    chain on ``P``/``sd``. Returns the four ``(F, ncols)`` sums.

    Everything derived from ``(cov, unbounded, out_idx)`` -- the BLUP map, the
    kept-block conditional-regression factors and the output column mapping --
    is constant across a convergence loop's rounds. A caller re-entering with
    the same covariance (as `_estimate_group` does each round) may pass a dict
    as ``_cache`` to compute those once, keyed on the unbounded mask; the sums
    are unchanged. ``None`` (the default) recomputes them every call."""
    F = int(lowers.shape[0])
    ncols = int(out_idx.shape[0])
    total_sum = np.zeros((F, ncols), dtype=np.float64)
    total_sumsq = np.zeros((F, ncols), dtype=np.float64)
    bm_sum = np.zeros((F, ncols), dtype=np.float64)
    bm_sumsq = np.zeros((F, ncols), dtype=np.float64)
    keep_idx = np.flatnonzero(~unbounded)
    coll_idx = np.flatnonzero(unbounded)
    if keep_idx.size == 0:
        for c, j in enumerate(out_idx):
            total_sumsq[:, c] = n_sim * cov[int(j), int(j)]
        return total_sum, total_sumsq, bm_sum, bm_sumsq
    if coll_idx.size == 0:
        _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                                np.ascontiguousarray(sd0), _as_bounds(lowers),
                                _as_bounds(uppers), out_idx, n_sim, burn_in,
                                batch_size, n_batch, np.ascontiguousarray(seeds),
                                total_sum, total_sumsq, bm_sum, bm_sumsq)
        return total_sum, total_sumsq, bm_sum, bm_sumsq
    prepared = None if _cache is None else _cache.get(unbounded.tobytes())
    if prepared is None:
        W, cond_var = _blup_from_keep(cov, keep_idx, coll_idx)
        keep_pos = {int(j): i for i, j in enumerate(keep_idx)}
        coll_pos = {int(j): i for i, j in enumerate(coll_idx)}
        out_kind = np.empty(ncols, dtype=np.int64)
        out_local = np.empty(ncols, dtype=np.int64)
        for c, j in enumerate(out_idx):
            j = int(j)
            if unbounded[j]:
                out_kind[c] = 1
                out_local[c] = coll_pos[j]
            else:
                out_kind[c] = 0
                out_local[c] = keep_pos[j]
        P_k, sd_k = gibbs_params(cov[np.ix_(keep_idx, keep_idx)])
        sd0_k = np.sqrt(np.diag(cov)[keep_idx])
        prepared = (W, cond_var, out_kind, out_local, P_k, sd_k, sd0_k)
        if _cache is not None:
            _cache[unbounded.tobytes()] = prepared
    W, cond_var, out_kind, out_local, P_k, sd_k, sd0_k = prepared
    lo_k = _as_bounds(np.ascontiguousarray(lowers[:, keep_idx]))
    hi_k = _as_bounds(np.ascontiguousarray(uppers[:, keep_idx]))
    _gibbs_estimate_batched_collapsed(
        P_k, sd_k, np.ascontiguousarray(sd0_k), lo_k, hi_k, out_kind,
        out_local, W, cond_var, n_sim, burn_in, batch_size, n_batch,
        np.ascontiguousarray(seeds), total_sum, total_sumsq, bm_sum, bm_sumsq)
    return total_sum, total_sumsq, bm_sum, bm_sumsq


def _gibbs_round(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                           batch_size, n_batch, seeds, cov=None, collapse=None,
                           _collapse_cache=None):
    """Run one fresh-chain sampling round and return streamed sufficient sums.

    Returns ``(sum, sumsq, batch_mean_sum, batch_mean_sumsq)`` for requested
    coordinates. When ``cov`` is supplied, each family's unbounded coordinates
    are collapsed analytically; their output streams ``E[z|y]`` and includes
    constant ``Var(z|y)`` in its second moment. ``collapse=False`` forces full
    chains. ``_collapse_cache`` reuses invariant factors across rounds without
    changing results. This internal primitive assumes validated arguments.
    """
    out_idx = np.asarray(out_idx, dtype=np.int64)
    F = int(np.asarray(lowers).shape[0])
    ncols = int(out_idx.shape[0])
    total_sum = np.zeros((F, ncols), dtype=np.float64)
    total_sumsq = np.zeros((F, ncols), dtype=np.float64)
    bm_sum = np.zeros((F, ncols), dtype=np.float64)
    bm_sumsq = np.zeros((F, ncols), dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.int64)
    n_sim = int(n_sim)
    burn_in = int(burn_in)
    batch_size = int(batch_size)
    n_batch = int(n_batch)

    use_collapse = bool(collapse) if collapse is not None else cov is not None
    if use_collapse:
        if cov is None:
            raise ValueError(
                "collapse=True requires the group covariance ``cov``")
        cov = np.ascontiguousarray(cov, dtype=np.float64)
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)
        patterns = _unbounded_patterns(lowers, uppers)
        if len(patterns) == 1 and not patterns[0][0].any():
            use_collapse = False                 # nothing to collapse: full chain
        else:
            for unbounded, rows in patterns:
                sums = _collapsed_round(
                    P, sd, sd0, cov, unbounded, lowers[rows], uppers[rows],
                    out_idx, n_sim, burn_in, batch_size, n_batch, seeds[rows],
                    _cache=_collapse_cache)
                for target, value in zip(
                        (total_sum, total_sumsq, bm_sum, bm_sumsq), sums):
                    target[rows] = value
            return total_sum, total_sumsq, bm_sum, bm_sumsq

    _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                            np.ascontiguousarray(sd0),
                            _as_bounds(lowers), _as_bounds(uppers),
                            out_idx, n_sim, burn_in, batch_size, n_batch,
                            seeds, total_sum, total_sumsq, bm_sum, bm_sumsq)
    return total_sum, total_sumsq, bm_sum, bm_sumsq


@_jit_parallel
def _gibbs_advance(P, sd, lowers, uppers, fixed, x, uniforms):
    """Advance ``x[f]`` by ``uniforms.shape[1]`` sweeps from supplied uniforms.

    Random-free: ``uniforms[f, sweep, j]`` feeds coordinate ``j`` (unused when
    ``fixed[f, j]``), so the result is independent of ``prange`` scheduling.
    Numba-parallel when available."""
    F = x.shape[0]
    d = x.shape[1]
    for f in prange(F):
        xf = x[f]
        for sweep in range(uniforms.shape[1]):
            for j in range(d):
                if not fixed[f, j]:
                    xf[j] = _gibbs_conditional_draw(P, sd, xf, j, lowers[f, j],
                                                    uppers[f, j],
                                                    uniforms[f, sweep, j])


@_jit
def _seed_numba_rng(seed):
    """Seed serial jitted samplers (or NumPy in the pure-Python fallback)."""
    np.random.seed(seed)


def _validate_seed(seed):
    """Return a user-supplied ``seed`` as a plain in-range int, or raise.

    The single gate for every public ``seed=`` argument that reaches a sampler, so
    the estimator and the fitters accept exactly the same values. ``bool`` is a
    subclass of ``int`` and is rejected on purpose: ``seed=True`` is a mistake, not
    a request for stream 1."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer in [0, {_MAX_SEED}]")
    seed = int(seed)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")
    return seed


def _validate_burn_in(burn_in):
    """Return a user-supplied ``burn_in`` as a plain non-negative int, or raise.

    The sweep kernels loop ``range(-burn_in, n_sim)``, so a negative burn-in
    silently drops initial output rows while ``n_sim`` draws are still credited.
    As for `_validate_seed`, ``bool`` and non-integers are rejected rather
    than truncated."""
    if isinstance(burn_in, (bool, np.bool_)) \
            or not isinstance(burn_in, (int, np.integer)):
        raise TypeError("burn_in must be a non-negative integer")
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer")
    return burn_in


def _seed_rng(seed):
    """Seed serial and parallel sampler streams; ``None`` reseeds from entropy.

    ``None`` (an unseeded call) draws a fresh OS-entropy seed and a fresh
    generator for both streams. Without it, a later unseeded call would replay
    whatever stream the previous seeded call left in this thread's Numba RNG
    and persistent-chain generator, making ``seed=None`` results deterministic
    after any seeded call in the same process.
    """
    if seed is None:
        generator = np.random.default_rng()
        seed = int(generator.integers(0, _MAX_SEED + 1))
    else:
        seed = _validate_seed(seed)
        generator = np.random.default_rng(seed)

    # Mutate the serial and parallel states only after all validation has
    # succeeded.
    _seed_numba_rng(seed)
    _advance_rng_state.generator = generator


def _advance_rng():
    """Return this calling thread's parallel-sampler generator."""
    generator = getattr(_advance_rng_state, "generator", None)
    if generator is None:
        generator = np.random.default_rng()
        _advance_rng_state.generator = generator
    return generator


def _gibbs_advance_chain(P, sd, lowers, uppers, fixed, x, n_sweeps):
    """Advance many families' truncated-MVN chains in place by ``n_sweeps`` sweeps.

    An internal fitting primitive of `ltpred.fit`; arguments are not validated,
    and public callers control reproducibility through the fitters' ``seed``.
    Unlike `_gibbs_round` (which runs fresh independent chains and
    returns their running sums), this keeps a **persistent** state ``x``
    (``F x d``, updated in place) that the caller carries across outer
    iterations — the data-augmentation step of a variance-component fit,
    where the covariance (hence ``P``/``sd``, from `gibbs_params`) changes
    between calls.
    ``lowers``/``uppers`` are the ``(F, d)`` bounds, and ``fixed[f, j]``
    coordinates (pinned cases) are held. ``n_sweeps <= 0`` does nothing. The
    family loop is parallel when Numba is installed and serial otherwise.

    Uniforms are generated before entering ``prange`` so a seeded fit is exact
    regardless of how Numba schedules families across worker threads. Generation
    is thread-local (isolating concurrent fits) and chunked to cap temporary memory.
    """
    n_sweeps = int(n_sweeps)
    n_families, d = x.shape
    if n_sweeps <= 0 or n_families == 0 or d == 0:
        return

    for start, stop, uniforms in _advance_uniform_blocks(n_families, d, n_sweeps):
        _gibbs_advance(P, sd, lowers[start:stop], uppers[start:stop],
                       fixed[start:stop], x[start:stop], uniforms)


def _advance_uniform_blocks(n_families, d, n_sweeps):
    """Yield ``(start, stop, uniforms)`` blocks for a persistent-chain advance.

    Draws from this thread's generator in a fixed family-then-sweep order, with
    at most ``_MAX_ADVANCE_UNIFORMS`` uniforms per block, so every advance kernel
    that consumes these blocks sees the same stream."""
    generator = _advance_rng()
    family_chunk = max(1, min(n_families, _MAX_ADVANCE_UNIFORMS // d))
    for start in range(0, n_families, family_chunk):
        stop = min(start + family_chunk, n_families)
        block_size = stop - start
        sweep_chunk = max(1, _MAX_ADVANCE_UNIFORMS // (block_size * d))
        for first_sweep in range(0, n_sweeps, sweep_chunk):
            this_sweeps = min(sweep_chunk, n_sweeps - first_sweep)
            yield start, stop, generator.random((block_size, this_sweeps, d))


def rtmvnorm_gibbs(covmat: ArrayLike, lower: ArrayLike = -np.inf,
                   upper: ArrayLike = np.inf, *, fixed: ArrayLike | None = None,
                   out: Sequence[int] = (0,), n_sim: int = 100_000,
                   burn_in: int = 1000, seed: int | None = None,
                   params: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> np.ndarray:
    """Draw truncated-MVN samples of the coordinates in ``out``.

    Parameters
    ----------
    covmat : (d, d) array
        Finite, symmetric, strictly positive-definite covariance.
    lower, upper : float or (d,) array
        Truncation bounds; scalars broadcast. Bounds must be ordered and not NaN.
    fixed : (d,) bool array, optional
        Coordinates to hold at their marginal truncated medians. Defaults to
        intervals narrower than ``1e-8``.
    out : sequence of int
        Coordinate indices; duplicates are removed and columns sorted.
    n_sim : int
        Positive number of retained draws; Boolean and non-integer values raise.
    burn_in : int
        Non-negative number of initial sweeps to discard.
    seed : int, optional
        Uint32 seed or ``None`` for fresh entropy. Each call reseeds this
        thread's serial sampler and persistent-chain stream. The pure-Python
        fallback shares NumPy's global RNG and is not safe for concurrent
        seeded calls.
    params : (P, sd), optional
        Precomputed `gibbs_params` for this same covariance; shapes are checked.

    Returns
    -------
    (n_sim, len(set(out))) ndarray
        Retained draws. The high-level estimators return summaries instead.
    """
    if isinstance(n_sim, (bool, np.bool_)):
        raise TypeError("n_sim must be a positive integer, not bool")
    try:
        n_sim = operator.index(n_sim)
    except TypeError:
        raise TypeError("n_sim must be a positive integer") from None
    if n_sim <= 0:
        raise ValueError("n_sim must be a positive integer")
    cov = (np.ascontiguousarray(covmat, dtype=np.float64)
           if params is None else _validate_covmat(covmat))
    if params is None:
        # ``gibbs_params`` performs the covariance validation in this path.
        params = gibbs_params(cov)
    else:
        params = _validate_params(params, cov.shape[0])
    d = cov.shape[0]
    burn_in = _validate_burn_in(burn_in)

    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (d,)).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (d,)).copy()
    validate_bounds(lower, upper, context="rtmvnorm_gibbs bounds")

    if fixed is None:
        fixed = (upper - lower) < _FIXED_TOL
    fixed = np.broadcast_to(np.asarray(fixed, dtype=bool), (d,)).copy()

    try:
        requested = list(out)
    except TypeError:
        raise TypeError("out must be a non-empty sequence of integer indices") from None
    if not requested:
        raise ValueError("out must contain at least one coordinate index")
    validated = []
    for value in requested:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("out indices must be integers, not bool")
        try:
            index = operator.index(value)
        except TypeError:
            raise TypeError(f"out index {value!r} must be an integer") from None
        if not 0 <= index < d:
            raise ValueError(f"out index {index} is outside the valid range [0, {d})")
        validated.append(index)
    out = sorted(set(validated))
    to_return = np.full(d, -1, dtype=np.int64)
    for col, o in enumerate(out):
        to_return[o] = col

    P, sd = params

    # start each coordinate at the median of its *marginal* truncated normal
    # (LTFHPlus's init); fixed coords collapse to their pinned value.
    sd0 = np.sqrt(np.diag(cov))
    x = _init_chain(lower, upper, sd0)
    directions, dual = _mixing_directions(P, sd, sd0, fixed)

    # Unconditional: seed=None reseeds from entropy (see _seed_rng), so an
    # unseeded call never replays a previous seeded call's leftover stream.
    _seed_rng(seed)
    res = np.empty((n_sim, len(out)), dtype=np.float64)
    _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, n_sim,
                 burn_in, res, directions, dual)
    return res
