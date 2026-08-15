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

In the family-model estimators the coordinates are ordered ``g`` (genetic
liability), ``o`` (proband full liability), then one per relative; ``out`` picks
which posterior samples to return (0 = genetic, 1 = full). Coordinates whose
bounds coincide are ``fixed`` (for example an onset-pinned case in LT-FH++, or
in family-free ADuLT) and are held constant rather than resampled.
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

__all__ = ["rtmvnorm_gibbs", "gibbs_params", "gibbs_estimate_batched"]

# Numba's ``np.random`` state is tied to worker threads, so seeding the calling
# thread cannot make a ``prange`` kernel scheduler-independent. Generate trusted
# float64 uniforms here instead and pass them into a random-free kernel. The RNG
# is thread-local so concurrent seeded fits cannot overwrite one another.
_advance_rng_state = threading.local()
_MAX_ADVANCE_UNIFORMS = 1 << 20       # 8 MiB of float64 temporary storage
_MAX_SEED = (1 << 32) - 1
# Coordinates whose bounds span less than this are held fixed (a pinned point
# mass, e.g. an onset-pinned case in LT-FH++) rather than resampled.
_FIXED_TOL = 1e-8


@_jit
def _far_right_std_tnorm_quantile(a, b, u):
    """Quantile on ``[a, b]`` with ``a`` so far right that ``Phi(-a)`` underflows.

    Beyond ``a ~ 38.5`` the survival probability is below the smallest positive
    double, so *every* quantity on the probability scale is exactly 0 and the
    ordinary route returns ``+inf`` -- which then poisons the whole sweep with
    NaN.  Work on the log scale instead, via the Gaussian tail ratio
    ``S(a + t) / S(a) ~ exp(-a t - t^2 / 2)``.  That inverts in closed form,
    ``t = -a + sqrt(a^2 - 2 log(1 - u'))``, which is the exponential/Rayleigh
    tail sampler underlying Devroye's and Robert's rejection schemes.  ``u'``
    rescales ``u`` by the interval's share of the tail so a finite ``b`` is
    honoured.  This mirrors the PA engine's ``_far_right_std_tnorm_moments``.
    """
    span = b - a
    if span <= 0.0:
        return a
    if np.isinf(span):
        share = 1.0
    else:
        share = 1.0 - np.exp(-a * span - 0.5 * span * span)
    p = u * share
    if p > 1.0 - 1e-16:
        p = 1.0 - 1e-16
    t = -a + np.sqrt(a * a - 2.0 * np.log(1.0 - p))
    x = a + t
    if x < a:
        return a
    if x > b:
        return b
    return x


@_jit
def _std_tnorm_quantile(a, b, u):
    """Quantile of ``N(0, 1)`` truncated to ``[a, b]``.

    In a positive tail, interpolating between ``Phi(a)`` and ``Phi(b)`` loses
    the interval when both CDFs round to one.  Work on survival probabilities
    instead and use normal symmetry, ``isf(q) = -ppf(q)``.  Lower-tail and
    central intervals are safe on the ordinary CDF scale.  The final clamp only
    guards the last-bit error of the inverse approximation; it never moves a
    draw across a truncation boundary.

    Past ``|z| ~ 38.5`` even the survival scale underflows to exactly 0 and the
    inverse returns an infinity that no clamp can catch (the opposite bound is
    typically infinite too).  Those intervals hand off to
    :func:`_far_right_std_tnorm_quantile`, mirrored for the left tail.
    """
    if a >= 0.0:
        sa = _norm_cdf(-a)
        if sa <= 0.0:                       # right tail underflowed to zero
            return _far_right_std_tnorm_quantile(a, b, u)
        sb = _norm_cdf(-b)
        q = (1.0 - u) * sa + u * sb
        x = -_norm_ppf(q)
        if not np.isfinite(x):
            return _far_right_std_tnorm_quantile(a, b, u)
    else:
        fb = _norm_cdf(b)
        if fb <= 0.0:                       # far left tail: reflect to the right
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
    """Initial chain state: each coordinate's marginal truncated median.

    LTFHPlus's init -- the median of the *marginal* truncated normal (the bounds
    standardised by the marginal SD ``sd0``), so fixed coordinates collapse to
    their pinned value; a non-finite quantile falls back to 0. Works on one
    chain's ``(d,)`` arrays, so the fit paths can share it with a unit ``sd0``."""
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
    """Precompute the sweep's conditional-regression matrix ``P`` and SDs ``sd``.

    The Gibbs conditionals come from the **precision** matrix ``Q = Sigma^-1``:
    ``sd[j] = sqrt(1 / Q[j,j])`` (conditional SD) and ``P[i,j] = -Q[i,j] / Q[j,j]``
    (0 on the diagonal), so ``mu_j = sum_i P[i,j] x_i`` is the conditional mean.
    This is one ``O(d^3)`` inverse instead of ``d`` size-``(d-1)`` solves
    (``O(d^4)``) -- a few-fold speed-up that matters for multi-trait / large
    pedigrees. Computed once and reused across the convergence loop in
    :mod:`ltpred.estimate`. Mathematically identical to the conditional-regression
    form; ``Sigma`` is strictly PD (see :func:`correct_positive_definite`).

    ``covmat`` must be finite, symmetric to ``1e-10`` relative to its largest
    absolute entry, and
    strictly positive-definite (smallest eigenvalue greater than ``1e-12``
    times the covariance's spectral scale);
    a merely invertible but indefinite covariance would otherwise silently
    yield NaN conditional SDs and garbage draws, so it raises ``ValueError``.
    """
    cov = _validate_covmat(covmat)
    Q = np.linalg.inv(cov)
    qdiag = np.diag(Q).copy()
    P = -Q / qdiag[np.newaxis, :]              # P[i,j] = -Q[i,j] / Q[j,j]
    np.fill_diagonal(P, 0.0)
    sd = np.sqrt(1.0 / qdiag)
    return np.ascontiguousarray(P), np.ascontiguousarray(sd)


@_jit
def _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, n_sim, burn_in, res):
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
            if k >= 0 and to_return[j] >= 0:
                res[k, to_return[j]] = x[j]
    return res


@_jit_parallel
def _gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                            batch_size, n_batch, seeds, total_sum, total_sumsq,
                            bm_sum, bm_sumsq):
    """Sample many families, accumulating means online (no sample store).

    All families share the conditional-regression factorisation ``(P, sd)`` (they
    have the same covariance structure); only their truncation bounds differ. Each
    ``prange`` iteration runs one family's ``burn_in + n_sim`` sweeps and writes,
    per output coordinate, the running sum ``total_sum[f]`` (for the mean), the
    running sum of squares ``total_sumsq[f]`` (for the **posterior** variance of
    the target -- the spread of the truncated-MVN itself, not the sampler's error)
    and two
    **batch-mean summaries** -- ``bm_sum[f]`` (sum of the ``n_batch`` batch means)
    and ``bm_sumsq[f]`` (sum of their squares) -- from which the batch-means
    Monte-Carlo SE is reconstructed without storing the batch means themselves.
    Because each family seeds its own RNG (``seeds[f]``) at the top of the
    iteration, results are deterministic regardless of how ``prange`` maps families
    to threads. Streaming these summaries (instead of the ``(ncols, n_batch)`` array)
    keeps the SE memory at ``O(ncols)`` per family. The family loop is parallel
    when Numba is installed and serial in the pure-Python fallback."""
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


def as_bounds(a):
    """Contiguous float array for a kernel: **keep float32** (to halve the memory
    of large per-family bound arrays), otherwise coerce to float64. The kernels'
    internal state and accumulators stay float64 regardless, so only the big
    ``(F, d)`` inputs shrink; the covariance / conditional-regression factors are
    unaffected."""
    a = np.ascontiguousarray(a)
    return a if a.dtype == np.float32 else np.ascontiguousarray(a, dtype=np.float64)


def gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                           batch_size, n_batch, seeds):
    """Thin wrapper over the batched kernel.

    Returns ``(total_sum, total_sumsq, bm_sum, bm_sumsq)``.
    ``total_sum[f, c]`` is the sum of ``n_sim`` post-burn-in draws (divide by
    ``n_sim`` for the posterior mean) and ``total_sumsq[f, c]`` the sum of their
    squares, from which the **posterior variance** follows as
    ``total_sumsq / N - (total_sum / N)^2``. ``bm_sum`` / ``bm_sumsq`` are the sum and
    sum-of-squares of the ``n_batch`` batch means, from which the batch-means SE is
    formed: with ``M`` batches of size ``b``, ``se = sqrt(b * (bm_sumsq - bm_sum^2/M)
    / (M-1) / N)``. The two are different quantities: the posterior variance is a
    property of the truncated MVN and does not shrink with more draws, while the
    batch-means SE is the sampler's own error and does.
    ``lowers``/``uppers`` may be float32 to halve their memory."""
    F = lowers.shape[0]
    ncols = out_idx.shape[0]
    total_sum = np.zeros((F, ncols), dtype=np.float64)
    total_sumsq = np.zeros((F, ncols), dtype=np.float64)
    bm_sum = np.zeros((F, ncols), dtype=np.float64)
    bm_sumsq = np.zeros((F, ncols), dtype=np.float64)
    _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                            np.ascontiguousarray(sd0),
                            as_bounds(lowers), as_bounds(uppers),
                            np.asarray(out_idx, dtype=np.int64),
                            int(n_sim), int(burn_in), int(batch_size),
                            int(n_batch), np.asarray(seeds, dtype=np.int64),
                            total_sum, total_sumsq, bm_sum, bm_sumsq)
    return total_sum, total_sumsq, bm_sum, bm_sumsq


@_jit_parallel
def _gibbs_advance(P, sd, lowers, uppers, fixed, x, uniforms):
    """Random-free kernel for one bounded block (Numba-parallel when available)."""
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
    As for :func:`_validate_seed`, ``bool`` and non-integers are rejected rather
    than truncated."""
    if isinstance(burn_in, (bool, np.bool_)) \
            or not isinstance(burn_in, (int, np.integer)):
        raise TypeError("burn_in must be a non-negative integer")
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer")
    return burn_in


def _offset_seed(seed, offset):
    """Derive a deterministic uint32 seed without overflowing its public range."""
    if seed is None:
        return None
    return (operator.index(seed) + int(offset)) % (_MAX_SEED + 1)


def _seed_rng(seed):
    """Seed serial and parallel sampler streams for reproducibility."""
    seed = _validate_seed(seed)

    # Construct first, then mutate the serial and parallel states only after all
    # validation has succeeded.
    generator = np.random.default_rng(seed)
    _seed_numba_rng(seed)
    _advance_rng_state.generator = generator


def _advance_rng():
    """Return this calling thread's parallel-sampler generator."""
    generator = getattr(_advance_rng_state, "generator", None)
    if generator is None:
        generator = np.random.default_rng()
        _advance_rng_state.generator = generator
    return generator


def gibbs_advance(P, sd, lowers, uppers, fixed, x, n_sweeps):
    """Advance many families' truncated-MVN chains in place by ``n_sweeps`` sweeps.

    Unlike :func:`gibbs_estimate_batched` (which runs independent short chains and
    returns their means), this keeps a **persistent** state ``x`` (``F x d``) that
    the caller carries across outer iterations — the data-augmentation step of a
    variance-component fit (:mod:`ltpred.fit`), where the covariance (hence ``P`` /
    ``sd``) changes between calls. ``fixed[f, j]`` coordinates (pinned cases) are
    held. The family loop is parallel when Numba is installed and serial otherwise.
    This is an internal fitting primitive; public callers should control
    reproducibility through the ``seed`` argument of the fitters rather than the
    private RNG helpers.

    Uniforms are generated before entering ``prange`` so a seeded fit is exact
    regardless of how Numba schedules families across worker threads. Generation
    is thread-local (isolating concurrent fits) and chunked to cap temporary memory.
    """
    n_sweeps = int(n_sweeps)
    n_families, d = x.shape
    if n_sweeps <= 0 or n_families == 0 or d == 0:
        return

    generator = _advance_rng()
    family_chunk = max(1, min(n_families, _MAX_ADVANCE_UNIFORMS // d))
    for start in range(0, n_families, family_chunk):
        stop = min(start + family_chunk, n_families)
        block_size = stop - start
        sweep_chunk = max(1, _MAX_ADVANCE_UNIFORMS // (block_size * d))
        for first_sweep in range(0, n_sweeps, sweep_chunk):
            this_sweeps = min(sweep_chunk, n_sweeps - first_sweep)
            uniforms = generator.random((block_size, this_sweeps, d))
            _gibbs_advance(P, sd, lowers[start:stop], uppers[start:stop],
                           fixed[start:stop], x[start:stop], uniforms)


@_jit_parallel
def _gibbs_advance_m2(P, sd, lowers, uppers, fixed, x, uniforms, out_m):
    """Random-free kernel: advance and accumulate ``sum_sweep outer(x, x)``.

    Same in-place advance as :func:`_gibbs_advance`, but after each full sweep it
    adds the current state's outer product into ``out_m`` (the caller divides by
    the sweep count). Used by moment-accumulating variance-component fits that
    need the average second moment ``E[x x']`` over the chain, not just draws."""
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
            for a in range(d):
                xa = xf[a]
                for b in range(d):
                    out_m[f, a, b] += xa * xf[b]


def gibbs_advance_moment(P, sd, lowers, uppers, fixed, x, n_sweeps):
    """Advance chains in place by ``n_sweeps`` and return the mean outer product.

    Returns ``out_m[f] = (1/n_sweeps) * sum_sweep outer(x_f, x_f)`` -- the average
    second moment over the chain, the sufficient statistic a moment/EM
    variance-component M-step needs. Far cheaper than calling
    :func:`gibbs_advance` once per draw and accumulating in Python (one RNG /
    dispatch instead of ``n_sweeps``). ``x`` is carried across calls exactly as
    for :func:`gibbs_advance`."""
    n_sweeps = int(n_sweeps)
    n_families, d = x.shape
    out_m = np.zeros((n_families, d, d))
    if n_sweeps <= 0 or n_families == 0 or d == 0:
        return out_m

    generator = _advance_rng()
    family_chunk = max(1, min(n_families, _MAX_ADVANCE_UNIFORMS // d))
    for start in range(0, n_families, family_chunk):
        stop = min(start + family_chunk, n_families)
        block_size = stop - start
        sweep_chunk = max(1, _MAX_ADVANCE_UNIFORMS // (block_size * d))
        for first_sweep in range(0, n_sweeps, sweep_chunk):
            this_sweeps = min(sweep_chunk, n_sweeps - first_sweep)
            uniforms = generator.random((block_size, this_sweeps, d))
            _gibbs_advance_m2(P, sd, lowers[start:stop], uppers[start:stop],
                              fixed[start:stop], x[start:stop], uniforms,
                              out_m[start:stop])
    out_m /= n_sweeps
    return out_m


def rtmvnorm_gibbs(covmat: ArrayLike, lower: ArrayLike = -np.inf,
                   upper: ArrayLike = np.inf, *, fixed: ArrayLike | None = None,
                   out: Sequence[int] = (0,), n_sim: int = 100_000,
                   burn_in: int = 1000, seed: int | None = None,
                   params: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> np.ndarray:
    """Draw truncated-MVN samples of the coordinates listed in ``out``.

    Parameters
    ----------
    covmat : (d, d) array
        Symmetric covariance of the (untruncated) multivariate normal.
    lower, upper : float or (d,) array
        Per-coordinate truncation bounds; scalars are broadcast. Every ``upper``
        must be greater than or equal to its corresponding ``lower``; reversed
        bounds raise :class:`ValueError`.
    fixed : (d,) bool array, optional
        Coordinates to hold constant instead of resampling. Defaults to
        ``upper - lower < 1e-8`` (a pinned point mass, e.g. an age-of-onset case).
        If a caller explicitly fixes a non-point interval, that coordinate starts
        at its marginal truncated median and remains there.
    out : sequence of int
        Zero-based coordinate indices to return (0 = genetic, 1 = proband full
        liability in the family-model ordering). Indices must be integers in
        ``[0, d)``. Duplicates are dropped and the returned order is sorted.
    n_sim, burn_in : int
        Post-burn-in draws to keep and sweeps to discard first. ``burn_in`` must
        be a non-boolean non-negative integer.
    seed : int, optional
        Non-boolean integer in ``[0, 2**32 - 1]`` for reproducibility, or ``None``.
    params : (P, sd), optional
        Precomputed :func:`gibbs_params` output; recomputed from ``covmat`` when
        omitted. The supplied ``covmat`` is still validated because it defines
        the marginal initialisation; ``params`` must have been computed from
        that same matrix.

    Returns
    -------
    (n_sim, len(out)) ndarray
        Samples, columns in the sorted order of ``out``.
    """
    cov = (np.ascontiguousarray(covmat, dtype=np.float64)
           if params is None else _validate_covmat(covmat))
    if params is None:
        # ``gibbs_params`` performs the covariance validation in this path.
        params = gibbs_params(cov)
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

    if seed is not None:
        _seed_rng(seed)

    res = np.empty((int(n_sim), len(out)), dtype=np.float64)
    _gibbs_sweep(P, sd, lower, upper, fixed, to_return, x, int(n_sim),
                 burn_in, res)
    return res
