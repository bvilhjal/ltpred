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

import threading

import numpy as np

from ._numba import _jit, _jit_parallel, prange
from ._mathfun import _norm_cdf, _norm_ppf

__all__ = ["rtmvnorm_gibbs", "gibbs_params", "gibbs_estimate_batched",
           "gibbs_advance"]

# U(Fa, Fb) draws are clamped this far off {0, 1} before the inverse-CDF step so
# a boundary draw (np.random.random() can return exactly 0.0) cannot map to an
# infinite liability. Statistically negligible; the R/Rcpp code omits it only
# because R's runif never returns its endpoints.
_U_EPS = 1e-15

# Numba's ``np.random`` state is tied to worker threads, so seeding the calling
# thread cannot make a ``prange`` kernel scheduler-independent. Generate trusted
# float64 uniforms here instead and pass them into a random-free kernel. The RNG
# is thread-local so concurrent seeded fits cannot overwrite one another.
_advance_rng_state = threading.local()
_MAX_ADVANCE_UNIFORMS = 1 << 20       # 8 MiB of float64 temporary storage
_MAX_SEED = (1 << 32) - 1


def gibbs_params(covmat):
    """Precompute the sweep's conditional-regression matrix ``P`` and SDs ``sd``.

    The Gibbs conditionals come from the **precision** matrix ``Q = Sigma^-1``:
    ``sd[j] = sqrt(1 / Q[j,j])`` (conditional SD) and ``P[i,j] = -Q[i,j] / Q[j,j]``
    (0 on the diagonal), so ``mu_j = sum_i P[i,j] x_i`` is the conditional mean.
    This is one ``O(d^3)`` inverse instead of ``d`` size-``(d-1)`` solves
    (``O(d^4)``) -- a few-fold speed-up that matters for multi-trait / large
    pedigrees. Computed once and reused across the convergence loop in
    :mod:`ltpred.estimate`. Mathematically identical to the conditional-regression
    form; ``Sigma`` is strictly PD (see :func:`correct_positive_definite`)."""
    cov = np.ascontiguousarray(covmat, dtype=np.float64)
    d = cov.shape[0]
    if cov.shape != (d, d):
        raise ValueError("covmat must be square")
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
                            batch_size, n_batch, seeds, total_sum, bm_sum, bm_sumsq):
    """Sample many families in parallel, accumulating means online (no sample store).

    All families share the conditional-regression factorisation ``(P, sd)`` (they
    have the same covariance structure); only their truncation bounds differ. Each
    ``prange`` iteration runs one family's ``burn_in + n_sim`` sweeps and writes,
    per output coordinate, the running sum ``total_sum[f]`` (for the mean) and two
    **batch-mean summaries** -- ``bm_sum[f]`` (sum of the ``n_batch`` batch means)
    and ``bm_sumsq[f]`` (sum of their squares) -- from which the batch-means
    Monte-Carlo SE is reconstructed without storing the batch means themselves.
    Because each family seeds its own RNG (``seeds[f]``) at the top of the
    iteration, results are deterministic regardless of how ``prange`` maps families
    to threads. Streaming these summaries (instead of the ``(ncols, n_batch)`` array)
    keeps the SE memory at ``O(ncols)`` per family."""
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
        s1 = np.zeros(ncols)          # sum of batch means Y_k
        s2 = np.zeros(ncols)          # sum of Y_k^2
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
                            y = batch_sum[c] / batch_size
                            s1[c] += y
                            s2[c] += y * y
                            batch_sum[c] = 0.0
                        bidx += 1
                        in_batch = 0

        for c in range(ncols):
            total_sum[f, c] = tot[c]
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
    """Thin wrapper over the parallel kernel; returns ``(total_sum, bm_sum, bm_sumsq)``.

    ``total_sum[f, c]`` is the sum of ``n_sim`` post-burn-in draws (divide by
    ``n_sim`` for the posterior mean); ``bm_sum`` / ``bm_sumsq`` are the sum and
    sum-of-squares of the ``n_batch`` batch means, from which the batch-means SE is
    formed: with ``M`` batches of size ``b``, ``se = sqrt(b * (bm_sumsq - bm_sum^2/M)
    / (M-1) / N)``. ``lowers``/``uppers`` may be float32 to halve their memory."""
    F = lowers.shape[0]
    ncols = out_idx.shape[0]
    total_sum = np.zeros((F, ncols), dtype=np.float64)
    bm_sum = np.zeros((F, ncols), dtype=np.float64)
    bm_sumsq = np.zeros((F, ncols), dtype=np.float64)
    _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                            np.ascontiguousarray(sd0),
                            as_bounds(lowers), as_bounds(uppers),
                            np.asarray(out_idx, dtype=np.int64),
                            int(n_sim), int(burn_in), int(batch_size),
                            int(n_batch), np.asarray(seeds, dtype=np.int64),
                            total_sum, bm_sum, bm_sumsq)
    return total_sum, bm_sum, bm_sumsq


@_jit_parallel
def _gibbs_advance(P, sd, lowers, uppers, fixed, x, uniforms):
    """Parallel random-free kernel for one bounded block of uniforms."""
    F = x.shape[0]
    d = x.shape[1]
    for f in prange(F):
        for sweep in range(uniforms.shape[1]):
            for j in range(d):
                if not fixed[f, j]:
                    mu_j = 0.0
                    for i in range(d):
                        mu_j += P[i, j] * x[f, i]
                    sd_j = sd[j]
                    fa = _norm_cdf((lowers[f, j] - mu_j) / sd_j)
                    fb = _norm_cdf((uppers[f, j] - mu_j) / sd_j)
                    u = fa + uniforms[f, sweep, j] * (fb - fa)
                    if u < _U_EPS:
                        u = _U_EPS
                    elif u > 1.0 - _U_EPS:
                        u = 1.0 - _U_EPS
                    x[f, j] = mu_j + sd_j * _norm_ppf(u)


@_jit
def _seed_numba_rng(seed):
    """Seed serial jitted samplers (or NumPy in the pure-Python fallback)."""
    np.random.seed(seed)


def _seed_rng(seed):
    """Seed serial and parallel sampler streams for reproducibility."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer in [0, {_MAX_SEED}]")
    seed = int(seed)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")

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
    held. Parallel over families; seed once beforehand with :func:`_seed_rng`.

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
        Zero-based coordinate indices to return (0 = genetic, 1 = proband full
        liability in the family-model ordering). Duplicates dropped, order sorted.
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
        _seed_rng(seed)

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
