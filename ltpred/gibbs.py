"""Gibbs sampler for the truncated multivariate normal (TMVN).

Port of LTFHPlus's ``rtmvnorm.gibbs`` and its Rcpp inner loop. With
``Q = Sigma^-1``, `gibbs_params` forms ``P[i, j] = -Q[i, j] / Q[j, j]``
(``P[j, j] = 0``) and ``sd[j] = sqrt(1 / Q[j, j])``; the full conditional of
coordinate ``j`` is ``N(P[:, j] . x, sd[j]^2)`` restricted to
``(lower[j], upper[j])``. ``Sigma`` must be finite, symmetric and strictly
positive definite. A coordinate whose interval spans less than ``1e-8`` is
*fixed* (a pinned point mass, e.g. an onset-pinned case) and never resampled.

Algorithm G (methods report, "Computing the truncated mean"), as implemented.

1. **G1 group.** Families with one role set share ``Sigma``;
   `ltpred.estimate._group_by_structure` buckets them and
   `ltpred.estimate._estimate_group` forms ``(P, sd)`` once per group.
2. **G2 collapse.** `gibbs_estimate_batched` integrates out every coordinate
   that is ``(-inf, inf)`` in *all* families passed to that call (always the
   genetic rows on the public path; also ``o`` or a relative when
   unobserved throughout). The chain runs on the kept block ``y`` with
   ``(P, sd)`` of its marginal covariance, recomputed per call. A collapsed
   output streams ``E[z | y] = W y`` and adds the constant ``Var(z | y)`` to
   its sum of squares (`_blup_from_keep`), so the reported variance is
   ``Var(E[z | y]) + Var(z | y)`` (Rao--Blackwell). With nothing kept the
   prior (mean 0, variance ``Sigma[j, j]``, SE 0) is returned unsampled.
   `rtmvnorm_gibbs` never collapses.
3. **G3 initialise.** `_init_chain` starts each coordinate at the median of
   its *marginal* truncated normal (SD ``sqrt(Sigma[j, j])``): a feasible
   point of the rectangle. Fixed coordinates stay there.
4. **G4 sweep.** Free coordinates are redrawn in covariance order by
   `_gibbs_conditional_draw`, ``x_j = mu_j + sd_j * Phi^-1(U(Phi(a), Phi(b)))``
   (Kotecha & Djuric 1999), one uniform each, on the survival scale when
   ``a >= 0``. Where even that underflows (``|z|`` beyond about 38.5)
   `_far_right_std_tnorm_quantile` inverts the leading-order Gaussian tail
   ratio on the log scale (a Rayleigh construction).
5. **G5 stream.** After ``burn_in`` sweeps, each of ``n_sim`` sweeps adds the
   outputs to running sums and sums of squares; the first
   ``n_batch * batch_size`` of them also form batch means, of which only the
   sum and sum of squares are kept. No draw array: ``O(ncols)`` per family.
6. **G6 stop.** `ltpred.estimate._estimate_group` reruns the unconverged
   families in rounds, pooling those sums, until every requested batch-means
   SE is ``<= tol``; converged families leave the active set.

Seeding. The batched kernels reseed the kernel RNG with ``seeds[f]`` at the
top of family ``f``'s ``prange`` iteration (``-1``: unseeded), so a family's
draws do not depend on thread scheduling; seed derivation is in
`ltpred.estimate._base_seeds`. `gibbs_advance` (persistent chains for
`ltpred.fit`) instead draws every uniform from a thread-local
``numpy.random.Generator`` before entering ``prange``. `rtmvnorm_gibbs`
seeds the process-global RNG.
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
# thread cannot make a ``prange`` kernel scheduler-independent. For
# `gibbs_advance`, generate float64 uniforms here instead and pass them into a
# random-free kernel (the estimator kernels reseed per family instead). The
# RNG is thread-local so concurrent seeded fits cannot overwrite one another.
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
    NaN.  Work on the log scale instead, via the leading-order Gaussian tail
    ratio ``S(a + t) / S(a) ~ exp(-a t - t^2 / 2)`` (it drops the factor
    ``a / (a + t)``, so the draw is approximate, with error vanishing as ``a``
    grows).  That inverts in closed form,
    ``t = -a + sqrt(a^2 - 2 log(1 - u'))``, which is the exponential/Rayleigh
    tail sampler underlying Devroye's and Robert's rejection schemes.  ``u'``
    rescales ``u`` by the interval's share of the tail so a finite ``b`` is
    honoured; the result is clamped to ``[a, b]``.  This mirrors the PA
    engine's ``_far_right_std_tnorm_moments`` (methods report, Algorithm G,
    step G4).
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
    `_far_right_std_tnorm_quantile`, mirrored for the left tail.
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
    """Precompute the sweep's conditional-regression matrix ``P`` and SDs ``sd``.

    The Gibbs conditionals come from the **precision** matrix ``Q = Sigma^-1``:
    ``sd[j] = sqrt(1 / Q[j,j])`` (conditional SD) and ``P[i,j] = -Q[i,j] / Q[j,j]``
    (0 on the diagonal), so ``mu_j = sum_i P[i,j] x_i`` is the conditional mean.
    This is one ``O(d^3)`` inverse instead of ``d`` size-``(d-1)`` solves
    (``O(d^4)``) -- a few-fold speed-up that matters for multi-trait / large
    pedigrees. Computed once and reused across the convergence loop in
    `ltpred.estimate`. Mathematically identical to the conditional-regression
    form; ``Sigma`` is strictly PD (see `correct_positive_definite`).

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


def _group_unbounded_mask(lowers, uppers):
    """Coordinates with ``(-inf, inf)`` bounds in every family of the group.

    The collapse set of step G2. It is a property of the families passed
    together, so a different subset (a later round, a chunk) can collapse a
    different set of coordinates."""
    lo = np.asarray(lowers)
    hi = np.asarray(uppers)
    return (~np.isfinite(lo)).all(axis=0) & (~np.isfinite(hi)).all(axis=0)


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
    """Sample many families, accumulating summaries online (steps G3--G5).

    All families share ``(P, sd)``; only their bounds differ. Each ``prange``
    iteration reseeds the kernel RNG with ``seeds[f]`` (unless ``-1``), starts a
    fresh chain at `_init_chain`, runs ``burn_in + n_sim`` sweeps and writes per
    output coordinate: ``total_sum[f]`` and ``total_sumsq[f]`` over the ``n_sim``
    retained draws (posterior mean and variance of the TMVN itself), and
    ``bm_sum[f]`` / ``bm_sumsq[f]``, the sum and sum of squares of the means of
    the first ``n_batch`` consecutive batches of ``batch_size`` draws (the
    batch-means Monte-Carlo SE). Draws past ``n_batch * batch_size`` enter the
    totals only. State is ``O(ncols)`` per family; the family loop is parallel
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


def as_bounds(a):
    """Contiguous float array for a kernel: **keep float32** (to halve the memory
    of large per-family bound arrays), otherwise coerce to float64. The kernels'
    internal state and accumulators stay float64 regardless, so only the big
    ``(F, d)`` inputs shrink; the covariance / conditional-regression factors are
    unaffected."""
    a = np.ascontiguousarray(a)
    return a if a.dtype == np.float32 else np.ascontiguousarray(a, dtype=np.float64)


def gibbs_estimate_batched(P, sd, sd0, lowers, uppers, out_idx, n_sim, burn_in,
                           batch_size, n_batch, seeds, cov=None, collapse=None):
    """Run one round of the batched sampler for same-covariance families.

    Returns ``(total_sum, total_sumsq, bm_sum, bm_sumsq)``, each ``(F, ncols)``.
    ``total_sum[f, c]`` is the sum of ``N = n_sim`` post-burn-in values (divide
    by ``N`` for the posterior mean) and ``total_sumsq[f, c]`` the sum of their
    squares, so the **posterior variance** is
    ``total_sumsq / N - (total_sum / N)^2``. ``bm_sum`` / ``bm_sumsq`` are the
    sum and sum of squares of the ``M = n_batch`` batch means of size
    ``b = batch_size`` (``M * b <= N``), giving the batch-means SE
    ``se = sqrt(b * (bm_sumsq - bm_sum^2 / M) / (M - 1) / (M * b))``: the
    denominator counts the batched draws only, as in R ``batchmeans``. The
    posterior variance is a property of the TMVN and does not shrink with more
    draws; the SE is the sampler's own error and does. ``seeds[f]`` seeds
    family ``f`` (``-1``: unseeded). ``lowers``/``uppers`` may be float32.

    When ``collapse`` is true (the default whenever ``cov`` is supplied),
    coordinates that are ``(-inf, inf)`` in every family of this call are
    integrated out of the sweep (methods report, Algorithm G, step G2). A
    collapsed output streams ``E[z | y]`` and its sum of squares includes the
    constant ``Var(z | y)``, so ``var`` is still ``Var(target | C_F)``, while
    its SE is that of the Rao--Blackwellised mean. ``P``/``sd`` are then
    unused: the kept block's are recomputed from ``cov``. If every coordinate
    is collapsed, no sampling happens (sums 0, ``total_sumsq = N * cov[j, j]``).
    Pass ``collapse=False`` to force the full chain. ``rtmvnorm_gibbs`` is
    never collapsed.
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
        unbounded = _group_unbounded_mask(lowers, uppers)
        keep_idx = np.flatnonzero(~unbounded)
        coll_idx = np.flatnonzero(unbounded)
        if keep_idx.size == 0:
            for c, j in enumerate(out_idx):
                total_sumsq[:, c] = n_sim * cov[int(j), int(j)]
            return total_sum, total_sumsq, bm_sum, bm_sumsq
        if coll_idx.size > 0:
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
            lo_k = as_bounds(np.ascontiguousarray(
                np.asarray(lowers)[:, keep_idx]))
            hi_k = as_bounds(np.ascontiguousarray(
                np.asarray(uppers)[:, keep_idx]))
            P_k, sd_k = gibbs_params(cov[np.ix_(keep_idx, keep_idx)])
            sd0_k = np.sqrt(np.diag(cov)[keep_idx])
            _gibbs_estimate_batched_collapsed(
                P_k, sd_k, np.ascontiguousarray(sd0_k),
                lo_k, hi_k, out_kind, out_local, W, cond_var,
                n_sim, burn_in, batch_size, n_batch, seeds,
                total_sum, total_sumsq, bm_sum, bm_sumsq)
            return total_sum, total_sumsq, bm_sum, bm_sumsq

    _gibbs_estimate_batched(np.ascontiguousarray(P), np.ascontiguousarray(sd),
                            np.ascontiguousarray(sd0),
                            as_bounds(lowers), as_bounds(uppers),
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

    Unlike `gibbs_estimate_batched` (which runs independent short chains and
    returns their means), this keeps a **persistent** state ``x`` (``F x d``) that
    the caller carries across outer iterations — the data-augmentation step of a
    variance-component fit (`ltpred.fit`), where the covariance (hence ``P`` /
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
    """Draw truncated-MVN samples of the coordinates listed in ``out``.

    Parameters
    ----------
    covmat : (d, d) array
        Symmetric covariance of the (untruncated) multivariate normal.
    lower, upper : float or (d,) array
        Per-coordinate truncation bounds; scalars are broadcast. Every ``upper``
        must be greater than or equal to its corresponding ``lower``; reversed
        bounds raise `ValueError`.
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
        Seeding mutates the process-global NumPy RNG (the serial sweep draws from
        it), so two *concurrent* seeded calls to this low-level sampler from
        different threads can interfere with one another. The high-level
        estimators seed each family inside the parallel kernel and are safe for
        concurrent seeded use.
    params : (P, sd), optional
        Precomputed `gibbs_params` output; recomputed from ``covmat`` when
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
