"""Tetrachoric correlation: latent liability correlation from binary pairs.

Under the liability-threshold model, two binary observations (two traits, or
the case/control statuses of two relatives) are read as a 2x2 table of a
latent **bivariate normal** cut at two thresholds. The tetrachoric correlation
is the latent Pearson correlation implied by that table -- the pairwise face
of the family model used throughout this package: for relatives with additive
relationship ``A`` and liability-scale heritability ``h2`` the expected
tetrachoric correlation under an A+E model is ``h2 * A`` (Falconer's route:
parent-offspring pairs give ``h2 / 2``, so ``h2 ~ 2 * tetrachoric``; full sibs
also share any sibship ``c2``).

The estimator is the maximum-likelihood tetrachoric (Kirk 1973; Tallis 1962):
thresholds ``t = Phi^-1(1 - p)`` from the marginal case proportions. The 2x2
model is saturated, so with plug-in thresholds matching both margins the MLE
is the unique root of ``p_both(rho) = a / n`` (``dp_both/drho`` is the
bivariate density, hence positive), found by Brent's method on
``(-1 + 1e-9, 1 - 1e-9)`` with cell probabilities from Plackett integration
(`ltpred.pairwise_multi._bivariate_probabilities`, the same engine the
pairwise fitters use). The standard error is a Wald SE from the closed-form
observed information for ``rho`` alone,
``-ll'' = -sum n (p''/p - (p'/p)^2)``, conditional on the thresholds. The
quick cosine approximation
(``cos(pi / (1 + sqrt(a*d / (b*c))))``-style estimators) is not used -- the
MLE is cheap enough.

Use it as a diagnostic and cross-check: estimate liability correlations
directly from relative pairs' statuses and compare with the model's
``h2 * A`` expectation, or sanity-check a fitted ``h2`` against the pairwise
tetrachorics in the same data (``benchmarks/bench_tetrachoric.py``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from ._validation import validate_binary
from .pairwise_multi import _bivariate_probabilities

__all__ = ["TetrachoricResult", "tetrachoric", "tetrachoric_table",
           "tetrachoric_matrix"]

_RHO_EPS = 1e-9


@dataclass
class TetrachoricResult:
    """One tetrachoric estimate.

    ``rho`` the MLE; ``se`` the observed-information standard error,
    conditional on the estimated marginal thresholds and Wald-shaped, so it is
    optimistic near the ``|rho| -> 1`` boundary and in small samples;
    ``thresholds`` the two marginal thresholds; ``n`` the observed pair count
    (the raw table total, never inflated by the continuity correction);
    ``corrected`` whether the 0.5 continuity correction was applied to a
    zero cell."""
    rho: float
    se: float
    thresholds: tuple
    n: int
    corrected: bool


def _plackett(t1, t2, rho):
    """Plackett cell probabilities, or ``None`` where that engine objects.

    The pairwise engine raises when a cell underflows to exactly zero and
    its quadrature warns when it cannot certify its tolerance (both happen
    at extreme thresholds/correlations the root bracket visits). Either way
    the caller falls back to SciPy's Genz CDF for that evaluation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _bivariate_probabilities(float(t1), float(t2), float(rho))
        except FloatingPointError:
            return None
    if any(w.category.__name__ == "IntegrationWarning" for w in caught):
        return None
    for w in caught:                       # unrelated warnings still surface
        warnings.showwarning(w.message, w.category, w.filename, w.lineno)
    return result


def _bvn_cdf(a, b, rho):
    """P(X < a, Y < b) for standard bivariate normal with correlation rho.

    The ``neither`` cell of the Plackett probabilities in
    `ltpred.pairwise_multi`; the rho -> +-1 limits and infinite bounds are
    handled here (the Plackett engine needs finite thresholds, ``|rho| < 1``)."""
    if a == -math.inf or b == -math.inf:
        return 0.0
    if a == math.inf:
        return float(ndtr(b))
    if b == math.inf:
        return float(ndtr(a))
    if rho >= 1.0 - 1e-12:
        return float(ndtr(min(a, b)))
    if rho <= -1.0 + 1e-12:
        return max(0.0, float(ndtr(a) - ndtr(-b)))
    cells = _plackett(a, b, rho)
    if cells is not None:
        return float(cells[0][3])
    return float(multivariate_normal.cdf(
        [a, b], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]],
        maxpts=1_000_000, abseps=1e-10, releps=1e-10))


def tetrachoric_table(a: float, b: float, c: float, d: float, *,
                      continuity_correction: bool = True) -> TetrachoricResult:
    """Tetrachoric MLE from a 2x2 table of counts.

    The table reads ``a`` = both cases, ``b`` = first case only, ``c`` =
    second case only, ``d`` = neither. Cell counts must be finite,
    nonnegative integers; fractional or non-finite counts are rejected. A
    zero cell gets a 0.5 continuity correction added to all four cells when
    ``continuity_correction`` (the standard handling; the estimate is then
    boundary-avoiding rather than exactly +/-1); with
    ``continuity_correction=False`` a zero cell raises. Monomorphic margins (one
    variable all-one-level) are rejected: there is no correlation information.
    ``se`` is NaN when the numeric curvature at the optimum is not positive.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) < 0:
        raise ValueError("cell counts must be nonnegative")
    # The finiteness gate precedes the integer gate: math.floor(inf) raises
    # OverflowError and math.floor(nan) a bare-messaged ValueError.
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise ValueError("cell counts must be finite")
    # Counts are data, so fractional cells are rejected rather than fitted.
    # (Integral floats such as 5.0 pass: they are the same number.)
    if any(v != math.floor(v) for v in (a, b, c, d)):
        raise ValueError("cell counts must be nonnegative integers")
    n_raw = a + b + c + d
    if n_raw == 0:
        raise ValueError("empty 2x2 table")
    # monomorphism is structural (no correlation information); it is checked
    # on the raw table -- the continuity correction must not mask it
    p1_raw = (a + b) / n_raw
    p2_raw = (a + c) / n_raw
    if not (0.0 < p1_raw < 1.0) or not (0.0 < p2_raw < 1.0):
        raise ValueError("monomorphic variable: a tetrachoric correlation "
                         "needs both levels of both variables present")
    n_pairs = int(round(n_raw))
    corrected = False
    if min(a, b, c, d) == 0.0:
        if not continuity_correction:
            raise ValueError("zero cell in the 2x2 table (enable "
                             "continuity_correction or supply more data)")
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        corrected = True
    n = a + b + c + d
    p1 = (a + b) / n
    p2 = (a + c) / n

    t1 = float(-ndtri(p1))   # Phi^-1(1-p) == -Phi^-1(p), no 1-p cancellation
    t2 = float(-ndtri(p2))

    # Saturated model: the MLE matches the one free cell. p_both is strictly
    # increasing in rho (derivative = bivariate density > 0) and the valid
    # table puts a/n between its Frechet endpoints, so the root is bracketed;
    # the endpoint fallbacks only cover last-bit bracket failures.
    target = a / n
    lo, hi = -1.0 + _RHO_EPS, 1.0 - _RHO_EPS
    k1, c2 = float(ndtr(-t1)), float(ndtr(t2))

    def gap(rho):
        # p_both = P(X > t1) - P(X > t1, Y < t2) = k1 - (c2 - below); routed
        # through _bvn_cdf so the +-1 bracket ends use its limit handling.
        return k1 - c2 + _bvn_cdf(t1, t2, rho) - target

    if gap(lo) >= 0.0:
        rho = lo
    elif gap(hi) <= 0.0:
        rho = hi
    else:
        rho = brentq(gap, lo, hi, xtol=1e-13, rtol=1e-13)

    # Closed-form observed information: -ll'' = -sum n (p''/p - (p'/p)^2).
    counts = np.array([a, b, c, d])
    cells = _plackett(t1, t2, float(rho))
    if cells is not None:
        prob, first, second = cells
        info = -float(np.sum(counts * (second / prob - (first / prob) ** 2)))
    else:
        # Extreme corner (see _plackett): numeric Hessian of the log-likelihood.
        def nll(r):
            below = _bvn_cdf(t1, t2, r)
            cells = (k1 - c2 + below, c2 - below,
                     float(ndtr(t1)) - below, below)
            return -sum(nij * math.log(max(pij, 1e-300))
                        for nij, pij in zip(counts, cells) if nij)
        delta = 1e-4
        info = (nll(rho + delta) - 2.0 * nll(rho)
                + nll(rho - delta)) / (delta * delta)
    se = math.sqrt(1.0 / info) if info > 0 else math.nan
    return TetrachoricResult(rho=float(rho), se=float(se),
                             thresholds=(t1, t2), n=n_pairs, corrected=corrected)


def tetrachoric(x: ArrayLike, y: ArrayLike, *,
                continuity_correction: bool = True) -> TetrachoricResult:
    """Tetrachoric MLE between two binary arrays of equal length.

    ``x`` and ``y`` are 1-D boolean or 0/1 arrays (NaN or other codes raise),
    one entry per pair. They are tabulated with ``x`` as the first variable,
    so ``x``-only cases form cell ``b``, and fitted by `tetrachoric_table`,
    which sets the zero-cell handling (``continuity_correction``) and rejects a
    monomorphic array.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if x.size == 0:
        raise ValueError("need at least one pair")
    xb = validate_binary(x, name="x", ndim=1)
    yb = validate_binary(y, name="y", ndim=1)
    a = float(np.sum(xb & yb))
    b = float(np.sum(xb & ~yb))
    c = float(np.sum(~xb & yb))
    d = float(np.sum(~xb & ~yb))
    return tetrachoric_table(a, b, c, d,
                             continuity_correction=continuity_correction)


def tetrachoric_matrix(X: ArrayLike, *, continuity_correction: bool = True,
                       check_psd: bool = True) -> np.ndarray:
    """Pairwise tetrachoric correlations among the columns of ``X``.

    ``X`` is an ``(n, m)`` boolean or 0/1 array, one row per unit (e.g. family)
    and one column per variable; returns the symmetric ``(m, m)`` matrix of
    pairwise estimates with unit diagonal. Each pair of
    columns is fitted by `tetrachoric_table`, so ``continuity_correction`` and
    the monomorphic-column error behave as there. Because the off-diagonal
    entries are fitted independently, the result is **not guaranteed to be a
    positive-semidefinite correlation matrix**. With ``check_psd=True`` (the
    default), a materially negative eigenvalue emits a warning; do not pass such
    a result to a covariance model without a scientifically justified joint fit
    or projection.
    """
    if not isinstance(check_psd, (bool, np.bool_)):
        raise TypeError("check_psd must be bool")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D (pairs, variables) array")
    Xb = validate_binary(X, name="X", ndim=2)
    m = Xb.shape[1]
    R = np.eye(m)
    Xi = Xb.astype(np.float64)
    n_a = Xi.T @ Xi                      # both 1
    col = Xi.sum(0)[:, None]             # per-variable case count
    n_b = col - n_a                      # first-only
    n_c = n_b.T                          # second-only
    n_d = Xb.shape[0] - (n_a + n_b + n_c)
    for i in range(m):
        for j in range(i + 1, m):
            r = tetrachoric_table(n_a[i, j], n_b[i, j], n_c[i, j], n_d[i, j],
                                  continuity_correction=continuity_correction)
            R[i, j] = R[j, i] = r.rho
    min_eig = float(np.linalg.eigvalsh(R).min()) if m else 0.0
    if check_psd and min_eig < -1e-8:
        warnings.warn(
            "pairwise tetrachoric estimates are not positive-semidefinite "
            f"(minimum eigenvalue {min_eig:.3g}); do not use this matrix as a "
            "covariance without a justified joint fit or projection",
            RuntimeWarning, stacklevel=2)
    return R
