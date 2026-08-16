"""Tetrachoric correlation: latent liability correlation from binary pairs.

Under the liability-threshold model, two binary observations (two traits, or
the case/control statuses of two relatives) are read as a 2x2 table of a
latent **bivariate normal** cut at two thresholds. The tetrachoric correlation
is the latent Pearson correlation implied by that table -- the pairwise face
of the family model used throughout this package: for relatives with additive
relationship ``A`` and liability-scale heritability ``h2`` the expected
tetrachoric correlation is ``h2 * A`` (Falconer's classic route: parent-
offspring and full sibs give ``h2 / 2``, so ``h2 ~ 2 * tetrachoric``).

The estimator is the maximum-likelihood tetrachoric (Kirk 1973; Tallis 1962):
thresholds from the marginals, then a bounded 1-D search over the correlation
maximising the 2x2 multinomial log-likelihood, with the bivariate-normal CDF
from SciPy (``multivariate_normal``, with explicit ``1e-10`` requested
integration tolerances). The pointwise standard error comes
from the observed information (a numeric Hessian of the log-likelihood). The
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
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from ._validation import validate_binary

__all__ = ["TetrachoricResult", "tetrachoric", "tetrachoric_table",
           "tetrachoric_matrix"]

_RHO_EPS = 1e-9
_CDF_TOL = 1e-10


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


def _bvn_cdf(a, b, rho):
    """P(X < a, Y < b) for standard bivariate normal with correlation rho.

    Delegates to SciPy's bivariate-normal CDF (``multivariate_normal.cdf``)
    with explicit absolute/relative tolerances of ``1e-10``. Numerical accuracy
    remains subject to SciPy's integration routine. Handles the rho -> +-1
    limits and infinite bounds itself."""
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
    return float(multivariate_normal.cdf(
        [a, b], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]],
        maxpts=1_000_000, abseps=_CDF_TOL, releps=_CDF_TOL))


def _neg_loglik(rho, t1, t2, a, b, c, d):
    """2x2 log-likelihood; a case lies ABOVE its threshold."""
    below = _bvn_cdf(t1, t2, rho)           # P(X < t1, Y < t2) = neither
    p_neither = below
    p_x_only = float(ndtr(t2)) - below      # X > t1, Y < t2
    p_y_only = float(ndtr(t1)) - below      # X < t1, Y > t2
    p_both = 1.0 - float(ndtr(t1)) - float(ndtr(t2)) + below
    ll = 0.0
    for n_ij, p_ij in ((a, p_both), (b, p_x_only), (c, p_y_only), (d, p_neither)):
        if n_ij:
            ll += n_ij * math.log(max(p_ij, 1e-300))
    return -ll


def tetrachoric_table(a: float, b: float, c: float, d: float, *,
                      continuity_correction: bool = True) -> TetrachoricResult:
    """Tetrachoric MLE from a 2x2 table of counts.

    The table reads ``a`` = both cases, ``b`` = first case only, ``c`` =
    second case only, ``d`` = neither. A zero cell gets a 0.5 continuity
    correction added to all four cells when ``continuity_correction`` (the
    standard handling; the estimate is then boundary-avoiding rather than
    exactly +/-1). Monomorphic margins (one variable all-one-level) are
    rejected: there is no correlation information.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) < 0:
        raise ValueError("cell counts must be nonnegative")
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

    t1 = float(ndtri(1.0 - p1))
    t2 = float(ndtri(1.0 - p2))

    def f(rho):
        return _neg_loglik(rho, t1, t2, a, b, c, d)

    res = minimize_scalar(f, bounds=(-1.0 + _RHO_EPS, 1.0 - _RHO_EPS),
                          method="bounded", options={"xatol": 1e-12})
    rho, fmin = res.x, res.fun

    # observed information via a numeric Hessian
    delta = 1e-4
    hess = (f(rho + delta) - 2.0 * fmin + f(rho - delta)) / (delta * delta)
    se = math.sqrt(1.0 / hess) if hess > 0 else math.nan
    return TetrachoricResult(rho=float(rho), se=float(se),
                             thresholds=(t1, t2), n=n_pairs, corrected=corrected)


def tetrachoric(x: ArrayLike, y: ArrayLike, *,
                continuity_correction: bool = True) -> TetrachoricResult:
    """Tetrachoric MLE between two binary arrays of equal length."""
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

    ``X`` is an ``(n_pairs, m)`` binary array; returns the symmetric ``(m, m)``
    matrix of pairwise estimates with unit diagonal. Because the off-diagonal
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
