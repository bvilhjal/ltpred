"""Numerical posterior moments for additive, noninbred nuclear families.

Offspring are independent conditional on the two parental breeding values.
Thus the number of siblings changes the number of likelihood factors, not the
dimension of integration. Point observations are conditioned analytically;
uninformative factor directions and zero/one-interval cases are integrated out.
The remaining one- or two-dimensional integral uses Gaussian quadrature centred
at its posterior mode. This is a numerical approximation to the same no-mixture
liability-threshold posterior as Gibbs, with a resolution-change diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import log_ndtr, logsumexp, roots_hermitenorm, roots_legendre

from ._validation import validate_bounds
from .pearson_aitken import _tnorm_moments_loc


@dataclass(frozen=True)
class QuadratureResult:
    """Per-family posterior ``est`` and ``var``, and numerical diagnostics.

    ``error`` is the largest change in either moment across the last two
    quadrature refinements. It is not a certified error bound or Monte-Carlo
    standard error. ``n_nodes`` is the final number of nodes per active factor
    dimension (zero for analytic answers), not the total tensor-grid size.
    """

    est: np.ndarray
    var: np.ndarray
    error: np.ndarray
    n_nodes: np.ndarray


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_GL_X, _GL_W = roots_legendre(12)
_EPS = float(np.finfo(float).eps)


def _log_mass(mean, variance, lower, upper):
    """Log normal interval mass, retaining narrow intervals and both tails."""
    mean = np.asarray(mean, dtype=float)
    sd = math.sqrt(variance)
    if lower == -np.inf and upper == np.inf:
        return np.zeros_like(mean)
    if np.isfinite(lower) and np.isfinite(upper):
        width = upper - lower
        # Local quadrature is exceptionally accurate for a nearly flat narrow
        # interval. A far-away Gaussian can instead vary by many log units
        # across even this width; use reflected log CDFs for those nodes.
        if width / sd <= 1e-3 and np.all(np.abs((lower + 0.5 * width - mean) * width / variance) <= 10.0):
            center = lower + 0.5 * width
            z = (center - mean) / sd
            delta = (0.5 * width / sd) * _GL_X
            correction = -z[..., None] * delta - 0.5 * delta * delta
            return (math.log(width / sd) - math.log(2.0) - _LOG_SQRT_2PI
                    - 0.5 * z * z
                    + logsumexp(np.log(_GL_W) + correction, axis=-1))
    a = (lower - mean) / sd
    b = (upper - mean) / sd
    # Reflect positive intervals before subtracting CDFs. expm1 avoids losing
    # finite widths when the two log CDF values are close.
    reflect = a > 0.0
    left = np.where(reflect, -b, a)
    right = np.where(reflect, -a, b)
    log_hi, log_lo = log_ndtr(right), log_ndtr(left)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = log_hi + np.log(-np.expm1(log_lo - log_hi))
    return value


def _moments(mean, variance, lower, upper):
    """Scalar interval moments, with centred integration for tiny widths."""
    if lower != upper and np.isfinite(lower) and np.isfinite(upper):
        width = upper - lower
        if width / math.sqrt(variance) <= 1e-3 and abs((lower + 0.5 * width - mean) * width / variance) <= 10.0:
            center = lower + 0.5 * width
            delta = 0.5 * width * _GL_X
            logw = np.log(_GL_W) - ((center - mean) * delta + 0.5 * delta**2) / variance
            weight = np.exp(logw - logsumexp(logw))
            shift = float(weight @ delta)
            return center + shift, float(weight @ (delta - shift)**2)
    return _tnorm_moments_loc(float(mean), math.sqrt(variance), float(lower), float(upper))


@lru_cache(maxsize=16)
def _grid(n, dimension):
    nodes, weight = roots_hermitenorm(n)
    if np.all(weight > 0.0):
        logweight = np.log(weight)
    else:
        # SciPy's largest-order tail weights underflow although their logs
        # remain useful after importance reweighting. Normalized probabilists'
        # Hermite polynomials stay finite through the public limit n=512;
        # w_i = sqrt(2*pi)/(n*p_{n-1}(x_i)^2).
        previous, current = np.ones(n), nodes.copy()
        for k in range(2, n):
            previous, current = current, (nodes * current - math.sqrt(k - 1) * previous) / math.sqrt(k)
        logweight = _LOG_SQRT_2PI - math.log(n) - 2.0 * np.log(np.abs(current))
    if dimension == 1:
        return nodes[:, None], logweight
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())), (logweight[:, None] + logweight[None, :]).ravel()


def _mode(offset, load, noise, lower, upper):
    """Newton iterations for a log-concave, whitened factor posterior."""
    dimension = load.shape[1]

    def evaluate(x, derivatives=True):
        means = offset + load @ x
        value = 0.5 * float(x @ x)
        gradient = x.copy()
        hessian = np.eye(dimension)
        for i, mean in enumerate(means):
            value -= float(_log_mass(mean, noise[i], lower[i], upper[i]))
            if derivatives:
                m, v = _moments(mean, noise[i], lower[i], upper[i])
                gradient -= load[i] * ((m - mean) / noise[i])
                hessian += np.outer(load[i], load[i]) * max(0.0, (noise[i] - v) / noise[i]**2)
        return value, gradient, hessian

    x = np.zeros(dimension)
    for _ in range(80):
        value, gradient, hessian = evaluate(x)
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) <= 1e-10 * (1.0 + np.max(np.abs(x))):
            return x, hessian
        scale = 1.0
        for _ in range(30):
            candidate = x - scale * step
            candidate_value, _, _ = evaluate(candidate, derivatives=False)
            if candidate_value <= value - 1e-4 * scale * float(gradient @ step):
                break
            scale *= 0.5
        else:
            # At floating-point resolution the objective may stop changing
            # before the derivative. This point remains a valid importance
            # proposal; quadrature refinement still determines acceptance.
            return x, hessian
        if value - candidate_value <= 8.0 * _EPS * (1.0 + abs(value)):
            # The same floor reached through a successful search. _moments
            # sets the accuracy of the exact gradient, so the Newton step can
            # stop shrinking while still above the step tolerance. The
            # sufficient decrease then underflows and admits scales that move
            # x by nothing, so no later iteration can do better than this.
            return candidate, hessian
        x = candidate
    raise RuntimeError("quadrature posterior-mode iteration did not converge")


def _family(roles, lower, upper, h2, out, atol, max_nodes):
    own = roles.index("o") if "o" in roles else None
    own_lo, own_hi = (-np.inf, np.inf) if own is None else (lower[own], upper[own])
    if out == "full" and own_lo == own_hi:
        return float(own_lo), 0.0, 0.0, 0
    if h2 == 0.0:
        m, v = _moments(0.0, 1.0, own_lo, own_hi)
        return (0.0, 0.0, 0.0, 0) if out == "genetic" else (m, v, 0.0, 0)
    informative = (lower != -np.inf) | (upper != np.inf)
    if all(roles[i] in ("m", "f") for i in np.flatnonzero(informative)):
        # The two parental full liabilities are independent. Their interval
        # moments therefore suffice, even at h2 close to one where integrating
        # latent-parent likelihood steps is unnecessarily difficult.
        estimate, variance = 0.0, h2 if out == "genetic" else 1.0
        for i in np.flatnonzero(informative):
            m, v = _moments(0.0, 1.0, lower[i], upper[i])
            estimate += 0.5 * h2 * m
            variance += (0.5 * h2)**2 * (v - 1.0)
        return float(estimate), float(variance), 0.0, 0

    common = np.array([0.5, 0.5])
    load = np.array([[1.0, 0.0] if r == "m" else [0.0, 1.0] if r == "f" else common for r in roles]).reshape(-1, 2)
    noise = np.array([1.0 - h2 if r in ("m", "f") else 1.0 - 0.5 * h2 for r in roles])
    mean = np.zeros(2)
    cov = h2 * np.eye(2)
    pins = lower == upper
    for i in np.flatnonzero(pins):
        cross = cov @ load[i]
        variance = noise[i] + load[i] @ cross
        mean += cross * ((lower[i] - load[i] @ mean) / variance)
        cov -= np.outer(cross, cross) / variance
        cov = 0.5 * (cov + cov.T)

    # Given the parental factor and own observation, the genetic residual is
    # Mendelian sampling (variance a), correlated a with own full residual v.
    a, v = 0.5 * h2, 1.0 - 0.5 * h2
    own_pin = own_lo == own_hi
    if own_pin:
        coefficient = 1.0 - a / v if out == "genetic" else 0.0
        constant = (a / v) * own_lo if out == "genetic" else own_lo
        residual = a - a*a / v if out == "genetic" else 0.0
    else:
        coefficient, constant = 1.0, 0.0
        residual = a if out == "genetic" else v
    base_mean = coefficient * (common @ mean) + constant
    base_var = coefficient**2 * (common @ cov @ common) + residual
    active = np.flatnonzero(~pins & ((lower != -np.inf) | (upper != np.inf)))
    if not active.size:
        return float(base_mean), float(base_var), 0.0, 0
    if active.size == 1:
        i = active[0]
        my = load[i] @ mean
        vy = noise[i] + load[i] @ cov @ load[i]
        cross = coefficient * (common @ cov @ load[i])
        if i == own:
            cross += a if out == "genetic" else v
        tm, tv = _moments(my, vy, lower[i], upper[i])
        weight = cross / vy
        return float(base_mean + weight * (tm - my)), float(max(0.0, base_var + weight**2 * (tv - vy))), 0.0, 0

    # Project onto the span of observed factor directions. Remaining Gaussian
    # directions contribute only their analytic variance to a linear target.
    chol = np.linalg.cholesky(cov)
    full_load = load[active] @ chol
    _, singular, vectors = np.linalg.svd(full_load, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    basis = vectors[:rank].T
    reduced = full_load @ basis
    target_load = common @ chol
    discarded = max(0.0, float(target_load @ target_load - (target_load @ basis) @ (target_load @ basis)))
    lo, hi, variances = lower[active], upper[active], noise[active]
    offset = load[active] @ mean
    mode, precision = _mode(offset, reduced, variances, lo, hi)
    proposal = np.linalg.cholesky(np.linalg.solve(precision, np.eye(rank)))
    previous = None
    changes = []
    n = 16
    while True:
        z, logw = _grid(n, rank)
        x = mode + z @ proposal.T
        logweight = logw - 0.5 * np.sum(x*x, axis=1) + 0.5 * np.sum(z*z, axis=1)
        for i in range(len(active)):
            logweight += _log_mass(offset[i] + x @ reduced[i], variances[i], lo[i], hi[i])
        if not np.all(np.isfinite(logweight)):
            raise RuntimeError("quadrature encountered non-finite log weights")
        weight = np.exp(logweight - logsumexp(logweight))
        s = common @ mean + x @ (target_load @ basis)
        if own_pin or (own_lo == -np.inf and own_hi == np.inf):
            target_mean = coefficient * s + constant
            target_var = np.full(len(s), residual + coefficient**2 * discarded)
        else:
            moments = np.array([_moments(value, v, own_lo, own_hi) for value in s])
            if out == "genetic":
                target_mean = s + (a / v) * (moments[:, 0] - s)
                target_var = a - a*a/v + (a/v)**2 * moments[:, 1]
            else:
                target_mean, target_var = moments[:, 0], moments[:, 1]
        estimate = float(weight @ target_mean)
        variance = float(weight @ (target_var + (target_mean - estimate)**2))
        if previous is not None:
            changes.append(max(abs(estimate - previous[0]), abs(variance - previous[1])))
            if len(changes) >= 2 and max(changes[-2:]) <= atol:
                return estimate, variance, max(changes[-2:]), n
        if n >= max_nodes:
            # Report what the criterion tests. The final change alone can
            # satisfy atol while the pair does not; at max_nodes=64 only two
            # changes exist, so the coarse first refinement is then binding.
            error = max(changes[-2:]) if changes else np.inf
            raise RuntimeError(f"quadrature did not converge by {max_nodes} nodes per dimension "
                               f"(largest of the last two changes {error:.3g}, atol {atol:.3g})")
        previous = estimate, variance
        n = min(n * 2, max_nodes)


def estimate_liability_quadrature_arrays(roles: Sequence[str], lower: ArrayLike,
                                         upper: ArrayLike, h2: float = 0.5, *,
                                         out: str = "genetic", atol: float = 1e-8,
                                         max_nodes: int = 128) -> QuadratureResult:
    """Compute additive nuclear-family posterior moments by factor quadrature.

    ``roles`` contains unique ``o``, ``m``, ``f`` and/or ``s1``, ``s2``, ...;
    ``lower`` and ``upper`` have shape ``(n_families, len(roles))``. The target
    is always the proband, even when ``o`` is absent (then own status is omitted).
    Parents are unrelated and noninbred. Only ``0 <= h2 < 1`` is supported:
    ``h2=1`` gives zero parent residual variance and needs a different integrator.
    Shared environment, other pedigree roles and censoring mixtures are outside
    this model. Point pins are exact observations, not narrow intervals.

    ``out`` selects ``"genetic"`` or ``"full"``. ``atol`` bounds successive
    changes in both posterior moments, not their unknown true numerical error.
    Two successive refinements must meet it; ``max_nodes`` (64 through 512)
    limits nodes per active factor dimension. The upper limit bounds the
    largest tensor grid to 262144 points. Failure raises ``RuntimeError`` with the
    family index, rather than returning an unchecked estimate. Analytic zero/
    one-interval cases and families with only parental observations require no
    nodes. No Monte-Carlo SE is defined.
    """
    roles = list(roles)
    if any(not isinstance(r, str) or re.fullmatch(r"o|m|f|s[1-9][0-9]*", r) is None for r in roles):
        raise ValueError("quadrature supports only nuclear-family roles o, m, f, s1, s2, ...")
    if len(set(roles)) != len(roles):
        raise ValueError("quadrature roles must be unique")
    if isinstance(h2, (bool, np.bool_)) or np.ndim(h2) != 0 or not np.isfinite(h2) or not 0 <= h2 < 1:
        raise ValueError("quadrature requires scalar h2 in [0, 1)")
    if out not in ("genetic", "full"):
        raise ValueError("out must be 'genetic' or 'full'")
    if isinstance(atol, (bool, np.bool_)) or np.ndim(atol) != 0 or not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and strictly positive")
    if isinstance(max_nodes, (bool, np.bool_)) or not isinstance(max_nodes, (int, np.integer)) or not 64 <= max_nodes <= 512:
        raise ValueError("max_nodes must be an integer from 64 through 512")
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[1] != len(roles):
        raise ValueError("lower and upper must have shape (n_families, len(roles))")
    validate_bounds(lower, upper, context="quadrature bounds")
    n = lower.shape[0]
    est, var, error = np.empty(n), np.empty(n), np.empty(n)
    nodes = np.empty(n, dtype=int)
    for i in range(n):
        try:
            est[i], var[i], error[i], nodes[i] = _family(roles, lower[i], upper[i], float(h2), out, float(atol), int(max_nodes))
        except RuntimeError as exc:
            raise RuntimeError(f"family {i}: {exc}") from exc
    return QuadratureResult(est, var, error, nodes)
