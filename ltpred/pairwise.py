"""Deterministic variance components from binary relative-pair likelihoods.

`fit_pairwise` estimates the proportions ``theta_c`` of `ltpred.fit` -- same
``A``/``C``/``M`` kernels, same input and sampling contract -- by maximising a
composite likelihood instead of augmenting liabilities (see the efficient
inference report, "Fit covariance parameters by pairwise likelihood"). A pair
of relatives has liability correlation ``rho_ij = x_ij' theta`` with
``x_ij = (K_c[i, j])_c``, and at the common threshold ``t`` its binary
outcome has one of four bivariate-normal cell probabilities ``p_ab(t, rho)``.

**Algorithm L.**

- **L1. Screen.** As Algorithm H1 (`ltpred.fit`). IPW weights are rescaled to
  mean one; the case-rate screen uses the rescaled weights.
- **L2. Tabulate.** In each role-set group, every within-family pair ``i < j``
  with a nonzero row ``x_ij`` contributes the families in which both members are
  observed (a finite bound). Pairs sharing ``x_ij`` share one weighted 2x2 table
  ``n_ab = sum w_f`` (cells: both cases, first only, second only, neither).
  Raise unless the distinct observed rows have full rank.
- **L3. Criterion.** ``-ell_c(theta) = -sum_x sum_ab n_ab log p_ab(t, x' theta)``
  with analytic gradient and exact Hessian.
- **L4. Optimise.** SLSQP on ``-ell_c / n_families`` from
  ``theta_c = (1 - eps) / (2 p)`` with ``theta_c >= 0`` and
  ``sum theta <= 1 - eps``. Raise on solver failure, infeasibility, or a KKT
  violation (reduced gradient with the sum-constraint multiplier) above
  ``max(1e-5, 10 sqrt(tol))``.
- **L5. Covariance.** ``V = H^-1 J H^-1``: ``H`` is the
  Hessian of ``-ell_c`` (observed sensitivity), ``J = n/(n-1) sum_f (s_f -
  s_bar)(s_f - s_bar)'`` over all ``n`` input families, ``s_f`` the weighted sum
  of family ``f``'s pair scores. ``V`` and the SEs are NaN when some ``theta_c``
  or ``1 - eps - sum theta`` is within ``max(1e-7, 10 tol)`` of zero, when
  ``n <= p``, or when ``H`` is not positive definite or ``H``/``J`` lack full rank.

Pair tables compress the composite objective only; they are not sufficient for
the full family likelihood. Pairs within a family share members, so the inverse
Hessian alone understates sampling variance; the sandwich treats families as
the independent units. Thresholds and weights are held fixed, so ``V`` is an
asymptotic covariance conditional on them -- not a finite-sample calibration
claim, an ascertainment likelihood, or a basis for ordinary likelihood-ratio
tests.
"""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import ndtr

from .estimate import _assert_nonempty_families, _check_unique_roles, _group_by_structure
from .fit import (_validate_components, _component_matrix, _assert_common_thresholds,
                  _assert_nonoverlapping_pids, _assert_population_case_rate,
                  _member_bounds, _validate_population_sampling,
                  _validate_weights, _validate_update_controls)

__all__ = ["PairwiseFitResult", "fit_pairwise"]


@dataclass
class PairwiseFitResult:
    """A constrained composite-likelihood fit on independent family clusters.

    ``components`` and ``residual`` are liability-variance fractions.
    ``covariance`` follows ``component_order``; ``se`` is its square-root
    diagonal. At a constraint boundary, or with insufficient observed
    information, both are NaN and ``inference_status`` explains why: it is
    ``"interior_cluster_sandwich"`` when they are available, otherwise
    ``"unavailable_boundary"``, ``"unavailable_clusters"`` (no more families
    than components) or ``"unavailable_information"``. ``at_boundary`` is true
    when some fraction, or the residual margin ``1 - eps - sum``, lies within
    the boundary tolerance (module docstring, L5). Normal
    intervals at such boundaries are inappropriate; a design-appropriate
    bootstrap or profile analysis needs its own boundary calibration.

    ``composite_loglik`` uses IPW weights normalized to mean one. Its magnitude
    is not a full-data log likelihood and must not be used for ordinary
    likelihood-ratio tests. ``n_pairs`` counts informative observed pairs;
    ``n_families`` counts input clusters, including those contributing no pairs.
    ``n_iter`` is the SLSQP iteration count.
    """

    components: dict[str, float]
    residual: float
    component_order: tuple[str, ...]
    covariance: np.ndarray
    se: dict[str, float]
    at_boundary: bool
    inference_status: str
    composite_loglik: float
    n_families: int
    n_pairs: int
    n_iter: int


def _pair_probabilities(threshold, rho):
    """Same-threshold probabilities and first two correlation derivatives.

    Cell order is both cases, first only, second only, neither. With
    ``I(a, b) = int_a^b exp(-t^2 / (1 + sin u)) / (2 pi) du`` (Plackett's
    identity with ``rho = sin u``, which removes the endpoint singularity),
    ``p11 = K^2 + I(0, asin rho)``, ``p00 = (1-K)^2 + I(0, asin rho)`` and
    ``p10 = p01 = I(asin rho, pi/2)``: discordant cells are integrated *from rho
    to one*, avoiding subtraction of nearly equal tails. Derivatives are
    ``+-phi2(t, t; rho)`` and its ``rho`` derivative. Requires ``0 <= rho < 1``;
    no probability flooring silently changes the likelihood.
    """
    if not np.isfinite(threshold) or not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("pair probabilities require a finite threshold and 0 <= rho < 1")
    threshold = float(threshold)
    rho = float(rho)
    case, control = float(ndtr(-threshold)), float(ndtr(threshold))
    if rho == 0:
        prob = np.array([case * case, case * control, case * control, control * control])
    else:
        angle = math.asin(rho)
        t2 = threshold * threshold

        def integrand(theta):
            """``phi2(t, t; rho) d rho / d theta`` with ``rho = sin(theta)``."""
            return math.exp(-t2 / (1 + math.sin(theta))) / (2 * math.pi)

        extra = quad(integrand, 0, angle, epsabs=0, epsrel=5e-12)[0]
        discordant = quad(integrand, angle, math.pi / 2, epsabs=0, epsrel=5e-12)[0]
        prob = np.array([case * case + extra, discordant, discordant,
                         control * control + extra])
    density = math.exp(-threshold * threshold / (1 + rho)) / (
        2 * math.pi * math.sqrt(1 - rho * rho))
    derivative = density * np.array([1., -1., -1., 1.])
    second = derivative * (rho / (1 - rho * rho) + threshold * threshold / (1 + rho) ** 2)
    if np.any(prob <= 0) or not np.all(np.isfinite(prob)):
        raise FloatingPointError("pair probabilities underflowed; the threshold is too extreme for this fitter")
    return prob, derivative, second


def _prepare_pairs(families, components, weights):
    """Algorithm L2: aggregate weighted 2x2 pair tables by kernel row.

    Returns ``(threshold, design rows, counts, groups, n_pairs)``; ``groups``
    keeps each role-set's Boolean case/observed masks and pair-to-row map for
    the family scores, and ``n_pairs`` counts unweighted observed pairs."""
    patterns, counts, groups = [], [], []
    lookup = {}
    threshold = None
    n_pairs = 0
    for _structure, indices in _group_by_structure(families):
        roles = sorted(member.role for member in families[indices[0]].members)
        kernels = [_component_matrix(roles, component) for component in components]
        cases = np.zeros((len(indices), len(roles)), dtype=bool)
        observed = np.zeros_like(cases)
        for slot, family_index in enumerate(indices):
            for j, member in enumerate(sorted(families[family_index].members, key=lambda m: m.role)):
                lo = float(np.asarray(member.lower).reshape(-1)[0])
                hi = float(np.asarray(member.upper).reshape(-1)[0])
                cases[slot, j] = np.isfinite(lo)
                observed[slot, j] = np.isfinite(lo) or np.isfinite(hi)
                if threshold is None and observed[slot, j]:
                    threshold = lo if cases[slot, j] else hi
        indices = np.asarray(indices, dtype=int)
        pairs = []
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                pattern = tuple(float(kernel[i, j]) for kernel in kernels)
                keep = observed[:, i] & observed[:, j]
                if not any(pattern) or not np.any(keep):
                    continue
                if pattern not in lookup:
                    lookup[pattern] = len(patterns)
                    patterns.append(pattern)
                    counts.append(np.zeros(4))
                k = lookup[pattern]
                # Boolean subtraction is disallowed by NumPy; cell 0 means 11.
                cells = 2 * (~cases[keep, i]).astype(int) + (~cases[keep, j])
                counts[k] += np.bincount(cells, weights=weights[indices[keep]], minlength=4)
                n_pairs += int(keep.sum())
                pairs.append((i, j, k))
        groups.append((indices, cases, observed, pairs))
    design = np.asarray(patterns, dtype=float).reshape(-1, len(components))
    if design.shape[0] < len(components) or np.linalg.matrix_rank(design) < len(components):
        raise ValueError("variance components not identified from observed pairs: relationship design is rank-deficient")
    return float(threshold), design, np.asarray(counts), groups, n_pairs


def _criterion(values, threshold, design, counts):
    """Negative composite log likelihood, gradient and observed sensitivity.

    The sensitivity is the exact Hessian ``sum n_ab (s^2 - p''/p) x x'`` with
    ``s = p'/p``; ``cell_scores[k, ab] = s_ab x_k`` is ``d log p_ab / d theta``."""
    value = 0.
    gradient = np.zeros(len(values))
    hessian = np.zeros((len(values), len(values)))
    cell_scores = []
    for row, cells in zip(design, counts):
        prob, derivative, second = _pair_probabilities(threshold, float(row @ values))
        score = derivative / prob
        value -= float(cells @ np.log(prob))
        gradient -= float(cells @ score) * row
        hessian += float(cells @ (score * score - second / prob)) * np.outer(row, row)
        cell_scores.append(score[:, None] * row)
    return value, gradient, hessian, np.asarray(cell_scores)


def _family_scores(groups, cell_scores, weights):
    """Per-family composite scores ``w_f sum_(pairs in f) d log p / d theta``.

    One row per input family; families without observed pairs score zero."""
    scores = np.zeros((len(weights), cell_scores.shape[-1]))
    for indices, cases, observed, pairs in groups:
        for i, j, k in pairs:
            keep = observed[:, i] & observed[:, j]
            cells = 2 * (~cases[keep, i]).astype(int) + (~cases[keep, j])
            scores[indices[keep]] += cell_scores[k, cells]
    return scores * weights[:, None]


def _cluster_sandwich(hessian, groups, cell_scores, weights, at_boundary):
    """Sampling covariance from independent family scores, or a reason to withhold it.

    Algorithm L5: ``H^-1 J H^-1`` with centred family scores and factor
    ``n/(n-1)``. Returns ``(covariance, inference_status)``; the status is
    ``interior_cluster_sandwich`` or ``unavailable_boundary``/``_clusters``/``_information``."""
    n, p = len(weights), hessian.shape[0]
    covariance = np.full((p, p), np.nan)
    if at_boundary:
        inference_status = "unavailable_boundary"
    elif n <= p:
        inference_status = "unavailable_clusters"
    else:
        score = _family_scores(groups, cell_scores, weights)
        score -= score.mean(axis=0)
        meat = (score.T @ score) * n / (n - 1)
        if (not np.all(np.isfinite(hessian)) or np.linalg.eigvalsh(hessian).min() <= 0
                or np.linalg.matrix_rank(hessian) < p or np.linalg.matrix_rank(meat) < p):
            inference_status = "unavailable_information"
        else:
            left = np.linalg.solve(hessian, meat)
            covariance = np.linalg.solve(hessian, left.T).T
            covariance = (covariance + covariance.T) / 2
            if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0):
                covariance[:] = np.nan
                inference_status = "unavailable_information"
            else:
                inference_status = "interior_cluster_sandwich"
    return covariance, inference_status


def fit_pairwise(families: Sequence, *, components: Sequence[str] = ("A",),
                 sampling: str | None = None, weights: ArrayLike | None = None,
                 eps: float = 1e-6, maxiter: int = 200,
                 tol: float = 1e-9) -> PairwiseFitResult:
    """Fit A, C and/or M by deterministic binary-pair composite likelihood.

    This opt-in estimator keeps the current fitting input contract: independent,
    non-overlapping families, a common single-trait case/control threshold, and
    population sampling or valid strictly positive family-level IPW weights.
    It rejects personalized thresholds, onset pins and two-sided intervals.
    Uninformative members contribute no pairs. The known threshold and weights
    are treated as fixed: estimating either from these families introduces
    uncertainty that the returned sandwich does not include.

    ``components`` selects the existing valid A/C/M covariance kernels. Fractions
    are nonnegative and their sum is at most ``1 - eps``. Independent identifying
    relationship contrasts must be observed. ``sampling=None`` uses the existing
    compatibility warning; explicitly pass ``"population"`` or ``"ipw"``.
    The marginal case-rate screen can reject an inconsistent design but cannot
    establish joint positivity, independent clusters, or correct weights.

    ``families`` is a list of `ltpred.family.Family`; ``weights`` (with
    ``sampling="ipw"`` only) holds one reciprocal inclusion probability per
    family and is rescaled to mean one. ``eps`` must lie in ``[1e-8, 0.5)``.

    The fit is Algorithm L of the module docstring: SLSQP with analytic scores
    and a deterministic probability calculation. ``maxiter`` (a positive
    integer) caps the SLSQP iterations and ``tol`` in ``(0, 1)`` is its
    ``ftol`` -- not the Monte-Carlo ``tol`` of the estimators; it also scales
    the KKT and boundary tolerances of L4-L5. Failed, infeasible or
    non-stationary optimizer results raise ``RuntimeError`` instead of being
    returned; withheld SEs emit a ``RuntimeWarning``.
    Interior sampling SEs use the observed sensitivity and centered, weighted
    family score outer products (finite-cluster factor n/(n-1)). They are
    asymptotic, conditional on the supplied threshold/weights; no ordinary
    chi-square likelihood-ratio calibration is claimed for this composite fit.
    """
    components = tuple(_validate_components(components))
    _, eps = _validate_update_controls(0.2, eps)
    if isinstance(maxiter, (bool, np.bool_)):
        raise TypeError("maxiter must be a positive integer")
    try:
        maxiter = operator.index(maxiter)
    except TypeError:
        raise TypeError("maxiter must be a positive integer") from None
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if isinstance(tol, (bool, np.bool_)):
        raise TypeError("tol must be a finite positive real number")
    tol = float(tol)
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and in (0, 1)")
    families = list(families)
    weights = _validate_weights(weights, len(families), "fit_pairwise")
    _validate_population_sampling(sampling, "fit_pairwise", weights=weights)
    _check_unique_roles(families, check_pids=False)
    _assert_nonempty_families(families)
    _assert_nonoverlapping_pids(families, "fit_pairwise")
    member_bounds = _member_bounds(families, 1)
    _assert_common_thresholds(families, 1, context="fit_pairwise",
                              member_bounds=member_bounds)
    if weights is None:
        normalized_weights = np.ones(len(families))
    else:
        normalized_weights = weights / weights.max()
        normalized_weights /= normalized_weights.mean()
        if np.any(normalized_weights == 0):
            raise ValueError("IPW weights span too wide a numerical range to retain positive contributions")
    _assert_population_case_rate(families, 1, context="fit_pairwise",
                                 weights=None if weights is None else normalized_weights,
                                 member_bounds=member_bounds)
    threshold, design, counts, groups, n_pairs = _prepare_pairs(
        families, components, normalized_weights)
    n, p = len(families), len(components)
    cap = 1 - eps

    def objective(values):
        """Per-family criterion and gradient for SLSQP."""
        value, gradient, _hessian, _scores = _criterion(values, threshold, design, counts)
        return value / n, gradient / n

    result = minimize(objective, np.full(p, cap / (2 * p)), jac=True,
                      method="SLSQP", bounds=[(0, cap)] * p,
                      constraints={"type": "ineq", "fun": lambda x: cap - x.sum(),
                                   "jac": lambda x: -np.ones(p)},
                      options={"maxiter": maxiter, "ftol": tol})
    values = np.asarray(result.x)
    if (not result.success or values.shape != (p,) or not np.all(np.isfinite(values))
            or not np.isfinite(result.fun) or np.min(values) < -1e-10
            or values.sum() > cap + 1e-10):
        raise RuntimeError(f"pairwise optimization failed or returned an invalid result: {result.message}")
    # Remove only solver roundoff at a constraint, then evaluate that exact state.
    values = np.maximum(values, 0)
    if values.sum() > cap:
        values *= cap / values.sum()
    value, gradient, hessian, cell_scores = _criterion(values, threshold, design, counts)
    boundary_tol = max(1e-7, 10 * tol)
    at_boundary = bool(np.any(values <= boundary_tol) or cap - values.sum() <= boundary_tol)
    # Solver success alone is not a numerical certificate. Check the simplex
    # KKT equations, on the same per-family scale used in optimization.
    gradient /= n
    positive = values > boundary_tol
    multiplier = (max(0., -float(gradient[positive].mean()))
                  if cap - values.sum() <= boundary_tol and np.any(positive) else 0.)
    reduced = gradient + multiplier
    violation = max(float(np.max(np.abs(reduced[positive]))) if np.any(positive) else 0.,
                    float(np.max(np.maximum(-reduced[~positive], 0))) if np.any(~positive) else 0.)
    if not np.isfinite(value) or violation > max(1e-5, 10 * math.sqrt(tol)):
        raise RuntimeError("pairwise optimization failed its constrained score check")
    covariance, inference_status = _cluster_sandwich(
        hessian, groups, cell_scores, normalized_weights, at_boundary)
    if inference_status != "interior_cluster_sandwich":
        warnings.warn("pairwise sampling SEs unavailable: " + inference_status
                      + "; boundary-aware profile or design-appropriate bootstrap inference needs separate calibration",
                      RuntimeWarning, stacklevel=2)
    return PairwiseFitResult(
        components=dict(zip(components, map(float, values))),
        residual=float(1 - values.sum()), component_order=components,
        covariance=covariance, se=dict(zip(components, map(float, np.sqrt(np.diag(covariance))))),
        at_boundary=at_boundary, inference_status=inference_status,
        composite_loglik=float(-value), n_families=n, n_pairs=n_pairs,
        n_iter=int(result.nit))
