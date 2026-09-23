"""Joint liability covariance from observed binary pairs.

Write Sigma = G (x) A + S (x) C + T (x) M + E (x) I, omitting
unrequested shared components. Each trait has unit marginal variance. The
free parameters are covariance entries, so every pair correlation is linear
and the observed-pair design exposes nonidentification before optimization.
Positive-semidefinite component constraints keep the decomposition coherent.

Only pair counts enter the optimizer. Boolean observations are retained for
family-cluster scores; no latent liabilities or Monte-Carlo traces are stored.
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

from .estimate import (_assert_nonempty_families, _check_unique_roles,
                       _group_by_structure, _validate_multitrait_bounds)
from .fit import (_component_matrix, _assert_common_thresholds,
                  _assert_nonoverlapping_pids, _assert_population_case_rate,
                  _validate_population_sampling, _validate_weights,
                  _validate_update_controls)
from .pairwise import _cluster_sandwich

__all__ = ["MultiTraitPairwiseResult", "fit_pairwise_multi"]


def _correlation(cov, variance_tol=1e-8):
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    active = np.diag(cov) > variance_tol
    denom = np.outer(sd, sd)
    out = np.full_like(cov, np.nan)
    np.divide(cov, denom, out=out, where=np.outer(active, active))
    # Only solver roundoff: a zero-variance component remains undefined.
    return np.clip(out, -1., 1.)


@dataclass
class MultiTraitPairwiseResult:
    """Joint A[/C/M]/E fit, conditional on thresholds and sampling weights.

    ``components`` includes residual ``E``. ``re`` is its within-person
    cross-trait correlation; shared sibship/couple correlations are separately
    available in ``correlations['C']``/``['M']``. ``rp`` sums *all* components.
    Zero-variance correlations are NaN. ``se`` contains sampling SEs for
    ``h2``, ``rg``, ``re`` and ``rp`` (delta method, family-cluster sandwich).
    All SEs are withheld at any covariance constraint boundary.

    ``parameter_covariance`` follows ``parameter_order``: (component, trait,
    trait) triples for the free symmetric entries. Residual diagonals are
    determined by unit total variance and are not free parameters. These are
    conditional asymptotic SEs; estimated thresholds/weights, informative
    missingness, overlapping clusters and boundary inference need other methods.
    """
    components: dict[str, np.ndarray]
    parameter_order: tuple
    parameter_covariance: np.ndarray
    se: dict[str, np.ndarray]
    phen_names: list[str]
    at_boundary: bool
    inference_status: str
    composite_loglik: float
    n_families: int
    n_pairs: int
    n_iter: int
    boundary_tolerance: float

    @property
    def h2(self):
        return np.diag(self.components["A"]).copy()

    @property
    def genetic_cov(self):
        return self.components["A"]

    @property
    def env_cov(self):
        return self.components["E"]

    @property
    def correlations(self):
        return {name: _correlation(cov, self.boundary_tolerance) for name, cov in self.components.items()}

    @property
    def rg(self):
        return _correlation(self.genetic_cov, self.boundary_tolerance)

    @property
    def re(self):
        return _correlation(self.env_cov, self.boundary_tolerance)

    @property
    def rp(self):
        return sum(self.components.values())


def _bivariate_probabilities(t1, t2, rho):
    """Four binary cells and two rho derivatives, with unequal thresholds.

    Plackett integration from independence is fast away from cancellation.
    Small cells are recomputed as positive integrals from the Frechet endpoints.
    The angle transform removes the density's square-root singularity.
    """
    if not np.all(np.isfinite([t1, t2, rho])) or not -1 < rho < 1:
        raise ValueError("pair probabilities need finite thresholds and -1 < rho < 1")
    k1, k2 = float(ndtr(-t1)), float(ndtr(-t2))
    c1, c2 = float(ndtr(t1)), float(ndtr(t2))
    angle = math.asin(rho)

    def integrand(theta):
        r = math.sin(theta)
        exponent = ((t1 + t2)**2 / (4 * (1 + r))
                    + (t1 - t2)**2 / (4 * (1 - r)))
        return math.exp(-exponent) / (2 * math.pi)

    extra = quad(integrand, 0., angle, epsabs=0., epsrel=2e-11)[0] if rho else 0.
    prob = np.array([k1*k2 + extra, k1*c2 - extra,
                     c1*k2 - extra, c1*c2 + extra])
    if np.min(prob) < 1e-8:
        # Stable marginal differences, including thresholds deep in either tail.
        d12 = k1-k2 if k1 <= .5 else c2-c1
        base11 = k1-c2 if k1 <= .5 else k2-c1
        concordant = quad(integrand, -math.pi/2, angle, epsabs=0., epsrel=2e-11)[0]
        discordant = quad(integrand, angle, math.pi/2, epsabs=0., epsrel=2e-11)[0]
        prob = np.array([max(base11, 0.) + concordant,
                         max(d12, 0.) + discordant,
                         max(-d12, 0.) + discordant,
                         max(-base11, 0.) + concordant])
    if np.any(prob <= 0) or not np.all(np.isfinite(prob)):
        raise FloatingPointError("binary pair probability underflow; thresholds/correlation too extreme")
    one_minus = 1 - rho*rho
    density = integrand(angle) / math.sqrt(one_minus)
    first = density * np.array([1., -1., -1., 1.])
    log_derivative = (rho / one_minus
                      + ((1 + rho*rho)*t1*t2 - rho*(t1*t1 + t2*t2)) / one_minus**2)
    return prob, first, first * log_derivative


def _covariance_basis(components, p):
    order = tuple((c, i, j) for c in components for i in range(p) for j in range(i, p))
    order += tuple(("E", i, j) for i in range(p) for j in range(i+1, p))
    basis = np.zeros((len(components)+1, p, p, len(order)))
    offset = np.zeros(basis.shape[:-1])
    offset[-1] = np.eye(p)
    for t, (c, i, j) in enumerate(order):
        ci = len(components) if c == "E" else components.index(c)
        basis[ci, i, j, t] = basis[ci, j, i, t] = 1.
        if i == j:
            basis[-1, i, i, t] = -1.
    return order, offset, basis


def _prepare_multi_pairs(families, components, p, order, weights):
    patterns, counts, groups, thresholds = [], [], [], np.full(p, np.nan)
    lookup = {}
    n_pairs = 0
    for _key, indices in _group_by_structure(families):
        roles = sorted(m.role for m in families[indices[0]].members)
        k = len(roles)
        kernels = {c: _component_matrix(roles, c) for c in components}
        kernels["E"] = np.eye(k)
        cases = np.zeros((len(indices), p*k), dtype=bool)
        observed = np.zeros_like(cases)
        for slot, fi in enumerate(indices):
            for i, member in enumerate(sorted(families[fi].members, key=lambda m: m.role)):
                lo, hi = np.asarray(member.lower), np.asarray(member.upper)
                cases[slot, i::k] = np.isfinite(lo)
                observed[slot, i::k] = np.isfinite(lo) | np.isfinite(hi)
                for a in range(p):
                    if observed[slot, a*k+i]:
                        thresholds[a] = lo[a] if cases[slot, a*k+i] else hi[a]
        indices = np.asarray(indices, dtype=int)
        pairs = []
        for u in range(p*k):
            a, i = divmod(u, k)
            for v in range(u+1, p*k):
                b, j = divmod(v, k)
                keep = observed[:, u] & observed[:, v]
                if not np.any(keep):
                    continue
                row = tuple(float(kernels[c][i, j]) if (a, b) == (pa, pb) else 0.
                            for c, pa, pb in order)
                if not any(row):
                    continue
                key = (a, b, row)
                if key not in lookup:
                    lookup[key] = len(patterns)
                    patterns.append(key)
                    counts.append(np.zeros(4))
                pi = lookup[key]
                cells = 2*(~cases[keep, u]).astype(int) + (~cases[keep, v])
                counts[pi] += np.bincount(cells, weights=weights[indices[keep]], minlength=4)
                pairs.append((u, v, pi))
                n_pairs += int(keep.sum())
        groups.append((indices, cases, observed, pairs))
    design = np.asarray([r for _, _, r in patterns]).reshape(-1, len(order))
    if len(design) < len(order) or np.linalg.matrix_rank(design) < len(order):
        raise ValueError("joint covariance components not identified from observed pairs: rank-deficient design")
    pair_thresholds = np.array([(thresholds[a], thresholds[b]) for a, b, _ in patterns])
    return design, np.asarray(counts), pair_thresholds, groups, n_pairs


def _probability_limits(thresholds, eps):
    """Locate safe trial-step continuation points without flooring any cell.

    Unequal thresholds make discordant cells exponentially small near |rho|=1.
    Leave a large floating-point margin for optimization; the returned fit is
    always re-evaluated with the unextended probability and score.
    """
    cache = {}
    for t1, t2 in thresholds:
        key = (float(t1), float(t2))
        if key in cache:
            continue
        limits = []
        for sign in (-1., 1.):
            lo, hi = 0., 1-eps
            for _ in range(32):
                value = hi if _ == 0 else (lo+hi)/2
                try:
                    safe = np.min(_bivariate_probabilities(t1, t2, sign*value)[0]) >= 1e-100
                except FloatingPointError:
                    safe = False
                if safe:
                    lo = value
                    if _ == 0:
                        break
                else:
                    hi = value
            limits.append(sign*lo)
        cache[key] = limits
    return np.asarray([cache[tuple(t)] for t in thresholds])


def _multi_criterion(values, thresholds, design, counts, rho_limits=None):
    value = 0.
    gradient = np.zeros(len(values))
    hessian = np.zeros((len(values), len(values)))
    cell_scores = []
    for pi, (row, cells, (t1, t2)) in enumerate(zip(design, counts, thresholds)):
        raw_rho = float(row @ values)
        rho = raw_rho if rho_limits is None else float(np.clip(raw_rho, *rho_limits[pi]))
        prob, first, second = _bivariate_probabilities(t1, t2, rho)
        score = first / prob
        loss = -float(cells @ np.log(prob))
        slope = -float(cells @ score)
        curvature = float(cells @ (score*score - second/prob))
        if raw_rho != rho:
            # SLSQP can evaluate infeasible trial points even with linear
            # constraints. A C1 quadratic extension supplies a finite restoring
            # score there; final estimates are checked and evaluated unextended.
            excess = raw_rho-rho
            curvature = max(abs(curvature), 1.)
            loss += slope*excess + .5*curvature*excess**2
            slope += curvature*excess
        value += loss
        gradient += slope * row
        hessian += curvature * np.outer(row, row)
        cell_scores.append(score[:, None] * row)
    return value, gradient, hessian, np.asarray(cell_scores)


def _derived_se(matrices, basis, covariance, ai):
    p = matrices.shape[1]
    if not np.all(np.isfinite(covariance)):
        return {name: np.full(shape, np.nan) for name, shape in
                (("h2", (p,)), ("rg", (p, p)), ("re", (p, p)), ("rp", (p, p)))}
    def propagate(derivative):
        return np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', derivative,
                                            covariance, derivative), 0.))

    def corr_gradient(ci):
        matrix, deriv = matrices[ci], basis[ci]
        diagonal = np.diag(matrix)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.sqrt(np.outer(diagonal, diagonal))
            corr = matrix / denom
            grad = deriv / denom[:, :, None]
            for i in range(p):
                for j in range(p):
                    grad[i, j] -= .5*corr[i, j]*(deriv[i, i]/diagonal[i]
                                                + deriv[j, j]/diagonal[j])
        return grad

    return dict(h2=propagate(basis[ai, np.arange(p), np.arange(p)]),
                rg=propagate(corr_gradient(ai)), re=propagate(corr_gradient(-1)),
                rp=propagate(basis.sum(axis=0)))


def fit_pairwise_multi(families: Sequence, *, components: Sequence[str] = ("A",),
                       sampling: str | None = None, weights: ArrayLike | None = None,
                       phen_names: Sequence[str] | None = None, eps: float = 1e-6,
                       maxiter: int = 300, tol: float = 1e-10) -> MultiTraitPairwiseResult:
    """Fit joint h², genetic and residual environmental correlations.

    Each member supplies explicit length-P bounds, P >= 2. A common threshold
    per trait is required; different traits may have different prevalences.
    Missing phenotypes use (-inf, inf) and contribute no pairs. Missingness must
    not select the pair's statuses, unless valid family IPW corrects that design.
    Families must be independent and non-overlapping. ``sampling='ipw'`` needs
    known, positive family inclusion probabilities; this is not an ascertainment
    likelihood or a method for overlapping register pedigrees.

    ``components`` contains A and optionally C (full sibship) and M (couple).
    Every component is a PSD trait covariance; residual E has eigenvalues at
    least eps. The marginal variance of every trait is one. C and M consume
    liability variance and have their own cross-trait covariances; ``re`` refers
    specifically to E. M describes spousal resemblance, not a generative model
    of assortative mating. Relationship kernels follow the single-trait fitter.

    The optimizer uses aggregated pair counts, analytic scores and sensitivity.
    Failed, infeasible or nonstationary solutions raise. Interior sampling SEs
    use family-cluster score variation and a delta method for correlations;
    any PSD/residual boundary withholds all normal SEs. A composite likelihood
    does not support ordinary likelihood-ratio tests. Broad finite-sample
    calibration, especially for rare traits or weak components, is not implied.
    """
    components = tuple(components)
    if ("A" not in components or any(c not in ("A", "C", "M") for c in components)
            or len(set(components)) != len(components)):
        raise ValueError("components must contain A and optionally C/M, without duplicates")
    _, eps = _validate_update_controls(.2, eps)
    if isinstance(maxiter, (bool, np.bool_)):
        raise TypeError("maxiter must be a positive integer")
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and in (0, 1)")
    families = list(families)
    if not families:
        raise ValueError("fit_pairwise_multi needs at least one family")
    _assert_nonempty_families(families)
    _check_unique_roles(families)
    _assert_nonoverlapping_pids(families, "fit_pairwise_multi")
    p = np.asarray(families[0].members[0].lower).size
    if p < 2:
        raise ValueError("fit_pairwise_multi needs at least 2 traits")
    _validate_multitrait_bounds(families, p)
    _assert_common_thresholds(families, p, context="fit_pairwise_multi")
    names = [f"phenotype{i+1}" for i in range(p)] if phen_names is None else list(phen_names)
    if len(names) != p:
        raise ValueError("phen_names length must match number of traits")
    weights = _validate_weights(weights, len(families), "fit_pairwise_multi")
    _validate_population_sampling(sampling, "fit_pairwise_multi", weights=weights)
    w = np.ones(len(families)) if weights is None else weights / weights.max()
    w /= w.mean()
    if np.any(w == 0):
        raise ValueError("IPW weights span too wide a numerical range")
    _assert_population_case_rate(families, p, context="fit_pairwise_multi",
                                 weights=None if weights is None else w)
    order, offset, basis = _covariance_basis(components, p)
    design, counts, thresholds, groups, n_pairs = _prepare_multi_pairs(
        families, components, p, order, w)
    probability_limits = _probability_limits(thresholds, eps)
    n, d = len(families), len(order)
    def matrices(x):
        return offset + np.einsum('cpqt,t->cpq', basis, x)

    def cone(x):
        values = np.linalg.eigvalsh(matrices(x))
        values[-1] -= eps
        return values.ravel()

    def cone_jac(x):
        _, vectors = np.linalg.eigh(matrices(x))
        return np.einsum('cik,cijt,cjk->ckt', vectors, basis, vectors).reshape(-1, d)

    constraints = [dict(type="ineq", fun=cone, jac=cone_jac),
                   dict(type="ineq", fun=lambda x: np.r_[1-eps-design@x, 1-eps+design@x],
                        jac=lambda x: np.r_[-design, design])]
    def objective(x):
        value, gradient, _, _ = _multi_criterion(x, thresholds, design, counts, probability_limits)
        return value/n, gradient/n

    initial = np.array([.5/len(components) if i == j else 0. for _, i, j in order])
    bounds = [(0. if i == j else -1+eps, 1-eps) for _, i, j in order]
    fit = minimize(objective, initial, jac=True, method="SLSQP", constraints=constraints, bounds=bounds,
                   options=dict(maxiter=maxiter, ftol=tol))
    x = np.asarray(fit.x)
    if (not fit.success or x.shape != (d,) or not np.all(np.isfinite(x))
            or not np.isfinite(fit.fun) or np.min(cone(x)) < -1e-8
            or np.max(np.abs(design@x)) > 1-eps+1e-8):
        raise RuntimeError(f"joint pairwise optimization failed or returned an invalid result: {fit.message}")
    value, gradient, hessian, cell_scores = _multi_criterion(x, thresholds, design, counts)
    # Projected score check for the PSD cone, including repeated zero eigenvalues.
    target = x - gradient/n
    def projection(z):
        delta = z-target
        return .5*float(delta@delta), delta
    projected = minimize(projection, x, jac=True, method="SLSQP", constraints=constraints, bounds=bounds,
                         options=dict(maxiter=maxiter, ftol=min(tol, 1e-12)))
    if (not projected.success or not np.all(np.isfinite(projected.x))
            or np.max(np.abs(projected.x-x)) > max(1e-5, 10*math.sqrt(tol))):
        raise RuntimeError("joint pairwise optimization failed its constrained score check")
    # An objective tolerance bounds squared parameter error locally. Use its
    # square root when deciding whether an eigenvalue is numerically interior.
    boundary_tol = max(1e-7, 10*math.sqrt(tol))
    boundary = bool(np.min(cone(x)) <= boundary_tol)
    covariance, status = _cluster_sandwich(hessian, groups, cell_scores, w, boundary)
    if status != "interior_cluster_sandwich":
        warnings.warn("joint pairwise sampling SEs unavailable: " + status,
                      RuntimeWarning, stacklevel=2)
    mats = matrices(x)
    se = _derived_se(mats, basis, covariance, components.index("A"))
    return MultiTraitPairwiseResult(
        components=dict(zip((*components, "E"), mats)), parameter_order=order,
        parameter_covariance=covariance, se=se, phen_names=names,
        at_boundary=boundary, inference_status=status, composite_loglik=-float(value),
        n_families=n, n_pairs=n_pairs, n_iter=int(fit.nit), boundary_tolerance=boundary_tol)
