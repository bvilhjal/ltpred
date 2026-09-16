"""Independent probability, identification and covariance-recovery oracles."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from ltpred.family import Family, Member
from ltpred.pairwise_multi import (fit_pairwise_multi, _bivariate_probabilities,
                                   _covariance_basis, _derived_se, _multi_criterion)


@pytest.mark.parametrize("t1,t2,rho", [(0., 0., -.8), (.4, 1.5, -.4),
                                       (-2., .3, .7), (4., 3., .6),
                                       (6., -5., -.8), (1., 2., .999),
                                       (1., -2., -.999)])
def test_binary_cells_match_independent_conditional_integrals(t1, t2, rho):
    prob, first, second = _bivariate_probabilities(t1, t2, rho)
    sd = math.sqrt(1-rho*rho)
    expected = []
    for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        lo, hi = (t1, np.inf) if a else (-np.inf, t1)
        sign = 1 if b else -1
        expected.append(quad(lambda z: norm.pdf(z)*norm.cdf(sign*(rho*z-t2)/sd),
                             lo, hi, epsabs=0, epsrel=1e-10)[0])
    assert prob == pytest.approx(expected, rel=1e-7, abs=1e-100)
    assert prob.sum() == pytest.approx(1., abs=2e-14)
    step = 1e-6 if abs(rho) < .99 else 1e-7
    plus, minus = (_bivariate_probabilities(t1, t2, rho+s*step) for s in (1, -1))
    assert first == pytest.approx((plus[0]-minus[0])/(2*step), rel=1e-5, abs=1e-9)
    assert second == pytest.approx((plus[1]-minus[1])/(2*step), rel=1e-5, abs=1e-9)


def exact_pair_tables(components):
    """Expected counts from the independent arcsine law at threshold zero.

    Each family observes exactly two coordinates; missing coordinates cannot
    manufacture extra pairs. Weighted four-cell tables have their exact
    population probabilities, avoiding Monte Carlo error in this oracle.
    """
    p = len(next(iter(components.values())))
    families, weights = [], []
    role_pairs = [("o", "m", dict(A=.5, C=0, M=0, E=0)),
                  ("o", "s1", dict(A=.5, C=1, M=0, E=0)),
                  ("m", "f", dict(A=0, C=0, M=1, E=0)),
                  ("o", "o", dict(A=1, C=1, M=1, E=1))]
    for a in range(p):
        for b in range(a, p):
            for ri, rj, kernel in role_pairs:
                if ri == rj and a == b:
                    continue
                rho = sum(kernel[c]*matrix[a, b] for c, matrix in components.items())
                if not any(kernel[c] for c in components if c != "E") and ri != rj:
                    continue
                both = .25 + math.asin(rho)/(2*math.pi)
                for (s1, s2), weight in zip([(1, 1), (1, 0), (0, 1), (0, 0)],
                                           [both, .5-both, .5-both, both]):
                    rows = {r: [np.full(p, -np.inf), np.full(p, np.inf)] for r in (ri, rj)}
                    rows[ri][0 if s1 else 1][a] = 0.
                    rows[rj][0 if s2 else 1][b] = 0.
                    families.append(Family(len(families), [Member(r, *bounds) for r, bounds in rows.items()]))
                    weights.append(weight*100)
    return families, np.array(weights)


@pytest.mark.parametrize("shared", [False, True])
def test_exact_population_tables_recover_signed_genetic_and_environmental_components(shared):
    components = {"A": np.array([[.4, -.12], [-.12, .3]])}
    if shared:
        components.update(C=np.array([[.15, .07], [.07, .2]]),
                          M=np.array([[.12, -.025], [-.025, .1]]))
    components["E"] = np.diag(1-sum(np.diag(v) for v in components.values()))
    components["E"][0, 1] = components["E"][1, 0] = .13
    families, weights = exact_pair_tables(components)
    result = fit_pairwise_multi(families, components=tuple(c for c in components if c != "E"),
                                sampling="ipw", weights=weights, tol=1e-14)
    for name, matrix in components.items():
        assert result.components[name] == pytest.approx(matrix, abs=3e-6)
    assert result.rg[0, 1] == pytest.approx(-.12/math.sqrt(.4*.3), abs=1e-5)
    expected_re = .13/math.sqrt(components["E"][0, 0]*components["E"][1, 1])
    assert result.re[0, 1] == pytest.approx(expected_re, abs=1e-5)
    assert np.diag(result.rp) == pytest.approx(np.ones(2))
    assert result.inference_status == "interior_cluster_sandwich"
    assert np.isfinite(result.se["re"]).all()
    assert result.n_pairs == len(families)
    scaled = fit_pairwise_multi(families, components=tuple(c for c in components if c != "E"),
                                sampling="ipw", weights=weights*1e80, tol=1e-14)
    assert scaled.h2 == pytest.approx(result.h2, abs=1e-10)
    assert scaled.parameter_covariance == pytest.approx(result.parameter_covariance, rel=1e-8)


def test_three_trait_decomposition_is_coherent():
    components = {"A": np.array([[.4, .1, -.05], [.1, .3, .08], [-.05, .08, .5]]),
                  "E": np.array([[.6, -.1, .08], [-.1, .7, .15], [.08, .15, .5]])}
    families, weights = exact_pair_tables(components)
    result = fit_pairwise_multi(families, sampling="ipw", weights=weights, tol=1e-14)
    for key, truth in components.items():
        assert result.components[key] == pytest.approx(truth, abs=4e-6)
        assert np.linalg.eigvalsh(result.components[key]).min() > 0
    assert result.rp == pytest.approx(components["A"]+components["E"], abs=4e-6)


def test_trait_and_role_permutations_preserve_estimates():
    components = {"A": np.array([[.4, -.1], [-.1, .3]]),
                  "E": np.array([[.6, .2], [.2, .7]])}
    families, weights = exact_pair_tables(components)
    reference = fit_pairwise_multi(families, sampling="ipw", weights=weights, tol=1e-14)
    permuted = [Family(f.fam_id, [Member(m.role, np.asarray(m.lower)[::-1],
                                        np.asarray(m.upper)[::-1]) for m in f.members[::-1]])
                for f in families]
    result = fit_pairwise_multi(permuted, sampling="ipw", weights=weights, tol=1e-14)
    for key, matrix in reference.components.items():
        assert result.components[key] == pytest.approx(matrix[::-1, ::-1], abs=2e-6)


def test_safe_trial_extension_has_the_stated_gradient():
    x = np.array([.97])
    thresholds, design, counts = np.array([[1.3, .4]]), np.ones((1, 1)), np.array([[2., 3., 4., 10.]])
    limits = np.array([[-.8, .8]])
    _, grad, _, _ = _multi_criterion(x, thresholds, design, counts, limits)
    step = 1e-6
    plus = _multi_criterion(x+step, thresholds, design, counts, limits)[0]
    minus = _multi_criterion(x-step, thresholds, design, counts, limits)[0]
    assert grad[0] == pytest.approx((plus-minus)/(2*step), rel=1e-8)


def test_analytic_objective_and_delta_method_match_finite_differences():
    order, offset, basis = _covariance_basis(("A",), 2)
    x = np.array([.4, -.1, .3, .15])
    design = np.array([[.5, 0, 0, 0], [0, .5, 0, 0], [0, 0, .5, 0], [0, 1, 0, 1]])
    thresholds = np.array([[1., 1.], [1., .4], [.4, .4], [1., .4]])
    counts = np.array([[10, 20, 30, 80], [20, 30, 15, 75], [40, 20, 20, 60], [30, 35, 10, 65]])
    _, gradient, hessian, _ = _multi_criterion(x, thresholds, design, counts)
    step = 1e-6
    for j in range(len(x)):
        hi, lo = [_multi_criterion(x+s*np.eye(len(x))[j]*step, thresholds, design, counts) for s in (1, -1)]
        assert gradient[j] == pytest.approx((hi[0]-lo[0])/(2*step), rel=1e-6)
        assert hessian[:, j] == pytest.approx((hi[1]-lo[1])/(2*step), rel=1e-6, abs=1e-7)
    mats = offset + np.einsum('cpqt,t->cpq', basis, x)
    covariance = np.diag([.01, .02, .03, .04])
    se = _derived_se(mats, basis, covariance, 0)
    for key, ci in [("rg", 0), ("re", 1)]:
        def corr(z):
            m = offset + np.einsum('cpqt,t->cpq', basis, z)
            return m[ci, 0, 1]/math.sqrt(m[ci, 0, 0]*m[ci, 1, 1])
        numeric = np.array([(corr(x+step*v)-corr(x-step*v))/(2*step) for v in np.eye(len(order))])
        assert se[key][0, 1] == pytest.approx(math.sqrt(numeric@covariance@numeric), rel=1e-8)


def test_missing_traits_and_absent_shared_environment_contrasts_are_rejected():
    families = [Family(i, [Member('o', [0., -np.inf], [np.inf, np.inf]),
                            Member('m', [-np.inf, -np.inf], [0., np.inf])]) for i in range(4)]
    with pytest.raises(ValueError, match="not identified"):
        fit_pairwise_multi(families, sampling="population")
    components = {"A": np.array([[.4, .1], [.1, .3]]), "E": np.diag([.6, .7])}
    families, weights = exact_pair_tables(components)
    # Drop the sibship contrasts but retain within-person and parent-offspring.
    keep = [i for i, f in enumerate(families) if 's1' not in [m.role for m in f.members]]
    with pytest.raises(ValueError, match="not identified"):
        fit_pairwise_multi([families[i] for i in keep], components=("A", "C"),
                           sampling="ipw", weights=weights[keep])


def test_overlap_and_disabled_optimization_are_rejected():
    components = {"A": np.array([[.4, .1], [.1, .3]]), "E": np.diag([.6, .7])}
    families, weights = exact_pair_tables(components)
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_pairwise_multi(families, sampling="ipw", weights=weights, maxiter=1)
    families[0].members[0].pid = families[1].members[0].pid = "same-person"
    with pytest.raises(ValueError, match="appears in families"):
        fit_pairwise_multi(families, sampling="ipw", weights=weights)


def test_boundary_withholds_uncertainty_and_zero_variance_correlations():
    components = {"A": np.zeros((2, 2)), "E": np.array([[1., .2], [.2, 1.]])}
    families, weights = exact_pair_tables(components)
    with pytest.warns(RuntimeWarning, match="unavailable_boundary"):
        result = fit_pairwise_multi(families, sampling="ipw", weights=weights, tol=1e-14)
    assert result.at_boundary
    assert result.h2 == pytest.approx([0., 0.], abs=result.boundary_tolerance)
    assert np.isnan(result.rg).all()
    assert np.isnan(result.se["rg"]).all()
    assert np.isnan(result.parameter_covariance).all()
