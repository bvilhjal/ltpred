"""Independent bivariate oracles and rejected-state checks for pairwise fitting."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from ltpred.family import Family, Member
from ltpred.pairwise import fit_pairwise, _pair_probabilities, _criterion


def table_families(counts, roles=("o", "m"), threshold=0., start=0):
    families = []
    for count, (first, second) in zip(counts, [(1, 1), (1, 0), (0, 1), (0, 0)]):
        for _ in range(count):
            members = [Member(role, threshold if status else -np.inf,
                              np.inf if status else threshold)
                       for role, status in zip(roles, [first, second])]
            families.append(Family(start + len(families), members))
    return families


@pytest.mark.parametrize("rho", [0., 0.1, 0.8, 0.999999])
def test_zero_threshold_arcsine_probability_oracle(rho):
    prob, first, _second = _pair_probabilities(0., rho)
    both = 0.25 + math.asin(rho) / (2 * math.pi)
    assert prob == pytest.approx([both, 0.5 - both, 0.5 - both, both], abs=1e-15)
    density = 1 / (2 * math.pi * math.sqrt(1 - rho * rho))
    assert first == pytest.approx(density * np.array([1, -1, -1, 1]))


@pytest.mark.parametrize("threshold,rho", [(1.5, 0.3), (8., 0.9), (-8., 0.9), (3., 0.999999)])
def test_probability_matches_conditional_normal_integral_in_tails(threshold, rho):
    prob, _first, _second = _pair_probabilities(threshold, rho)
    # An independent conditional-normal integral, not Plackett's formula.
    sd = math.sqrt(1 - rho * rho)
    oracle = quad(lambda x: norm.pdf(x) * ndtr((rho * x - threshold) / sd),
                  threshold, np.inf, epsabs=0, epsrel=2e-10)[0]
    assert prob[0] == pytest.approx(oracle, rel=2e-8, abs=1e-100)
    assert np.all(prob > 0)
    assert prob.sum() == pytest.approx(1., abs=2e-14)
    assert prob[0] + prob[1] == pytest.approx(ndtr(-threshold), rel=2e-12)


def test_probability_derivatives_and_objective_hessian():
    threshold, rho, step = 1.2, .35, 1e-5
    prob, first, second = _pair_probabilities(threshold, rho)
    plus = _pair_probabilities(threshold, rho + step)
    minus = _pair_probabilities(threshold, rho - step)
    assert first == pytest.approx((plus[0] - minus[0]) / (2 * step), rel=2e-8, abs=1e-10)
    assert second == pytest.approx((plus[1] - minus[1]) / (2 * step), rel=2e-8, abs=1e-10)
    design = np.array([[.5, 0], [.5, 1]])
    counts = np.array([[15., 70, 70, 345], [25., 60, 60, 355]])
    x = np.array([.4, .15])
    value, gradient, hessian, _scores = _criterion(x, threshold, design, counts)
    for j in range(2):
        delta = np.eye(2)[j] * step
        hi, lo = _criterion(x + delta, threshold, design, counts), _criterion(x - delta, threshold, design, counts)
        assert gradient[j] == pytest.approx((hi[0] - lo[0]) / (2 * step), rel=1e-7, abs=1e-7)
        assert hessian[:, j] == pytest.approx((hi[1] - lo[1]) / (2 * step), rel=1e-7, abs=1e-7)
    assert np.isfinite(value) and np.all(prob > 0)


def test_exact_table_mle_and_independent_binomial_sandwich_oracle():
    counts = [300, 200, 200, 300]
    families = table_families(counts)
    result = fit_pairwise(families, sampling="population", tol=1e-12)
    concordance = .6
    expected = 2 * math.sin(math.pi * (concordance - .5))
    # Delta method for the binomial concordance proportion, with the documented
    # n/(n-1) empirical-score correction.
    variance = (2 * math.pi * math.cos(math.pi * (concordance - .5))) ** 2 \
        * concordance * (1 - concordance) / (len(families) - 1)
    assert result.components["A"] == pytest.approx(expected, abs=2e-7)
    assert result.covariance[0, 0] == pytest.approx(variance, rel=2e-6)
    assert result.se["A"] == pytest.approx(math.sqrt(variance), rel=1e-6)
    assert result.inference_status == "interior_cluster_sandwich"
    assert not result.at_boundary
    assert result.n_families == result.n_pairs == 1000


def test_sandwich_keeps_dependence_of_pairs_within_families():
    # Siblings share exactly the mother's status in these deliberately dependent
    # synthetic observations. Naive pairwise Hessian-only SEs must not be used.
    families = table_families([100, 150, 150, 100])
    for family in families:
        mother = family.members[1]
        family.members.append(Member("s1", mother.lower, mother.upper))
    result = fit_pairwise(families, sampling="population", tol=1e-12)
    assert result.inference_status == "interior_cluster_sandwich"
    x = result.components["A"]
    score_by_family = []
    bread = 0.
    for family in families:
        status = [np.isfinite(member.lower) for member in family.members]
        score = 0.
        for i in range(3):
            for j in range(i + 1, 3):
                cell = 2 * int(not status[i]) + int(not status[j])
                prob, first, second = _pair_probabilities(0., .5 * x)
                score += .5 * first[cell] / prob[cell]
                bread += .25 * ((first[cell] / prob[cell]) ** 2 - second[cell] / prob[cell])
        score_by_family.append(score)
    score_by_family = np.asarray(score_by_family)
    expected = np.var(score_by_family, ddof=1) * len(families) / bread ** 2
    assert result.covariance[0, 0] == pytest.approx(expected, rel=1e-10)
    assert result.covariance[0, 0] != pytest.approx(1 / bread, rel=.05)


def test_multiple_components_recover_exact_independent_pair_optima():
    specifications = [([112, 88, 88, 112], ("o", "m")),
                      ([125, 75, 75, 125], ("o", "s1")),
                      ([110, 90, 90, 110], ("m", "f"))]
    families = []
    for counts, roles in specifications:
        families += table_families(counts, roles, start=len(families))
    result = fit_pairwise(families, components=("A", "C", "M"), sampling="population", tol=1e-12)
    rho_parent = math.sin(math.pi * (.56 - .5))
    expected = {"A": 2 * rho_parent, "C": math.sin(math.pi * (.625 - .5)) - rho_parent,
                "M": math.sin(math.pi * (.55 - .5))}
    assert result.components == pytest.approx(expected, abs=3e-6)
    assert result.component_order == ("A", "C", "M")
    assert result.residual == pytest.approx(1 - sum(expected.values()), abs=3e-6)
    assert result.inference_status == "interior_cluster_sandwich"
    assert np.linalg.eigvalsh(result.covariance).min() > 0


@pytest.mark.parametrize("counts,expected", [([100, 100, 100, 100], 0.), ([180, 20, 20, 180], 1 - 1e-6)])
def test_boundaries_withhold_normal_uncertainty(counts, expected):
    with pytest.warns(RuntimeWarning, match="unavailable_boundary"):
        result = fit_pairwise(table_families(counts), sampling="population", tol=1e-12)
    assert result.components["A"] == pytest.approx(expected, abs=1e-7)
    assert result.at_boundary
    assert np.isnan(result.covariance).all() and np.isnan(result.se["A"])
    assert result.residual > 0


def test_ipw_recovers_population_table_and_is_scale_invariant():
    population = table_families([30, 20, 20, 30])
    selected = table_families([90, 60, 20, 30])
    weights = np.array([1 / 3] * 150 + [1.] * 50)
    expected = fit_pairwise(population, sampling="population", tol=1e-12)
    actual = fit_pairwise(selected, sampling="ipw", weights=weights, tol=1e-12)
    scaled = fit_pairwise(selected, sampling="ipw", weights=weights * 1e100, tol=1e-12)
    assert actual.components == pytest.approx(expected.components, abs=1e-6)
    assert scaled.components == pytest.approx(actual.components, abs=1e-10)
    assert scaled.covariance == pytest.approx(actual.covariance, rel=1e-12)


@pytest.mark.parametrize("sampling,weights,match", [
    ("ipw", None, "requires weights"),
    ("population", np.ones(4), "weights are not accepted"),
    ("ipw", [1., 1., 0., 1.], "strictly positive"),
    ("ipw", [1., 1., np.inf, 1.], "finite"),
    ("ipw", [1.], "one entry per family"),
    ("case_control", None, "supports sampling"),
])
def test_rejects_invalid_sampling_contracts(sampling, weights, match):
    with pytest.raises(ValueError, match=match):
        fit_pairwise(table_families([1, 1, 1, 1]), sampling=sampling, weights=weights)


def test_population_screen_and_overlap_are_not_bypassed():
    with pytest.raises(ValueError, match="not consistent"):
        fit_pairwise(table_families([100, 100, 100, 100], threshold=norm.isf(.01)), sampling="population")
    families = table_families([1, 1, 1, 1])
    families[0].members[1].pid = "shared"
    families[1].members[1].pid = "shared"
    with pytest.raises(ValueError, match="appears in families"):
        fit_pairwise(families, sampling="population")


@pytest.mark.parametrize("lo,hi,match", [(0., 0., "onset-pinned"), (-1., 1., "two-sided"),
                                         (-np.inf, .3, "differ between"), (np.nan, np.inf, "NaN")])
def test_rejects_unsupported_or_invalid_bounds(lo, hi, match):
    families = table_families([1, 1, 1, 1])
    families[0].members[0].lower = lo
    families[0].members[0].upper = hi
    with pytest.raises(ValueError, match=match):
        fit_pairwise(families, sampling="population")


def test_identifiability_uses_observed_pairs_only():
    families = table_families([3, 2, 2, 3])
    for family in families:
        family.members.append(Member("s1", -np.inf, np.inf))
    with pytest.raises(ValueError, match="not identified"):
        fit_pairwise(families, components=("A", "C"), sampling="population")
    result = fit_pairwise(families, sampling="population")
    reference = fit_pairwise(table_families([3, 2, 2, 3]), sampling="population")
    assert result.components == pytest.approx(reference.components)
    assert result.n_pairs == reference.n_pairs
    lone = [Family(0, [Member("o", -np.inf, 0.)])]
    with pytest.raises(ValueError, match="not identified"):
        fit_pairwise(lone, sampling="population")


def test_failed_and_invalid_optimizer_results_are_rejected(monkeypatch):
    import ltpred.pairwise as pairwise
    families = table_families([3, 2, 2, 3])
    for success, values in [(False, [.5]), (True, [np.nan]), (True, [1.2]), (True, [-.2])]:
        monkeypatch.setattr(pairwise, "minimize", lambda *a, **k: SimpleNamespace(
            success=success, x=np.asarray(values), fun=1., message="test failure", nit=1))
        with pytest.raises(RuntimeError, match="optimization failed"):
            fit_pairwise(families, sampling="population")


def test_iteration_failure_does_not_return_a_candidate():
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_pairwise(table_families([30, 20, 20, 30]), sampling="population", maxiter=1)


def test_success_flag_does_not_hide_nonstationary_candidate(monkeypatch):
    import ltpred.pairwise as pairwise
    monkeypatch.setattr(pairwise, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, x=np.array([.1]), fun=1., message="success", nit=1))
    with pytest.raises(RuntimeError, match="constrained score check"):
        fit_pairwise(table_families([30, 20, 20, 30]), sampling="population")


@pytest.mark.parametrize("kwargs", [{"components": ()}, {"components": ("A", "A")},
                                    {"components": ("D",)}, {"eps": 0.},
                                    {"maxiter": 0}, {"maxiter": True}, {"tol": np.nan}])
def test_invalid_controls(kwargs):
    with pytest.raises((ValueError, TypeError)):
        fit_pairwise(table_families([3, 2, 2, 3]), sampling="population", **kwargs)
