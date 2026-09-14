"""Known-truth posterior moments for nuclear-family factor quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp

from ltpred import (Family, Member, QuadratureResult, estimate_liability,
                    estimate_liability_from_kinship)
from ltpred.covariance import construct_covmat_single
from ltpred.quadrature import estimate_liability_quadrature_arrays as estimate
from ltpred.quadrature import _grid


def _dense_two_interval_oracle(cov, lo, hi, observed, target):
    """Integrate two observed liabilities directly, without parental factors."""
    c = cov[np.ix_(observed, observed)]
    cross = cov[target, observed]
    beta = np.linalg.solve(c, cross)
    residual = cov[target, target] - cross @ beta
    sd1 = np.sqrt(c[0, 0])
    sd2 = np.sqrt(c[1, 1] - c[1, 0]**2 / c[0, 0])

    def integrands(x):
        mu2 = c[1, 0] / c[0, 0] * x
        a, b = (lo[1] - mu2) / sd2, (hi[1] - mu2) / sd2
        mass = norm.cdf(b) - norm.cdf(a)
        if mass == 0.0:
            return np.zeros(3)
        m2, v2 = truncnorm.stats(a, b, loc=mu2, scale=sd2, moments="mv")
        m = beta[0] * x + beta[1] * m2
        wt = norm.pdf(x / sd1) / sd1 * mass
        return wt * np.array([1.0, m, residual + beta[1]**2 * v2 + m*m])

    lower, upper = max(lo[0], -12.0), min(hi[0], 12.0)
    values = np.array([quad(lambda x: integrands(x)[i], lower, upper,
                            epsabs=1e-11, epsrel=1e-11)[0] for i in range(3)])
    mean = values[1] / values[0]
    return mean, values[2] / values[0] - mean**2


@pytest.mark.parametrize("out", ["genetic", "full"])
@pytest.mark.parametrize("h2", [0.0, 0.2, 0.8])
@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), (1.3, np.inf), (-2.0, 0.4), (3.0, 3.0)])
def test_adult_is_analytic(out, h2, bounds):
    lo, hi = bounds
    m, v = (lo, 0.0) if lo == hi else truncnorm.stats(lo, hi, moments="mv")
    expected = (h2*m, h2 - h2*h2 + h2*h2*v) if out == "genetic" else (m, v)
    result = estimate(["o"], [[lo]], [[hi]], h2=h2, out=out)
    assert result.est[0] == pytest.approx(expected[0], abs=1e-12)
    assert result.var[0] == pytest.approx(expected[1], abs=1e-12)
    assert result.error[0] == 0.0
    assert result.n_nodes[0] == 0


@pytest.mark.parametrize("out", ["genetic", "full"])
def test_all_pins_equal_dense_gaussian_conditioning(out):
    roles = ["o", "m", "f", "s1", "s2"]
    values = np.array([2.3, -0.4, 1.4, 0.2, 4.0])
    cov = construct_covmat_single(fam_vec=roles, h2=0.7).matrix
    target = 0 if out == "genetic" else 1
    weight = np.linalg.solve(cov[1:, 1:], cov[1:, target])
    mean = weight @ values
    variance = cov[target, target] - weight @ cov[1:, target]
    result = estimate(roles, values[None, :], values[None, :], h2=0.7, out=out)
    assert result.est[0] == pytest.approx(mean, abs=1e-12)
    assert result.var[0] == pytest.approx(variance, abs=1e-12)
    assert result.n_nodes[0] == 0


def test_pinned_then_one_interval_matches_independent_closed_form():
    result = estimate(["o", "m"], [[3.0, -np.inf]], [[3.0, norm.isf(0.05)]], h2=0.5)
    assert result.est[0] == pytest.approx(1.4591376235349836, abs=2e-12)
    assert result.var[0] == pytest.approx(0.24345482008488772, abs=2e-12)
    assert result.n_nodes[0] == 0


@pytest.mark.parametrize("roles,bounds", [
    (["o", "s1"], ([1.0, -np.inf], [np.inf, 0.1])),
    (["m", "s1"], ([-0.5, 1.2], [0.2, np.inf])),
    (["o", "m"], ([1.1, -1.2], [1.10001, 0.5])),
    (["m", "f"], ([1.0, -0.6], [2.0, 0.4])),
])
@pytest.mark.parametrize("out", ["genetic", "full"])
def test_two_intervals_match_direct_liability_integration(roles, bounds, out):
    lower, upper = bounds
    cov_obj = construct_covmat_single(fam_vec=roles, h2=0.65)
    observed = [cov_obj.roles.index(role) for role in roles]
    target = cov_obj.roles.index("g" if out == "genetic" else "o")
    expected = _dense_two_interval_oracle(cov_obj.matrix, lower, upper, observed, target)
    result = estimate(roles, [lower], [upper], h2=0.65, out=out)
    assert result.est[0] == pytest.approx(expected[0], abs=2e-8)
    assert result.var[0] == pytest.approx(expected[1], abs=2e-8)
    assert result.error[0] <= 1e-8
    assert result.n_nodes[0] == 0 if roles == ["m", "f"] else 64 <= result.n_nodes[0] <= 128


def test_nuclear_family_matches_reference_and_role_permutations():
    threshold = norm.isf(0.05)
    roles = ["o", "m", "f", "s1", "s2", "s3", "s4"]
    lower = np.array([[2.0, -np.inf, threshold, threshold, -np.inf, -np.inf, -np.inf]])
    upper = np.array([[2.0, threshold, np.inf, np.inf, threshold, threshold, threshold]])
    result = estimate(roles, lower, upper, h2=0.5)
    # Independently checked against 1e6-draw collapsed Gibbs (MCSE .0001473)
    # and the 20/40/80-node non-adaptive parental-factor prototype.
    assert result.est[0] == pytest.approx(1.239036113244667, abs=2e-10)
    assert result.var[0] == pytest.approx(0.213147155892779, abs=2e-10)
    order = [6, 2, 0, 4, 1, 5, 3]
    shuffled = estimate([roles[i] for i in order], lower[:, order], upper[:, order], h2=0.5)
    assert shuffled.est == pytest.approx(result.est, abs=2e-12)
    assert shuffled.var == pytest.approx(result.var, abs=2e-12)
    swapped_roles = ["f" if r == "m" else "m" if r == "f" else r for r in roles]
    swapped = estimate(swapped_roles, lower, upper, h2=0.5)
    assert swapped.est == pytest.approx(result.est, abs=2e-12)
    assert swapped.var == pytest.approx(result.var, abs=2e-12)


def test_absent_proband_and_unobserved_parents_do_not_change_answer():
    result = estimate(["s1", "s2"], [[1.0, -np.inf]], [[np.inf, 0.5]], h2=0.5)
    augmented = estimate(["o", "m", "f", "s1", "s2"],
                         [[-np.inf, -np.inf, -np.inf, 1.0, -np.inf]],
                         [[np.inf, np.inf, np.inf, np.inf, 0.5]], h2=0.5)
    assert result.est == pytest.approx(augmented.est, abs=1e-13)
    assert result.var == pytest.approx(augmented.var, abs=1e-13)
    assert result.n_nodes.tolist() == augmented.n_nodes.tolist()


@pytest.mark.parametrize("h2", [0.8, 0.99, 0.9999])
def test_independent_parents_known_truth(h2):
    threshold = norm.isf(0.01)
    m, v = truncnorm.stats(threshold, np.inf, moments="mv")
    result = estimate(["m", "f"], [[threshold, threshold]], [[np.inf, np.inf]], h2=h2)
    assert result.est[0] == pytest.approx(h2*m, abs=1e-10)
    assert result.var[0] == pytest.approx(h2 - 0.5*h2*h2*(1-v), abs=1e-10)
    assert result.n_nodes[0] == 0


def test_rare_opposing_intervals_reflect_and_remain_finite():
    lo, hi = np.array([[8.0, -np.inf]]), np.array([[np.inf, -8.0]])
    positive = estimate(["o", "s1"], lo, hi, h2=0.5)
    negative = estimate(["o", "s1"], -hi, -lo, h2=0.5)
    assert positive.est == pytest.approx(-negative.est, abs=1e-10)
    assert positive.var == pytest.approx(negative.var, abs=1e-10)
    assert np.all(np.isfinite(positive.est))
    assert np.all(positive.var > 0)
    # Pinning far into the tail must recenter the Gaussian, not rely on nodes
    # around the unconditioned prior. One remaining interval is then analytic.
    pinned = estimate(["o", "s1"], [[100.0, -np.inf]], [[100.0, -8.0]], h2=0.5)
    assert np.isfinite(pinned.est[0]) and pinned.var[0] > 0.0
    assert pinned.n_nodes[0] == 0


def test_narrow_interval_converges_to_pin_without_collapsing_it():
    lo = np.array([[2.0, 0.5, -np.inf]])
    hi = np.array([[2.0 + 1e-6, np.inf, 0.0]])
    narrow = estimate(["o", "m", "s1"], lo, hi, h2=0.5, out="full")
    assert 2.0 <= narrow.est[0] <= 2.0 + 1e-6
    assert narrow.var[0] == pytest.approx(1e-12 / 12.0, rel=1e-4)
    pinned = estimate(["o", "m", "s1"], lo, [[2.0, np.inf, 0.0]], h2=0.5, out="full")
    assert pinned.est[0] == 2.0 and pinned.var[0] == 0.0


def test_batch_and_empty_arrays():
    result = estimate(["m", "f"], [[1.0, 0.5], [-np.inf, -np.inf]],
                      [[np.inf, np.inf], [np.inf, np.inf]], h2=0.5)
    assert result.est.shape == result.var.shape == result.error.shape == result.n_nodes.shape == (2,)
    assert result.est[1] == 0.0 and result.var[1] == 0.5
    assert estimate([], np.empty((1, 0)), np.empty((1, 0)), h2=0.5).var[0] == 0.5
    assert estimate([], np.empty((0, 0)), np.empty((0, 0)), h2=0.5).est.size == 0


@pytest.mark.parametrize("kwargs", [{"h2": 1.0}, {"h2": -0.1}, {"h2": np.nan},
                                     {"h2": True}, {"h2": 0.5, "out": "risk"}, {"h2": 0.5, "atol": 0.0},
                                     {"h2": 0.5, "atol": np.inf}, {"h2": 0.5, "max_nodes": 32},
                                     {"h2": 0.5, "max_nodes": 64.5}, {"h2": 0.5, "max_nodes": True},
                                     {"h2": 0.5, "max_nodes": 513}])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        estimate(["o"], [[0.0]], [[np.inf]], **kwargs)


@pytest.mark.parametrize("roles", [["mgm"], ["s0"], ["m", "m"], ["g"], [1]])
def test_unsupported_or_duplicate_roles(roles):
    with pytest.raises(ValueError):
        estimate(roles, np.zeros((1, len(roles))), np.ones((1, len(roles))), h2=0.5)


@pytest.mark.parametrize("lo,hi", [([[np.nan]], [[1.0]]), ([[2.0]], [[1.0]]),
                                  ([[np.inf]], [[np.inf]]), ([0.0], [1.0]),
                                  ([[0.0]], [[1.0, 2.0]])])
def test_invalid_bounds(lo, hi):
    with pytest.raises(ValueError):
        estimate(["o"], lo, hi, h2=0.5)


def test_nonconvergence_raises_instead_of_returning_unchecked_value():
    with pytest.raises(RuntimeError, match="family 0.*did not converge"):
        estimate(["o", "m", "f", "s1"], [[1.2, -0.3, 2.0, -np.inf]],
                 [[np.inf, 0.4, np.inf, 0.0]], h2=0.95,
                 atol=1e-15, max_nodes=64)


def test_largest_grid_keeps_log_tail_weights_and_normal_moments():
    nodes, logweights = _grid(512, 1)
    assert np.all(np.isfinite(logweights))
    assert logsumexp(logweights) == pytest.approx(0.5 * np.log(2*np.pi), abs=1e-12)
    weight = np.exp(logweights - logsumexp(logweights))
    assert weight @ nodes[:, 0] == pytest.approx(0.0, abs=1e-13)
    assert weight @ nodes[:, 0]**2 == pytest.approx(1.0, abs=1e-12)


def test_high_heritability_rare_family_fails_at_insufficient_resolution():
    with pytest.raises(RuntimeError, match="family 0.*did not converge"):
        estimate(["o", "m", "f", "s1"], [[3.0, 2.0, 2.0, -np.inf]],
                 [[np.inf, np.inf, np.inf, -1.0]], h2=0.9999, max_nodes=128)


@pytest.mark.parametrize("lower, upper, out, est, var", [
    # The Newton step can stop shrinking above the step tolerance while the
    # objective is already stationary, because _moments sets the accuracy of
    # the exact gradient. These four refused before the mode iteration tested
    # progress as well as step length. Values independently reproduced by an
    # 800x800 Gauss-Legendre integration over the two parental breeding values
    # (agreement <=2e-15) and 40M-draw rejection sampling (|z|<=1.8).
    ([-1.324, -1.238, -1.508], [0.796, 1.2, 0.958], "genetic",
     -0.10999237341204565, 0.2967520244504461),
    ([-1.324, -1.238, -1.508], [0.796, 1.2, 0.958], "full",
     -0.18845903980960627, 0.31476283058652166),
    ([-2.1959, -0.9243, 0.1858], [-2.1344, 0.667, 1.1224], "genetic",
     -0.8795880576944444, 0.21947931082299538),
    ([0.8492, 0.5973, -0.9887], [2.065, 1.4144, 0.5168], "genetic",
     0.6745461290871979, 0.23838389999239584),
])
def test_stationary_objective_is_the_mode_not_a_failure(lower, upper, out, est, var):
    result = estimate(["o", "m", "f"], [lower], [upper], h2=0.5, out=out)
    assert result.est[0] == pytest.approx(est, abs=2e-12)
    assert result.var[0] == pytest.approx(var, abs=2e-12)
    assert result.error[0] <= 1e-8


@pytest.mark.parametrize("style", ["two_sided", "one_sided", "pinned_proband"])
def test_randomised_bounds_do_not_refuse_at_the_posterior_mode(style):
    # Hand-picked round bounds miss this: the stall needs a mode whose Newton
    # step lands near the gradient's own noise floor, which is generic rather
    # than special. Only max_nodes can rescue a refinement refusal, so a mode
    # refusal must not stand in for one.
    rng = np.random.default_rng(20260910)
    roles = ["o", "m", "f"]
    for _ in range(200):
        if style == "two_sided":
            lower = rng.normal(-0.3, 1.0, 3)
            upper = lower + np.abs(rng.normal(0.0, 1.0, 3)) + 1e-3
        elif style == "one_sided":
            lower = rng.normal(0.0, 1.0, 3)
            tail = rng.random(3) < 0.5
            upper = np.where(tail, lower, np.inf)
            lower = np.where(tail, -np.inf, lower)
        else:
            lower = rng.normal(0.0, 1.0, 3)
            upper = np.full(3, np.inf)
            lower[0] = upper[0] = rng.normal(0.0, 1.0)
        try:
            estimate(roles, [lower], [upper], h2=0.5, max_nodes=64)
        except RuntimeError as exc:
            assert "posterior-mode" not in str(exc), (lower, upper)
            assert "nodes per dimension" in str(exc)


def test_refusal_reports_the_change_pair_the_criterion_tests():
    # Acceptance needs both of the last two changes below atol. At max_nodes=64
    # only two exist, so the coarse 16->32 refinement is binding and the final
    # change alone can look convergent.
    with pytest.raises(RuntimeError, match=r"largest of the last two changes 7\.76e-06"):
        estimate(["o", "m", "f"], [[4.600652, 3.112669, 3.918868]],
                 [[np.inf, np.inf, np.inf]], h2=0.95, max_nodes=64)
    relaxed = estimate(["o", "m", "f"], [[4.600652, 3.112669, 3.918868]],
                       [[np.inf, np.inf, np.inf]], h2=0.95, max_nodes=128)
    assert relaxed.error[0] <= 1e-8


def test_quadrature_object_adapter_preserves_alignment_and_diagnostics():
    families = [Family('a', [Member('o', 2, 2, pid='x'), Member('m', -np.inf, 1.5)]),
                Family('b', [Member('o', -np.inf, 1, pid='y')]),
                Family('c', [Member('m', -np.inf, 1.5), Member('o', 1, 1, pid='z')])]
    result = estimate_liability(families, h2=.5, method='quadrature',
                                out=['genetic', 'full'])
    assert result.fam_ids.tolist() == ['a', 'b', 'c']
    assert result.pids.tolist() == ['x', 'y', 'z']
    for name in ('genetic', 'full'):
        for i, family in enumerate(families):
            ref = estimate([m.role for m in family.members],
                           [[m.lower for m in family.members]],
                           [[m.upper for m in family.members]], h2=.5, out=name)
            assert isinstance(ref, QuadratureResult)
            assert result.est[name][i] == pytest.approx(ref.est[0], abs=1e-14)
            assert result.var[name][i] == pytest.approx(ref.var[0], abs=1e-14)
            assert result.quadrature_error[name][i] == ref.error[0]
            assert result.quadrature_nodes[name][i] == ref.n_nodes[0]
        assert not result.se[name].any()


@pytest.mark.parametrize('kwargs,match', [
    ({'h2': [.5, .5]}, 'single-trait'), ({'h2': .5, 'use_mixture': True}, 'without'),
    ({'h2': .5, 'c2': .1}, 'without'), ({'h2': .5, 'm2': .1}, 'without'),
    ({'h2': .5, 'quadrature_max_nodes': 32}, 'max_nodes'),
    ({'h2': .5, 'quadrature_atol': -1}, 'atol')])
def test_quadrature_dispatch_rejects_unsupported_requests(kwargs, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        estimate_liability([Family('f', [Member('o', -np.inf, 1)])],
                           method='quadrature', **kwargs)


def test_quadrature_is_not_silently_used_as_gibbs_for_kinship():
    with pytest.raises(NotImplementedError, match='nuclear-family'):
        estimate_liability_from_kinship(np.eye(1), [[-np.inf]], [[1]], h2=.5,
                                        method='quadrature')
