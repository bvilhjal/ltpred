"""End-to-end liability estimation and the batch-means convergence rule."""

import warnings

import numpy as np
import pytest
from scipy import stats

from ltpred.covariance import construct_covmat_multi, construct_covmat_single
from ltpred.family import Family, Member, families_from_columns
from ltpred.estimate import (batch_means, estimate_liability,
                             estimate_liability_from_kinship,
                             estimate_liability_gibbs_arrays,
                             estimate_liability_pa_arrays,
                             _estimate_liability_multi, _base_seeds)
from ltpred.fit import fit_heritability, fit_variance_components
from ltpred.simulate import simulate_under_LTM_single
from ltpred.thresholds import age_thresholds

from _helpers import imr


def test_batch_means_mean_and_se():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, size=10_000)
    est, se = batch_means(x)
    assert est[0] == pytest.approx(x.mean())
    assert 0 < se[0] < 0.1


def test_batch_means_se_shrinks_with_n():
    rng = np.random.default_rng(1)
    small = batch_means(rng.normal(0, 1, 2_500))[1][0]
    big = batch_means(rng.normal(0, 1, 40_000))[1][0]
    assert big < small


@pytest.mark.parametrize("n_sim", [0, 1, 2, 3])
def test_gibbs_estimation_rejects_draw_counts_without_two_batches(n_sim):
    fam = Family("f", [Member("o", -np.inf, 0.0)])
    with pytest.raises(ValueError, match="n_sim must be at least 4"):
        estimate_liability(
            [fam], h2=0.5, method="gibbs", n_sim=n_sim, burn_in=0, max_rounds=1,
        )


def test_gibbs_minimum_draw_boundary_has_finite_mc_se():
    from ltpred.estimate import estimate_liability_gibbs_arrays

    est, se = estimate_liability_gibbs_arrays(
        ["o"], np.array([[-np.inf]]), np.array([[0.0]]), 0.5,
        n_sim=4, burn_in=0, max_rounds=1, tol=1e9, seed=1,
    )
    assert np.isfinite(est[0])
    assert np.isfinite(se[0])


def test_single_proband_case_only():
    # a lone case with no relatives -> E[g] = h2 * IMR(T)
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    fam = Family("f1", [Member("o", lower=t, upper=np.inf)])
    res = estimate_liability([fam], h2=h2, method="gibbs", out=("genetic", "full"),
                             tol=0.02, n_sim=40_000, burn_in=800, seed=1)
    assert res.est["full"][0] == pytest.approx(imr(t), abs=0.05)
    assert res.est["genetic"][0] == pytest.approx(h2 * imr(t), abs=0.05)
    assert res.se["genetic"][0] <= 0.02


def test_pa_full_matches_gibbs_estimand_on_a_lone_case():
    # PA used to skip the target's own bound, so out="full" was 0 for a lone
    # case. Both engines now return E[l_o | own interval].
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    fam = Family("f1", [Member("o", lower=t, upper=np.inf)])
    pa = estimate_liability([fam], h2=h2, method="pa", out=("genetic", "full"))
    assert pa.est["full"][0] == pytest.approx(imr(t), abs=1e-9)
    assert pa.est["genetic"][0] == pytest.approx(h2 * imr(t), abs=1e-9)
    unbound = Family("f2", [Member("o", lower=-np.inf, upper=np.inf)])
    rel_only = estimate_liability([unbound], h2=h2, method="pa", out="full")
    assert rel_only.est["full"][0] == pytest.approx(0.0)


def test_high_level_gibbs_handles_nine_sigma_case_bound():
    fam = Family("tail", [Member("o", lower=9.0, upper=np.inf)])
    res = estimate_liability([fam], h2=0.5, method="gibbs", out="full",
                             tol=1e9, n_sim=10_000, burn_in=100,
                             max_rounds=1, seed=7)

    assert 9.0 <= res.est["full"][0] < 9.3


def test_adult_is_personalized_bounds_without_family_history():
    """ADuLT uses the proband's age-dependent bound and no relative rows."""
    h2 = 0.5
    lower, upper = age_thresholds([1], [45], pop_prev=0.1)
    fam = Family("adult", [Member("o", lower[0], upper[0])])
    res = estimate_liability([fam], h2=h2, method="pa", out=("genetic",))
    assert res.est["genetic"][0] == pytest.approx(h2 * lower[0])


def test_personalized_family_model_adds_relatives_to_adult_proband_bound():
    """Adding an affected relative changes the score; ADuLT itself has none."""
    lower, upper = age_thresholds([1, 1], [45, 35], pop_prev=0.1)
    adult = Family("adult", [Member("o", lower[0], upper[0])])
    family = Family("family", [Member("o", lower[0], upper[0]),
                               Member("m", lower[1], upper[1])])
    res = estimate_liability([adult, family], h2=0.5, out=("genetic",))
    assert res.est["genetic"][1] > res.est["genetic"][0]


def test_family_history_raises_genetic_estimate():
    # same proband (case); affected relatives should push the genetic
    # liability estimate up relative to unaffected relatives.
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))

    def fam(relative_case, fid):
        lo, hi = (t, np.inf) if relative_case else (-np.inf, t)
        members = [Member("o", t, np.inf)]  # proband is a case
        members += [Member(r, lo, hi) for r in ("m", "f", "s1")]
        return Family(fid, members)

    res = estimate_liability([fam(True, "affected"), fam(False, "healthy")],
                             h2=h2, out=("genetic",), tol=0.02,
                             n_sim=40_000, burn_in=800, seed=2)
    g_affected, g_healthy = res.est["genetic"]
    assert g_affected > g_healthy + 0.1


def test_pa_is_invariant_to_member_and_first_family_order():
    t = float(stats.norm.isf(0.1))
    bounds = {
        "a": {"o": (t, np.inf), "m": (t, np.inf),
              "f": (-np.inf, t), "s1": (t, np.inf)},
        "b": {"o": (-np.inf, t), "m": (t, np.inf),
              "f": (t, np.inf), "s1": (-np.inf, t)},
    }

    def family(fid, roles):
        return Family(fid, [Member(role, *bounds[fid][role], pid=f"{fid}-{role}")
                            for role in roles])

    original = [family("a", ["o", "m", "f", "s1"]),
                family("b", ["s1", "f", "m", "o"])]
    reordered = [family("b", ["o", "m", "f", "s1"]),
                 family("a", ["s1", "f", "m", "o"])]
    expected = estimate_liability(original, h2=0.5, method="pa",
                                  out=("genetic", "full"))
    actual = estimate_liability(reordered, h2=0.5, method="pa",
                                out=("genetic", "full"))

    for name in ("genetic", "full"):
        expected_est = dict(zip(expected.fam_ids, expected.est[name]))
        actual_est = dict(zip(actual.fam_ids, actual.est[name]))
        expected_var = dict(zip(expected.fam_ids, expected.var[name]))
        actual_var = dict(zip(actual.fam_ids, actual.var[name]))
        assert actual_est == expected_est
        assert actual_var == expected_var
    assert dict(zip(actual.fam_ids, actual.pids)) == dict(zip(expected.fam_ids,
                                                             expected.pids))


def test_families_from_columns_round_trip():
    fams = families_from_columns(
        fam_id=["a", "a", "b"],
        role=["o", "m", "o"],
        lower=[1.6, -np.inf, -np.inf],
        upper=[np.inf, 1.6, 1.6],
        pid=["a_o", "a_m", "b_o"],
    )
    assert [f.fam_id for f in fams] == ["a", "b"]
    assert [m.role for m in fams[0].members] == ["o", "m"]
    assert fams[0].members[0].pid == "a_o"


def test_result_pids_default_to_o_member():
    fam = Family("f1", [Member("o", 1.6, np.inf, pid="proband1")])
    res = estimate_liability([fam], h2=0.5, out=("genetic",),
                             n_sim=10_000, burn_in=300, tol=0.1, seed=1)
    assert res.pids[0] == "proband1"


def _duplicate_role_family():
    t = float(stats.norm.isf(0.05))
    return [Family("f1", [Member("o", t, np.inf), Member("s1", t, np.inf),
                          Member("s1", -np.inf, t)])]        # two 's1'


def _duplicate_role_fit_family():
    return [Family(0, [Member("m", -np.inf, 1.0), Member("m", -np.inf, 1.0)])]


@pytest.mark.parametrize("call", [
    pytest.param(lambda: estimate_liability(_duplicate_role_family(), h2=0.5,
                                            out=("genetic",)), id="object-default"),
    pytest.param(lambda: estimate_liability(_duplicate_role_family(), h2=0.5,
                                            method="gibbs", out=("genetic",)), id="object-gibbs"),
    pytest.param(lambda: estimate_liability_pa_arrays(
        ["o", "m", "m"], np.full((2, 3), -np.inf), np.full((2, 3), np.inf), 0.5), id="array-pa"),
    pytest.param(lambda: estimate_liability_gibbs_arrays(
        ["o", "m", "m"], np.full((2, 3), -np.inf), np.full((2, 3), np.inf), 0.5), id="array-gibbs"),
    pytest.param(lambda: construct_covmat_single(fam_vec=["m", "m"], h2=0.5), id="covmat-single-m"),
    pytest.param(lambda: construct_covmat_single(fam_vec=["s1", "s1"], h2=0.5), id="covmat-single-s1"),
    pytest.param(lambda: construct_covmat_multi(fam_vec=["m", "m"], genetic_corrmat=np.eye(2),
                                                full_corrmat=np.eye(2), h2_vec=[0.5, 0.4]), id="covmat-multi"),
    pytest.param(lambda: fit_heritability(_duplicate_role_fit_family(), n_iter=10, burn_in=6,
                                          inner_sweeps=1, sampling="population"), id="fit-heritability"),
    pytest.param(lambda: fit_variance_components(_duplicate_role_fit_family(), ("A",), n_iter=10,
                                                 burn_in=6, inner_sweeps=1, sampling="population"),
                 id="fit-variance-components"),
])
def test_duplicate_roles_are_rejected_at_every_entry_point(call):
    # duplicate singleton roles used to build a singular matrix with two
    # perfectly-correlated "mothers"; every path that assembles a family
    # covariance must refuse them, not just the multi-trait one
    with pytest.raises(ValueError, match="duplicate role"):
        call()


def test_warns_when_not_converged():
    # a punishing tol with max_rounds=1 cannot converge -> warning, but still returns
    t = float(stats.norm.isf(0.05))
    fam = Family("f1", [Member("o", t, np.inf), Member("m", t, np.inf)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = estimate_liability([fam], h2=0.5, method="gibbs", out=("genetic",),
                                 tol=1e-6, n_sim=5000, burn_in=200, max_rounds=1, seed=1)
    assert any("did not reach tol" in str(x.message) for x in w)
    assert np.isfinite(res.est["genetic"][0])


def test_multi_trait_runs_and_shapes():
    h2 = [0.5, 0.4]
    gcorr = np.array([[1.0, 0.3], [0.3, 1.0]])
    fcorr = np.array([[1.0, 0.2], [0.2, 1.0]])
    t = float(stats.norm.isf(0.05))
    # per-member (lower, upper) are length-2 (one per phenotype)
    fam = Family("f1", [
        Member("o", lower=[t, -np.inf], upper=[np.inf, t]),  # case for A, control for B
        Member("m", lower=[-np.inf, t], upper=[t, np.inf]),  # control A, case B
    ])
    res = _estimate_liability_multi([fam], h2_vec=h2, genetic_corrmat=gcorr,
                                   full_corrmat=fcorr, phen_names=["A", "B"],
                                   out=("genetic",), tol=0.05,
                                   n_sim=20_000, burn_in=500, seed=3)
    assert set(res.est) == {"genetic_A", "genetic_B"}
    assert res.est["genetic_A"].shape == (1,)
    # proband is a case for A but not B -> higher genetic liability for A
    assert res.est["genetic_A"][0] > res.est["genetic_B"][0]


@pytest.mark.parametrize("attribute", ["lower", "upper"])
@pytest.mark.parametrize("bad_bound", [0.0, [0.0], [0.0, 0.0, 0.0]])
def test_multitrait_estimation_rejects_scalar_and_wrong_length_bounds(attribute,
                                                                     bad_bound):
    bounds = {
        "lower": [-np.inf, -np.inf],
        "upper": [np.inf, np.inf],
    }
    bounds[attribute] = bad_bound
    fam = Family("f1", [Member("o", bounds["lower"], bounds["upper"])])

    with pytest.raises(ValueError, match=r"must be a length-2 .*sequence"):
        _estimate_liability_multi(
            [fam], h2_vec=[0.5, 0.4], genetic_corrmat=np.eye(2),
            full_corrmat=np.eye(2), n_sim=10, burn_in=0,
        )


def test_multi_trait_rejects_incoherent_genetic_and_full_correlations():
    t = float(stats.norm.isf(0.05))
    fam = Family("f1", [Member("o", [t, t], [np.inf, np.inf])])
    genetic_corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    full_corr = np.array([[1.0, 0.1], [0.1, 1.0]])

    with pytest.raises(ValueError, match="incoherent"):
        _estimate_liability_multi([fam], h2_vec=[0.8, 0.8],
                                 genetic_corrmat=genetic_corr,
                                 full_corrmat=full_corr)


def test_inbred_kinship_covariance_stays_on_unit_liability_scale():
    from ltpred.covariance import construct_covmat_from_kinship

    h2 = 0.5
    A = np.array([[1.25, 0.5], [0.5, 1.0]])
    cov = construct_covmat_from_kinship(A, h2=h2, target=0).matrix
    raw_target_var = h2 * A[0, 0] + (1.0 - h2)

    assert np.array_equal(np.diag(cov)[1:], np.ones(2))
    assert cov[0, 0] == pytest.approx(h2 * A[0, 0] / raw_target_var)
    assert cov[0, 1] == pytest.approx(h2 * A[0, 0] / raw_target_var)


def test_estimate_from_kinship_matches_role_based():
    # the kinship path reproduces the role-based estimator on the same data
    from ltpred import (simulate_under_LTM_single,
                        estimate_liability_from_kinship, kinship_from_pedigree)
    from ltpred.estimate import _estimate_liability_single
    h2 = 0.5
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=h2,
                                    n_sim=400, pop_prev=0.1, seed=11)
    role = _estimate_liability_single(sim.families, h2=h2, out=("genetic",),
                                     n_sim=20000, burn_in=500, seed=1)
    ids    = ["o", "m", "f", "s1", "s2"]
    father = ["f", None, None, "f", "f"]
    mother = ["m", None, None, "m", "m"]
    _, A = kinship_from_pedigree(ids, father, mother)
    F = len(sim.families)
    lower = np.empty((F, len(ids))); upper = np.empty((F, len(ids)))
    for fi, fam in enumerate(sim.families):
        byrole = {m.role: m for m in fam.members}
        for c, r in enumerate(ids):
            lower[fi, c], upper[fi, c] = byrole[r].lower, byrole[r].upper
    kin, _, _ = estimate_liability_from_kinship(A, lower, upper, h2=h2, target=0,
                                                out="genetic", method="gibbs",
                                                n_sim=20000, burn_in=500, seed=1)
    # identical covariance + identical seeds -> identical draws
    assert np.corrcoef(role.est["genetic"], kin)[0, 1] > 0.999
    assert np.max(np.abs(role.est["genetic"] - kin)) < 1e-9


def test_kinship_estimator_threads_environment_kernels_to_both_engines():
    from ltpred import (construct_covmat_from_kinship,
                        estimate_liability_from_kinship,
                        estimate_liability_gibbs_arrays,
                        kinship_from_pedigree, pa_estimate_batched)
    from ltpred.fit import _component_matrix

    roles = ["o", "m", "f", "s1"]
    _, A = kinship_from_pedigree(
        roles, ["f", None, None, "f"], ["m", None, None, "m"])
    C = _component_matrix(roles, "C")
    M = _component_matrix(roles, "M")
    h2, c2, m2 = 0.4, 0.1, 0.05
    t = float(stats.norm.isf(0.1))
    lower = np.array([[t, -np.inf, -np.inf, t],
                      [-np.inf, t, -np.inf, -np.inf]])
    upper = np.array([[np.inf, t, t, np.inf],
                      [t, np.inf, t, t]])

    cov = construct_covmat_from_kinship(
        A, h2=h2, c2=c2, c_kernel=C, m2=m2, m_kernel=M).matrix
    lo = np.column_stack([np.full(len(lower), -np.inf), lower])
    hi = np.column_stack([np.full(len(upper), np.inf), upper])
    pa_expected, pa_expected_var = pa_estimate_batched(cov, lo, hi, target=0)
    pa_kin, pa_se, pa_kin_var = estimate_liability_from_kinship(
        A, lower, upper, h2=h2, c2=c2, c_kernel=C, m2=m2, m_kernel=M)
    np.testing.assert_allclose(pa_kin, pa_expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pa_kin_var, pa_expected_var, rtol=0, atol=1e-12)
    assert np.array_equal(pa_se, np.zeros_like(pa_se))

    gibbs_role = estimate_liability_gibbs_arrays(
        roles, lower, upper, h2=h2, c2=c2, m2=m2, seed=7,
        n_sim=40, burn_in=0, tol=1e9, max_rounds=1, return_var=True)
    gibbs_kin = estimate_liability_from_kinship(
        A, lower, upper, h2=h2, c2=c2, c_kernel=C, m2=m2, m_kernel=M,
        method="gibbs", seed=7, n_sim=40, burn_in=0, tol=1e9, max_rounds=1)
    for role_value, kin_value in zip(gibbs_role, gibbs_kin):
        np.testing.assert_allclose(kin_value, role_value, rtol=0, atol=1e-12)


def test_kinship_estimator_accepts_near_symmetric_relationship_inputs():
    from ltpred import estimate_liability_from_kinship

    noisy = np.array([[1.0, 5e-9], [0.0, 1.0]])
    exact = 0.5 * (noisy + noisy.T)
    lower = np.array([[-np.inf, 1.0]])
    upper = np.array([[0.0, np.inf]])
    observed = estimate_liability_from_kinship(
        noisy, lower, upper, h2=0.4, c2=0.2, c_kernel=noisy)
    expected = estimate_liability_from_kinship(
        exact, lower, upper, h2=0.4, c2=0.2, c_kernel=exact)
    for observed_value, expected_value in zip(observed, expected):
        np.testing.assert_array_equal(observed_value, expected_value)


def test_estimate_from_kinship_validation():
    from ltpred import estimate_liability_from_kinship, kinship_from_pedigree
    _, A = kinship_from_pedigree(["o", "m", "f"], ["f", None, None], ["m", None, None])
    with pytest.raises(ValueError, match="columns"):
        estimate_liability_from_kinship(A, np.zeros((5, 2)), np.ones((5, 2)), 0.5)   # wrong n cols
    with pytest.raises(ValueError, match="square"):
        estimate_liability_from_kinship(np.zeros((3, 2)), np.zeros((5, 3)), np.ones((5, 3)), 0.5)


def test_estimate_from_kinship_defaults_to_pa():
    from ltpred import estimate_liability_from_kinship, kinship_from_pedigree
    _, A = kinship_from_pedigree(["o", "m", "f"], ["f", None, None], ["m", None, None])
    lower = np.array([[1.2, -np.inf, -np.inf], [-np.inf, 1.2, -np.inf]])
    upper = np.array([[1.2, 1.2, 1.2], [1.2, np.inf, 1.2]])
    default = estimate_liability_from_kinship(A, lower, upper, 0.5)
    explicit = estimate_liability_from_kinship(A, lower, upper, 0.5, method="pa")
    assert np.array_equal(default[0], explicit[0])
    assert np.array_equal(default[1], explicit[1])
    with pytest.raises(ValueError, match="unknown method"):
        estimate_liability_from_kinship(A, lower, upper, 0.5, method="magic")


def test_kinship_estimator_accepts_pa_mixture():
    from ltpred import estimate_liability_from_kinship, kinship_from_pedigree
    from ltpred.thresholds import pa_thresholds
    _, A = kinship_from_pedigree(
        ["o", "m"], [None, None], [None, None])
    status = np.array([[0, 0], [1, 0]])
    age = np.array([[40.0, 70.0], [45.0, 70.0]])
    # two families, two members; build K from the control rows
    lower = np.empty((2, 2))
    upper = np.empty((2, 2))
    K_i = np.empty((2, 2))
    K_pop = np.empty((2, 2))
    for j in range(2):
        lo, hi, ki, kp = pa_thresholds(status[:, j], age[:, j], pop_prev=0.1)
        lower[:, j], upper[:, j], K_i[:, j], K_pop[:, j] = lo, hi, ki, kp
    est_mix, _, var_mix = estimate_liability_from_kinship(
        A, lower, upper, h2=0.5, use_mixture=True, K_i=K_i, K_pop=K_pop)
    est_plain, _, _ = estimate_liability_from_kinship(A, lower, upper, h2=0.5)
    assert np.all(np.isfinite(est_mix))
    assert np.all(var_mix >= 0)
    # first family is two censored controls: the mixture should raise the score
    # relative to a naive age truncation (weaker evidence of low liability)
    assert est_mix[0] > est_plain[0]
    with pytest.raises(ValueError, match="only supported by Pearson-Aitken"):
        estimate_liability_from_kinship(
            A, lower, upper, h2=0.5, method="gibbs", use_mixture=True,
            K_i=K_i, K_pop=K_pop, n_sim=20, burn_in=0)


@pytest.mark.parametrize("spelling", ["genetic", ("genetic",)])
def test_out_accepts_scalar_and_sequence(spelling):
    # a bare "genetic" (documented) must work everywhere, not just the tuple form
    from ltpred import simulate_under_LTM_single, estimate_liability_from_kinship
    from ltpred.estimate import (estimate_liability_pa_arrays,
                                 estimate_liability_gibbs_arrays)
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=80,
                                    pop_prev=0.1, seed=1)
    # object API, both back-ends
    assert "genetic" in estimate_liability(sim.families, h2=0.5, method="pa",
                                           out=spelling).est
    assert "genetic" in estimate_liability(sim.families, h2=0.5, method="gibbs",
                                           out=spelling, n_sim=2000, seed=1).est
    # single-column APIs (array, kinship) take the same spellings
    roles = ["o", "m", "f"]
    lo = np.array([[1.2, -9.0, 1.2]]); hi = np.array([[9.0, 1.2, 9.0]])
    estimate_liability_pa_arrays(roles, lo, hi, h2=0.5, out=spelling)
    estimate_liability_gibbs_arrays(roles, lo, hi, h2=0.5, out=spelling, n_sim=2000, seed=1)
    A = np.array([[1.0, .5, .5], [.5, 1, 0], [.5, 0, 1]])
    klo = np.array([[-9.0, 1.2, 1.2]]); khi = np.array([[9.0, 9.0, 9.0]])
    estimate_liability_from_kinship(A, klo, khi, h2=0.5, out=spelling)


def test_out_invalid_and_multicolumn_errors():
    from ltpred import simulate_under_LTM_single
    from ltpred.estimate import estimate_liability_pa_arrays
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=60,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="genetic/full"):
        estimate_liability(sim.families, h2=0.5, method="pa", out="bogus")
    # a single-column API cannot return two columns at once
    roles = ["o", "m", "f"]
    lo = np.array([[1.2, -9.0, 1.2]]); hi = np.array([[9.0, 1.2, 9.0]])
    with pytest.raises(ValueError, match="single column"):
        estimate_liability_pa_arrays(roles, lo, hi, 0.5, out=("genetic", "full"))


@pytest.mark.parametrize(
    "invalid,error",
    [([], ValueError), ((), ValueError), (True, TypeError), (False, TypeError),
     (0.0, TypeError), (1.0, TypeError),
     ("g", ValueError), ("o", ValueError), (0, TypeError), (1, TypeError),
     ((0,), TypeError)],          # the dropped "g"/"o"/0/1 spellings
)
def test_out_rejects_empty_bool_float_and_dropped_aliases(invalid, error):
    from ltpred.estimate import estimate_liability_pa_arrays

    fam = Family("f1", [Member("o", -np.inf, 1.0)])
    with pytest.raises(error):
        estimate_liability([fam], h2=0.5, method="pa", out=invalid)

    with pytest.raises(error):
        estimate_liability_pa_arrays(
            ["o"], np.array([[-np.inf]]), np.array([[1.0]]), 0.5, out=invalid)


def test_shared_bounds_validation_allows_infinities_and_point_pins():
    from ltpred._validation import validate_bounds

    validate_bounds(
        np.array([[-np.inf, 1.25]]),
        np.array([[np.inf, 1.25]]),
    )


@pytest.mark.parametrize(
    "lower,upper,match",
    [(np.nan, np.inf, "NaN"), (1.0, 0.0, "reversed bounds")],
)
def test_public_estimators_reject_invalid_bounds(lower, upper, match):
    from ltpred import estimate_liability_from_kinship
    from ltpred.estimate import (estimate_liability_gibbs_arrays,
                                 estimate_liability_pa_arrays)

    fam = Family("bad", [Member("o", lower, upper)])
    for method in ("pa", "gibbs"):
        with pytest.raises(ValueError, match=match):
            estimate_liability([fam], h2=0.5, method=method, n_sim=20, burn_in=0)

    lo = np.array([[lower]])
    hi = np.array([[upper]])
    with pytest.raises(ValueError, match=match):
        estimate_liability_pa_arrays(["o"], lo, hi, 0.5)
    with pytest.raises(ValueError, match=match):
        estimate_liability_gibbs_arrays(["o"], lo, hi, 0.5, n_sim=20, burn_in=0)
    with pytest.raises(ValueError, match=match):
        estimate_liability_from_kinship(np.eye(1), lo, hi, 0.5)


def test_default_method_is_pa_and_multitrait_falls_back_to_gibbs():
    from ltpred import simulate_under_LTM_single
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=200,
                                    pop_prev=0.1, seed=1)
    # single-trait default -> PA (deterministic: se == 0)
    res = estimate_liability(sim.families, h2=0.5)
    assert np.all(res.se["genetic"] == 0.0)
    assert res.genetic is res.est["genetic"]        # convenience property
    # multi-trait default -> Gibbs fallback (PA can't); works without method=
    rg = np.array([[1.0, 0.3], [0.3, 1.0]]); rp = np.array([[1.0, 0.2], [0.2, 1.0]])
    m = [Family(f.fam_id, [Member(mm.role, [mm.lower, mm.lower], [mm.upper, mm.upper])
                           for mm in f.members]) for f in sim.families[:40]]
    multi = estimate_liability(m, h2=[0.5, 0.4], genetic_corrmat=rg, full_corrmat=rp,
                               n_sim=3000, seed=1)
    assert "genetic_phenotype1" in multi.est
    # but an *explicit* PA request for multi-trait still raises
    with pytest.raises(NotImplementedError, match="single-trait"):
        estimate_liability(m, h2=[0.5, 0.4], method="pa", genetic_corrmat=rg,
                           full_corrmat=rp)
    # PA-FGRS names a threshold/censoring model, not an inference engine.
    with pytest.raises(ValueError, match="unknown method"):
        estimate_liability(sim.families, h2=0.5, method="pa-fgrs")


@pytest.mark.parametrize("component", [
    {"c2": 0.1}, {"m2": 0.1}, {"c2": [0.1, 0.0]},
])
def test_multitrait_rejects_unsupported_shared_environment_components(component):
    fam = Family("f", [Member("o", [-np.inf, -np.inf], [0.0, 0.0])])

    with pytest.raises(
        NotImplementedError,
        match="do not specify cross-trait environmental covariance",
    ):
        estimate_liability(
            [fam], h2=[0.5, 0.4], genetic_corrmat=np.eye(2),
            full_corrmat=np.eye(2), **component,
        )


@pytest.mark.jit_required
def test_multitrait_gibbs_matches_single_truncation_closed_form():
    """Multi-trait Gibbs against the closed form under one bounded coordinate.

    With exactly one truncated coordinate the truncated-MVN conditional mean of
    every other coordinate is exact in closed form,
    ``E[x_j | x_i in (a, b)] = cov_ij / var_i * E[Z | Z in (a, b)]``, so a
    converged Gibbs run must land on it. This is the multi-trait counterpart of
    the single-trait inverse-Mills oracles."""
    from ltpred.covariance import construct_covmat_multi

    K = 0.10
    t = float(stats.norm.isf(K))
    lam = imr(t)                                  # E[Z | Z > t]
    rg = np.array([[1.0, 0.6], [0.6, 1.0]])
    rp = np.array([[1.0, 0.5], [0.5, 1.0]])
    # only trait 1's mother is a case; every other coordinate uninformative
    fam = [Family("f", [Member("o", [-np.inf, -np.inf], [np.inf, np.inf]),
                        Member("m", [t, -np.inf], [np.inf, np.inf])])]
    cov = construct_covmat_multi(fam_vec=["o", "m"], add_ind=True,
                                 genetic_corrmat=rg, full_corrmat=rp,
                                 h2_vec=[0.5, 0.4]).matrix
    # rows are phenotype-major: g1, o1, m1, g2, o2, m2; only m1 (row 2) is bounded
    expected_g1 = cov[0, 2] / cov[2, 2] * lam
    expected_g2 = cov[3, 2] / cov[2, 2] * lam
    res = estimate_liability(fam, h2=[0.5, 0.4], genetic_corrmat=rg,
                             full_corrmat=rp, out=("genetic",), tol=0.002,
                             n_sim=100_000, burn_in=1000, seed=11)
    assert abs(res.est["genetic_phenotype1"][0] - expected_g1) < 0.01
    assert abs(res.est["genetic_phenotype2"][0] - expected_g2) < 0.01


def test_use_mixture_without_K_raises():
    from ltpred.estimate import estimate_liability_pa_arrays

    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", -np.inf, t), Member("m", -np.inf, t)])  # no K_i
    with pytest.raises(ValueError, match="at least one active censored-control"):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)
    with pytest.raises(ValueError, match="at least one active censored-control"):
        estimate_liability_pa_arrays(
            ["o"], np.array([[-np.inf]]), np.array([[t]]), 0.5,
            K_i=np.array([[np.nan]]), K_pop=np.array([[np.nan]]),
            use_mixture=True)


@pytest.mark.parametrize("bound_type", [list, np.asarray])
def test_mixture_vector_bounds_get_single_trait_shape_error(bound_type):
    fam = Family(
        "f", [Member("o", bound_type([-np.inf, -np.inf]),
                           bound_type([1.0, 1.0]), K_i=0.05, K_pop=0.10)])
    with pytest.raises(ValueError, match="single-trait estimator needs scalar bounds"):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)


@pytest.mark.parametrize("lower", [-np.inf, 1.0])
def test_mixture_rejects_pair_on_case_or_unbounded_row(lower):
    """K pairs are invalid, rather than merely inactive, when upper is +inf."""
    from ltpred.estimate import estimate_liability_pa_arrays

    fam = Family("f", [Member("o", lower, np.inf, K_i=0.02, K_pop=0.10)])
    with pytest.raises(ValueError, match=r"upper == \+inf"):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)
    with pytest.raises(ValueError, match=r"upper == \+inf"):
        estimate_liability_pa_arrays(
            ["o"], np.array([[lower]]), np.array([[np.inf]]), 0.5,
            K_i=np.array([[0.02]]), K_pop=np.array([[0.10]]),
            use_mixture=True)


@pytest.mark.parametrize("lower", [2.0, float(stats.norm.isf(0.10))])
def test_mixture_rejects_control_lower_not_below_lifetime_split(lower):
    """An active control's lower bound must leave a lifetime-control component."""
    from ltpred.estimate import estimate_liability_pa_arrays

    upper = 3.0
    fam = Family("f", [Member("o", lower, upper, K_i=0.02, K_pop=0.10)])
    match = r"lower < Phi\^-1\(1 - K_pop\)"
    with pytest.raises(ValueError, match=match):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)
    with pytest.raises(ValueError, match=match):
        estimate_liability_pa_arrays(
            ["o"], np.array([[lower]]), np.array([[upper]]), 0.5,
            K_i=np.array([[0.02]]), K_pop=np.array([[0.10]]),
            use_mixture=True)


def test_mixture_rejects_case_pair_even_when_an_active_control_exists():
    """The global active-row gate must not hide a bad pair on another row."""
    t = float(stats.norm.isf(0.10))
    fam = Family("f", [Member("o", -np.inf, t, K_i=0.02, K_pop=0.10),
                       Member("m", t, np.inf, K_i=0.05, K_pop=0.10)])
    with pytest.raises(ValueError, match=r"upper == \+inf"):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)


def test_use_mixture_case_only_structure_group_estimates():
    """A structure group with no mixture pair of its own is not an error.

    The at-least-one-pair gate is a property of the whole call (it already ran
    on the flattened member list), not of each role-set group: an all-case
    family — every ``K`` NaN because observed cases carry no censoring
    mixture — must estimate alongside families that do supply valid pairs,
    taking the plain truncated-normal fold for its case rows."""
    t = float(stats.norm.isf(0.10))
    control = Family("a", [Member("o", -np.inf, t, K_i=0.05, K_pop=0.10),
                           Member("m", -np.inf, t, K_i=0.08, K_pop=0.10)])
    cases = Family("b", [Member("o", t, np.inf), Member("s1", t, np.inf)])
    res = estimate_liability([control, cases], h2=0.5, method="pa",
                             use_mixture=True)
    assert np.all(np.isfinite(res.est["genetic"]))
    assert res.est["genetic"][1] > res.est["genetic"][0]   # two cases vs none
    # the same all-case group alone (no valid pair anywhere) still raises
    with pytest.raises(ValueError, match="at least one active censored-control"):
        estimate_liability([cases], h2=0.5, method="pa", use_mixture=True)


@pytest.mark.parametrize(
    "K_i,K_pop,match",
    [(0.01, np.nan, "finite K_i and K_pop"),
     (np.nan, 0.10, "finite K_i and K_pop"),
     (np.inf, 0.10, "finite K_i and K_pop"),
     (-0.01, 0.10, "0 <= K_i <= K_pop < 1"),
     (0.20, 0.10, "0 <= K_i <= K_pop < 1"),
     (0.00, 0.00, "K_pop > 0"),
     (0.10, 1.00, "K_pop > 0")],
)
def test_mixture_pair_validation_object_and_array_apis(K_i, K_pop, match):
    from ltpred.estimate import estimate_liability_pa_arrays

    t = float(stats.norm.isf(0.10))
    fam = Family("f", [Member("o", -np.inf, t, K_i=K_i, K_pop=K_pop)])
    with pytest.raises(ValueError, match=match):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)

    with pytest.raises(ValueError, match=match):
        estimate_liability_pa_arrays(
            ["o"], np.array([[-np.inf]]), np.array([[t]]), 0.5,
            K_i=np.array([[K_i]]), K_pop=np.array([[K_pop]]),
            use_mixture=True)


def test_gibbs_rejects_unsupported_mixture():
    t = float(stats.norm.isf(0.10))
    fam = Family("f", [Member("o", -np.inf, t, K_i=0.05, K_pop=0.10)])
    with pytest.raises(ValueError, match="only supported by Pearson-Aitken"):
        estimate_liability([fam], h2=0.5, method="gibbs", use_mixture=True)


def _seed_probe_families(n=4):
    t = float(stats.norm.isf(0.05))
    return [Family(str(i), [Member("o", -np.inf, t), Member("m", t, np.inf),
                            Member("f", -np.inf, t)]) for i in range(n)]


def _seed_probe(seed):
    return estimate_liability(_seed_probe_families(), h2=0.5, method="gibbs",
                              n_sim=2000, burn_in=100, tol=1e9,
                              seed=seed).est["genetic"]


@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.0, np.float64(2), "3"])
def test_estimator_seed_rejects_non_integer_types(seed):
    # the estimator gates seed exactly as the fitters do, rather than truncating
    with pytest.raises(TypeError, match="seed must be an integer"):
        _seed_probe(seed)


@pytest.mark.parametrize("seed", [-1, np.int64(-1), 1 << 32])
def test_estimator_seed_rejects_values_outside_uint32(seed):
    # -1 used to slip through and land on the kernel's *unseeded* sentinel,
    # silently leaving family 0 non-reproducible while the rest stayed seeded
    with pytest.raises(ValueError, match="seed must be in"):
        _seed_probe(seed)


def test_base_seeds_sentinel_only_from_none():
    assert np.array_equal(_base_seeds(None, 3, 100), [-1, -1, -1])
    assert (_base_seeds(0, 3, 100) >= 0).all()


def test_base_seeds_stay_in_uint32_range():
    # seed=2**32-1 is a value the fitters accept, so the derived per-family block
    # must wrap rather than run past what the kernel's RNG accepts
    seeds = _base_seeds((1 << 32) - 1, 5, 100)
    assert seeds.min() >= 0 and seeds.max() <= (1 << 32) - 1
    assert len(set(seeds.tolist())) == 5           # still distinct per family


def test_max_seed_is_reproducible_and_distinct_from_other_seeds():
    top = (1 << 32) - 1
    assert np.allclose(_seed_probe(top), _seed_probe(top))
    assert not np.allclose(_seed_probe(top), _seed_probe(1))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_batch_means_requires_at_least_four_samples(n):
    # below two batches the estimator's rule is undefined; the public helper
    # now guards it exactly like the estimator guards n_sim
    with pytest.raises(ValueError, match="at least 4"):
        batch_means(np.arange(n, dtype=float))


def test_batch_means_accepts_four_samples():
    est, se = batch_means(np.arange(4, dtype=float))
    assert est[0] == pytest.approx(1.5)
    assert np.isfinite(se[0])


def test_single_trait_gibbs_rejects_multitrait_shaped_bounds():
    # a length-2 (per-phenotype) bound used to be silently collapsed to its
    # first column; the single-trait Gibbs path now points at the multi-trait
    # estimator instead (the PA path already rejects it)
    fam = Family("f1", [Member("o", [1.0, -np.inf], [np.inf, 1.0])])
    with pytest.raises(ValueError, match="multi-trait"):
        estimate_liability([fam], h2=0.5, method="gibbs", n_sim=20, burn_in=0)
    # scalar bounds still pass
    fam_ok = Family("f1", [Member("o", 1.0, np.inf)])
    res = estimate_liability([fam_ok], h2=0.5, method="gibbs", n_sim=20,
                             burn_in=0, tol=1e9, max_rounds=1, seed=1)
    assert np.isfinite(res.est["genetic"][0])


_GIBBS_SMOKE = dict(n_sim=20, burn_in=0, tol=1e9, max_rounds=1)


@pytest.mark.parametrize("call", [
    pytest.param(lambda fam, lo, hi: estimate_liability(fam, h2=1.0, method="pa"), id="object-pa"),
    pytest.param(lambda fam, lo, hi: estimate_liability(fam, h2=1.0, method="gibbs", **_GIBBS_SMOKE),
                 id="object-gibbs"),
    pytest.param(lambda fam, lo, hi: estimate_liability_pa_arrays(["o", "m"], lo, hi, h2=1.0),
                 id="array-pa"),
    pytest.param(lambda fam, lo, hi: estimate_liability_gibbs_arrays(["o", "m"], lo, hi, h2=1.0,
                                                                     **_GIBBS_SMOKE), id="array-gibbs"),
    pytest.param(lambda fam, lo, hi: estimate_liability_from_kinship(np.ones((2, 2)), lo, hi, h2=1.0),
                 id="kinship"),
])
def test_every_estimator_warns_when_covariance_is_corrected(call):
    # h2 = 1 makes g and o perfectly correlated, so the assembled covariance is
    # singular and correct_positive_definite fires; every path must report the
    # nudge, not just the multi-trait Gibbs one
    fam = [Family("f1", [Member("o", 1.0, np.inf), Member("m", -np.inf, 1.0)])]
    lo, hi = np.array([[1.0, -np.inf]]), np.array([[np.inf, 1.0]])
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        call(fam, lo, hi)


_BURN_IN = "burn_in must be a non-negative integer"
_TOL = "tol must be finite and > 0"


@pytest.mark.parametrize("api", ["object", "array"])
@pytest.mark.parametrize("kwargs,error,match", [
    # burn_in: validated once in _estimate_group, the choke point of every Gibbs path
    ({"burn_in": -1}, ValueError, _BURN_IN),
    ({"burn_in": np.int64(-5)}, ValueError, _BURN_IN),
    ({"burn_in": True}, TypeError, _BURN_IN),
    ({"burn_in": np.bool_(False)}, TypeError, _BURN_IN),
    ({"burn_in": 2.5}, TypeError, _BURN_IN),
    # tol=NaN never satisfied se <= tol, so the sampler ran to max_rounds while
    # the unconverged warning (keyed on the same comparison) stayed silent
    ({"burn_in": 0, "tol": 0.0}, ValueError, _TOL),
    ({"burn_in": 0, "tol": -0.5}, ValueError, _TOL),
    ({"burn_in": 0, "tol": np.nan}, ValueError, _TOL),
    ({"burn_in": 0, "tol": np.inf}, ValueError, _TOL),
    # max_rounds < 1 skipped the sampling loop and returned all-zero estimates
    ({"burn_in": 0, "max_rounds": 0}, ValueError, "max_rounds"),
    ({"burn_in": 0, "max_rounds": -3}, ValueError, "max_rounds"),
    ({"burn_in": 0, "max_rounds": True}, TypeError, "max_rounds"),
    ({"burn_in": 0, "max_rounds": 1.5}, TypeError, "max_rounds"),
])
def test_gibbs_estimators_reject_invalid_sampler_controls(api, kwargs, error, match):
    with pytest.raises(error, match=match):
        if api == "object":
            estimate_liability([Family("f", [Member("o", 1.0, np.inf)])], h2=0.5,
                               method="gibbs", n_sim=20, **kwargs)
        else:
            estimate_liability_gibbs_arrays(["o"], np.array([[1.0]]), np.array([[np.inf]]),
                                            0.5, n_sim=20, **kwargs)


def test_multi_trait_rejects_duplicate_phen_names():
    # result columns are keyed by (output, phenotype) name; duplicates silently
    # collapsed two traits onto one key (last write won)
    fam = Family("f1", [Member("o", [1.0, -np.inf], [np.inf, 1.0])])
    with pytest.raises(ValueError, match="phen_names contains duplicates"):
        estimate_liability([fam], h2=[0.5, 0.4], genetic_corrmat=np.eye(2),
                           full_corrmat=np.eye(2), phen_names=["A", "A"],
                           n_sim=20, burn_in=0)


def test_member_role_g_is_rejected_on_object_and_array_paths():
    # "g" is never legitimate user input: the estimator adds the genetic
    # coordinate itself, so a supplied "g" row silently conditioned it
    from ltpred.estimate import (estimate_liability_gibbs_arrays,
                                 estimate_liability_pa_arrays)
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("g", t, np.inf), Member("o", -np.inf, t)])
    for method in ("pa", "gibbs"):
        with pytest.raises(ValueError, match="role 'g'"):
            estimate_liability([fam], h2=0.5, method=method, n_sim=20, burn_in=0)
    lo = np.array([[-9.0, t]])
    hi = np.array([[9.0, np.inf]])
    with pytest.raises(ValueError, match="must not contain 'g'"):
        estimate_liability_pa_arrays(["g", "o"], lo, hi, h2=0.5)
    with pytest.raises(ValueError, match="must not contain 'g'"):
        estimate_liability_gibbs_arrays(["g", "o"], lo, hi, h2=0.5, n_sim=20,
                                        burn_in=0)


def test_family_without_members_raises_instead_of_a_silent_zero():
    # A family with no observed rows has nothing to condition on, so every
    # engine would return the prior mean 0 -- indistinguishable from a real
    # estimate near zero. A join that dropped the rows must not leave that
    # zero in a GWAS phenotype.
    t = float(stats.norm.isf(0.05))
    fams = [Family("empty", []),
            Family("ok", [Member("o", t, np.inf), Member("m", -np.inf, t)])]
    for method in ("pa", "gibbs"):
        with pytest.raises(ValueError, match="no members"):
            estimate_liability(fams, h2=0.5, method=method,
                               n_sim=200, burn_in=0, seed=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = estimate_liability(fams[1:], h2=0.5)
    assert res.est["genetic"][0] > 0.3


def test_gibbs_reports_posterior_variance_alongside_the_mc_error():
    # `var` is the spread of the posterior itself and does not shrink with more
    # draws; `se` is the sampler's error in the mean and does. Both engines now
    # report `var`, so they must agree to PA's approximation error.
    t = float(stats.norm.isf(0.05))
    fams = [Family("f", [Member("o", t, np.inf), Member("m", -np.inf, t),
                         Member("f", t, np.inf)])]
    pa = estimate_liability(fams, h2=0.5, out=("genetic", "full"))
    # tol below any reachable SE with max_rounds=1 fixes the draw count at n_sim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the expected "did not reach tol"
        small = estimate_liability(fams, h2=0.5, method="gibbs",
                                   out=("genetic", "full"), n_sim=20_000,
                                   burn_in=500, seed=2, tol=1e-9, max_rounds=1)
        big = estimate_liability(fams, h2=0.5, method="gibbs",
                                 out=("genetic", "full"), n_sim=500_000,
                                 burn_in=500, seed=2, tol=1e-9, max_rounds=1)
    for name in ("genetic", "full"):
        assert big.var[name][0] == pytest.approx(pa.var[name][0], rel=0.05)
        # posterior variance is stable in n_sim; the Monte-Carlo SE shrinks
        assert big.var[name][0] == pytest.approx(small.var[name][0], rel=0.05)
        assert big.se[name][0] < 0.5 * small.se[name][0]
        assert pa.se[name][0] == 0.0


@pytest.mark.parametrize("fam_id,role", [
    # NaN != NaN, so a missing id would otherwise fragment silently into
    # one-member families (review 2026-08, F3)
    pytest.param([1.0, np.nan, np.nan, 1.0], ["o", "m", "m", "f"], id="float-nan"),
    pytest.param(["a", None, "a"], ["o", "m", "f"], id="none"),
    # numpy non-finite sentinels hiding in an object column
    *[pytest.param(np.array(["a", s, "a"], dtype=object), ["o", "m", "f"],
                   id=f"object-{type(s).__name__}-{s}")
      for s in (np.float16(np.nan), np.float32(np.nan), np.float64(np.nan),
                np.float16(np.inf), np.float32(-np.inf), np.float64(np.inf))],
    # The F3 fix caught NaN/None but not the textual sentinels a CSV loader
    # produces. Those are worse than NaN: NaN != NaN fragments records into
    # singletons, whereas every "" or "NA" compares EQUAL and merges unrelated
    # probands into one family -- a wrong-but-finite score with no signal.
    *[pytest.param(["a", s, s, "a"], ["o", "m", "m", "f"], id=f"string-{s!r}")
      for s in ("", "  ", "NA", "nan", "None", "null", ".")],
])
def test_families_from_columns_rejects_missing_or_sentinel_fam_id(fam_id, role):
    n = len(role)
    with pytest.raises(ValueError, match="fam_id"):
        families_from_columns(fam_id=fam_id, role=role,
                              lower=np.zeros(n), upper=np.ones(n))


def test_families_from_columns_keeps_finite_numpy_numeric_object_ids():
    fam_id = np.array([np.float32(1.5), np.float64(1.5), "2", "2"],
                      dtype=object)
    fams = families_from_columns(fam_id=fam_id, role=["o", "m", "o", "m"],
                                 lower=np.zeros(4), upper=np.ones(4))
    assert [family.fam_id for family in fams] == [np.float32(1.5), "2"]
    assert [len(family.members) for family in fams] == [2, 2]


@pytest.mark.parametrize("name", ["pid", "K_i", "K_pop", "aod"])
@pytest.mark.parametrize("bad_len", [2, 8])
def test_families_from_columns_length_checks_optional_columns(name, bad_len):
    # The optional columns are indexed positionally alongside the mandatory
    # ones, so before this check an over-long column was silently truncated to
    # the first n rows (shifting mixture estimates) and a short one raised a
    # bare IndexError from deep inside the loop.
    kw = dict(fam_id=["a", "a", "b", "b"], role=["o", "m", "o", "m"],
              lower=np.zeros(4), upper=np.ones(4))
    fill = ["p"] * bad_len if name == "pid" else np.full(bad_len, 0.1)
    with pytest.raises(ValueError, match=name):
        families_from_columns(**kw, **{name: fill})


def test_families_from_columns_keeps_legitimate_string_ids():
    # ...and the sentinel check must not swallow real ids that merely look
    # wordy. "NAME" is not "NA".
    fams = families_from_columns(fam_id=["NAME", "NAME", "0", "0"],
                                 role=["o", "m", "o", "m"],
                                 lower=np.zeros(4), upper=np.ones(4))
    assert [f.fam_id for f in fams] == ["NAME", "0"]


@pytest.mark.parametrize("fam_id", [[1, "1"], [np.int64(2), "2"], [1.0, "b"]])
def test_families_from_columns_rejects_mixed_string_and_numeric_ids(fam_id):
    # NumPy would stringify both to "1" and merge two unrelated families.
    with pytest.raises(ValueError, match="mixes string and non-string"):
        families_from_columns(fam_id=fam_id, role=["o", "o"],
                              lower=np.zeros(2), upper=np.ones(2))


def test_same_pid_under_two_roles_is_rejected_but_shared_relatives_are_not():
    t = float(stats.norm.isf(0.05))
    base = [Member("o", -np.inf, t, pid="p"), Member("m", -np.inf, t, pid=7),
            Member("s1", t, np.inf, pid="sib")]
    for dup in (Member("s2", t, np.inf, pid="sib"), Member("f", -np.inf, t,
                                                            pid=np.int64(7))):
        with pytest.raises(ValueError, match="more than once in family"):
            estimate_liability([Family("F", base + [dup])], h2=0.5)
    # Missing ids witness nothing; one person in two families is a shared
    # register relative, not a duplicate.
    ok = [Family("F", base + [Member("s2", t, np.inf, pid="NA"),
                              Member("f", -np.inf, t, pid=None)]),
          Family("G", [Member("o", -np.inf, t, pid="sib")])]
    assert estimate_liability(ok, h2=0.5).genetic.shape == (2,)


@pytest.mark.parametrize("method", ["pa", "gibbs", "quadrature"])
@pytest.mark.parametrize("pid", [" sib ", b"sib", np.bytes_("sib")])
def test_duplicate_pid_check_uses_normalized_identity(method, pid):
    family = Family("F", [Member("s1", 1.6, np.inf, pid="sib"),
                          Member("s2", 1.6, np.inf, pid=pid)])
    with pytest.raises(ValueError, match="more than once in family"):
        estimate_liability([family], h2=0.5, method=method)


def test_families_from_columns_rejects_mismatched_bound_shapes():
    with pytest.raises(ValueError, match="same shape"):
        families_from_columns(fam_id=[1, 1], role=["o", "m"],
                              lower=np.zeros(2), upper=np.ones((2, 1)))


def test_array_estimators_validate_column_count():
    # extra columns must not be silently dropped and short or 1-D inputs must
    # not surface a raw IndexError (review 2026-08, F4)
    from ltpred import estimate_liability_pa_arrays
    from ltpred import estimate_liability_gibbs_arrays
    t = float(stats.norm.isf(0.05))
    lower = np.array([[t, -np.inf, -np.inf]])
    upper = np.array([[np.inf, t, t]])
    wide_lo = np.hstack([lower, [[0.0]]])
    wide_hi = np.hstack([upper, [[1.0]]])
    with pytest.raises(ValueError, match="one column per role"):
        estimate_liability_pa_arrays(["o", "m", "f"], wide_lo, wide_hi, h2=0.5)
    with pytest.raises(ValueError, match="one column per role"):
        estimate_liability_pa_arrays(["o", "m", "f"], lower[:, :2],
                                     upper[:, :2], h2=0.5)
    with pytest.raises(ValueError, match="one column per role"):
        estimate_liability_pa_arrays(["o", "m", "f"], lower[0], upper[0],
                                     h2=0.5)
    with pytest.raises(ValueError, match="one column per role"):
        estimate_liability_gibbs_arrays(["o", "m", "f"], wide_lo, wide_hi,
                                        h2=0.5, n_sim=100, seed=1)


def test_kinship_estimator_returns_est_se_var_on_both_engines():
    # the second array used to change statistical meaning with `method`
    # (review 2026-08, F6); both engines now return (est, se, var)
    from ltpred import estimate_liability_from_kinship
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([[1.6, -np.inf]])
    upper = np.array([[np.inf, 1.6]])
    est, se, var = estimate_liability_from_kinship(A, lower, upper, h2=0.5,
                                                   method="pa")
    assert se[0] == 0.0            # deterministic PA: no Monte-Carlo error
    assert var[0] > 0.0            # posterior variance, reported by both engines
    est_g, se_g, var_g = estimate_liability_from_kinship(
        A, lower, upper, h2=0.5, method="gibbs", n_sim=20_000, burn_in=500,
        seed=1)
    assert se_g[0] > 0.0
    assert var_g[0] == pytest.approx(var[0], rel=0.05)
    # PA carries a sequential-approximation error, so the engines agree to
    # better than a percent but not to Monte-Carlo precision
    assert est_g[0] == pytest.approx(est[0], abs=0.02)


def test_length_one_h2_is_a_single_trait_request():
    fam = Family("f1", [Member("o", 1.6, np.inf), Member("m", -np.inf, 1.6)])
    scalar = estimate_liability([fam], h2=0.5)
    vector = estimate_liability([fam], h2=[0.5])
    assert vector.est["genetic"] == pytest.approx(scalar.est["genetic"])


def test_batch_means_se_uses_batched_draw_count():
    # the bmmat SE divides by the a*b draws that enter the batches, not by the
    # full retained n (review 2026-08, F18); n = 10 tiles as a = 3, b = 3
    x = np.arange(10, dtype=float)
    _, se = batch_means(x)
    batch_mean = x[:9].reshape(3, 3).mean(axis=1)
    sigma2 = 3.0 * np.sum((batch_mean - batch_mean.mean()) ** 2) / 2
    assert se[0] == pytest.approx(np.sqrt(sigma2 / 9.0))


def test_multi_trait_var_reports_posterior_variance():
    # pin trait A of the proband's own liability: Var(g_A | o_A = v) has the
    # closed form h2_A - h2_A^2, and trait B keeps positive variance
    # (review 2026-08, F36)
    h2 = [0.5, 0.4]
    gcorr = np.array([[1.0, 0.3], [0.3, 1.0]])
    fcorr = np.array([[1.0, 0.2], [0.2, 1.0]])
    fam = Family("f1", [
        Member("o", lower=[1.2, -np.inf], upper=[1.2, np.inf]),
    ])
    res = _estimate_liability_multi([fam], h2_vec=h2, genetic_corrmat=gcorr,
                                    full_corrmat=fcorr, phen_names=["A", "B"],
                                    out=("genetic",), tol=0.05,
                                    n_sim=20_000, burn_in=500, seed=3)
    assert res.var["genetic_A"][0] == pytest.approx(0.5 - 0.5 ** 2, abs=0.02)
    assert res.var["genetic_B"][0] > 0.0


@pytest.mark.parametrize("h2", [.2, .5, .8])
def test_adult_public_scalar_dispatch_matches_analytic_posterior(h2):
    # family-free (ADuLT) probands never need a family covariance: the scalar
    # Gaussian-regression closed form must be what the public dispatcher returns
    bounds = [(-np.inf, np.inf), (2., 2.), (-np.inf, 1.), (1., np.inf), (-.5, 1.5)]
    families = [Family(str(i), [Member('o', lo, hi)]) for i, (lo, hi) in enumerate(bounds)]
    result = estimate_liability(families, h2=h2, out=['genetic', 'full'])
    assert result.quadrature_error is None
    for i, (lo, hi) in enumerate(bounds):
        if lo == hi:
            m, v = lo, 0.
        elif lo == -np.inf and hi == np.inf:
            m, v = 0., 1.
        else:
            m, v = stats.truncnorm.stats(lo, hi, moments='mv')
        assert result.est['genetic'][i] == pytest.approx(h2 * m, abs=1e-14)
        assert result.var['genetic'][i] == pytest.approx(h2 * (1 - h2) + h2 * h2 * v, abs=1e-14)
        assert result.est['full'][i] == pytest.approx(m, abs=1e-14)
        assert result.var['full'][i] == pytest.approx(v, abs=1e-14)


def test_estimator_accepts_c2_m2_and_treats_none_as_absent():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.4,
                                    pop_prev=0.1, n_sim=50, seed=1)
    base = estimate_liability(sim.families, h2=0.4)
    explicit_none = estimate_liability(sim.families, h2=0.4, c2=None, m2=None)
    wired = estimate_liability(sim.families, h2=0.4, c2=0.1, m2=0.05)
    np.testing.assert_allclose(explicit_none.est["genetic"], base.est["genetic"],
                               rtol=0, atol=1e-12)
    assert np.all(np.isfinite(wired.est["genetic"]))
    assert not np.allclose(base.est["genetic"], wired.est["genetic"])


def test_pa_estimation_under_c2_matches_closed_form():
    # pinning a full sib at v is exact Gaussian conditioning:
    # E[o | s1 = v] = (0.5*h2 + c2)*v and E[g | s1 = v] = 0.5*h2*v
    # (review 2026-08, F36: first exact oracle for estimation under c2)
    h2, c2, v = 0.4, 0.1, 1.1
    lower = np.array([[-np.inf, v]])
    upper = np.array([[np.inf, v]])
    est_o, var_o = estimate_liability_pa_arrays(["o", "s1"], lower, upper,
                                                h2=h2, c2=c2, out="full")
    assert est_o[0] == pytest.approx((0.5 * h2 + c2) * v, abs=1e-12)
    assert var_o[0] == pytest.approx(1.0 - (0.5 * h2 + c2) ** 2, abs=1e-12)
    est_g, var_g = estimate_liability_pa_arrays(["o", "s1"], lower, upper,
                                                h2=h2, c2=c2, out="genetic")
    assert est_g[0] == pytest.approx(0.5 * h2 * v, abs=1e-12)
