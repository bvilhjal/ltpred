"""End-to-end liability estimation and the batch-means convergence rule."""

import warnings

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member, families_from_columns
from ltpred.estimate import (batch_means, estimate_liability,
                             estimate_liability_multi, _base_seeds)
from ltpred.thresholds import age_thresholds


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


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
            [fam], method="gibbs", n_sim=n_sim, burn_in=0, max_rounds=1,
        )


def test_gibbs_minimum_draw_boundary_has_finite_mc_se():
    from ltpred.estimate import estimate_liability_gibbs_arrays

    est, se = estimate_liability_gibbs_arrays(
        ["o"], np.array([[-np.inf]]), np.array([[0.0]]),
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
    assert res.est["full"][0] == pytest.approx(_imr(t), abs=0.05)
    assert res.est["genetic"][0] == pytest.approx(h2 * _imr(t), abs=0.05)
    assert res.se["genetic"][0] <= 0.02


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


def test_duplicate_role_raises():
    t = float(stats.norm.isf(0.05))
    fam = Family("f1", [Member("o", t, np.inf), Member("s1", t, np.inf),
                        Member("s1", -np.inf, t)])           # two 's1' -> error
    with pytest.raises(ValueError, match="duplicate role"):
        estimate_liability([fam], h2=0.5, out=("genetic",))
    # PA path validates too
    with pytest.raises(ValueError, match="duplicate role"):
        estimate_liability([fam], h2=0.5, method="pa", out=("genetic",))


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
    res = estimate_liability_multi([fam], h2_vec=h2, genetic_corrmat=gcorr,
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
        estimate_liability_multi(
            [fam], h2_vec=[0.5, 0.4], genetic_corrmat=np.eye(2),
            full_corrmat=np.eye(2), n_sim=10, burn_in=0,
        )


def test_multi_trait_rejects_incoherent_genetic_and_full_correlations():
    t = float(stats.norm.isf(0.05))
    fam = Family("f1", [Member("o", [t, t], [np.inf, np.inf])])
    genetic_corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    full_corr = np.array([[1.0, 0.1], [0.1, 1.0]])

    with pytest.raises(ValueError, match="incoherent"):
        estimate_liability_multi([fam], h2_vec=[0.8, 0.8],
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
    from ltpred.estimate import estimate_liability_single
    h2 = 0.5
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=h2,
                                    n_sim=400, pop_prev=0.1, seed=11)
    role = estimate_liability_single(sim.families, h2=h2, out=("genetic",),
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
    kin, _ = estimate_liability_from_kinship(A, lower, upper, h2=h2, target=0,
                                             out="genetic", method="gibbs",
                                             n_sim=20000, burn_in=500, seed=1)
    # identical covariance + identical seeds -> identical draws
    assert np.corrcoef(role.est["genetic"], kin)[0, 1] > 0.999
    assert np.max(np.abs(role.est["genetic"] - kin)) < 1e-9


def test_estimate_from_kinship_validation():
    from ltpred import estimate_liability_from_kinship, kinship_from_pedigree
    _, A = kinship_from_pedigree(["o", "m", "f"], ["f", None, None], ["m", None, None])
    with pytest.raises(ValueError, match="columns"):
        estimate_liability_from_kinship(A, np.zeros((5, 2)), np.ones((5, 2)))   # wrong n cols
    with pytest.raises(ValueError, match="square"):
        estimate_liability_from_kinship(np.zeros((3, 2)), np.zeros((5, 3)), np.ones((5, 3)))


def test_estimate_from_kinship_defaults_to_pa():
    from ltpred import estimate_liability_from_kinship, kinship_from_pedigree
    _, A = kinship_from_pedigree(["o", "m", "f"], ["f", None, None], ["m", None, None])
    lower = np.array([[1.2, -np.inf, -np.inf], [-np.inf, 1.2, -np.inf]])
    upper = np.array([[1.2, 1.2, 1.2], [1.2, np.inf, 1.2]])
    default = estimate_liability_from_kinship(A, lower, upper)
    explicit = estimate_liability_from_kinship(A, lower, upper, method="pa")
    assert np.array_equal(default[0], explicit[0])
    assert np.array_equal(default[1], explicit[1])
    with pytest.raises(ValueError, match="unknown method"):
        estimate_liability_from_kinship(A, lower, upper, method="magic")


def test_liability_sensitivity_h2_grid():
    from ltpred import simulate_under_LTM_single, liability_sensitivity
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=800, pop_prev=0.1, seed=9)
    grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    r = liability_sensitivity(sim.families, grid, method="pa", out="genetic")
    default = liability_sensitivity(sim.families, grid, out="genetic")
    assert r.estimates.shape == (len(grid), 800)
    assert r.corr.shape == (len(grid), len(grid))
    assert np.allclose(np.diag(r.corr), 1.0)
    assert r.mean.shape == r.sd.shape == (len(grid),)
    assert r.out == "genetic"
    assert np.array_equal(default.estimates, r.estimates)
    # ranking is highly stable to the assumed h2 (the reassuring result)
    assert r.min_corr > 0.9
    # scale grows with h2 even though ranking does not
    assert r.sd[-1] > r.sd[0]


def test_liability_sensitivity_validation():
    from ltpred import simulate_under_LTM_single, liability_sensitivity
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    n_sim=100, pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="at least 2"):
        liability_sensitivity(sim.families, [0.5], method="pa")
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        liability_sensitivity(sim.families, [0.5, 1.5], method="pa")


@pytest.mark.parametrize("spelling", ["genetic", ("genetic",), "g", 0, np.int64(0)])
def test_out_accepts_scalar_and_sequence(spelling):
    # a bare "genetic" (documented) must work everywhere, not just the tuple form
    from ltpred import (simulate_under_LTM_single, liability_sensitivity,
                        estimate_liability_from_kinship)
    from ltpred.estimate import (estimate_liability_pa_arrays,
                                 estimate_liability_gibbs_arrays)
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=80,
                                    pop_prev=0.1, seed=1)
    # object API, both back-ends
    assert "genetic" in estimate_liability(sim.families, h2=0.5, method="pa",
                                           out=spelling).est
    assert "genetic" in estimate_liability(sim.families, h2=0.5, method="gibbs",
                                           out=spelling, n_sim=2000, seed=1).est
    # single-column APIs (array, kinship, sensitivity) take the same spellings
    roles = ["g", "o", "m", "f"]
    lo = np.array([[-9.0, 1.2, -9.0, 1.2]]); hi = np.array([[9.0, 9.0, 1.2, 9.0]])
    estimate_liability_pa_arrays(roles, lo, hi, h2=0.5, out=spelling)
    estimate_liability_gibbs_arrays(roles, lo, hi, h2=0.5, out=spelling, n_sim=2000, seed=1)
    A = np.array([[1.0, .5, .5], [.5, 1, 0], [.5, 0, 1]])
    klo = np.array([[-9.0, 1.2, 1.2]]); khi = np.array([[9.0, 9.0, 9.0]])
    estimate_liability_from_kinship(A, klo, khi, h2=0.5, out=spelling)
    liability_sensitivity(sim.families, [0.3, 0.5], method="pa", out=spelling)


def test_out_invalid_and_multicolumn_errors():
    from ltpred import simulate_under_LTM_single
    from ltpred.estimate import estimate_liability_pa_arrays
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=60,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="genetic/full"):
        estimate_liability(sim.families, h2=0.5, method="pa", out="bogus")
    # a single-column API cannot return two columns at once
    roles = ["g", "o", "m", "f"]
    lo = np.array([[-9.0, 1.2, -9.0, 1.2]]); hi = np.array([[9.0, 9.0, 1.2, 9.0]])
    with pytest.raises(ValueError, match="single column"):
        estimate_liability_pa_arrays(roles, lo, hi, out=("genetic", "full"))


@pytest.mark.parametrize(
    "invalid,error",
    [([], ValueError), ((), ValueError), (True, TypeError), (False, TypeError),
     (0.0, TypeError), (1.0, TypeError)],
)
def test_out_rejects_empty_bool_and_float_aliases(invalid, error):
    from ltpred.estimate import estimate_liability_pa_arrays

    fam = Family("f1", [Member("o", -np.inf, 1.0)])
    with pytest.raises(error):
        estimate_liability([fam], h2=0.5, method="pa", out=invalid)

    with pytest.raises(error):
        estimate_liability_pa_arrays(
            ["o"], np.array([[-np.inf]]), np.array([[1.0]]), out=invalid)


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
        estimate_liability_pa_arrays(["o"], lo, hi)
    with pytest.raises(ValueError, match=match):
        estimate_liability_gibbs_arrays(["o"], lo, hi, n_sim=20, burn_in=0)
    with pytest.raises(ValueError, match=match):
        estimate_liability_from_kinship(np.eye(1), lo, hi)


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


def test_use_mixture_without_K_raises():
    from ltpred.estimate import estimate_liability_pa_arrays

    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", -np.inf, t), Member("m", -np.inf, t)])  # no K_i
    with pytest.raises(ValueError, match="at least one valid K_i/K_pop pair"):
        estimate_liability([fam], h2=0.5, method="pa", use_mixture=True)
    with pytest.raises(ValueError, match="at least one valid K_i/K_pop pair"):
        estimate_liability_pa_arrays(
            ["o"], np.array([[-np.inf]]), np.array([[t]]),
            K_i=np.array([[np.nan]]), K_pop=np.array([[np.nan]]),
            use_mixture=True)


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
            ["o"], np.array([[-np.inf]]), np.array([[t]]),
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


def test_estimators_warn_when_covariance_is_corrected():
    # h2 = 1 makes g and o perfectly correlated, so the assembled covariance is
    # singular and correct_positive_definite fires; every path must report the
    # nudge, not just the multi-trait Gibbs one
    fam = Family("f1", [Member("o", 1.0, np.inf), Member("m", -np.inf, 1.0)])
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        estimate_liability([fam], h2=1.0, method="pa")
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        estimate_liability([fam], h2=1.0, method="gibbs", n_sim=20, burn_in=0,
                           tol=1e9, max_rounds=1)


def test_array_and_kinship_estimators_warn_when_covariance_is_corrected():
    from ltpred import estimate_liability_from_kinship
    from ltpred.estimate import (estimate_liability_pa_arrays,
                                 estimate_liability_gibbs_arrays)
    lo = np.array([[1.0, -np.inf]])
    hi = np.array([[np.inf, 1.0]])
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        estimate_liability_pa_arrays(["o", "m"], lo, hi, h2=1.0)
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        estimate_liability_gibbs_arrays(["o", "m"], lo, hi, h2=1.0, n_sim=20,
                                        burn_in=0, tol=1e9, max_rounds=1)
    with pytest.warns(RuntimeWarning, match="nudged to strict positive definiteness"):
        estimate_liability_from_kinship(np.ones((2, 2)), np.array([[1.0, -np.inf]]),
                                        np.array([[np.inf, 1.0]]), h2=1.0)
