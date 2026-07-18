"""The batched, parallel estimation path: agreement, grouping, determinism."""

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member
from ltpred.gibbs import gibbs_estimate_batched, gibbs_params
from ltpred.estimate import estimate_liability


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


def test_batched_kernel_matches_batch_means_single_round():
    # streamed batch-mean summaries reconstruct est + SE, matching the analytic
    # case posterior (g = h2*IMR, o = IMR), single round, one family.
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    t = float(stats.norm.isf(0.05))
    lowers = np.array([[-np.inf, t]])
    uppers = np.array([[np.inf, np.inf]])
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))
    n_sim = 40_000
    b = int(np.floor(np.sqrt(n_sim)))
    nb = n_sim // b
    tot, bm_sum, bm_sumsq = gibbs_estimate_batched(
        P, sd, sd0, lowers, uppers, np.array([0, 1]), n_sim, 1000, b, nb,
        np.array([7]))
    est = tot[0] / n_sim
    ss = bm_sumsq[0] - bm_sum[0] ** 2 / nb        # sum((Y - Ybar)^2)
    se = np.sqrt(b * ss / (nb - 1) / n_sim)
    assert est[0] == pytest.approx(0.5 * _imr(t), abs=0.03)
    assert est[1] == pytest.approx(_imr(t), abs=0.03)
    assert np.all(se > 0)


def test_pa_arrays_match_object_api():
    from ltpred.estimate import estimate_liability_pa_arrays, estimate_liability_pa
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(1)
    case = rng.random((50, 4)) < 0.3
    lower = np.where(case, t, -np.inf)
    upper = np.where(case, np.inf, t)
    est_a, var_a = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    fams = [Family(i, [Member(r, lower[i, k], upper[i, k]) for k, r in enumerate(roles)])
            for i in range(50)]
    obj = estimate_liability_pa(fams, h2=0.5, out=("genetic",))
    assert np.allclose(est_a, obj.est["genetic"])
    assert np.allclose(var_a, obj.var["genetic"])


def test_gibbs_arrays_match_object_api():
    from ltpred.estimate import estimate_liability_gibbs_arrays, estimate_liability
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m"]
    lower = np.array([[t, -np.inf], [-np.inf, t]])
    upper = np.array([[np.inf, t], [t, np.inf]])
    est_a, se_a = estimate_liability_gibbs_arrays(roles, lower, upper, h2=0.5,
                                                  n_sim=20_000, burn_in=400, seed=3)
    fams = [Family(i, [Member(r, lower[i, k], upper[i, k]) for k, r in enumerate(roles)])
            for i in range(2)]
    obj = estimate_liability(fams, h2=0.5, method="gibbs", out=("genetic",),
                             n_sim=20_000, burn_in=400, seed=3)
    assert np.allclose(est_a, obj.est["genetic"])
    assert np.allclose(se_a, obj.se["genetic"])


@pytest.mark.parametrize("name", ["pa", "gibbs"])
def test_array_estimators_reject_duplicate_role_labels(name):
    from ltpred.estimate import (estimate_liability_gibbs_arrays,
                                 estimate_liability_pa_arrays)

    estimator = (estimate_liability_pa_arrays if name == "pa"
                 else estimate_liability_gibbs_arrays)
    roles = ["o", "m", "m"]
    lower = np.full((2, 3), -np.inf)
    upper = np.full((2, 3), np.inf)

    with pytest.raises(ValueError, match="duplicate role"):
        estimator(roles, lower, upper)


def test_pa_nomix_equals_mixture_with_nan_K():
    # the no-mixture fast path must equal the mixture path fed all-NaN K
    from ltpred.pearson_aitken import pa_estimate_batched
    t = float(stats.norm.isf(0.05))
    cov = np.array([[0.5, 0.5, 0.25], [0.5, 1.0, 0.25], [0.25, 0.25, 1.0]])
    lowers = np.array([[-np.inf, t, -np.inf], [-np.inf, -np.inf, t]])
    uppers = np.array([[np.inf, np.inf, t], [np.inf, t, np.inf]])
    e0, v0 = pa_estimate_batched(cov, lowers, uppers, target=0)      # nomix path
    nan = np.full_like(lowers, np.nan)
    e1, v1 = pa_estimate_batched(cov, lowers, uppers, target=0, K_is=nan, K_pops=nan)
    assert np.allclose(e0, e1) and np.allclose(v0, v1)


def test_canonical_grouping_permuted_members():
    # families with the same relatives in different row order agree per-family
    t = float(stats.norm.isf(0.05))
    a = Family("a", [Member("o", t, np.inf), Member("m", -np.inf, t), Member("f", t, np.inf)])
    b = Family("b", [Member("f", t, np.inf), Member("o", t, np.inf), Member("m", -np.inf, t)])
    res = estimate_liability([a, b], h2=0.5, method="pa", out=("genetic",))
    # a and b are the same family with reordered rows -> identical estimate
    assert res.est["genetic"][0] == pytest.approx(res.est["genetic"][1])


def test_float32_bounds_match_float64():
    # float32 per-family bounds halve memory and match float64 to f32 precision
    from ltpred.simulate import simulate_under_LTM_single
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=400,
                                    pop_prev=0.05, seed=1)
    g64 = estimate_liability(sim.families, h2=0.5, method="gibbs", out=("genetic",),
                             n_sim=20_000, burn_in=400, seed=0)
    g32 = estimate_liability(sim.families, h2=0.5, method="gibbs", out=("genetic",),
                             n_sim=20_000, burn_in=400, seed=0, dtype=np.float32)
    assert np.allclose(g64.est["genetic"], g32.est["genetic"], atol=1e-5)
    p64 = estimate_liability(sim.families, h2=0.5, method="pa")
    p32 = estimate_liability(sim.families, h2=0.5, method="pa", dtype=np.float32)
    assert np.allclose(p64.est["genetic"], p32.est["genetic"], atol=1e-5)


def test_array_api_preserves_float32():
    from ltpred.estimate import estimate_liability_pa_arrays
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m", "f"]
    case = np.random.default_rng(0).random((100, 3)) < 0.2
    lo = np.where(case, t, -np.inf).astype(np.float32)
    hi = np.where(case, np.inf, t).astype(np.float32)
    e32, v32 = estimate_liability_pa_arrays(roles, lo, hi, h2=0.5)
    e64, v64 = estimate_liability_pa_arrays(roles, lo.astype(np.float64),
                                            hi.astype(np.float64), h2=0.5)
    assert np.allclose(e32, e64, atol=1e-5)


def test_bad_dtype_rejected():
    from ltpred.family import Family, Member
    fam = Family("f", [Member("o", 1.6, np.inf)])
    with pytest.raises(ValueError, match="float32 or float64"):
        estimate_liability([fam], h2=0.5, method="pa", dtype=np.int32)


def test_grouping_gives_same_answer_regardless_of_order():
    # mix of two structures; shuffling families must not change per-family results
    t = float(stats.norm.isf(0.05))
    trio = lambda i: Family(f"t{i}", [Member("o", t, np.inf),
                                      Member("m", -np.inf, t),
                                      Member("f", t, np.inf)])
    solo = lambda i: Family(f"s{i}", [Member("o", -np.inf, t)])
    fams = [trio(0), solo(0), trio(1), solo(1), trio(2)]

    res = estimate_liability(fams, h2=0.5, method="gibbs", out=("genetic",),
                             tol=0.02, n_sim=40_000, burn_in=800, seed=5)
    order = [3, 0, 4, 1, 2]
    res2 = estimate_liability([fams[i] for i in order], h2=0.5, method="gibbs",
                              out=("genetic",), tol=0.02, n_sim=40_000, burn_in=800, seed=5)
    # same fam -> same estimate; seed is tied to global position, so compare by id
    by_id = dict(zip(res.fam_ids, res.est["genetic"]))
    for fid, val in zip(res2.fam_ids, res2.est["genetic"]):
        # different global seed offset, but must agree within Monte-Carlo error
        assert abs(by_id[fid] - val) < 0.1


def test_batched_estimator_is_deterministic():
    t = float(stats.norm.isf(0.05))
    fams = [Family(f"f{i}", [Member("o", t if i % 2 else -np.inf,
                                    np.inf if i % 2 else t),
                             Member("m", -np.inf, t)]) for i in range(8)]
    a = estimate_liability(fams, h2=0.5, method="gibbs", out=("genetic", "full"),
                           tol=0.05, n_sim=20_000, burn_in=400, seed=3)
    b = estimate_liability(fams, h2=0.5, method="gibbs", out=("genetic", "full"),
                           tol=0.05, n_sim=20_000, burn_in=400, seed=3)
    assert np.array_equal(a.est["genetic"], b.est["genetic"])
    assert np.array_equal(a.est["full"], b.est["full"])


def test_multi_round_convergence_tightens_se():
    # a tight tolerance forces extra rounds; the reported SE must respect it
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", t, np.inf)])
    res = estimate_liability([fam], h2=0.5, method="gibbs", out=("genetic",),
                             tol=0.005, n_sim=20_000, burn_in=500, seed=1, max_rounds=20)
    assert res.se["genetic"][0] <= 0.005
