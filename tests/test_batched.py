"""The batched, parallel estimation path: agreement, grouping, determinism."""

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member
from ltpred.gibbs import gibbs_estimate_batched, gibbs_params
from ltpred.estimate import estimate_liability


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


@pytest.mark.jit_required
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
    tot, tot_sq, bm_sum, bm_sumsq = gibbs_estimate_batched(
        P, sd, sd0, lowers, uppers, np.array([0, 1]), n_sim, 1000, b, nb,
        np.array([7]), cov=cov)
    est = tot[0] / n_sim
    ss = bm_sumsq[0] - bm_sum[0] ** 2 / nb        # sum((Y - Ybar)^2)
    se = np.sqrt(b * ss / (nb - 1) / n_sim)
    assert est[0] == pytest.approx(0.5 * _imr(t), abs=0.03)
    assert est[1] == pytest.approx(_imr(t), abs=0.03)
    assert np.all(se > 0)

    # The streamed sums of squares give the *posterior* variance, a different
    # quantity from the batch-means SE above: for the analytic one-truncation
    # case, Var(o | o > t) = 1 + t*IMR - IMR^2 and Var(g | o > t) =
    # h2 + h2^2 * (Var(o | o > t) - 1) by the Pearson-Aitken rank-1 update.
    var = tot_sq[0] / n_sim - est ** 2
    var_o = 1.0 + t * _imr(t) - _imr(t) ** 2
    assert var[1] == pytest.approx(var_o, rel=0.05)
    assert var[0] == pytest.approx(0.5 + 0.25 * (var_o - 1.0), rel=0.05)
    # posterior variance is a property of the target, not of how long we sampled
    tot2, tot_sq2, _, _ = gibbs_estimate_batched(
        P, sd, sd0, lowers, uppers, np.array([0, 1]), 4 * n_sim, 1000,
        b, nb, np.array([7]), cov=cov)
    var2 = tot_sq2[0] / (4 * n_sim) - (tot2[0] / (4 * n_sim)) ** 2
    assert var2 == pytest.approx(var, rel=0.05)


@pytest.mark.jit_required
def test_collapsed_genetic_matches_full_chain_mean():
    # Public estimate path collapses unbounded g. The mean of E[g|y] equals
    # the mean of sampled g; the streamed var still matches the PA update.
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    t = float(stats.norm.isf(0.05))
    lowers = np.array([[-np.inf, t], [-np.inf, -np.inf]])
    uppers = np.array([[np.inf, np.inf], [np.inf, t]])
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))
    n_sim, burn, seed = 30_000, 800, np.array([3, 4])
    b = int(np.floor(np.sqrt(n_sim)))
    nb = n_sim // b
    kwargs = dict(lowers=lowers, uppers=uppers, out_idx=np.array([0, 1]),
                  n_sim=n_sim, burn_in=burn, batch_size=b, n_batch=nb,
                  seeds=seed)
    tot_c, tsq_c, _, _ = gibbs_estimate_batched(
        P, sd, sd0, cov=cov, collapse=True, **kwargs)
    tot_f, tsq_f, _, _ = gibbs_estimate_batched(
        P, sd, sd0, collapse=False, **kwargs)
    est_c = tot_c / n_sim
    est_f = tot_f / n_sim
    var_c = tsq_c / n_sim - est_c ** 2
    var_f = tsq_f / n_sim - est_f ** 2
    np.testing.assert_allclose(est_c, est_f, atol=0.03)
    np.testing.assert_allclose(var_c, var_f, atol=0.03)
    imr = _imr(t)
    assert est_c[0, 0] == pytest.approx(0.5 * imr, abs=0.03)
    assert est_c[0, 1] == pytest.approx(imr, abs=0.03)


def test_collapse_all_unbounded_is_the_prior():
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))
    lowers = np.full((4, 2), -np.inf)
    uppers = np.full((4, 2), np.inf)
    n_sim, b, nb = 20, 4, 5
    tot, tsq, bm, bmsq = gibbs_estimate_batched(
        P, sd, sd0, lowers, uppers, np.array([0, 1]), n_sim, 0, b, nb,
        np.arange(4), cov=cov)
    assert np.array_equal(tot, np.zeros((4, 2)))
    np.testing.assert_allclose(
        tsq / n_sim, np.broadcast_to(np.diag(cov), tsq.shape))
    assert np.array_equal(bm, np.zeros((4, 2)))
    assert np.array_equal(bmsq, np.zeros((4, 2)))


def test_multi_round_pooling_matches_offline_batch_means():
    # _estimate_group pools the streamed batch-mean sums (bm_s1/bm_s2) across
    # rounds. With n_sim = 4 the per-round batching (b = 2, nb = 2) tiles the
    # concatenated draws exactly, so two tol-forced rounds must reproduce the
    # offline batch_means of the same-seeded rtmvnorm_gibbs draws to machine
    # precision.
    from ltpred.estimate import _estimate_group, _base_seeds, batch_means
    from ltpred.gibbs import rtmvnorm_gibbs
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    t = float(stats.norm.isf(0.05))
    lowers = np.array([[-np.inf, t], [t, -np.inf], [-np.inf, -np.inf]])
    uppers = np.array([[np.inf, np.inf], [np.inf, t], [t, t]])
    seed, max_rounds = 11, 2
    base = _base_seeds(seed, 3, max_rounds)
    est, se, var = _estimate_group(cov, [0, 1], lowers, uppers, base,
                                   tol=1e-12, n_sim=4, burn_in=25,
                                   max_rounds=max_rounds)
    for f in range(3):
        draws = np.vstack([
            rtmvnorm_gibbs(cov, lowers[f], uppers[f], out=(0, 1), n_sim=4,
                           burn_in=25, seed=int((base[f] + r) % (2 ** 32)))
            for r in range(max_rounds)
        ])
        est_off, se_off = batch_means(draws)
        np.testing.assert_allclose(est[f], est_off, atol=1e-12)
        np.testing.assert_allclose(se[f], se_off, atol=1e-12)


def test_pa_arrays_match_object_api():
    from ltpred.estimate import estimate_liability_pa_arrays, _estimate_liability_pa
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(1)
    case = rng.random((50, 4)) < 0.3
    lower = np.where(case, t, -np.inf)
    upper = np.where(case, np.inf, t)
    est_a, var_a = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    fams = [Family(i, [Member(r, lower[i, k], upper[i, k]) for k, r in enumerate(roles)])
            for i in range(50)]
    obj = _estimate_liability_pa(fams, h2=0.5, out=("genetic",))
    assert np.allclose(est_a, obj.est["genetic"])
    assert np.allclose(var_a, obj.var["genetic"])


@pytest.mark.jit_required
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
        estimator(roles, lower, upper, 0.5)


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


@pytest.mark.jit_required
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


@pytest.mark.jit_required
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


@pytest.mark.jit_required
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


@pytest.mark.jit_required
def test_multi_round_convergence_tightens_se():
    # a tight tolerance forces extra rounds; the reported SE must respect it
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", t, np.inf)])
    res = estimate_liability([fam], h2=0.5, method="gibbs", out=("genetic",),
                             tol=0.005, n_sim=20_000, burn_in=500, seed=1, max_rounds=20)
    assert res.se["genetic"][0] <= 0.005


@pytest.mark.jit_required
def test_gibbs_arrays_coerces_integer_and_list_bounds():
    # The `lower = as_bounds(lower)` line sat inside the duplicate-role `raise`
    # block, so it never ran: integer bounds failed when the auto-added genetic
    # row needed a -inf default, and list-of-lists raised AttributeError.
    from ltpred.estimate import (estimate_liability_gibbs_arrays,
                                 estimate_liability_pa_arrays)
    roles = ["o", "m"]
    lo_int = np.array([[-2, 1], [1, -2]])
    lo = lo_int.astype(float)
    hi = np.array([[1.0, np.inf], [np.inf, 1.0]])
    est, _ = estimate_liability_gibbs_arrays(roles, lo_int, hi, h2=0.5,
                                             tol=0.05, n_sim=20_000,
                                             burn_in=400, seed=1)
    est_float, _ = estimate_liability_gibbs_arrays(
        roles, lo, hi, h2=0.5, tol=0.05, n_sim=20_000, burn_in=400, seed=1)
    pa_est, _ = estimate_liability_pa_arrays(roles, lo, hi, h2=0.5)
    assert np.all(np.isfinite(est)) and np.all(np.abs(est) < 5.0)
    np.testing.assert_allclose(est, est_float, atol=1e-12)
    np.testing.assert_allclose(est, pa_est, atol=0.05)
    # list-of-lists must work too, as the docstring promises
    est_list, _ = estimate_liability_gibbs_arrays(
        roles, lo.tolist(), hi.tolist(), h2=0.5, tol=0.05, n_sim=20_000,
        burn_in=400, seed=1)
    np.testing.assert_allclose(est_list, est, atol=1e-12)


def test_mixed_precision_bounds_keep_their_own_dtype():
    # _align_to_cov used columns[0].dtype for every output, silently demoting a
    # float64 `upper` to float32 whenever `lower` was float32.
    from ltpred.estimate import _align_to_cov
    lo = np.zeros((2, 2), dtype=np.float32)
    hi = np.zeros((2, 2), dtype=np.float64)
    out_lo, out_hi = _align_to_cov(["o", "m"], ["g", "o", "m"], (lo, hi),
                                   (-np.inf, np.inf))
    assert out_lo.dtype == np.float32
    assert out_hi.dtype == np.float64


@pytest.mark.jit_required
def test_multi_trait_full_output_smoke():
    # out=("full",) on the multi-trait path returns finite per-trait estimates
    # under the expected (output, phenotype) column names
    gcorr = np.array([[1.0, 0.3], [0.3, 1.0]])
    fcorr = np.array([[1.0, 0.2], [0.2, 1.0]])
    t = float(stats.norm.isf(0.05))
    fam = Family("f1", [Member("o", [t, -np.inf], [np.inf, t]),
                        Member("m", [-np.inf, t], [t, np.inf])])
    res = estimate_liability([fam], h2=[0.5, 0.4], genetic_corrmat=gcorr,
                             full_corrmat=fcorr, phen_names=["A", "B"],
                             out=("full",), tol=0.1, n_sim=10_000,
                             burn_in=200, seed=1)
    assert set(res.est) == {"full_A", "full_B"}
    assert np.isfinite(res.est["full_A"]).all()
    assert np.isfinite(res.est["full_B"]).all()


@pytest.mark.parametrize("method", ["pa", "gibbs"])
def test_family_without_proband_own_status_estimates(method):
    # no member with role "o": the proband's own liability is inserted as
    # uninformative, so the estimate is driven by the relatives alone.
    #
    # Asserting only isfinite here would pass against an implementation that
    # discarded every relative and returned the prior mean 0, so pin the value
    # instead. With a single truncated relative PA is exact, and the whole
    # family reduces to one Pearson-Aitken fold:
    #     E[g | l_m > t] = Cov(g, l_m)/Var(l_m) * E[l_m | l_m > t]
    #                    = (h2/2) * phi(t)/(1 - Phi(t)).
    h2 = 0.5
    t = float(stats.norm.isf(0.05))
    lam = float(stats.norm.pdf(t) / stats.norm.sf(t))       # inverse Mills ratio
    expected = (h2 / 2.0) * lam
    kwargs = (dict(n_sim=40_000, burn_in=500, tol=0.002, seed=1)
              if method == "gibbs" else {})
    atol = 0.02 if method == "gibbs" else 1e-9              # Monte-Carlo slack

    one = Family("f", [Member("m", t, np.inf)])
    got = estimate_liability([one], h2=h2, method=method,
                             out=("genetic",), **kwargs).est["genetic"][0]
    assert got == pytest.approx(expected, abs=atol)

    # ...and the relatives must actually drive the sign: an affected pair pulls
    # the genetic estimate up, an unaffected pair pulls it down, and the mixed
    # family lands strictly between. This ordering is what fails if the
    # relative rows stop reaching the covariance.
    def est(members):
        return estimate_liability([Family("f", members)], h2=h2, method=method,
                                  out=("genetic",), **kwargs).est["genetic"][0]

    both_cases = est([Member("m", t, np.inf), Member("f", t, np.inf)])
    mixed = est([Member("m", t, np.inf), Member("f", -np.inf, t)])
    both_controls = est([Member("m", -np.inf, t), Member("f", -np.inf, t)])
    assert both_controls < 0.0 < mixed < both_cases
    assert both_cases - both_controls > 0.5
