"""Pearson-Aitken (PA-FGRS) estimation: exactness, agreement with Gibbs, mixture."""

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member
from ltpred.pearson_aitken import (pa_algorithm, pa_estimate_batched,
                                   tnorm_moments, tnorm_mixture_conditional)
from ltpred.thresholds import pa_thresholds
from ltpred.estimate import estimate_liability, estimate_liability_pa
from ltpred.simulate import simulate_under_LTM_single


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


@pytest.mark.parametrize("lo,hi", [(-np.inf, np.inf), (0.5, 2.0), (-1.0, 1.0),
                                   (1.5, np.inf), (-np.inf, -0.5)])
def test_tnorm_moments_match_scipy(lo, hi):
    m, v = tnorm_moments(0.3, 1.7, lo, hi)
    ref = stats.truncnorm((lo - 0.3) / np.sqrt(1.7), (hi - 0.3) / np.sqrt(1.7),
                          loc=0.3, scale=np.sqrt(1.7))
    assert m == pytest.approx(ref.mean(), abs=1e-9)
    assert v == pytest.approx(ref.var(), abs=1e-9)


def test_tnorm_moments_point_mass_and_infinite():
    assert tnorm_moments(0.0, 1.0, 1.3, 1.3) == (1.3, 0.0)
    m, v = tnorm_moments(0.7, 2.0, -np.inf, np.inf)
    assert m == pytest.approx(0.7) and v == pytest.approx(2.0)


def test_pa_single_case_is_exact():
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    cov = np.array([[h2, h2], [h2, 1.0]])  # roles g, o
    est, var = pa_algorithm(cov, lower=[-np.inf, t], upper=[np.inf, np.inf], target=0)
    assert est == pytest.approx(h2 * _imr(t), abs=1e-9)
    assert var > 0


def test_pa_single_control_is_negative():
    h2, prev = 0.5, 0.2
    t = float(stats.norm.isf(prev))
    cov = np.array([[h2, h2], [h2, 1.0]])
    est, _ = pa_algorithm(cov, lower=[-np.inf, -np.inf], upper=[np.inf, t], target=0)
    assert est == pytest.approx(h2 * (-stats.norm.pdf(t) / stats.norm.cdf(t)), abs=1e-9)


def test_pa_pinned_case_conditions_exactly():
    # ADuLT point-mass: o pinned at c -> E[g] = h2 * c exactly
    h2, c = 0.5, 1.3
    cov = np.array([[h2, h2], [h2, 1.0]])
    est, var = pa_algorithm(cov, lower=[-np.inf, c], upper=[np.inf, c], target=0)
    assert est == pytest.approx(h2 * c)
    assert var == pytest.approx(h2 - h2 ** 2)


def test_pa_matches_gibbs_on_families():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    pop_prev=0.1, n_sim=500, seed=3)
    pa = estimate_liability(sim.families, h2=0.5, method="pa", out=("genetic",))
    gb = estimate_liability(sim.families, h2=0.5, method="gibbs", out=("genetic",),
                            tol=0.02, n_sim=40_000, burn_in=800, seed=1)
    diff = np.abs(pa.est["genetic"] - gb.est["genetic"])
    assert np.corrcoef(pa.est["genetic"], gb.est["genetic"])[0, 1] > 0.99
    assert diff.mean() < 0.02


def test_pa_family_history_raises_estimate():
    t = float(stats.norm.isf(0.05))
    def fam(rel_case, fid):
        lo, hi = (t, np.inf) if rel_case else (-np.inf, t)
        return Family(fid, [Member("o", t, np.inf)] +
                      [Member(r, lo, hi) for r in ("m", "f", "s1")])
    res = estimate_liability_pa([fam(True, "aff"), fam(False, "healthy")],
                                h2=0.5, out=("genetic",))
    assert res.est["genetic"][0] > res.est["genetic"][1] + 0.1


def test_pa_result_has_zero_se_and_variance():
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", -np.inf, t)])
    res = estimate_liability_pa([fam], h2=0.5, out=("genetic", "full"))
    assert np.all(res.se["genetic"] == 0)
    assert res.var is not None and res.var["genetic"][0] > 0


def test_batched_matches_single_pa():
    t = float(stats.norm.isf(0.05))
    cov = np.array([[0.5, 0.5, 0.25], [0.5, 1.0, 0.25], [0.25, 0.25, 1.0]])
    lowers = np.array([[-np.inf, t, -np.inf], [-np.inf, -np.inf, t]])
    uppers = np.array([[np.inf, np.inf, t], [np.inf, t, np.inf]])
    est, var = pa_estimate_batched(cov, lowers, uppers, target=0)
    for i in range(2):
        e, v = pa_algorithm(cov, lowers[i], uppers[i], target=0)
        assert est[i] == pytest.approx(e)
        assert var[i] == pytest.approx(v)


def test_mixture_raises_young_control_liability():
    # a young control (K_i << K_pop) is weaker evidence of low liability
    K_i, K_pop = 0.01, 0.15
    t_i = float(stats.norm.isf(K_i))
    nomix, _ = tnorm_moments(0.0, 1.0, -np.inf, t_i)
    mix, _ = tnorm_mixture_conditional(0.0, 1.0, -np.inf, t_i, K_i=K_i, K_pop=K_pop)
    assert mix > nomix


def test_mixture_changes_genetic_estimate():
    # a proband case with a single young control sibling: the mixture (accounting
    # for the sibling's residual risk) should not decrease the genetic estimate.
    status = np.array([1, 0])          # proband case, sibling control
    age = np.array([55, 25])
    lo, hi, ki, kp = pa_thresholds(status, age, pop_prev=0.1)
    fam = Family("f", [Member("o", lo[0], hi[0]),
                       Member("s1", lo[1], hi[1], K_i=ki[1], K_pop=kp[1])])
    with_mix = estimate_liability_pa([fam], h2=0.5, out=("genetic",), use_mixture=True)
    no_mix = estimate_liability_pa([fam], h2=0.5, out=("genetic",), use_mixture=False)
    assert with_mix.est["genetic"][0] >= no_mix.est["genetic"][0] - 1e-9
    assert with_mix.est["genetic"][0] != no_mix.est["genetic"][0]


def test_method_dispatch_and_multi_trait_error():
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf)])
    for m in ("pa", "pearson-aitken", "PA-FGRS"):
        res = estimate_liability([fam], h2=0.5, method=m, out=("genetic",))
        assert res.est["genetic"][0] > 0
    with pytest.raises(NotImplementedError):
        estimate_liability([fam], h2=[0.5, 0.5], method="pa",
                           genetic_corrmat=np.eye(2), full_corrmat=np.eye(2))
    with pytest.raises(ValueError):
        estimate_liability([fam], h2=0.5, method="bogus")
