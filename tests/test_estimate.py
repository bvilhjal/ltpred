"""End-to-end liability estimation and the batch-means convergence rule."""

import warnings

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member, families_from_columns
from ltpred.estimate import (batch_means, estimate_liability,
                             estimate_liability_multi)


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


def test_single_proband_case_only():
    # a lone case with no relatives -> E[g] = h2 * IMR(T)
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    fam = Family("f1", [Member("o", lower=t, upper=np.inf)])
    res = estimate_liability([fam], h2=h2, out=("genetic", "full"),
                             tol=0.02, n_sim=40_000, burn_in=800, seed=1)
    assert res.est["full"][0] == pytest.approx(_imr(t), abs=0.05)
    assert res.est["genetic"][0] == pytest.approx(h2 * _imr(t), abs=0.05)
    assert res.se["genetic"][0] <= 0.02


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
        res = estimate_liability([fam], h2=0.5, out=("genetic",), tol=1e-6,
                                 n_sim=5000, burn_in=200, max_rounds=1, seed=1)
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
