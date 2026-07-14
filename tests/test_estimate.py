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
                                             out="genetic", n_sim=20000, burn_in=500, seed=1)
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


def test_liability_sensitivity_h2_grid():
    from ltpred import simulate_under_LTM_single, liability_sensitivity
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=800, pop_prev=0.1, seed=9)
    grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    r = liability_sensitivity(sim.families, grid, method="pa", out="genetic")
    assert r.estimates.shape == (len(grid), 800)
    assert r.corr.shape == (len(grid), len(grid))
    assert np.allclose(np.diag(r.corr), 1.0)
    assert r.mean.shape == r.sd.shape == (len(grid),)
    assert r.out == "genetic"
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
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        liability_sensitivity(sim.families, [0.5, 1.5], method="pa")


@pytest.mark.parametrize("spelling", ["genetic", ("genetic",), "g", 0])
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
