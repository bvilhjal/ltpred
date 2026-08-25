"""own_status in/out (use I vs II) tests."""
import numpy as np
import pytest
from scipy import stats

from ltpred import Family, Member, estimate_liability, families_from_columns


def _imr(t):
    """Inverse Mills ratio phi(t)/Phi-bar(t) for a unit normal."""
    return float(stats.norm.pdf(t) / stats.norm.sf(t))


def test_own_status_out_sign_lone_case_and_relatives():
    # Sign test, not just inequality. Lone case, no relatives, PA:
    # mu_in = h2 * IMR(T); the out estimand is the prior 0, which the API
    # refuses to return (see test_own_status_out_adult_raises). With
    # relatives, for a case mu_in > mu_out, and mu_out is the exact
    # single-truncation relatives-only score.
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    lone = Family("lone", [Member("o", t, np.inf)])
    mu_in = estimate_liability([lone], h2=h2, method="pa").genetic[0]
    assert mu_in == pytest.approx(h2 * _imr(t))

    fam = Family("r", [Member("o", t, np.inf, pid="person-1"),
                       Member("m", t, np.inf)])
    inn = estimate_liability([fam], h2=h2, method="pa")
    out = estimate_liability([fam], h2=h2, method="pa", own_status="out")
    assert inn.genetic[0] > out.genetic[0]
    assert out.genetic[0] == pytest.approx((h2 / 2) * _imr(t))
    assert out.genetic[0] > 0.0


def test_own_status_out_keeps_o_row_and_pid():
    t = float(stats.norm.isf(0.05))
    fam = Family("FAM", [Member("o", t, np.inf, pid="person-1"),
                         Member("m", -np.inf, t)])
    res = estimate_liability([fam], h2=0.5, method="pa", own_status="out")
    assert res.pids[0] == "person-1"
    assert res.fam_ids[0] == "FAM"
    # the input family is not mutated
    assert fam.members[0].lower == t

    built = families_from_columns(
        fam_id=["FAM", "FAM"], role=["o", "m"], lower=[t, -np.inf],
        upper=[np.inf, t], pid=["person-1", "mom"], own_status="out")
    o = next(m for m in built[0].members if m.role == "o")
    assert o.pid == "person-1"
    assert o.lower == -np.inf and o.upper == np.inf
    assert any(m.role == "o" for m in built[0].members)
    res2 = estimate_liability(built, h2=0.5, method="pa")
    assert res2.pids[0] == "person-1"


def test_omitting_o_without_flag_still_falls_back_pids_to_fam_id():
    t = float(stats.norm.isf(0.05))
    fam = Family("FAM", [Member("m", t, np.inf, pid="mom")])
    res = estimate_liability([fam], h2=0.5, method="pa")
    assert res.pids[0] == "FAM"


def test_array_and_kinship_honor_own_status_out():
    from ltpred.estimate import (estimate_liability_from_kinship,
                                 estimate_liability_pa_arrays)
    from ltpred.covariance import kinship_from_pedigree

    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    roles = ["o", "m"]
    lo = np.array([[t, t]])
    hi = np.array([[np.inf, np.inf]])
    est_in, _ = estimate_liability_pa_arrays(roles, lo, hi, h2=h2)
    est_out, _ = estimate_liability_pa_arrays(
        roles, lo, hi, h2=h2, own_status="out")
    assert est_in[0] > est_out[0]
    assert est_out[0] == pytest.approx((h2 / 2) * _imr(t))
    assert lo[0, 0] == t  # caller arrays are not mutated

    _, A = kinship_from_pedigree(
        ["o", "m"], father=[None, None], mother=["m", None])
    kin_in, _, _ = estimate_liability_from_kinship(A, lo, hi, h2=h2, target=0)
    kin_out, _, _ = estimate_liability_from_kinship(
        A, lo, hi, h2=h2, target=0, own_status="out")
    assert kin_in[0] > kin_out[0]
    assert kin_out[0] == pytest.approx((h2 / 2) * _imr(t))
    assert lo[0, 0] == t


def test_own_status_out_adult_raises():
    t = float(stats.norm.isf(0.05))
    fam = Family("adult", [Member("o", t, np.inf, pid="p")])
    with pytest.raises(ValueError, match="nothing to condition on"):
        estimate_liability([fam], h2=0.5, method="pa", own_status="out")
    from ltpred.estimate import estimate_liability_pa_arrays
    with pytest.raises(ValueError, match="nothing to condition on"):
        estimate_liability_pa_arrays(
            ["o"], np.array([[t]]), np.array([[np.inf]]), own_status="out")
    with pytest.raises(ValueError, match="nothing to condition on"):
        families_from_columns(
            fam_id=["adult"], role=["o"], lower=[t], upper=[np.inf],
            own_status="out")


@pytest.mark.parametrize("bad", ["IN", "Out", "unbound", "omit", True, None, 1])
def test_invalid_own_status_raises(bad):
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", -np.inf, t)])
    with pytest.raises(ValueError, match="own_status"):
        estimate_liability([fam], h2=0.5, own_status=bad)
    with pytest.raises(ValueError, match="own_status"):
        families_from_columns(
            fam_id=["f", "f"], role=["o", "m"], lower=[t, -np.inf],
            upper=[np.inf, t], own_status=bad)


def test_own_status_in_matches_default_bit_for_bit():
    from ltpred import simulate_under_LTM_single

    sim = simulate_under_LTM_single(
        fam_vec=["m", "f"], h2=0.5, pop_prev=0.1, n_sim=40,
        use_age=False, seed=3)
    default = estimate_liability(sim.families, h2=0.5, method="pa")
    explicit = estimate_liability(
        sim.families, h2=0.5, method="pa", own_status="in")
    np.testing.assert_array_equal(default.genetic, explicit.genetic)
    np.testing.assert_array_equal(default.var["genetic"], explicit.var["genetic"])
    np.testing.assert_array_equal(default.pids, explicit.pids)


def test_gibbs_own_status_out_still_rejects_mixture():
    t = float(stats.norm.isf(0.10))
    fam = Family("f", [Member("o", -np.inf, t, K_i=0.05, K_pop=0.10),
                       Member("m", -np.inf, t, K_i=0.05, K_pop=0.10)])
    with pytest.raises(ValueError, match="only supported by Pearson-Aitken"):
        estimate_liability([fam], h2=0.5, method="gibbs", use_mixture=True,
                           own_status="out")
