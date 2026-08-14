"""Simulation under the LTM, and the estimate-recovers-truth end-to-end check."""

import numpy as np
import pytest

from ltpred.simulate import simulate_under_LTM_single
from ltpred.estimate import estimate_liability


def test_simulated_prevalence_matches():
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=20_000,
                                    pop_prev=0.1, seed=0)
    assert sim.status["o"].mean() == pytest.approx(0.1, abs=0.01)
    assert sim.genetic.var() == pytest.approx(0.5, abs=0.05)   # var(g) = h2
    assert sim.full.var() == pytest.approx(1.0, abs=0.05)      # var(o) = 1


def test_simulation_family_structure():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], n_sim=5, seed=1)
    assert len(sim.families) == 5
    roles = [m.role for m in sim.families[0].members]
    assert roles == ["o", "m", "f", "s1"]   # g omitted (estimator adds it)


def test_estimate_recovers_true_genetic_liability():
    # posterior mean genetic liability should correlate with the simulated truth
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    n_sim=400, pop_prev=0.1, seed=7)
    res = estimate_liability(sim.families, h2=0.5, out=("genetic",),
                             tol=0.05, n_sim=15_000, burn_in=400, seed=1)
    r = np.corrcoef(res.est["genetic"], sim.genetic)[0, 1]
    assert r > 0.4


def test_estimate_beats_raw_status():
    # Classic LT-FH posterior mean should track true genetic liability at least as
    # well as the raw case/control label of the proband.
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    n_sim=400, pop_prev=0.1, seed=9)
    res = estimate_liability(sim.families, h2=0.5, out=("genetic",),
                             tol=0.05, n_sim=15_000, burn_in=400, seed=2)
    r_ltfh = np.corrcoef(res.est["genetic"], sim.genetic)[0, 1]
    r_status = np.corrcoef(sim.status["o"].astype(float), sim.genetic)[0, 1]
    assert r_ltfh > r_status


def test_cases_have_higher_estimates_than_controls():
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=300,
                                    pop_prev=0.1, seed=11)
    res = estimate_liability(sim.families, h2=0.5, out=("genetic",),
                             tol=0.05, n_sim=15_000, burn_in=400, seed=3)
    g = res.est["genetic"]
    case = sim.status["o"]
    assert g[case].mean() > g[~case].mean() + 0.3


def test_age_flavour_runs_and_is_finite():
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=50,
                                    pop_prev=0.1, use_age=True, seed=13)
    res = estimate_liability(sim.families, h2=0.5, out=("genetic",),
                             tol=0.1, n_sim=10_000, burn_in=300, seed=4)
    assert np.all(np.isfinite(res.est["genetic"]))


def test_age_flavour_is_generation_consistent_and_followup_coherent():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=400,
                                    pop_prev=0.1, use_age=True, seed=13)
    assert np.all(sim.ages["m"] > sim.ages["o"])
    assert np.all(sim.ages["f"] > sim.ages["o"])
    for role in ("o", "m", "f", "s1"):
        is_case = sim.status[role]
        assert np.all(sim.onset[role][is_case] <= sim.ages[role][is_case])
        # a lifetime-affected person who is still too young is a control
        future = np.isfinite(sim.onset[role]) & ~is_case
        assert np.all(sim.onset[role][future] > sim.ages[role][future])


def test_stochastic_onset_does_not_pin_cases_at_true_liability():
    sim = simulate_under_LTM_single(
        fam_vec=["m"], h2=0.5, n_sim=800, pop_prev=0.2, use_age=True,
        onset_model="stochastic", case_encoding="lifetime", seed=17,
    )
    o = sim.roles.index("o")
    case = sim.status["o"]
    assert case.any()
    # lifetime encoding: every case shares the population threshold, not l_o
    t = next(m.lower for m in sim.families[int(np.flatnonzero(case)[0])].members
             if m.role == "o")
    pins = np.array([
        next(m.lower for m in fam.members if m.role == "o")
        for fam, is_case in zip(sim.families, case) if is_case
    ])
    assert np.allclose(pins, t)
    assert not np.allclose(sim.liabilities[case, o], t, atol=0.05)


def _proband_case_pins(sim):
    case = sim.status["o"]
    pins = np.array([next(m.lower for m in fam.members if m.role == "o")
                     for fam in sim.families])
    return case, pins, sim.liabilities[:, sim.roles.index("o")]


def test_onset_resolution_controls_pin_fidelity():
    # Under threshold_crossing the onset age is the CIP inverse of the true
    # liability, so a pin rebuilt from an *exactly* recorded onset returns that
    # liability. The default one-year recording grid is a deliberate register
    # approximation and must NOT be mistaken for exact recovery: it shifts the
    # pin by ~1e-2, which is a floor under any oracle comparison built on it.
    def run(resolution):
        sim = simulate_under_LTM_single(
            fam_vec=["m", "f"], h2=0.5, n_sim=1500, pop_prev=0.1,
            use_age=True, seed=1, onset_resolution=resolution)
        case, pins, true_l = _proband_case_pins(sim)
        assert case.any()
        return np.abs(pins[case] - true_l[case])

    assert np.max(run(None)) < 1e-9
    coarse = np.max(run(1.0))
    assert 1e-3 < coarse < 0.1
    assert np.max(run(0.25)) < coarse


def test_onset_resolution_applies_to_interval_cases_too():
    # pin and interval must describe the same observation process: both derive
    # the case bound from the recorded onset age.
    kwargs = dict(fam_vec=["m"], h2=0.5, n_sim=600, pop_prev=0.1, use_age=True,
                  seed=3, onset_model="threshold_crossing")
    pinned = simulate_under_LTM_single(case_encoding="pin", **kwargs)
    interval = simulate_under_LTM_single(case_encoding="interval", **kwargs)
    case, pin_lo, _ = _proband_case_pins(pinned)
    _, int_lo, _ = _proband_case_pins(interval)
    assert case.any()
    assert np.allclose(pin_lo[case], int_lo[case])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, True])
def test_onset_resolution_is_validated(bad):
    with pytest.raises((ValueError, TypeError), match="onset_resolution"):
        simulate_under_LTM_single(n_sim=2, use_age=True, onset_resolution=bad)


def test_age_options_rejected_without_use_age():
    with pytest.raises(ValueError, match="use_age"):
        simulate_under_LTM_single(use_age=False, onset_model="stochastic")


def test_liability_dependent_rho_one_matches_crossing():
    kwargs = dict(fam_vec=["m", "f"], h2=0.5, n_sim=2000, pop_prev=0.2,
                  use_age=True, seed=21, onset_resolution=None)
    crossing = simulate_under_LTM_single(onset_model="threshold_crossing",
                                         **kwargs)
    dependent = simulate_under_LTM_single(
        onset_model="liability_dependent", onset_rho=1.0, **kwargs)
    # Same liabilities (same seed) so lifetime cases match; rho=1 recovers
    # the crossing onset among those cases.
    assert np.allclose(crossing.liabilities, dependent.liabilities)
    for role in ("o", "m", "f"):
        lifetime = np.isfinite(crossing.onset[role])
        assert lifetime.any()
        assert np.allclose(crossing.onset[role][lifetime],
                           dependent.onset[role][lifetime], atol=1e-6)


def test_liability_dependent_rho_zero_matches_stochastic():
    kwargs = dict(fam_vec=["m"], h2=0.5, n_sim=1500, pop_prev=0.2,
                  use_age=True, seed=22, onset_resolution=None)
    stochastic = simulate_under_LTM_single(onset_model="stochastic", **kwargs)
    independent = simulate_under_LTM_single(
        onset_model="liability_dependent", onset_rho=0.0, **kwargs)
    assert np.allclose(stochastic.liabilities, independent.liabilities)
    for role in ("o", "m"):
        assert np.allclose(stochastic.onset[role], independent.onset[role],
                           equal_nan=True)


def test_liability_dependent_advances_onset_with_liability():
    sim = simulate_under_LTM_single(
        fam_vec=["m"], h2=0.5, n_sim=4000, pop_prev=0.25, use_age=True,
        onset_model="liability_dependent", onset_rho=0.6, seed=23,
        onset_resolution=None)
    o = sim.roles.index("o")
    lifetime = np.isfinite(sim.onset["o"])
    assert lifetime.sum() > 50
    corr = np.corrcoef(sim.liabilities[lifetime, o], sim.onset["o"][lifetime])[0, 1]
    # Higher liability -> earlier onset, but not the crossing identity.
    assert corr < -0.3
    case, pins, _ = _proband_case_pins(sim)
    assert case.any()
    # Default encoding is lifetime, so observed cases share the population T.
    assert np.allclose(pins[case], pins[case][0])


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, True])
def test_onset_rho_is_validated(bad):
    with pytest.raises((ValueError, TypeError), match="onset_rho"):
        simulate_under_LTM_single(n_sim=2, use_age=True,
                                  onset_model="liability_dependent",
                                  onset_rho=bad)


def test_onset_rho_rejected_on_other_models():
    with pytest.raises(ValueError, match="onset_rho"):
        simulate_under_LTM_single(n_sim=2, use_age=True,
                                  onset_model="stochastic", onset_rho=0.5)


def test_child_roles_get_the_child_age_range():
    # "c1.2".rstrip("0123456789") stops at the dot and yields "c1.", so the
    # _AGE_RANGES["c"] entry was unreachable and children drew adult ages.
    from ltpred.simulate import _age_range
    assert _age_range("c1.1") == (0, 30)
    assert _age_range("c2.10") == (0, 30)
    assert _age_range("o") == (10, 60)
    assert _age_range("s12") == (10, 60)
    assert _age_range("mgm") == (65, 95)
    assert _age_range("mau3") == (40, 80)
