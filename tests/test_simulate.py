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
