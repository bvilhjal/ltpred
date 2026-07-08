"""Fitting liability-scale heritability from family data (data-augmentation Gibbs)."""

import numpy as np
import pytest

from ltpred import (simulate_under_LTM_single, fit_heritability,
                    fit_variance_components)
from ltpred.family import Family, Member


@pytest.mark.parametrize("h2_true", [0.3, 0.6])
def test_recovers_simulated_heritability(h2_true):
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=h2_true,
                                    n_sim=2500, pop_prev=0.1, seed=3)
    res = fit_heritability(sim.families, n_iter=500, burn_in=150, inner_sweeps=5,
                           damp=0.2, seed=1)
    assert res.h2 == pytest.approx(h2_true, abs=0.07)
    assert 0.0 < res.h2 < 1.0
    assert res.h2_se > 0
    assert res.samples.shape == (350,)
    assert res.trace.shape == (500,)


def test_converges_from_different_inits():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=3000,
                                    pop_prev=0.1, seed=4)
    lo = fit_heritability(sim.families, h2_init=0.15, n_iter=500, burn_in=150, seed=1).h2
    hi = fit_heritability(sim.families, h2_init=0.85, n_iter=500, burn_in=150, seed=1).h2
    assert abs(lo - hi) < 0.05          # the fit forgets its starting point


def test_lone_probands_raise():
    # no relatives -> no related pairs -> h2 not identified
    t = 1.64
    fams = [Family(i, [Member("o", -np.inf, t)]) for i in range(20)]
    with pytest.raises(ValueError, match="no related pairs"):
        fit_heritability(fams, n_iter=50, burn_in=10)


def test_variance_components_additive_matches_heritability():
    # single-component fit == fit_heritability (same data-augmentation)
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=1500, pop_prev=0.1, seed=2)
    vc = fit_variance_components(sim.families, ("A",), n_iter=500, burn_in=150, seed=1)
    he = fit_heritability(sim.families, n_iter=500, burn_in=150, seed=1).h2
    assert vc.components["A"] == pytest.approx(he, abs=0.05)
    assert vc.residual == pytest.approx(1.0 - sum(vc.components.values()))
    assert vc.traces["A"].shape == (350,)


def test_variance_components_recovers_A_and_C():
    # additive + common-environment: multiple HE regression separates them.
    # simulate liabilities with a full-sib shared-environment component directly.
    import numpy as np
    from ltpred.covariance import correct_positive_definite
    from ltpred.thresholds import liability_threshold
    from ltpred.fit import _component_matrix
    roles = ["m", "f", "s1", "s2", "s3", "s4"]
    a2, c2 = 0.4, 0.2
    Sig = (1 - a2 - c2) * np.eye(len(roles)) + a2 * _component_matrix(roles, "A") \
        + c2 * _component_matrix(roles, "C")
    Sig, _ = correct_positive_definite(Sig)
    rng = np.random.default_rng(0)
    liab = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=3500)
    t = float(liability_threshold(0.1))
    fams = []
    for i in range(liab.shape[0]):
        members = [Member(r, (t if liab[i, c] > t else -np.inf),
                          (np.inf if liab[i, c] > t else t)) for c, r in enumerate(roles)]
        fams.append(Family(i, members))
    r = fit_variance_components(fams, ("A", "C"), n_iter=800, burn_in=250, seed=1)
    # a real C is recovered (not collapsed to 0, not absorbed into A); the sampler
    # uses numba's parallel RNG so the exact value drifts run-to-run -> band check
    assert 0.28 < r.components["A"] < 0.52          # true 0.40
    assert 0.12 < r.components["C"] < 0.30          # true 0.20
    assert r.residual == pytest.approx(1.0 - a2 - c2, abs=0.12)


def test_variance_components_validates_input():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=400, pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="unknown component"):
        fit_variance_components(sim.families, ("A", "Z"), n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="unknown component"):        # D not supported
        fit_variance_components(sim.families, ("A", "D"), n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="duplicate"):
        fit_variance_components(sim.families, ("A", "A"), n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="burn_in"):
        fit_variance_components(sim.families, ("A",), n_iter=50)
    # C with no full-sib pairs -> not identified
    par = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=300,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="not identified"):
        fit_variance_components(par.families, ("A", "C"), n_iter=100, burn_in=30)
