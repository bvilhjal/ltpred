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


def test_variance_components_smoke():
    # experimental animal-model Gibbs: runs, returns sane proportions, validates input
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=800, pop_prev=0.1, seed=1)
    r = fit_variance_components(sim.families, ("A",), n_iter=300, burn_in=100, seed=1)
    assert 0.0 <= r.components["A"] <= 1.0
    assert r.residual == pytest.approx(1.0 - sum(r.components.values()))
    with pytest.raises(ValueError, match="unknown component"):
        fit_variance_components(sim.families, ("A", "Z"), n_iter=10)
