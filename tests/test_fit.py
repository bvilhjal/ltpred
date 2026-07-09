"""Fitting liability-scale heritability from family data (data-augmentation Gibbs)."""

import numpy as np
import pytest

from ltpred import (simulate_under_LTM_single, fit_heritability,
                    fit_variance_components, fit_genetic_correlation)
from ltpred.family import Family, Member


def _simulate_two_trait(fam_vec, h2_vec, rg, rp, n_fam, prev, seed):
    """Two-trait families: each member gets one case/control interval per trait."""
    from ltpred.covariance import construct_covmat_multi, correct_positive_definite
    from ltpred.thresholds import liability_threshold
    P = len(h2_vec)
    cov = construct_covmat_multi(fam_vec=fam_vec, add_ind=True,
                                 genetic_corrmat=rg, full_corrmat=rp,
                                 h2_vec=np.asarray(h2_vec, float))
    roles = cov.roles
    k = len(roles) // P
    fam_roles = roles[:k]
    Sig, _ = correct_positive_definite(cov.matrix)
    rng = np.random.default_rng(seed)
    liab = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=n_fam)
    t = [float(liability_threshold(prev[p])) for p in range(P)]
    obs = [r for r in fam_roles if r != "g"]
    fams = []
    for i in range(n_fam):
        members = []
        for r in obs:
            lo, hi = [], []
            for p in range(P):
                case = liab[i, p * k + fam_roles.index(r)] > t[p]
                lo.append(t[p] if case else -np.inf)
                hi.append(np.inf if case else t[p])
            members.append(Member(role=r, lower=lo, upper=hi))
        fams.append(Family(fam_id=i, members=members))
    return fams


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


def test_genetic_correlation_recovers_rg():
    # two correlated traits: cross-trait HE recovers r_g and per-trait h2
    rg = np.array([[1.0, 0.5], [0.5, 1.0]])
    rp = np.array([[1.0, 0.2], [0.2, 1.0]])
    fams = _simulate_two_trait(["m", "f", "s1", "s2"], [0.5, 0.4], rg, rp,
                               n_fam=2500, prev=[0.1, 0.1], seed=101)
    r = fit_genetic_correlation(fams, n_iter=800, burn_in=250, seed=1,
                                phen_names=["A", "B"])
    assert r.rg.shape == (2, 2)
    assert r.phen_names == ["A", "B"]
    assert np.allclose(np.diag(r.rg), 1.0)
    assert r.rg[0, 1] == pytest.approx(r.rg[1, 0])            # symmetric
    assert 0.25 < r.rg[0, 1] < 0.75                          # true 0.50 (parallel-RNG band)
    assert r.h2[0] == pytest.approx(0.5, abs=0.12)
    assert r.h2[1] == pytest.approx(0.4, abs=0.12)
    assert np.allclose(np.diag(r.genetic_cov), r.h2)
    # phenotypic covariance decomposes into genetic + environmental (rp = G + E)
    assert np.allclose(r.rp, r.genetic_cov + r.env_cov)
    assert np.allclose(np.diag(r.env_cov), 1.0 - r.h2)
    assert r.re.shape == (2, 2) and np.allclose(np.diag(r.re), 1.0)


def test_genetic_correlation_null_no_false_positive():
    # genetically uncorrelated traits (r_g=0) but phenotypically correlated (r_p=0.3)
    rg = np.eye(2)
    rp = np.array([[1.0, 0.3], [0.3, 1.0]])
    fams = _simulate_two_trait(["m", "f", "s1", "s2"], [0.5, 0.5], rg, rp,
                               n_fam=2500, prev=[0.1, 0.1], seed=7)
    r = fit_genetic_correlation(fams, n_iter=800, burn_in=250, seed=1)
    assert abs(r.rg[0, 1]) < 0.30                            # no spurious genetic corr
    assert r.rp[0, 1] > 0.10                                 # phenotypic corr still seen
    # the phenotypic correlation shows up as an *environmental* one instead
    # (true r_e = 0.3 / (1-0.5) = 0.6)
    assert r.re[0, 1] > 0.30


def test_genetic_correlation_validates_input():
    # single-trait families (scalar bounds) -> needs >= 2 traits
    single = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                       n_sim=200, pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="2 traits"):
        fit_genetic_correlation(single.families, n_iter=50, burn_in=10)
    # lone probands (no related pairs)
    lone = [Family(i, [Member("o", [-np.inf, -np.inf], [1.0, 1.0])]) for i in range(20)]
    with pytest.raises(ValueError, match="related pairs"):
        fit_genetic_correlation(lone, n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="burn_in"):
        fit_genetic_correlation(
            _simulate_two_trait(["m", "f", "s1"], [0.5, 0.5], np.eye(2), np.eye(2),
                                50, [0.1, 0.1], seed=1), n_iter=50)


def test_bootstrap_fit_scalar_and_calibration():
    from ltpred import bootstrap_fit
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=1500, pop_prev=0.1, seed=5)
    full = fit_heritability(sim.families, n_iter=400, burn_in=120, seed=1)
    bs = bootstrap_fit(sim.families,
                       lambda f: fit_heritability(f, n_iter=400, burn_in=120, seed=1).h2,
                       n_boot=25, seed=0)
    assert bs.estimate.shape == ()                       # scalar estimator -> 0-d
    assert bs.samples.shape == (25,)
    assert bs.se > 0
    assert bs.ci_low < float(bs.estimate) < bs.ci_high
    assert bs.ci_level == 0.95
    # the whole point: the bootstrap SE is far larger than the within-dataset se
    assert bs.se > 5 * full.h2_se


def test_bootstrap_fit_vector_and_errors():
    import numpy as np
    from ltpred import bootstrap_fit, fit_variance_components
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2", "s3"], h2=0.5,
                                    n_sim=1200, pop_prev=0.1, seed=6)
    bs = bootstrap_fit(
        sim.families,
        lambda f: np.array(list(fit_variance_components(
            f, ("A", "C"), n_iter=300, burn_in=100, seed=1).components.values())),
        n_boot=12, seed=1)
    assert bs.estimate.shape == (2,)
    assert bs.se.shape == (2,) and np.all(bs.se > 0)
    assert bs.ci_low.shape == (2,) and bs.ci_high.shape == (2,)
    with pytest.raises(ValueError, match="at least 2 families"):
        bootstrap_fit(sim.families[:1], lambda f: 0.0, n_boot=3)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_fit(sim.families, lambda f: 0.0, n_boot=3, ci_level=1.5)


def _sim_ac(fam, a2, c2, n, seed, prev=0.1):
    import numpy as np
    from ltpred.covariance import correct_positive_definite
    from ltpred.thresholds import liability_threshold
    from ltpred.fit import _component_matrix
    Sig = (1 - a2 - c2) * np.eye(len(fam)) + a2 * _component_matrix(fam, "A")
    if c2 > 0:
        Sig = Sig + c2 * _component_matrix(fam, "C")
    Sig, _ = correct_positive_definite(Sig)
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal(np.zeros(len(fam)), Sig, size=n)
    t = float(liability_threshold(prev))
    return [Family(i, [Member(r, (t if L[i, c] > t else -np.inf),
                              (np.inf if L[i, c] > t else t)) for c, r in enumerate(fam)])
            for i in range(n)]


def test_component_test_detects_real_C():
    from ltpred import test_variance_component
    fams = _sim_ac(["m", "f", "s1", "s2", "s3", "s4"], 0.4, 0.2, 1500, 1)
    r = test_variance_component(fams, "C", n_boot=25, seed=1, n_iter=350, burn_in=100)
    assert r.estimate > 0.10                       # a real C is estimated
    assert r.null.shape == (25,)
    assert 0.0 < r.p_value <= 1.0
    assert r.p_value < 0.2                          # strong signal -> significant
    assert r.null.mean() < r.estimate              # null centred below the observed


def test_component_test_validates_and_rejects_pinned():
    from ltpred import test_variance_component
    fams = _sim_ac(["m", "f", "s1", "s2"], 0.5, 0.0, 300, 2)
    with pytest.raises(ValueError, match="component"):
        test_variance_component(fams, "A", n_boot=3)
    with pytest.raises(ValueError, match="component"):
        test_variance_component(fams, "D", n_boot=3)
    # pinned (age-of-onset) bounds are not supported by the parametric bootstrap;
    # reuse the identifiable A+C structure but encode cases as a point mass
    base = _sim_ac(["m", "f", "s1", "s2", "s3", "s4"], 0.5, 0.0, 50, 4)
    def _pin_cases(fam):
        mem = [Member(m.role, m.lower, m.lower)                 # case (t, inf) -> pinned (t, t)
               if (np.isfinite(m.lower) and not np.isfinite(m.upper)) else m
               for m in fam.members]
        return Family(fam.fam_id, mem)
    pinned = [_pin_cases(fam) for fam in base]
    with pytest.raises(NotImplementedError, match="case/control"):
        test_variance_component(pinned, "C", n_boot=3, n_iter=50, burn_in=10)


def test_genetic_correlation_test_detects_rg():
    from ltpred import test_genetic_correlation
    rg = np.array([[1.0, 0.5], [0.5, 1.0]])
    rp = np.array([[1.0, 0.2], [0.2, 1.0]])
    fams = _simulate_two_trait(["m", "f", "s1", "s2"], [0.5, 0.5], rg, rp,
                               n_fam=1500, prev=[0.1, 0.1], seed=3)
    r = test_genetic_correlation(fams, n_boot=25, seed=1, n_iter=350, burn_in=100)
    assert r.null.shape == (25,)
    assert 0.0 < r.p_value <= 1.0
    assert r.p_value < 0.2                          # real r_g -> significant
