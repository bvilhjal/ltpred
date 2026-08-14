"""Fitting liability-scale heritability from family data (data-augmentation Gibbs).

The multi-trait genetic-correlation, factor-model, MCEM variance-component, and
significance-test fits moved to ``research/advanced_fitting.py``; their tests live
in ``research/tests/test_advanced_fitting.py``.
"""

import numpy as np
import pytest

from ltpred import (simulate_under_LTM_single, fit_heritability,
                    fit_variance_components)
from ltpred.family import Family, Member


@pytest.mark.parametrize("h2_true", [0.3, 0.6])
def test_recovers_simulated_heritability(h2_true):
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=h2_true,
                                    n_sim=2500, pop_prev=0.1, seed=3)
    res = fit_heritability(
        sim.families, n_iter=500, burn_in=150, inner_sweeps=5,
        damp=0.2, seed=1, sampling="population")
    assert res.h2 == pytest.approx(h2_true, abs=0.07)
    assert 0.0 < res.h2 < 1.0
    assert res.h2_se > 0
    assert res.samples.shape == (350,)
    assert res.trace.shape == (500,)


def test_converges_from_different_inits():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5, n_sim=3000,
                                    pop_prev=0.1, seed=4)
    lo = fit_heritability(
        sim.families, h2_init=0.15, n_iter=500, burn_in=150, seed=1,
        sampling="population").h2
    hi = fit_heritability(
        sim.families, h2_init=0.85, n_iter=500, burn_in=150, seed=1,
        sampling="population").h2
    assert abs(lo - hi) < 0.05          # the fit forgets its starting point


def test_fit_preparers_align_bounds_by_role():
    import importlib
    fit_mod = importlib.import_module("ltpred.fit")

    scalar = [
        Family(0, [Member("m", -1.0, 0.0), Member("f", 0.0, 1.0)]),
        Family(1, [Member("f", 0.2, 1.2), Member("m", -0.8, 0.2)]),
    ]
    expected_scalar = np.array([[0.0, -1.0], [0.2, -0.8]])
    for group in (fit_mod._prepare_group(scalar, [0, 1]),
                  fit_mod._prepare_group_vc(scalar, [0, 1], ["A"])):
        assert group["roles"] == ["f", "m"]
        assert np.allclose(group["lowers"], expected_scalar)


def test_fit_preparers_reject_nan_and_reversed_bounds():
    import importlib
    fit_mod = importlib.import_module("ltpred.fit")

    scalar = [Family(0, [Member("m", np.nan, np.inf),
                         Member("f", -np.inf, np.inf)])]
    for prepare in (lambda: fit_mod._prepare_group(scalar, [0]),
                    lambda: fit_mod._prepare_group_vc(scalar, [0], ["A"])):
        with pytest.raises(ValueError, match="NaN"):
            prepare()


def test_lone_probands_raise():
    # no relatives -> no related pairs -> h2 not identified
    t = 1.64
    fams = [Family(i, [Member("o", -np.inf, t)]) for i in range(20)]
    with pytest.raises(ValueError, match="no related pairs"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="population")


def test_variance_components_additive_matches_heritability():
    # single-component fit == fit_heritability (same data-augmentation)
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=1500, pop_prev=0.1, seed=2)
    vc = fit_variance_components(
        sim.families, ("A",), n_iter=500, burn_in=150, seed=1,
        sampling="population")
    he = fit_heritability(
        sim.families, n_iter=500, burn_in=150, seed=1,
        sampling="population").h2
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
    r = fit_variance_components(
        fams, ("A", "C"), n_iter=800, burn_in=250, seed=1,
        sampling="population")
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
        fit_variance_components(
            par.families, ("A", "C"), n_iter=100, burn_in=30,
            sampling="population")


def test_variance_components_is_he_only():
    # the MCEM/likelihood route moved to
    # research.advanced_fitting.fit_variance_components_mcem; fit_variance_components
    # is the Haseman-Elston moment fit only (no `method` switch), and its result
    # carries no likelihood fields.
    import dataclasses
    from ltpred.fit import VarCompResult
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=200, pop_prev=0.1, seed=8)
    with pytest.raises(TypeError):
        fit_variance_components(sim.families, ("A",), method="mcem",
                                n_iter=50, burn_in=10)
    assert {f.name for f in dataclasses.fields(VarCompResult)} == {
        "components", "residual", "se", "traces", "n_iter", "burn_in"}


def test_bootstrap_fit_scalar_and_calibration():
    from ltpred import bootstrap_fit
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=1500, pop_prev=0.1, seed=5)
    full = fit_heritability(
        sim.families, n_iter=400, burn_in=120, seed=1,
        sampling="population")
    bs = bootstrap_fit(sim.families,
                       lambda f: fit_heritability(
                           f, n_iter=400, burn_in=120, seed=1,
                           sampling="population").h2,
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
            f, ("A", "C"), n_iter=300, burn_in=100, seed=1,
            sampling="population").components.values())),
        n_boot=12, seed=1)
    assert bs.estimate.shape == (2,)
    assert bs.se.shape == (2,) and np.all(bs.se > 0)
    assert bs.ci_low.shape == (2,) and bs.ci_high.shape == (2,)
    with pytest.raises(ValueError, match="at least 2 families"):
        bootstrap_fit(sim.families[:1], lambda f: 0.0, n_boot=3)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_fit(sim.families, lambda f: 0.0, n_boot=3, ci_level=1.5)


@pytest.mark.parametrize("n_boot", [0, 1])
def test_bootstrap_fit_requires_at_least_two_resamples(n_boot):
    from ltpred import bootstrap_fit

    families = [Family(0), Family(1)]
    with pytest.raises(ValueError, match="n_boot must be >= 2"):
        bootstrap_fit(families, lambda f: 0.0, n_boot=n_boot)


def test_bootstrap_fit_rejects_changing_statistic_shape():
    from ltpred import bootstrap_fit

    families = [Family(0), Family(1)]
    calls = 0

    def changing_shape(_families):
        nonlocal calls
        calls += 1
        return np.zeros(2 if calls == 1 else 3)

    with pytest.raises(ValueError, match="expected stable shape"):
        bootstrap_fit(families, changing_shape, n_boot=2, seed=0)


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


def _sim_vc(fam, props, n, seed, prev=0.1):
    """Families under liab = sum_c sqrt(props[c]) * N(0, K_c) + e, thresholded at
    prevalence ``prev``. ``props`` maps a component ('A', 'C', 'M') to its variance
    proportion; the residual ``e2 = 1 - sum(props)``."""
    import numpy as np
    from ltpred.covariance import correct_positive_definite
    from ltpred.thresholds import liability_threshold
    from ltpred.fit import _component_matrix
    tot = sum(props.values())
    Sig = (1.0 - tot) * np.eye(len(fam))
    for c, v in props.items():
        if v > 0:
            Sig = Sig + v * _component_matrix(fam, c)
    Sig, _ = correct_positive_definite(Sig)
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal(np.zeros(len(fam)), Sig, size=n)
    t = float(liability_threshold(prev))
    return [Family(i, [Member(r, (t if L[i, c] > t else -np.inf),
                              (np.inf if L[i, c] > t else t)) for c, r in enumerate(fam)])
            for i in range(n)]


def test_is_mates_predicate():
    from ltpred.fit import _is_mates
    assert _is_mates("m", "f") and _is_mates("f", "m")
    assert _is_mates("mgm", "mgf") and _is_mates("pgm", "pgf")
    # not mates: a parent and a grandparent, two sibs, an unrelated cross-couple
    assert not _is_mates("m", "mgm")
    assert not _is_mates("s1", "s2")
    assert not _is_mates("m", "pgf")
    assert not _is_mates("o", "m")


def test_component_matrix_rejects_non_psd(monkeypatch):
    # a bogus "vertical" environment shared by parent-offspring pairs chains across
    # generations (o-m and m-mgm share, but o-mgm do not) -> non-PSD -> rejected.
    from ltpred.fit import _component_matrix, _COMPONENT_OFFDIAG

    def vertical(a, b):
        chain = {frozenset({"o", "m"}), frozenset({"m", "mgm"})}
        return 1.0 if frozenset({a, b}) in chain else 0.0

    monkeypatch.setitem(_COMPONENT_OFFDIAG, "V", vertical)
    with pytest.raises(ValueError, match="positive-semidefinite"):
        _component_matrix(["o", "m", "mgm"], "V")
    # the two supported environment components are valid (PSD) partitions
    for comp in ("C", "M"):
        K = _component_matrix(["o", "s1", "m", "f", "mgm", "mgf"], comp)
        assert np.min(np.linalg.eigvalsh(K)) > -1e-8


def test_variance_component_preparer_keeps_exact_singular_kernels():
    import importlib

    fit_mod = importlib.import_module("ltpred.fit")
    families = _sim_vc(["o", "s1", "m", "f"], {"A": 0.4, "C": 0.2},
                       10, seed=21)
    group = fit_mod._prepare_group_vc(families, list(range(10)), ["A", "C"])
    roles = group["roles"]
    oi, si = roles.index("o"), roles.index("s1")
    assert group["K"]["C"][oi, si] == 1.0
    assert np.min(np.linalg.eigvalsh(group["K"]["C"])) > -1e-8
    mate_families = _sim_vc(["m", "f"], {"A": 0.4, "M": 0.2},
                            10, seed=23)
    mate_group = fit_mod._prepare_group_vc(
        mate_families, list(range(10)), ["A", "M"])
    np.testing.assert_array_equal(
        mate_group["K"]["M"], np.ones((2, 2)))


def test_scalar_preparer_keeps_exact_additive_kernel():
    import importlib

    fit_mod = importlib.import_module("ltpred.fit")
    families = _sim_vc(["o", "s1", "m", "f"], {"A": 0.4}, 10, seed=22)
    group = fit_mod._prepare_group(families, list(range(10)))
    roles = group["roles"]
    expected = np.array([
        [fit_mod.get_relatedness(a, b, h2=1.0) for b in roles]
        for a in roles
    ])
    np.testing.assert_array_equal(group["A"], expected)


def test_c_component_groups_same_side_avuncular():
    # a parent and their full sibs (aunts/uncles) form one sibship for C: the block
    # {m, mau1, mau2} must be a complete PSD block, not a non-PSD chain.
    from ltpred.fit import _is_full_sib, _component_matrix
    assert _is_full_sib("mau1", "mau2") and _is_full_sib("pau1", "pau2")
    assert _is_full_sib("m", "mau1") and _is_full_sib("f", "pau2")
    assert not _is_full_sib("mau1", "pau1")          # opposite sides: unrelated
    K = _component_matrix(["m", "mau1", "mau2"], "C")
    assert np.allclose(K, np.ones((3, 3)))           # full sibship block, not a chain
    assert np.min(np.linalg.eigvalsh(K)) > -1e-8
    # so fitting A+C on such a structure no longer trips the PSD guard
    fams = _sim_vc(["o", "m", "mau1", "mau2"], {"A": 0.4, "C": 0.2}, 300, seed=21)
    r = fit_variance_components(
        fams, ("A", "C"), n_iter=80, burn_in=25, seed=1,
        sampling="population")
    assert set(r.components) == {"A", "C"}


def test_variance_components_recovers_couple_env():
    # additive + couple (spousal) environment: M loads on the genetically-unrelated
    # mate pairs (m,f), (mgm,mgf), (pgm,pgf); A on the related pairs. The HE
    # regression separates them because mates have A=0.
    fam = ["o", "m", "f", "mgm", "mgf", "pgm", "pgf"]
    fams = _sim_vc(fam, {"A": 0.4, "M": 0.2}, 3500, seed=11)
    r = fit_variance_components(
        fams, ("A", "M"), n_iter=800, burn_in=250, seed=1,
        sampling="population")
    assert 0.28 < r.components["A"] < 0.52          # true 0.40
    assert 0.10 < r.components["M"] < 0.32          # true 0.20 (not collapsed / absorbed)
    assert r.residual == pytest.approx(1.0 - 0.4 - 0.2, abs=0.12)


def test_variance_components_no_spurious_couple_env():
    # purely additive data -> M should stay near 0 (no spurious couple environment)
    fam = ["o", "m", "f", "mgm", "mgf", "pgm", "pgf"]
    fams = _sim_vc(fam, {"A": 0.5}, 3500, seed=12)
    r = fit_variance_components(
        fams, ("A", "M"), n_iter=700, burn_in=200, seed=1,
        sampling="population")
    assert r.components["M"] < 0.10                 # true 0.0
    assert 0.38 < r.components["A"] < 0.62          # true 0.50, unbiased by fitting M


def test_variance_components_couple_env_needs_mate_pairs():
    # M is identified only from mate pairs; a sib-only structure has none -> singular
    fams = _sim_vc(["o", "s1", "s2"], {"A": 0.5}, 400, seed=13)
    with pytest.raises(ValueError, match="not identified"):
        fit_variance_components(
            fams, ("A", "M"), n_iter=100, burn_in=30,
            sampling="population")


def test_variance_components_joint_A_C_M_identified():
    # a 3-generation pedigree identifies all three at once: A from the related pairs,
    # C from the full-sib excess (o,s1,s2), M from the mate pairs (m,f)/(mgm,mgf).
    fam = ["o", "s1", "s2", "m", "f", "mgm", "mgf"]
    fams = _sim_vc(fam, {"A": 0.35, "C": 0.2, "M": 0.15}, 4000, seed=14)
    r = fit_variance_components(
        fams, ("A", "C", "M"), n_iter=900, burn_in=300, seed=1,
        sampling="population")
    for comp in ("A", "C", "M"):
        assert r.components[comp] > 0.03            # none collapsed to the boundary
    assert 0.20 < r.components["A"] < 0.55          # true 0.35
    assert r.components["C"] < 0.40 and r.components["M"] < 0.40
    assert r.residual == pytest.approx(1.0 - 0.7, abs=0.15)


def test_init_chain_holds_extreme_pin_at_its_value():
    # regression: a pinned coordinate at |z| >~ 8.2 saturated the CDF-average
    # init and fell back to 0.0 -- and fixed coordinates are never resampled,
    # so the family was conditioned on 0.0 forever. The fit paths share the
    # sampler's _init_chain with a unit marginal SD (pre-standardised bounds).
    from ltpred.gibbs import _init_chain
    lowers = np.array([9.0, -np.inf])
    uppers = np.array([9.0, 0.0])
    x = _init_chain(lowers, uppers, np.ones(2))
    assert x[0] == 9.0                     # the pin, not the old 0.0 fallback
    assert np.isfinite(x).all()
    # a free coordinate far into the tail starts inside its interval
    x = _init_chain(np.array([9.0]), np.array([np.inf]), np.ones(1))
    assert 9.0 <= x[0] < np.inf


def test_prepare_group_starts_pinned_member_at_pin():
    import importlib
    fit_mod = importlib.import_module("ltpred.fit")
    fams = [Family(0, [Member("m", 9.0, 9.0), Member("f", -np.inf, 0.0)])]
    group = fit_mod._prepare_group(fams, [0])
    assert group["x"][0, group["roles"].index("m")] == 9.0


def test_fit_heritability_validates_burn_in_and_h2_init():
    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=30,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="burn_in"):
        fit_heritability(sim.families, n_iter=10, burn_in=10)
    with pytest.raises(ValueError, match="h2_init"):
        fit_heritability(sim.families, h2_init=1.5, n_iter=10, burn_in=6)
    with pytest.raises(ValueError, match="h2_init"):
        fit_heritability(sim.families, h2_init=-0.1, n_iter=10, burn_in=6)
    # valid inputs still run (4 post-burn-in samples is the batch_means minimum)
    res = fit_heritability(
        sim.families, h2_init=0.5, n_iter=10, burn_in=6,
        inner_sweeps=1, seed=1, sampling="population")
    assert np.isfinite(res.h2)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("damp", 0.0, "damp"),
        ("damp", 1.1, "damp"),
        ("damp", np.nan, "damp"),
        ("eps", 0.0, "eps"),
        ("eps", 1e-13, "eps"),
        ("eps", 0.5, "eps"),
        ("eps", np.inf, "eps"),
    ],
)
def test_moment_fitters_validate_update_controls(name, value, match):
    sim = simulate_under_LTM_single(
        fam_vec=["m", "s1"], h2=0.5, n_sim=10, pop_prev=0.1, seed=1)
    kwargs = {name: value, "n_iter": 10, "burn_in": 6}
    with pytest.raises(ValueError, match=match):
        fit_heritability(sim.families, **kwargs)
    with pytest.raises(ValueError, match=match):
        fit_variance_components(sim.families, ("A",), **kwargs)


def test_moment_fitters_make_population_sampling_contract_explicit():
    import warnings

    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=30,
                                    pop_prev=0.1, seed=1)
    with pytest.warns(RuntimeWarning, match="unascertained"):
        fit_heritability(sim.families, n_iter=10, burn_in=6,
                         inner_sweeps=1, seed=1)
    with pytest.raises(ValueError, match=r"sampling='population' or sampling='ipw'"):
        fit_heritability(sim.families, sampling="case-control",
                         n_iter=10, burn_in=6)
    with pytest.raises(ValueError, match=r"sampling='population' or sampling='ipw'"):
        fit_variance_components(sim.families, ("A",),
                                sampling="family-history",
                                n_iter=10, burn_in=6)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fit_heritability(sim.families, sampling="population",
                         n_iter=10, burn_in=6, inner_sweeps=1, seed=1)
    assert not recorded


def _coherent_varying_bounds(n_fam=30, seed=1, *, mixed_geometry=False):
    """Simulated liabilities observed through exogenous person-specific bounds."""
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    pop_prev=0.1, n_sim=n_fam, seed=seed)
    roles = [role for role in sim.roles if role != "g"]
    families = []
    for i in range(n_fam):
        members = []
        for j, role in enumerate(roles):
            value = float(sim.liabilities[i, sim.roles.index(role)])
            geometry = (i + 3 * j) % 17
            if mixed_geometry and geometry == 0:
                lower = upper = value
            elif mixed_geometry and geometry == 1:
                lower, upper = value - 0.05, value + 0.05
            else:
                # Deterministic in family/member identity, not in the latent value.
                threshold = -0.4 + 0.15 * ((i + 2 * j) % 9)
                lower, upper = ((threshold, np.inf) if value > threshold
                                else (-np.inf, threshold))
            members.append(Member(role, lower, upper))
        families.append(Family(i, members))
    return families


def test_moment_fitters_reject_coherent_person_specific_rectangles():
    # Even perfectly coherent personalised/pinned LT-FH++ bounds bias the pooled
    # HE fixed point to h2~1 (and invent a spurious C), so they are refused
    # rather than silently fitted -- the geometry alone triggers the guard.
    fams = _coherent_varying_bounds(mixed_geometry=True)
    assert any(m.lower == m.upper for f in fams for m in f.members)
    assert any(np.isfinite(m.lower) and np.isfinite(m.upper) and m.lower < m.upper
               for f in fams for m in f.members)

    with pytest.raises(ValueError, match="single case/control threshold"):
        fit_heritability(fams, n_iter=20, burn_in=8, inner_sweeps=1, seed=1,
                         sampling="population")
    with pytest.raises(ValueError, match="single case/control threshold"):
        fit_variance_components(fams, ("A", "C"), n_iter=20, burn_in=8,
                                inner_sweeps=1, seed=1, sampling="population")


def test_moment_fitters_recover_h2_on_coherent_common_threshold_data():
    # the supported path: one case/control threshold per trait -> ~0.5, not ~1.0
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    pop_prev=0.1, n_sim=1500, seed=1)
    res = fit_heritability(sim.families, seed=1, sampling="population")
    assert 0.30 < res.h2 < 0.70


def test_general_rectangles_still_receive_standard_bounds_validation():
    fams = _coherent_varying_bounds()
    fams[0].members[0].lower = 1.0
    fams[0].members[0].upper = 0.0
    with pytest.raises(ValueError, match="reversed bounds"):
        fit_heritability(
            fams, n_iter=20, burn_in=8, seed=1, sampling="population")


def test_common_threshold_accepts_mixed_float32_float64_endpoints():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    pop_prev=0.1, n_sim=30, seed=1)
    threshold = next(float(endpoint)
                     for member in sim.families[0].members
                     for endpoint in (member.lower, member.upper)
                     if np.isfinite(endpoint))
    assert float(np.float32(threshold)) != float(np.float64(threshold))
    for i, family in enumerate(sim.families):
        endpoint = np.float32(threshold) if i % 2 else np.float64(threshold)
        for member in family.members:
            if np.isfinite(member.lower):
                member.lower = endpoint
            if np.isfinite(member.upper):
                member.upper = endpoint

    result = fit_heritability(
        sim.families, n_iter=20, burn_in=8, inner_sweeps=1, seed=1,
        sampling="population")
    assert np.isfinite(result.h2)


def test_moment_fitters_reject_negative_burn_in():
    # the burn_in >= n_iter guard used to let a negative burn_in through
    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=10,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        fit_heritability(sim.families, n_iter=10, burn_in=-1)
    with pytest.raises(ValueError, match="non-negative"):
        fit_variance_components(sim.families, ("A",), n_iter=10, burn_in=-1)


def _cc_family(fid, roles, statuses, threshold):
    """One family with common-threshold case/control bounds."""
    return Family(fid, [Member(r, threshold if st else -np.inf,
                               np.inf if st else threshold)
                        for r, st in zip(roles, statuses)])


def test_population_case_rate_guard_catches_proband_ascertainment():
    # `sampling="population"` used to be an honour system: it checked a string,
    # not the data, so a case/control cohort fitted straight through to a fixed
    # point at the clamp. The thresholds themselves assert the prevalence, so
    # the claim is checkable -- and must be checked, because the failure is
    # severe and silent (on ascertained data with true h2=0 the fitter returns
    # h2=1.0).
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(0)

    # every proband affected; relatives at the population rate
    fams = [_cc_family(i, roles, [True] + list(rng.random(3) < 0.05), t)
            for i in range(800)]
    with pytest.raises(ValueError, match="not consistent with sampling='population'"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="population")
    with pytest.raises(ValueError, match=r"Role 'o'"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="population")


def test_population_case_rate_guard_flags_relatives_not_just_the_proband():
    # Selection on family history leaves the proband at the population rate and
    # enriches the RELATIVES, so a guard that only looked at `o` would miss it.
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(1)
    fams = [_cc_family(i, roles,
                       [rng.random() < 0.05, True] + list(rng.random(2) < 0.05), t)
            for i in range(800)]
    with pytest.raises(ValueError, match=r"Role 'm'"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="population")


def test_population_case_rate_guard_passes_genuine_population_samples():
    # Specificity is the property that matters most: a false positive refuses a
    # legitimate analysis. Nothing here may raise.
    for n_fam, prev, seed in [(400, 0.05, 3), (1200, 0.10, 4), (1200, 0.20, 5)]:
        sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                        n_sim=n_fam, pop_prev=prev, seed=seed)
        res = fit_heritability(sim.families, n_iter=120, burn_in=40, seed=1,
                               sampling="population")
        assert 0.0 < res.h2 < 1.0


def test_population_case_rate_guard_survives_bootstrap_resampling():
    # bootstrap_fit re-runs the estimator on resamples centred on the COHORT's
    # rate rather than on K, so their z carries the cohort's own sampling error
    # as an offset. At a z bar of 4 this fired on a legitimate 1500-family
    # cohort; the bar is 6 so that the bootstrap stays usable.
    from ltpred import bootstrap_fit
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=1500, pop_prev=0.1, seed=5)
    bs = bootstrap_fit(sim.families,
                       lambda f: fit_heritability(f, n_iter=120, burn_in=40,
                                                  seed=1,
                                                  sampling="population").h2,
                       n_boot=12, seed=0)
    assert bs.samples.shape == (12,)


def test_population_case_rate_guard_covers_variance_components():
    from ltpred import fit_variance_components
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.05))
    roles = ["o", "m", "f", "s1", "s2", "s3"]
    rng = np.random.default_rng(2)
    fams = [_cc_family(i, roles, [True] + list(rng.random(5) < 0.05), t)
            for i in range(800)]
    with pytest.raises(ValueError, match="not consistent with sampling='population'"):
        fit_variance_components(fams, ("A", "C"), n_iter=50, burn_in=10,
                                sampling="population")


def test_ipw_recovers_h2_under_case_control_ascertainment():
    # The augmentation for a GIVEN family with GIVEN statuses is already the
    # right conditional distribution; selection breaks the *mix* of families,
    # which is exactly what inverse-probability weighting repairs.
    import ltpred.fit as fit_mod
    from ltpred.thresholds import liability_threshold
    K, q, n = 0.05, 0.5, 3000
    t = float(liability_threshold(K))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(4)
    from ltpred.covariance import correct_positive_definite
    from ltpred.fit import _component_matrix
    Sig, _ = correct_positive_definite(
        0.5 * _component_matrix(roles, "A") + 0.5 * np.eye(4))

    keep_p = K * (1 - q) / (q * (1 - K))
    fams, w = [], []
    while len(fams) < n:
        liab = rng.multivariate_normal(np.zeros(4), Sig, size=40_000)
        st = liab > t
        keep = st[:, 0] | (rng.random(st.shape[0]) < keep_p)
        for row in st[keep]:
            if len(fams) >= n:
                break
            fams.append(_cc_family(len(fams), roles, row, t))
            w.append(1.0 if row[0] else 1.0 / keep_p)
    w = np.asarray(w)

    # unguarded, this cohort pins at the clamp -- that is the thing being fixed
    original = fit_mod._assert_population_case_rate
    fit_mod._assert_population_case_rate = lambda *a, **k: None
    try:
        naive = fit_heritability(fams, n_iter=600, burn_in=200, seed=2,
                                 sampling="population").h2
    finally:
        fit_mod._assert_population_case_rate = original
    assert naive > 0.9

    ipw = fit_heritability(fams, n_iter=600, burn_in=200, seed=2,
                           sampling="ipw", weights=w).h2
    assert 0.35 < ipw < 0.65, ipw            # truth 0.5
    assert ipw < naive - 0.3


def test_ipw_rejects_positivity_failure_even_with_weights():
    # Ascertainment through an affected proband gives an entire stratum
    # inclusion probability zero. Correct weights must reproduce the asserted
    # prevalence, so the weighted case-rate check is also the positivity check.
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.05))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(6)
    fams = [_cc_family(i, roles, [True] + list(rng.random(3) < 0.05), t)
            for i in range(800)]
    with pytest.raises(ValueError, match="positivity failure"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="ipw",
                         weights=np.ones(len(fams)))


def test_ipw_sampling_and_weights_must_agree():
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.1))
    roles = ["o", "m", "f", "s1"]
    rng = np.random.default_rng(7)
    fams = [_cc_family(i, roles, rng.random(4) < 0.1, t) for i in range(200)]
    w = np.ones(len(fams))
    with pytest.raises(ValueError, match="requires weights"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="ipw")
    with pytest.raises(ValueError, match="weights are not accepted"):
        fit_heritability(fams, n_iter=50, burn_in=10, sampling="population",
                         weights=w)
    with pytest.raises(ValueError, match="only meaningful with sampling='ipw'"):
        fit_heritability(fams, n_iter=50, burn_in=10, weights=w)
    for bad, msg in [(np.zeros(len(fams)), "strictly positive"),
                     (np.ones(5), "one entry per family"),
                     (np.full(len(fams), np.nan), "finite")]:
        with pytest.raises(ValueError, match=msg):
            fit_heritability(fams, n_iter=50, burn_in=10, sampling="ipw",
                             weights=bad)


def test_ipw_weights_pass_through_variance_components():
    from ltpred import fit_variance_components
    from ltpred.thresholds import liability_threshold
    t = float(liability_threshold(0.1))
    roles = ["o", "m", "f", "s1", "s2", "s3"]
    rng = np.random.default_rng(8)
    fams = [_cc_family(i, roles, rng.random(6) < 0.1, t) for i in range(600)]
    res = fit_variance_components(fams, ("A", "C"), n_iter=120, burn_in=40,
                                  seed=1, sampling="ipw",
                                  weights=np.ones(len(fams)))
    assert 0.0 <= res.components["A"] <= 1.0


def test_unknown_sampling_mode_names_the_supported_ones():
    # The contract grew a second mode; an unknown value must still be refused,
    # and the message should say what IS accepted rather than only what is not.
    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=30,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match=r"sampling='population' or sampling='ipw'"):
        fit_heritability(sim.families, sampling="case-control",
                         n_iter=10, burn_in=6)
    # ...and it should point at why proband-ascertained designs are not simply
    # another mode: they cannot be reweighted at all.
    with pytest.raises(ValueError, match="cannot be reweighted at all"):
        fit_heritability(sim.families, sampling="proband", n_iter=10, burn_in=6)
