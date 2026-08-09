"""Tests for the moved inferential fitting machinery (``research.advanced_fitting``).

These cover the multi-trait genetic-correlation fit, the common-factor model, the
MCEM variance-component route, and the parametric-bootstrap significance tests —
split out of ``tests/test_fit.py`` when the estimators moved to
``research/advanced_fitting.py``.
"""

import numpy as np
import pytest

from ltpred import simulate_under_LTM_single
from ltpred.family import Family, Member
from ltpred.fit import fit_variance_components

from research.advanced_fitting import (
    GenCorrResult, fit_genetic_correlation, fit_genetic_factor,
    fit_variance_components_mcem)
# NOTE: test_variance_component / test_genetic_correlation are imported
# function-locally below -- a top-level import would let pytest collect them as
# test functions (their names start with "test_").


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


def test_prepare_group_multi_aligns_bounds_by_role():
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    multi = [
        Family(0, [Member("m", [-1.0, -0.5], [0.0, 0.5]),
                   Member("f", [0.0, 0.5], [1.0, 1.5])]),
        Family(1, [Member("f", [0.2, 0.6], [1.2, 1.6]),
                   Member("m", [-0.8, -0.4], [0.2, 0.6])]),
    ]
    group = af_mod._prepare_group_multi(multi, [0, 1], 2)
    assert group["roles"] == ["f", "m"]
    assert np.allclose(group["lowers"],
                       [[0.0, -1.0, 0.5, -0.5],
                        [0.2, -0.8, 0.6, -0.4]])


def test_prepare_group_multi_rejects_reversed_bounds():
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    multi = [Family(0, [Member("m", [0.0, 2.0], [1.0, 1.0]),
                        Member("f", [-np.inf, -np.inf], [np.inf, np.inf])])]
    with pytest.raises(ValueError, match="reversed bounds"):
        af_mod._prepare_group_multi(multi, [0], 2)


def test_genetic_correlation_recovers_rg():
    # two correlated traits: cross-trait HE recovers r_g and per-trait h2
    rg = np.array([[1.0, 0.5], [0.5, 1.0]])
    rp = np.array([[1.0, 0.2], [0.2, 1.0]])
    fams = _simulate_two_trait(["m", "f", "s1", "s2"], [0.5, 0.4], rg, rp,
                               n_fam=2500, prev=[0.1, 0.1], seed=101)
    r = fit_genetic_correlation(fams, n_iter=800, burn_in=250, seed=1,
                                phen_names=["A", "B"], sampling="population")
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
    r = fit_genetic_correlation(fams, n_iter=800, burn_in=250, seed=1,
                                sampling="population")
    assert abs(r.rg[0, 1]) < 0.30                            # no spurious genetic corr
    assert r.rp[0, 1] > 0.10                                 # phenotypic corr still seen
    # the phenotypic correlation shows up as an *environmental* one instead
    # (true r_e = 0.3 / (1-0.5) = 0.6)
    assert r.re[0, 1] > 0.30


def test_genetic_correlation_uses_and_reports_one_coherent_psd_model(monkeypatch):
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    # The fixed augmented liabilities imply an impossible pairwise genetic
    # correlation matrix (one negative eigenvalue), reproducing the old failure.
    n_pheno = n_fam = 3
    raw_rg = np.array([[1.0, 0.9, 0.9],
                       [0.9, 1.0, -0.9],
                       [0.9, -0.9, 1.0]])
    U = np.eye(n_pheno)
    V = (n_fam / 2.0) * 0.4 * raw_rg
    fixed_x = np.empty((n_fam, 2 * n_pheno))
    for p in range(n_pheno):
        fixed_x[:, 2 * p] = U[:, p]
        fixed_x[:, 2 * p + 1] = V[:, p]

    fams = [Family(f, [Member("o", [-np.inf] * n_pheno, [np.inf] * n_pheno),
                             Member("m", [-np.inf] * n_pheno, [np.inf] * n_pheno)])
            for f in range(n_fam)]
    sampled = []

    def fake_params(sigma):
        sampled.append(np.array(sigma, copy=True))
        return None, None

    def fake_advance(_P, _sd, _lo, _hi, _fixed, state, _sweeps):
        state[:] = fixed_x

    monkeypatch.setattr(af_mod, "gibbs_params", fake_params)
    monkeypatch.setattr(af_mod, "gibbs_advance", fake_advance)
    result = af_mod.fit_genetic_correlation(
        fams, n_iter=5, burn_in=1, inner_sweeps=1, damp=1.0,
        sampling="population")

    for matrix in (result.genetic_cov, result.env_cov, result.rg,
                   result.re, result.rp):
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    assert np.allclose(result.rp, result.genetic_cov + result.env_cov)
    assert np.allclose(np.diag(result.genetic_cov), result.h2)
    assert np.allclose(np.diag(result.env_cov), 1.0 - result.h2)

    group = af_mod._prepare_group_multi(fams, list(range(n_fam)), n_pheno)
    reported_model = af_mod._multi_cov(
        group["A"], result.h2, result.genetic_cov, result.rp)
    assert np.allclose(sampled[-1], reported_model)


def test_genetic_correlation_validates_input():
    # single-trait families (scalar bounds) -> needs >= 2 traits
    single = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                       n_sim=200, pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="2 traits"):
        fit_genetic_correlation(single.families, n_iter=50, burn_in=10)
    # lone probands (no related pairs)
    lone = [Family(i, [Member("o", [-np.inf, -np.inf], [1.0, 1.0])]) for i in range(20)]
    with pytest.raises(ValueError, match="related pairs"):
        fit_genetic_correlation(lone, n_iter=50, burn_in=10,
                                sampling="population")
    with pytest.raises(ValueError, match="burn_in"):
        fit_genetic_correlation(
            _simulate_two_trait(["m", "f", "s1"], [0.5, 0.5], np.eye(2), np.eye(2),
                                50, [0.1, 0.1], seed=1), n_iter=50)


def test_memberless_families_raise_clear_fit_errors():
    import importlib

    af_mod = importlib.import_module("research.advanced_fitting")
    families = [Family("observed", [
        Member("m", [-np.inf, -np.inf], [1.0, 1.0])]),
        Family("empty")]

    calls = (
        lambda: fit_genetic_correlation(families, n_iter=2, burn_in=1),
        lambda: af_mod.test_genetic_correlation(families, n_boot=1),
        lambda: af_mod.test_variance_component(families, "C", n_boot=1),
    )
    for call in calls:
        with pytest.raises(ValueError, match="family 'empty' has no members"):
            call()


@pytest.mark.parametrize("attribute", ["lower", "upper"])
@pytest.mark.parametrize("bad_bound", [0.0, [0.0], [0.0, 0.0, 0.0]])
def test_genetic_correlation_rejects_scalar_and_wrong_length_bounds(attribute,
                                                                    bad_bound):
    valid = Member("m", [-np.inf, -np.inf], [1.0, 1.0])
    bounds = {"lower": [-np.inf, -np.inf], "upper": [1.0, 1.0]}
    bounds[attribute] = bad_bound
    invalid = Member("f", bounds["lower"], bounds["upper"])
    families = [Family("f1", [valid, invalid])]

    with pytest.raises(ValueError, match=r"must be a length-2 .*sequence"):
        fit_genetic_correlation(families, n_iter=20, burn_in=5)


def test_genetic_factor_recovers_single_factor():
    # a planted one-factor genetic correlation r_g = lam lam' (diag 1): MINRES
    # recovers the loadings exactly (rank-1 off-diagonal is fit perfectly)
    lam = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
    R = np.outer(lam, lam)
    np.fill_diagonal(R, 1.0)
    r = fit_genetic_factor(R, 1, phen_names=list("ABCDE"))
    assert r.loadings.shape == (5, 1)
    assert np.allclose(r.loadings.ravel(), lam, atol=1e-3)   # exact up to sign (fixed +)
    assert np.allclose(r.communality, lam ** 2, atol=1e-3)   # variance explained
    assert np.allclose(r.uniqueness, 1.0 - lam ** 2, atol=1e-3)
    assert r.srmr < 1e-3 and r.prop_explained > 0.999        # one factor fits
    assert r.df == ((5 - 1) ** 2 - (5 + 1)) // 2 == 5
    assert r.input_correlation and r.phen_names == list("ABCDE")


def test_genetic_factor_detects_two_factors():
    # two independent blocks (a two-factor truth): one factor fits poorly, two well
    l1, l2 = np.array([0.8, 0.7, 0.6]), np.array([0.75, 0.65, 0.55])
    R = np.zeros((6, 6))
    R[:3, :3] = np.outer(l1, l1)
    R[3:, 3:] = np.outer(l2, l2)
    np.fill_diagonal(R, 1.0)
    one = fit_genetic_factor(R, 1)
    two = fit_genetic_factor(R, 2)
    assert one.srmr > 0.1 and one.prop_explained < 0.7        # one factor mis-specified
    assert two.srmr < 1e-3 and two.prop_explained > 0.999     # two factors fit
    assert two.loadings.shape == (6, 2)
    # each factor loads only its own block (cross-block loadings ~0)
    assert np.max(np.abs(two.loadings[3:, 0])) < 1e-2
    assert np.max(np.abs(two.loadings[:3, 1])) < 1e-2


def test_genetic_factor_standardizes_covariance():
    # a genetic *covariance* (diag = h2, not 1) is standardised to a correlation, so
    # the loadings come back on the correlation scale regardless
    lam = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
    h2 = np.array([0.5, 0.4, 0.3, 0.6, 0.45])
    G = np.outer(lam, lam) * np.outer(np.sqrt(h2), np.sqrt(h2))
    np.fill_diagonal(G, h2)
    r = fit_genetic_factor(G, 1)
    assert not r.input_correlation
    assert np.allclose(r.loadings.ravel(), lam, atol=1e-3)


def test_genetic_factor_no_structure_flags_zero_explained():
    # genetically uncorrelated traits: whatever the (degenerate) loadings, the honest
    # fit flags are srmr ~ 0 and prop_explained ~ 0 (no real structure to explain)
    r = fit_genetic_factor(np.eye(4), 1)
    assert r.srmr < 1e-6
    assert r.prop_explained == pytest.approx(0.0, abs=1e-6)


def _assert_coherent_factor_result(r):
    expected = r.loadings @ r.loadings.T + np.diag(r.uniqueness)
    assert np.allclose(r.fitted, expected, atol=1e-10)
    assert np.allclose(np.diag(r.fitted), 1.0, atol=1e-10)
    assert np.allclose(r.communality, np.sum(r.loadings ** 2, axis=1), atol=1e-10)
    assert np.allclose(r.uniqueness, 1.0 - r.communality, atol=1e-10)
    assert np.all(r.communality >= 0.0) and np.all(r.communality <= 1.0 + 1e-12)
    assert np.all(r.uniqueness >= -1e-12)


def test_genetic_factor_three_trait_incompatible_signs_leave_misfit():
    # This is a valid PSD correlation matrix, but one real factor cannot reproduce
    # the sign pattern: r12*r13*r23 < 0. Nominal df=0 does not make it exactly fit.
    R = np.array([[1.0, 0.5, 0.5],
                  [0.5, 1.0, -0.5],
                  [0.5, -0.5, 1.0]])
    assert np.min(np.linalg.eigvalsh(R)) > -1e-12
    r = fit_genetic_factor(R)
    _assert_coherent_factor_result(r)
    assert np.all(np.isfinite(r.loadings))
    assert r.df == 0
    assert r.srmr > 0.1


def test_genetic_factor_constrains_heywood_solution():
    # The unconstrained exact solution has lambda_1^2 = .5*.5/.1 = 2.5. The
    # admissible fit must put that communality on its boundary and retain misfit.
    R = np.array([[1.0, 0.5, 0.5],
                  [0.5, 1.0, 0.1],
                  [0.5, 0.1, 1.0]])
    assert np.min(np.linalg.eigvalsh(R)) > 0.0
    r = fit_genetic_factor(R)
    _assert_coherent_factor_result(r)
    assert np.max(r.communality) == pytest.approx(1.0, abs=1e-8)
    assert r.srmr > 0.01


def test_genetic_factor_from_gencorrresult():
    # accepts a GenCorrResult directly, pulling rg and phen_names off it
    lam = np.array([0.7, 0.6, 0.5, 0.4])
    R = np.outer(lam, lam)
    np.fill_diagonal(R, 1.0)
    gc = GenCorrResult(h2=np.full(4, 0.4), rg=R, re=np.eye(4), rp=np.eye(4),
                       genetic_cov=R * 0.4, env_cov=np.eye(4), se={},
                       phen_names=["w", "x", "y", "z"], traces={}, n_iter=1, burn_in=0)
    r = fit_genetic_factor(gc)
    assert r.phen_names == ["w", "x", "y", "z"]
    assert np.allclose(r.loadings.ravel(), lam, atol=1e-3)


def test_genetic_factor_validates_input():
    with pytest.raises(ValueError, match="3 traits"):            # P < 3
        fit_genetic_factor(np.eye(2), 1)
    with pytest.raises(ValueError, match="not identified"):      # df < 0
        fit_genetic_factor(np.eye(3), 2)
    with pytest.raises(ValueError, match="n_factors"):           # m < 1
        fit_genetic_factor(np.eye(4), 0)
    with pytest.raises(ValueError, match="square"):              # non-square
        fit_genetic_factor(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="phen_names"):          # name/size mismatch
        fit_genetic_factor(np.eye(4), 1, phen_names=["a", "b"])
    with pytest.raises(ValueError, match="weights"):             # wrong-shape weights
        fit_genetic_factor(np.eye(4), 1, weights=np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        bad = np.eye(4)
        bad[0, 1] = bad[1, 0] = np.nan
        fit_genetic_factor(bad)
    with pytest.raises(ValueError, match="symmetric"):
        bad = np.eye(4)
        bad[0, 1] = 0.2
        fit_genetic_factor(bad)
    with pytest.raises(ValueError, match="diagonal"):
        bad = np.eye(4)
        bad[0, 0] = 0.0
        fit_genetic_factor(bad)
    with pytest.raises(ValueError, match="positive-semidefinite"):
        bad = np.full((4, 4), 0.9)
        np.fill_diagonal(bad, 1.0)
        bad[0, 1] = bad[1, 0] = -0.9
        fit_genetic_factor(bad)
    with pytest.raises(ValueError, match="finite, symmetric, and non-negative"):
        bad_weights = np.ones((4, 4))
        bad_weights[0, 1] = -1.0
        fit_genetic_factor(np.eye(4), weights=bad_weights)


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


def test_component_test_accepts_M():
    # the significance test accepts 'M' (couple env) and rejects 'A' / 'D'
    from research.advanced_fitting import test_variance_component
    fam = ["o", "m", "f", "mgm", "mgf"]
    fams = _sim_vc(fam, {"A": 0.4, "M": 0.2}, 800, seed=15)
    r = test_variance_component(
        fams, "M", n_boot=10, seed=1, n_iter=250, burn_in=80,
        sampling="population")
    assert r.label == "M proportion > 0"
    assert r.null.shape == (10,)
    assert 0.0 < r.p_value <= 1.0
    for bad in ("A", "D"):
        with pytest.raises(ValueError, match="non-additive component"):
            test_variance_component(fams, bad, n_boot=3)


def test_component_test_detects_real_C():
    from research.advanced_fitting import test_variance_component
    fams = _sim_ac(["m", "f", "s1", "s2", "s3", "s4"], 0.4, 0.2, 1500, 1)
    r = test_variance_component(
        fams, "C", n_boot=25, seed=1, n_iter=350, burn_in=100,
        sampling="population")
    assert r.estimate > 0.10                       # a real C is estimated
    assert r.null.shape == (25,)
    assert 0.0 < r.p_value <= 1.0
    assert r.p_value < 0.2                          # strong signal -> significant
    assert r.null.mean() < r.estimate              # null centred below the observed


def test_component_test_validates_and_rejects_pinned():
    from research.advanced_fitting import test_variance_component
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


def test_significance_tests_reject_unverified_individualized_thresholds():
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    scalar = [Family(0, [Member("o", 0.5, np.inf)]),
              Family(1, [Member("o", -np.inf, 1.0)])]
    with pytest.raises(ValueError, match="fixed independently of observed status"):
        af_mod.test_variance_component(scalar, "C", n_boot=1)
    # The explicit assertion permits bounds made from external baseline covariates.
    af_mod._assert_case_control_bounds(
        scalar, thresholds_are_status_independent=True)

    multi = [Family(0, [Member("o", [0.5, -np.inf], [np.inf, 1.5])]),
             Family(1, [Member("o", [-np.inf, 0.8], [1.0, np.inf])])]
    with pytest.raises(ValueError, match="never age at onset"):
        af_mod.test_genetic_correlation(multi, n_boot=1)
    af_mod._assert_case_control_bounds(
        multi, thresholds_are_status_independent=True)

    malformed = [Family(0, [Member("o", np.inf, np.inf)])]
    with pytest.raises(NotImplementedError, match="case/control"):
        af_mod._assert_case_control_bounds(malformed)


def test_genetic_correlation_test_detects_rg():
    from research.advanced_fitting import test_genetic_correlation
    rg = np.array([[1.0, 0.5], [0.5, 1.0]])
    rp = np.array([[1.0, 0.2], [0.2, 1.0]])
    fams = _simulate_two_trait(["m", "f", "s1", "s2"], [0.5, 0.5], rg, rp,
                               n_fam=1500, prev=[0.1, 0.1], seed=3)
    r = test_genetic_correlation(fams, n_boot=25, seed=1, n_iter=350, burn_in=100,
                                 sampling="population")
    assert r.null.shape == (25,)
    assert 0.0 < r.p_value <= 1.0
    assert r.p_value < 0.2                          # real r_g -> significant


def test_simulate_null_aligns_reordered_family_bounds_by_role():
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    def members(order):
        by_role = {
            "m": Member("m", [-np.inf, 2.0], [1.0, np.inf]),
            "f": Member("f", [3.0, -np.inf], [np.inf, 4.0]),
        }
        return [by_role[role] for role in order]

    fams = [Family(0, members(["m", "f"])),
            Family(1, members(["f", "m"]))]

    class FixedRng:
        def multivariate_normal(self, _mean, _cov, size):
            assert size == 2
            # phenotype-major, canonical role order: first role high, second low
            return np.tile([10.0, -10.0, 10.0, -10.0], (size, 1))

    h2 = np.array([0.4, 0.5])
    simulated = af_mod._simulate_null(
        fams, h2, np.diag(h2), np.eye(2), FixedRng())
    for family in simulated:
        by_role = {member.role: member for member in family.members}
        assert np.array_equal(by_role["f"].lower, [3.0, 4.0])
        assert np.array_equal(by_role["f"].upper, [np.inf, np.inf])
        assert np.array_equal(by_role["m"].lower, [-np.inf, -np.inf])
        assert np.array_equal(by_role["m"].upper, [1.0, 2.0])


def test_genetic_correlation_test_validates_indices_before_fit(monkeypatch):
    import importlib
    af_mod = importlib.import_module("research.advanced_fitting")

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("fit must not run for an invalid trait index")

    monkeypatch.setattr(af_mod, "fit_genetic_correlation", unexpected_fit)
    fams = [Family(0, [Member("o", [-np.inf] * 3, [0.0] * 3)])]
    for bad in (True, np.bool_(False), 0.0, np.float64(1.0)):
        with pytest.raises(TypeError, match="integer trait index"):
            af_mod.test_genetic_correlation(fams, i=bad, j=1, n_boot=1)
    for i, j in ((0, 0), (-1, 1), (0, 3)):
        with pytest.raises(ValueError, match="distinct trait indices"):
            af_mod.test_genetic_correlation(fams, i=i, j=j, n_boot=1)


def test_genetic_correlation_test_preserves_nuisance_genetics(monkeypatch):
    import importlib
    from types import SimpleNamespace
    af_mod = importlib.import_module("research.advanced_fitting")

    h2 = np.array([0.5, 0.6, 0.7])
    rg = np.array([[1.0, 0.2, 0.25],
                   [0.2, 1.0, 0.3],
                   [0.25, 0.3, 1.0]])
    G = rg * np.sqrt(np.outer(h2, h2))
    e2 = 1.0 - h2
    re = np.array([[1.0, 0.1, -0.1],
                   [0.1, 1.0, 0.15],
                   [-0.1, 0.15, 1.0]])
    E = re * np.sqrt(np.outer(e2, e2))
    full = SimpleNamespace(h2=h2, rg=rg, genetic_cov=G, env_cov=E, rp=G + E)
    refits = iter([full, SimpleNamespace(rg=np.eye(3))])
    captured = []

    def fake_fit(_families, **_kwargs):
        return next(refits)

    def fake_sim(_families, h2_null, G_null, rp_null, _rng):
        captured.append((h2_null.copy(), G_null.copy(), rp_null.copy()))
        return _families

    monkeypatch.setattr(af_mod, "fit_genetic_correlation", fake_fit)
    monkeypatch.setattr(af_mod, "_simulate_null", fake_sim)
    fams = [Family(0, [Member("o", [-np.inf] * 3, [0.0] * 3)])]
    af_mod.test_genetic_correlation(
        fams, i=np.int64(0), j=np.int32(1), n_boot=1, seed=3)

    _, G0, rp0 = captured[0]
    assert G0[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert G0[0, 2] == pytest.approx(G[0, 2])
    assert G0[1, 2] == pytest.approx(G[1, 2])
    assert np.allclose(rp0 - G0, E)
    assert np.min(np.linalg.eigvalsh(G0)) > 0.0
    assert np.min(np.linalg.eigvalsh(rp0)) > 0.0


def test_offset_seed_wraps_within_uint32_range():
    from ltpred.gibbs import _offset_seed

    maximum = (1 << 32) - 1
    assert _offset_seed(None, 1) is None
    assert _offset_seed(maximum, 1) == 0
    assert _offset_seed(np.uint32(maximum), 2) == 1


def test_variance_components_mcem_matches_he_and_has_modelbased_se():
    # MCEM point estimate agrees with HE; its se is a model-based SE (~ the true
    # across-dataset SD ~0.05), far larger than HE's within-dataset MC error.
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1", "s2"], h2=0.5,
                                    n_sim=2000, pop_prev=0.1, seed=7)
    he = fit_variance_components(sim.families, ("A",),
                                 n_iter=400, burn_in=120, seed=1,
                                 sampling="population")
    mcem = fit_variance_components_mcem(sim.families, ("A",),
                                        n_iter=400, burn_in=120, seed=1,
                                        sampling="population")
    assert mcem.components["A"] == pytest.approx(he.components["A"], abs=0.06)
    assert mcem.se["A"] > 5 * he.se["A"]                # model-based >> within-dataset MC
    assert 0.01 < mcem.se["A"] < 0.15                   # in the plausible sampling-SD range


def test_variance_components_mcem_loglik_aic():
    # the MCEM fit reports a Monte-Carlo log-likelihood + AIC for model comparison;
    # on real A+C data AIC prefers A+C. The HE fit has no likelihood fields.
    import dataclasses
    fams = _sim_ac(["m", "f", "s1", "s2", "s3", "s4"], 0.4, 0.2, 2000, 5)
    rA = fit_variance_components_mcem(fams, ("A",), n_iter=350, burn_in=100,
                                      seed=1, sampling="population")
    rAC = fit_variance_components_mcem(fams, ("A", "C"), n_iter=350, burn_in=100,
                                       seed=1, sampling="population")
    assert rA.loglik is not None and np.isfinite(rA.loglik)
    assert rA.aic == pytest.approx(2 * 1 - 2 * rA.loglik)
    assert rAC.aic == pytest.approx(2 * 2 - 2 * rAC.loglik)
    assert rAC.loglik > rA.loglik                       # richer model fits better
    assert rAC.aic < rA.aic                              # AIC prefers A+C (real C=0.2)
    # the HE fit has no likelihood fields (they live on MCEMVarCompResult only)
    he = fit_variance_components(
        fams, ("A", "C"), n_iter=100, burn_in=30, seed=1,
        sampling="population")
    assert "loglik" not in {f.name for f in dataclasses.fields(he)}
    # the MCEM route validates its inputs the same way the HE route does
    with pytest.raises(ValueError, match="unknown component"):
        fit_variance_components_mcem(fams, ("A", "Z"), n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="duplicate"):
        fit_variance_components_mcem(fams, ("A", "A"), n_iter=50, burn_in=10)
    with pytest.raises(ValueError, match="burn_in"):
        fit_variance_components_mcem(fams, ("A",), n_iter=50)


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


def test_mcem_rejects_person_specific_thresholds():
    # MCEM shares the pooled-moment augmentation, so it inherits the same
    # personalised-threshold bias and refuses those bounds like the HE fit.
    fams = _coherent_varying_bounds(n_fam=20)
    with pytest.raises(ValueError, match="single case/control threshold"):
        fit_variance_components_mcem(fams, ("A",), n_iter=12, burn_in=6,
                                     inner_sweeps=1, seed=1,
                                     sampling="population")


def test_genetic_correlation_rejects_person_specific_rectangles():
    fams = _simulate_two_trait(
        ["m", "s1"], [0.4, 0.4], np.eye(2), np.eye(2),
        n_fam=20, prev=[0.1, 0.2], seed=3,
    )
    for i, family in enumerate(fams):
        for j, member in enumerate(family.members):
            delta = 0.01 * (i + 2 * j)
            lower = np.asarray(member.lower, dtype=float).copy()
            upper = np.asarray(member.upper, dtype=float).copy()
            lower[np.isfinite(lower)] += delta
            upper[np.isfinite(upper)] += delta
            member.lower, member.upper = lower, upper

    # The cross-trait HE regression is biased by personalised thresholds the same
    # way; onset-age structure is the job of fit_genetic_correlation_decay.
    with pytest.raises(ValueError, match="single case/control threshold"):
        fit_genetic_correlation(fams, n_iter=12, burn_in=6, inner_sweeps=1,
                                seed=1, sampling="population")


def test_research_fitters_make_population_sampling_contract_explicit():
    # the research fitters embed the same population-sampling assumption as the
    # core moment fitters, so they share the sampling= contract gate
    import warnings

    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=30,
                                    pop_prev=0.1, seed=1)
    with pytest.warns(RuntimeWarning, match="unascertained"):
        fit_variance_components_mcem(sim.families, ("A",), n_iter=10, burn_in=6,
                                     inner_sweeps=1, seed=1)
    with pytest.raises(ValueError, match="only sampling='population'"):
        fit_variance_components_mcem(sim.families, ("A",), n_iter=10, burn_in=6,
                                     sampling="case-control")
    multi = _simulate_two_trait(["m", "s1"], [0.5, 0.4], np.eye(2), np.eye(2),
                                n_fam=30, prev=[0.1, 0.1], seed=1)
    with pytest.warns(RuntimeWarning, match="unascertained"):
        fit_genetic_correlation(multi, n_iter=10, burn_in=6, inner_sweeps=1,
                                seed=1)
    with pytest.raises(ValueError, match="only sampling='population'"):
        fit_genetic_correlation(multi, n_iter=10, burn_in=6,
                                sampling="family-history")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fit_genetic_correlation(multi, n_iter=10, burn_in=6, inner_sweeps=1,
                                seed=1, sampling="population")
    assert not recorded


def test_research_fitters_reject_negative_burn_in():
    sim = simulate_under_LTM_single(fam_vec=["m", "s1"], h2=0.5, n_sim=30,
                                    pop_prev=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        fit_variance_components_mcem(sim.families, ("A",), n_iter=10, burn_in=-1)
    multi = _simulate_two_trait(["m", "s1"], [0.5, 0.4], np.eye(2), np.eye(2),
                                n_fam=30, prev=[0.1, 0.1], seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        fit_genetic_correlation(multi, n_iter=10, burn_in=-1)


class _GradientTests:
    """Finite-difference pins for analytic gradients hidden in optimizer closures.

    Same idea as the GradientTests of research/tests/test_decay.py: the analytic
    gradient must match central finite differences of its own objective. The
    objectives here are closures passed to ``scipy.optimize.minimize``, so a spy
    captures them as the fit runs."""

    @staticmethod
    def _capture_minimize(monkeypatch):
        import scipy.optimize
        captured = {}
        real_minimize = scipy.optimize.minimize

        def spy(fun, *args, **kwargs):
            captured.setdefault("obj_grad", fun)
            return real_minimize(fun, *args, **kwargs)

        monkeypatch.setattr(scipy.optimize, "minimize", spy)
        return captured

    @staticmethod
    def _check(obj_grad, x0):
        _, grad = obj_grad(x0)
        d = 1e-6
        for a in range(len(x0)):
            step = np.zeros_like(x0)
            step[a] = d
            fd = (obj_grad(x0 + step)[0] - obj_grad(x0 - step)[0]) / (2 * d)
            assert grad[a] == pytest.approx(fd, rel=1e-4, abs=1e-6), \
                f"grad[{a}]: {grad[a]:.6f} vs fd {fd:.6f}"


def test_mstep_reml_gradient_matches_finite_difference(monkeypatch):
    from research.advanced_fitting import _mstep_reml
    from ltpred.fit import _component_matrix

    roles = ["o", "s1", "m", "f"]
    Kmats = [_component_matrix(roles, "A"), _component_matrix(roles, "C")]
    k = len(roles)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, k))
    stats = [(X.T @ X / 30.0 + 0.3 * np.eye(k), 30, Kmats)]   # PD second moment
    captured = _GradientTests._capture_minimize(monkeypatch)
    _mstep_reml(np.array([0.3, 0.2]), stats, 1e-4)
    _GradientTests._check(captured["obj_grad"], np.array([0.35, 0.25]))


def test_minres_loadings_gradient_matches_finite_difference(monkeypatch):
    from research.advanced_fitting import _minres_loadings

    R = np.array([[1.0, 0.5, 0.4, 0.3],
                  [0.5, 1.0, 0.45, 0.35],
                  [0.4, 0.45, 1.0, 0.5],
                  [0.3, 0.35, 0.5, 1.0]])
    captured = _GradientTests._capture_minimize(monkeypatch)
    _minres_loadings(R, 1, None, 50, 1e-8)
    # an arbitrary interior loading vector, away from the warm start
    _GradientTests._check(captured["obj_grad"],
                          np.array([0.35, -0.2, 0.55, 0.1]))
