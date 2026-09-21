"""Simulation under the LTM, and the estimate-recovers-truth end-to-end check."""

import warnings

import numpy as np
import pytest

from ltpred.covariance import construct_covmat_single
from ltpred.simulate import simulate_under_LTM_single, _stable_factor
from ltpred.estimate import estimate_liability


def test_simulated_prevalence_matches():
    sim = simulate_under_LTM_single(fam_vec=["m", "f"], h2=0.5, n_sim=20_000,
                                    pop_prev=0.1, seed=0)
    assert sim.status["o"].mean() == pytest.approx(0.1, abs=0.01)
    assert sim.genetic.var() == pytest.approx(0.5, abs=0.05)   # var(g) = h2
    assert sim.full.var() == pytest.approx(1.0, abs=0.05)      # var(o) = 1


def test_seeded_draws_use_a_canonical_factorisation():
    """A seed has to reproduce on every machine, not just this one.

    ``Generator.multivariate_normal`` factors the covariance with an SVD, and
    an SVD has no canonical sign: the same seed drew different families on
    different LAPACK builds, which silently invalidated every figure quoted
    from a seeded run (docs/vignette.md). The draws must come from the unique
    Cholesky factor instead -- pinned here so a refactor cannot quietly go
    back to the platform-dependent path.
    """
    roles = ["m", "f", "s1"]
    cov = construct_covmat_single(fam_vec=roles, h2=0.5)
    sim = simulate_under_LTM_single(fam_vec=roles, h2=0.5, n_sim=64,
                                    pop_prev=0.05, seed=3)
    draws = np.random.default_rng(3).standard_normal((64, len(cov.roles)))
    expected = draws @ np.linalg.cholesky(cov.matrix).T
    np.testing.assert_array_equal(sim.liabilities, expected)


@pytest.mark.parametrize("h2", [0.2, 0.5, 1.0])   # h2=1 is exactly singular
@pytest.mark.parametrize("fam_vec", [["m", "f", "s1"],
                                     ["m", "f", "s1", "mgm", "mgf",
                                      "pgm", "pgf"]])
def test_stable_factor_reproduces_the_covariance(h2, fam_vec):
    cov = construct_covmat_single(fam_vec=fam_vec, h2=h2)
    L = _stable_factor(cov.matrix)
    # h2=1 is factored after lifting the spectrum by ~1e-12 of the mean
    # variance, so the reconstruction carries that ridge and no more.
    np.testing.assert_allclose(L @ L.T, cov.matrix, atol=1e-9)


def test_stable_factor_warns_only_on_a_genuinely_indefinite_covariance():
    """`multivariate_normal(check_valid="warn")` used to be the safety net."""
    singular = construct_covmat_single(fam_vec=["m", "f"], h2=1.0).matrix
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a rounding-level negative
        _stable_factor(singular)                # eigenvalue must stay silent
    with pytest.warns(RuntimeWarning, match="not positive-semidefinite"):
        _stable_factor(np.array([[1.0, 0.9], [0.9, 0.5]]))


@pytest.mark.parametrize("fam_vec", [["m", "f"],
                                     ["m", "f", "s1", "mgm", "mgf",
                                      "pgm", "pgf"]])
def test_singular_covariance_still_gets_the_canonical_cholesky_form(fam_vec):
    """h2=1 has no Cholesky factor, and eigenvectors are no way out.

    LAPACK picks an arbitrary basis inside a repeated eigenvalue's eigenspace,
    which is the same platform dependence `_stable_factor` exists to remove --
    and the default seven-relative pedigree at h2=1 has such a repetition. The
    factor must therefore still be a Cholesky factor: lower triangular with a
    non-negative diagonal is exactly the form that is unique.
    """
    cov = construct_covmat_single(fam_vec=fam_vec, h2=1.0).matrix
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(cov)                     # genuinely singular
    L = _stable_factor(cov)
    np.testing.assert_array_equal(L, np.tril(L))    # lower triangular
    assert np.all(np.diag(L) >= 0.0)                # positive diagonal
    np.testing.assert_allclose(L @ L.T, cov, atol=1e-9)


def test_repeated_eigenvalue_pedigree_is_the_case_that_needs_it():
    """Pins the premise of the test above, so it cannot quietly stop biting."""
    cov = construct_covmat_single(
        fam_vec=["m", "f", "s1", "mgm", "mgf", "pgm", "pgf"], h2=1.0).matrix
    evals = np.round(np.linalg.eigvalsh(cov), 9)
    assert len(set(evals)) < len(evals)


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


@pytest.mark.parametrize("case_encoding", ["pin", "interval"])
def test_rounded_threshold_crossing_bounds_contain_generating_liability(
        case_encoding):
    sim = simulate_under_LTM_single(
        fam_vec=["m", "f"], h2=0.5, n_sim=3000, pop_prev=0.1,
        use_age=True, seed=31, onset_model="threshold_crossing",
        case_encoding=case_encoding, onset_resolution=1.0,
    )
    for i, family in enumerate(sim.families):
        for member in family.members:
            liability = sim.liabilities[i, sim.roles.index(member.role)]
            assert member.lower <= liability + 1e-12
            assert liability <= member.upper + 1e-12


def test_exact_threshold_crossing_case_encoding_is_preserved():
    kwargs = dict(
        fam_vec=["m"], h2=0.5, n_sim=1000, pop_prev=0.1,
        use_age=True, seed=32, onset_model="threshold_crossing",
        onset_resolution=None,
    )
    pinned = simulate_under_LTM_single(case_encoding="pin", **kwargs)
    interval = simulate_under_LTM_single(case_encoding="interval", **kwargs)
    case = pinned.status["o"]
    assert case.any()
    for i in np.flatnonzero(case):
        pin = next(m for m in pinned.families[i].members if m.role == "o")
        one_sided = next(m for m in interval.families[i].members if m.role == "o")
        true_liability = pinned.liabilities[i, pinned.roles.index("o")]
        assert pin.lower == pytest.approx(true_liability, abs=1e-9)
        assert pin.upper == pytest.approx(true_liability, abs=1e-9)
        assert one_sided.lower == pytest.approx(true_liability, abs=1e-9)
        assert one_sided.upper == np.inf


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
    # "c" entry was unreachable and children drew adult ages. The range table
    # lives here (the package no longer ships the unused lookup); the
    # production behaviour is pinned through _role_stem, which _draw_ages uses.
    from ltpred.simulate import _role_stem
    _AGE_RANGES = {
        "o": (10, 60), "s": (10, 60), "mhs": (10, 60), "phs": (10, 60),
        "m": (40, 80), "f": (40, 80), "mau": (40, 80), "pau": (40, 80),
        "mgm": (65, 95), "mgf": (65, 95), "pgm": (65, 95), "pgf": (65, 95),
        "c": (0, 30),
    }
    age_range = lambda role: _AGE_RANGES.get(_role_stem(role), (10, 60))
    assert age_range("c1.1") == (0, 30)
    assert age_range("c2.10") == (0, 30)
    assert age_range("o") == (10, 60)
    assert age_range("s12") == (10, 60)
    assert age_range("mgm") == (65, 95)
    assert age_range("mau3") == (40, 80)


def test_pin_encoding_warns_without_threshold_crossing():
    # under stochastic/liability-dependent onset, T(onset) is not the case's
    # liability, so the pin fabricates information (review 2026-08, F21)
    with pytest.warns(UserWarning, match="pin"):
        simulate_under_LTM_single(fam_vec=["m"], n_sim=5, use_age=True,
                                  onset_model="stochastic",
                                  case_encoding="pin", seed=1)


# ---------------------------------------------------------------------------
# Promoted population-register, follow-up and multi-trait generators
# (docs/reviews/REVIEW_2026-09d.md, T1-1/T2-2). These were benchmark-private;
# the guards below are what make them safe to teach from.
# ---------------------------------------------------------------------------

from ltpred.simulate import (pedigree_birth_times,  # noqa: E402
                             simulate_followup_records, simulate_pedigree,
                             simulate_register_liabilities,
                             simulate_under_LTM_multi)

_CIP_AGES = np.arange(0, 121, 1.0)
_CIP_K, _CIP_MID, _CIP_SLOPE = 0.10, 60.0, 1.0 / 8.0
_CIP_VALUES = _CIP_K / (1.0 + np.exp((_CIP_MID - _CIP_AGES) * _CIP_SLOPE))


def test_simulate_pedigree_is_a_valid_trio_table():
    ids, father, mother = simulate_pedigree(np.random.default_rng(3),
                                            n_founder_pairs=12, gens=3)
    assert len(ids) == len(father) == len(mother)
    assert len(set(ids)) == len(ids), "person ids must be unique"
    known = set(ids)
    for column in (father, mother):
        assert all(p is None or p in known for p in column), \
            "a recorded parent must be a person in the pedigree"
    assert any(p is None for p in father), "founders must remain"
    assert any(p is not None for p in father), "and so must their descendants"


def test_pedigree_birth_times_put_coparents_together_and_children_later():
    ids, father, mother = simulate_pedigree(np.random.default_rng(3),
                                            n_founder_pairs=8, gens=3)
    birth = pedigree_birth_times(ids, father, mother)
    pos = {p: i for i, p in enumerate(ids)}
    unique = np.unique(birth)
    # founders plus one generation per `gens` step, evenly spaced
    np.testing.assert_array_equal(unique, 1920.0 + 30.0 * np.arange(len(unique)))
    assert len(unique) == 4, "three child generations on top of the founders"
    for child, (fa, mo) in enumerate(zip(father, mother)):
        if fa in pos and mo in pos:
            assert birth[pos[fa]] == birth[pos[mo]], "co-parents share a generation"
            assert birth[child] > birth[pos[fa]], "children are a later generation"


def test_pedigree_birth_times_honours_the_calendar_arguments():
    ids = ["a", "b", "c"]
    birth = pedigree_birth_times(ids, [None, None, "a"], [None, None, "b"],
                                 base_birth_year=1900.0, generation_years=25.0)
    np.testing.assert_array_equal(birth, [1900.0, 1900.0, 1925.0])


def test_simulate_register_liabilities_recovers_the_generating_parameters():
    ids, father, mother = simulate_pedigree(np.random.default_rng(11),
                                            n_founder_pairs=60, gens=2)
    sim = simulate_register_liabilities(np.random.default_rng(11), ids, father,
                                        mother, h2=0.5, cip_ages=_CIP_AGES,
                                        cip_values=_CIP_VALUES, eval_age=70.0)
    n = len(ids)
    assert sim.status.shape == sim.age.shape == sim.genetic.shape == (n,)
    # var(g) = h2 on the standardised liability scale
    assert sim.genetic.var() == pytest.approx(0.5, abs=0.08)
    # observed cases are exactly those diagnosed by the evaluation age, and the
    # rate is the CIP at that age
    np.testing.assert_array_equal(sim.status, sim.onset <= 70.0)
    np.testing.assert_allclose(sim.age, np.where(sim.status, sim.onset, 70.0))
    assert sim.status.mean() == pytest.approx(float(np.interp(70.0, _CIP_AGES,
                                                              _CIP_VALUES)),
                                              abs=0.02)
    # a case is diagnosed by the evaluation age; a non-case may still carry a
    # finite onset, meaning a lifetime case diagnosed *after* eval_age
    assert np.all(np.isfinite(sim.onset[sim.status]))
    assert np.all(sim.onset[sim.status] <= 70.0)
    assert np.all(sim.onset[~sim.status] > 70.0)
    assert np.all(sim.residual_var > 0.0)


def test_register_simulation_feeds_the_public_register_driver():
    """The point of promoting it: a tutorial can reach estimate_liabilities."""
    from ltpred.pipeline import estimate_liabilities

    ids, father, mother = simulate_pedigree(np.random.default_rng(5),
                                            n_founder_pairs=40, gens=2)
    sim = simulate_register_liabilities(np.random.default_rng(5), ids, father,
                                        mother, h2=0.5, cip_ages=_CIP_AGES,
                                        cip_values=_CIP_VALUES, eval_age=70.0)
    scores = estimate_liabilities(
        sim.ids, sim.father, sim.mother, probands=sim.ids,
        status=sim.status.astype(int), age=sim.age, use="gwas",
        cip_ages=_CIP_AGES, cip_values=_CIP_VALUES, k_pop=_CIP_K, h2=0.5)
    est = np.asarray(scores.est)
    assert est.shape[0] == len(ids)
    assert np.all(np.isfinite(est))
    # a real signal, not a perfect one: the truth is the standardised g
    assert np.corrcoef(est, sim.genetic)[0, 1] > 0.3

    # the calendar-time route (use I) is where birth_time/index_time belong;
    # `use="gwas"` rejects them, which is itself part of the contract
    with pytest.raises(ValueError, match="must be omitted"):
        estimate_liabilities(
            sim.ids, sim.father, sim.mother, probands=sim.ids,
            status=sim.status.astype(int), age=sim.age, use="gwas",
            cip_ages=_CIP_AGES, cip_values=_CIP_VALUES, k_pop=_CIP_K, h2=0.5,
            birth_time=sim.birth_time)
    # Restrict the prospective cohort to probands still disease-free at their
    # index time, as the pipeline's own prevalent-case warning instructs;
    # scoring an already-diagnosed proband prospectively is leakage.
    index_time = sim.birth_time + 40.0
    at_risk = sim.onset > 40.0
    assert 0 < at_risk.sum() < len(ids)
    predicted = estimate_liabilities(
        sim.ids, sim.father, sim.mother,
        probands=[p for p, keep in zip(sim.ids, at_risk) if keep],
        status=sim.status.astype(int), age=sim.age, use="prediction",
        cip_ages=_CIP_AGES, cip_values=_CIP_VALUES, k_pop=_CIP_K, h2=0.5,
        birth_time=sim.birth_time, index_time=index_time[at_risk])
    assert np.all(np.isfinite(np.asarray(predicted.est)))


def test_simulate_followup_records_codes_the_competing_risk():
    rec = simulate_followup_records(np.random.default_rng(2), 20_000,
                                    pop_prev=_CIP_K, mid_point=_CIP_MID,
                                    slope=_CIP_SLOPE, mortality=True)
    assert set(np.unique(rec.event)) <= {0, 1, 2}
    assert (rec.event == 2).sum() > 0, "mortality arm must produce deaths"
    assert (rec.event == 1).sum() > 0, "and diagnoses"
    assert rec.age_entry.shape == rec.age_exit.shape == rec.event.shape
    assert np.all(rec.age_exit > rec.age_entry)
    assert rec.n_dropped == 20_000 - len(rec.event)
    # a death is only coded 2 when it precedes onset
    died = rec.event == 2
    assert np.all(rec.death_age[died] <= rec.age_exit[died])


def test_simulate_followup_records_without_mortality_has_no_deaths():
    rec = simulate_followup_records(np.random.default_rng(2), 5_000,
                                    pop_prev=_CIP_K, mid_point=_CIP_MID,
                                    slope=_CIP_SLOPE, mortality=False)
    assert not np.any(rec.event == 2)
    assert np.all(np.isinf(rec.death_age))


def test_simulate_followup_records_left_truncates_on_delayed_entry():
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    full = simulate_followup_records(rng_a, 20_000, pop_prev=_CIP_K,
                                     mid_point=_CIP_MID, slope=_CIP_SLOPE,
                                     mortality=True)
    late = simulate_followup_records(rng_b, 20_000, pop_prev=_CIP_K,
                                     mid_point=_CIP_MID, slope=_CIP_SLOPE,
                                     mortality=True, register_start_year=1995)
    assert len(late.event) < len(full.event), "delayed entry drops records"
    assert late.n_dropped > full.n_dropped
    # entry ages are still 0 for people born after the register opens, but the
    # cohort as a whole is observed from later in life
    assert late.age_entry.mean() > full.age_entry.mean()


def test_simulate_under_LTM_multi_hits_its_target_trait_correlation():
    sim = simulate_under_LTM_multi(n_families=4_000, seed=1)
    n_traits = len(sim.pop_prev)
    assert n_traits == 2
    assert sim.liabilities.shape == (4_000, len(sim.roles), n_traits)
    assert sim.status.shape == (4_000 * len(sim.roles), n_traits)
    # unit liability variance per trait, and the documented cross-correlation
    for p in range(n_traits):
        assert sim.liabilities[:, :, p].var() == pytest.approx(1.0, abs=0.05)
        assert sim.status[:, p].mean() == pytest.approx(float(sim.pop_prev[p]),
                                                        abs=0.02)
    empirical = np.corrcoef(sim.liabilities[:, :, 0].ravel(),
                            sim.liabilities[:, :, 1].ravel())[0, 1]
    assert empirical == pytest.approx(sim.truth["target_trait_corr"], abs=0.02)
    # one bound interval per trait, so fit_pairwise_multi can consume it
    member = sim.families[0].members[0]
    assert len(member.lower) == len(member.upper) == n_traits


def test_simulate_under_LTM_multi_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="one prevalence per trait"):
        simulate_under_LTM_multi(n_families=4, seed=1, pop_prev=(0.1,))
    with pytest.raises(ValueError, match="sib_shared"):
        simulate_under_LTM_multi(n_families=4, seed=1,
                                 sib_shared=((0.15, 0.075),))


def test_register_liabilities_rejects_a_non_invertible_cip():
    ids, father, mother = ["a", "b"], [None, "a"], [None, None]
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_register_liabilities(np.random.default_rng(0), ids, father,
                                      mother, h2=0.5, cip_ages=[0.0, 1.0],
                                      cip_values=[0.1, 0.1], eval_age=70.0)
    with pytest.raises(ValueError, match="same shape"):
        simulate_register_liabilities(np.random.default_rng(0), ids, father,
                                      mother, h2=0.5, cip_ages=[0.0, 1.0],
                                      cip_values=[0.1], eval_age=70.0)
