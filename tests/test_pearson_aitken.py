"""Pearson-Aitken inference: exactness, agreement with Gibbs, and PA-FGRS mixture."""

import math

import numpy as np
import pytest
from scipy import stats

from ltpred.family import Family, Member
from ltpred.pearson_aitken import (pa_algorithm, pa_estimate_batched,
                                   _std_tnorm_moments, _tnorm_mixture)
from ltpred.thresholds import pa_thresholds
from ltpred.estimate import (estimate_liability, _estimate_liability_pa,
                             estimate_liability_pa_arrays,
                             estimate_liability_gibbs_arrays)
from ltpred.simulate import simulate_under_LTM_single


def _imr(t):
    return stats.norm.pdf(t) / stats.norm.sf(t)


def _tnorm_moments(mu=0.0, var=1.0, lower=-np.inf, upper=np.inf):
    """Mean and variance of ``N(mu, var)`` truncated to ``(lower, upper)``.

    Test-local stand-in for the removed public ``tnorm_moments`` wrapper,
    exercising the private standardized kernel directly. Mirrors the module's
    own ``_tnorm_mean``/``_tnorm_var`` guards: the degenerate point-mass and
    unbounded intervals never reach ``_std_tnorm_moments`` (the narrow-interval
    quadrature divides by the zero width)."""
    if lower == -np.inf and upper == np.inf:
        return mu, var
    if lower == upper:
        return lower, 0.0
    sd = math.sqrt(var)
    m, v = _std_tnorm_moments((lower - mu) / sd, (upper - mu) / sd)
    return mu + sd * m, var * v


@pytest.mark.parametrize("lo,hi", [(-np.inf, np.inf), (0.5, 2.0), (-1.0, 1.0),
                                   (1.5, np.inf), (-np.inf, -0.5)])
def test_tnorm_moments_match_scipy(lo, hi):
    m, v = _tnorm_moments(0.3, 1.7, lo, hi)
    ref = stats.truncnorm((lo - 0.3) / np.sqrt(1.7), (hi - 0.3) / np.sqrt(1.7),
                          loc=0.3, scale=np.sqrt(1.7))
    assert m == pytest.approx(ref.mean(), abs=1e-9)
    assert v == pytest.approx(ref.var(), abs=1e-9)


def test_tnorm_moments_point_mass_and_infinite():
    assert _tnorm_moments(0.0, 1.0, 1.3, 1.3) == (1.3, 0.0)
    m, v = _tnorm_moments(0.7, 2.0, -np.inf, np.inf)
    assert m == pytest.approx(0.7) and v == pytest.approx(2.0)


@pytest.mark.parametrize("threshold", [8.0, 9.0])
def test_tnorm_moments_are_stable_in_extreme_tails(threshold):
    expected = stats.truncnorm(threshold, np.inf)
    mean, var = _tnorm_moments(lower=threshold)
    assert mean == pytest.approx(expected.mean(), abs=2e-10)
    assert var == pytest.approx(expected.var(), abs=2e-10)
    assert mean > threshold
    assert 0.0 < var < 0.02


@pytest.mark.parametrize("lower,upper", [(8.0, 9.0), (9.0, 10.0)])
def test_tnorm_moments_are_stable_in_finite_tail_intervals(lower, upper):
    expected = stats.truncnorm(lower, upper)
    mean, var = _tnorm_moments(lower=lower, upper=upper)
    assert mean == pytest.approx(expected.mean(), abs=2e-10)
    assert var == pytest.approx(expected.var(), abs=2e-10)


@pytest.mark.parametrize("width", [1e-6, 1e-5])
def test_tnorm_moments_are_stable_in_narrow_tail_intervals(width):
    lower = 8.0
    upper = lower + width
    actual_width = upper - lower
    center = lower + 0.5 * actual_width
    # Over a microscopic interval the normal density is locally exponential.
    # These leading centered-series terms have errors far below the tolerances.
    expected_mean = center - center * actual_width ** 2 / 12.0
    expected_var = actual_width ** 2 / 12.0
    mean, var = _tnorm_moments(lower=lower, upper=upper)
    assert mean == pytest.approx(expected_mean, abs=1e-12)
    assert var == pytest.approx(expected_var, rel=1e-6)


@pytest.mark.parametrize("threshold", [8.0, 9.0])
def test_tnorm_moments_extreme_tail_symmetry(threshold):
    right_mean, right_var = _tnorm_moments(lower=threshold)
    left_mean, left_var = _tnorm_moments(upper=-threshold)
    assert left_mean == pytest.approx(-right_mean, abs=1e-12)
    assert left_var == pytest.approx(right_var, abs=1e-12)


@pytest.mark.parametrize("threshold", [1_000.0, 10_000.0, 100_000.0])
def test_tnorm_moments_are_stable_in_far_one_sided_tails(threshold):
    # Inverse-Mills asymptotics, through terms well below double precision at
    # these thresholds.  SciPy's direct variance formula also cancels here, so
    # it is not a usable reference for this regression.
    inv = 1.0 / threshold
    expected_mean = threshold + inv - 2.0 * inv ** 3 + 10.0 * inv ** 5
    expected_var = inv ** 2 - 6.0 * inv ** 4 + 50.0 * inv ** 6

    right_mean, right_var = _tnorm_moments(lower=threshold)
    left_mean, left_var = _tnorm_moments(upper=-threshold)

    assert right_mean == pytest.approx(expected_mean,
                                       abs=2.0 * np.spacing(expected_mean))
    assert right_var == pytest.approx(expected_var, rel=2e-13)
    assert right_mean > threshold
    assert 0.0 < right_var < inv ** 2
    assert left_mean == -right_mean
    assert left_var == right_var


@pytest.mark.parametrize("lower,upper", [
    (1_000.0, 1_001.0),
    (100_000.0, 100_000.0005),
])
def test_tnorm_moments_are_stable_in_far_finite_tail_intervals(lower, upper):
    one_sided_mean, one_sided_var = _tnorm_moments(lower=lower)
    right_mean, right_var = _tnorm_moments(lower=lower, upper=upper)
    left_mean, left_var = _tnorm_moments(lower=-upper, upper=-lower)

    # The excluded upper-tail mass is below double precision in both cases, so
    # these finite-interval moments equal the one-sided result numerically.  The
    # second case also guards routing before the generic narrow-interval path.
    assert right_mean == one_sided_mean
    assert right_var == one_sided_var
    assert lower < right_mean < upper
    assert 0.0 < right_var < 1.0 / lower ** 2
    assert left_mean == -right_mean
    assert left_var == right_var


def test_pa_single_case_is_exact():
    h2, prev = 0.5, 0.05
    t = float(stats.norm.isf(prev))
    cov = np.array([[h2, h2], [h2, 1.0]])  # roles g, o
    est, var = pa_algorithm(cov, lower=[-np.inf, t], upper=[np.inf, np.inf], target=0)
    assert est == pytest.approx(h2 * _imr(t), abs=1e-9)
    assert var > 0


def test_pa_single_extreme_case_propagates_stable_tail_moments():
    h2, threshold = 0.5, 9.0
    cov = np.array([[h2, h2], [h2, 1.0]])
    mean, selected_var = _tnorm_moments(lower=threshold)
    est, var = pa_algorithm(cov, lower=[-np.inf, threshold],
                            upper=[np.inf, np.inf], target=0)
    assert est == pytest.approx(h2 * mean, abs=1e-10)
    assert var == pytest.approx(h2 + h2 ** 2 * (selected_var - 1.0), abs=1e-10)
    assert np.isfinite(est) and 0.0 < var < h2


def test_pa_single_control_is_negative():
    h2, prev = 0.5, 0.2
    t = float(stats.norm.isf(prev))
    cov = np.array([[h2, h2], [h2, 1.0]])
    est, _ = pa_algorithm(cov, lower=[-np.inf, -np.inf], upper=[np.inf, t], target=0)
    assert est == pytest.approx(h2 * (-stats.norm.pdf(t) / stats.norm.cdf(t)), abs=1e-9)


def test_pa_pinned_case_conditions_exactly():
    # Family-free ADuLT point mass: only o is pinned at c -> E[g] = h2*c.
    h2, c = 0.5, 1.3
    cov = np.array([[h2, h2], [h2, 1.0]])
    est, var = pa_algorithm(cov, lower=[-np.inf, c], upper=[np.inf, c], target=0)
    assert est == pytest.approx(h2 * c)
    assert var == pytest.approx(h2 - h2 ** 2)


@pytest.mark.jit_required
def test_pa_matches_gibbs_on_families():
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=0.5,
                                    pop_prev=0.1, n_sim=500, seed=3)
    pa = estimate_liability(sim.families, h2=0.5, method="pa", out=("genetic",))
    gb = estimate_liability(sim.families, h2=0.5, method="gibbs", out=("genetic",),
                            tol=0.02, n_sim=40_000, burn_in=800, seed=1)
    diff = np.abs(pa.est["genetic"] - gb.est["genetic"])
    assert np.corrcoef(pa.est["genetic"], gb.est["genetic"])[0, 1] > 0.99
    assert diff.mean() < 0.02


@pytest.mark.jit_required
def test_pa_conditional_variance_matches_gibbs_multi_truncation():
    # PA's `var` is exact for a single truncation (pinned by the tests above),
    # but real families fold several, where the sequential approximation is only
    # approximate. Both engines now report a posterior variance, so check PA's
    # against the sampler's on a family that folds seven observations -- the
    # regime where `var` is actually used and was previously unchecked.
    t = float(stats.norm.isf(0.05))
    roles = ["o", "m", "f", "s1", "mgm", "mgf", "pgm"]
    rng = np.random.default_rng(0)
    status = rng.random((24, len(roles))) < 0.35
    lower = np.where(status, t, -np.inf)
    upper = np.where(status, np.inf, t)

    _pa_est, pa_var = estimate_liability_pa_arrays(roles, lower, upper, h2=0.5)
    _gb_est, _se, gb_var = estimate_liability_gibbs_arrays(
        roles, lower, upper, h2=0.5, n_sim=200_000, burn_in=2000, seed=5,
        tol=0.005, return_var=True)

    rel = np.abs(pa_var - gb_var) / gb_var
    assert np.all(pa_var > 0)
    assert rel.max() < 0.05, f"worst relative deviation {rel.max():.3f}"
    assert rel.mean() < 0.02


def test_pa_family_history_raises_estimate():
    t = float(stats.norm.isf(0.05))
    def fam(rel_case, fid):
        lo, hi = (t, np.inf) if rel_case else (-np.inf, t)
        return Family(fid, [Member("o", t, np.inf)] +
                      [Member(r, lo, hi) for r in ("m", "f", "s1")])
    res = _estimate_liability_pa([fam(True, "aff"), fam(False, "healthy")],
                                h2=0.5, out=("genetic",))
    assert res.est["genetic"][0] > res.est["genetic"][1] + 0.1


def test_pa_result_has_zero_se_and_variance():
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf), Member("m", -np.inf, t)])
    res = _estimate_liability_pa([fam], h2=0.5, out=("genetic", "full"))
    assert np.all(res.se["genetic"] == 0)
    assert res.var is not None and res.var["genetic"][0] > 0


def test_batched_matches_single_pa():
    t = float(stats.norm.isf(0.05))
    cov = np.array([[0.5, 0.5, 0.25], [0.5, 1.0, 0.25], [0.25, 0.25, 1.0]])
    lowers = np.array([[-np.inf, t, -np.inf], [-np.inf, -np.inf, t]])
    uppers = np.array([[np.inf, np.inf, t], [np.inf, t, np.inf]])
    est, var = pa_estimate_batched(cov, lowers, uppers, target=0)
    for i in range(2):
        e, v = pa_algorithm(cov, lowers[i], uppers[i], target=0)
        assert est[i] == pytest.approx(e)
        assert var[i] == pytest.approx(v)


def test_mixture_raises_young_control_liability():
    # a young control (K_i << K_pop) is weaker evidence of low liability
    K_i, K_pop = 0.01, 0.15
    t_i = float(stats.norm.isf(K_i))
    nomix, _ = _tnorm_moments(0.0, 1.0, -np.inf, t_i)
    mix, _ = _tnorm_mixture(0.0, 1.0, -np.inf, t_i, K_i=K_i, K_pop=K_pop)
    assert mix > nomix


def test_mixture_weight_stable_when_no_future_cases_and_extreme_mean():
    # K_i == K_pop is a legal input (0 <= K_i <= K_pop): the individual is old
    # enough that no future cases remain, so the "eventual case not yet onset"
    # component has weight (K_pop - K_i)/K_pop == 0. Combined with a conditional
    # mean so extreme that P(control) underflows to exactly 0.0, the mixture-weight
    # denominator becomes 0 and the ratio an unguarded 0/0 -> NaN that would then
    # poison the whole PA fold. The guard resolves that limit to a certain genuine
    # control (mixture_prob = 1), i.e. the plain truncated normal on the lifetime
    # interval (-inf, thr_pop).
    K = 0.10
    thr_pop = float(stats.norm.isf(K))          # split = Phi^-1(1 - K_pop)
    mean, var = _tnorm_mixture(50.0, 1.0, -np.inf, thr_pop,
                               K_i=K, K_pop=K)
    assert np.isfinite(mean) and np.isfinite(var)
    ref_mean, ref_var = _tnorm_moments(50.0, 1.0, -np.inf, thr_pop)
    assert mean == pytest.approx(ref_mean)
    assert var == pytest.approx(ref_var)


@pytest.mark.parametrize(
    "K_i,K_pop,match",
    [(0.01, np.nan, "finite K_i and K_pop"),
     (np.nan, 0.10, "finite K_i and K_pop"),
     (-0.01, 0.10, "0 <= K_i <= K_pop < 1"),
     (0.20, 0.10, "0 <= K_i <= K_pop < 1"),
     (0.00, 0.00, "K_pop > 0"),
     (0.10, 1.00, "K_pop > 0")],
)
def test_family_pa_validate_mixture_pairs(K_i, K_pop, match):
    t = float(stats.norm.isf(0.10))
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match=match):
        pa_algorithm(
            cov, [-np.inf, -np.inf], [np.inf, t], target=0,
            K_i=[np.nan, K_i], K_pop=[np.nan, K_pop])


def test_public_pa_allows_nan_nan_at_unused_coordinates():
    t = float(stats.norm.isf(0.10))
    plain = _tnorm_moments(0.0, 1.0, -np.inf, t)
    assert _tnorm_mixture(
        0.0, 1.0, -np.inf, t, K_i=np.nan, K_pop=np.nan) == pytest.approx(plain)

    cov = np.array([[0.5, 0.5, 0.25],
                    [0.5, 1.0, 0.25],
                    [0.25, 0.25, 1.0]])
    est, var = pa_algorithm(
        cov, [-np.inf, t, -np.inf], [np.inf, np.inf, t], target=0,
        K_i=[np.nan, np.nan, 0.02], K_pop=[np.nan, np.nan, 0.10])
    assert np.isfinite(est) and np.isfinite(var)


def test_public_pa_rejects_K_pair_on_observed_case_without_global_gate():
    """A supplied pair on upper=+inf is invalid even when require_pair is false."""
    t = float(stats.norm.isf(0.10))
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match=r"upper == \+inf"):
        pa_algorithm(
            cov, [-np.inf, t], [np.inf, np.inf], target=0,
            K_i=[np.nan, 0.02], K_pop=[np.nan, 0.10])


def test_pa_algorithm_normalises_float16_mixture_inputs_for_numba():
    t = float(stats.norm.isf(0.10))
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    K_i = np.array([np.nan, 0.02], dtype=np.float16)
    K_pop = np.array([np.nan, 0.10], dtype=np.float16)
    args = (cov, [-np.inf, -np.inf], [np.inf, t])

    got = pa_algorithm(*args, target=0, K_i=K_i, K_pop=K_pop)
    ref = pa_algorithm(
        *args, target=0, K_i=K_i.astype(np.float64),
        K_pop=K_pop.astype(np.float64))
    assert got == pytest.approx(ref)


def test_mixture_changes_genetic_estimate():
    # a proband case with a single young control sibling: the mixture (accounting
    # for the sibling's residual risk) should not decrease the genetic estimate.
    status = np.array([1, 0])          # proband case, sibling control
    age = np.array([55, 25])
    lo, hi, ki, kp = pa_thresholds(status, age, pop_prev=0.1)
    fam = Family("f", [Member("o", lo[0], hi[0]),
                       Member("s1", lo[1], hi[1], K_i=ki[1], K_pop=kp[1])])
    with_mix = _estimate_liability_pa([fam], h2=0.5, out=("genetic",), use_mixture=True)
    no_mix = _estimate_liability_pa([fam], h2=0.5, out=("genetic",), use_mixture=False)
    assert with_mix.est["genetic"][0] >= no_mix.est["genetic"][0] - 1e-9
    assert with_mix.est["genetic"][0] != no_mix.est["genetic"][0]


def test_mixture_split_is_lifetime_threshold_not_passed_upper():
    # PA-FGRS (Krebs et al. 2024, eqs S3-S5): the censored-control mixture splits at
    # the *lifetime* threshold Phi^-1(1-K_pop), not the passed `upper`. Age enters
    # only through K_i. So a censored control encoded with the age-specific bound
    # Phi^-1(1-K_i) must yield the SAME estimate as one encoded with the lifetime
    # bound Phi^-1(1-K_pop). Before the fix the split was taken from `upper`, so the
    # age bound got corrected a second time and the two disagreed.
    K_i, K_pop = 0.02, 0.10
    thr_age = float(stats.norm.isf(K_i))          # Phi^-1(1 - K_i), age-specific
    thr_life = float(stats.norm.isf(K_pop))       # Phi^-1(1 - K_pop), lifetime

    m_age, v_age = _tnorm_mixture(0.3, 0.8, -np.inf, thr_age,
                                  K_i=K_i, K_pop=K_pop)
    m_life, v_life = _tnorm_mixture(0.3, 0.8, -np.inf, thr_life,
                                    K_i=K_i, K_pop=K_pop)
    assert m_age == pytest.approx(m_life)
    assert v_age == pytest.approx(v_life)

    # ...and end-to-end through a family estimate (proband case + control sibling)
    fam_age = Family("f", [Member("o", thr_life, np.inf),
                           Member("s1", -np.inf, thr_age, K_i=K_i, K_pop=K_pop)])
    fam_life = Family("f", [Member("o", thr_life, np.inf),
                            Member("s1", -np.inf, thr_life, K_i=K_i, K_pop=K_pop)])
    e_age = _estimate_liability_pa([fam_age], h2=0.5, use_mixture=True).est["genetic"][0]
    e_life = _estimate_liability_pa([fam_life], h2=0.5, use_mixture=True).est["genetic"][0]
    assert e_age == pytest.approx(e_life)


def _simulate_pa_calibration(n_sim=8000, h2=0.5, pop_prev=0.1, seed=0):
    """proband + parents + 2 sibs, young ages; returns g_true and pa_thresholds cols."""
    from ltpred.covariance import construct_covmat_single
    from ltpred.thresholds import (liability_threshold, convert_liability_to_aoo)
    rng = np.random.default_rng(seed)
    cov = construct_covmat_single(fam_vec=("m", "f", "s1", "s2"), add_ind=True, h2=h2)
    roles, mat = cov.roles, cov.matrix
    liab = rng.multivariate_normal(np.zeros(len(roles)), mat, size=n_sim)
    t = float(liability_threshold(pop_prev))
    non_g = [r for r in roles if r != "g"]
    ranges = {"o": (10, 40), "s1": (10, 40), "s2": (10, 40), "m": (40, 70), "f": (40, 70)}

    fam_id, role, age, status = [], [], [], []
    for i in range(n_sim):
        for r in non_g:
            col = roles.index(r)
            is_case = liab[i, col] > t
            if is_case:
                aoo = convert_liability_to_aoo(liab[i, col], pop_prev=pop_prev)
                a = 0.0 if not np.isfinite(aoo) else round(float(aoo))
            else:
                a = int(rng.integers(*ranges[r]))
            fam_id.append(f"fam_{i}"); role.append(r); age.append(a)
            status.append(bool(is_case))
    return liab[:, roles.index("g")], (np.array(fam_id), np.array(role, dtype=object),
                                       np.array(status), np.array(age, dtype=float))


def test_mixture_is_calibrated_no_double_correction():
    # Calibration regression guard for the PA-FGRS censored-control mixture. On data
    # simulated under the LTM (young controls, so censoring bites), the mixture fed
    # `pa_thresholds`' mixture is checked against the same interval-case bounds with
    # no mixture, computed on the same data. Anchoring to that in-test
    # reference is seed-robust and independent of the absolute slope (~0.84 here,
    # set by the interval case encoding). The pre-fix double-correction inflated the
    # estimate badly -- bias +0.84 vs +0.25, corr(g) 0.47 vs 0.52, slope 0.77 vs
    # 0.84, and only ~0.98 agreement with the exact encoding -- so it fails below.
    from ltpred.family import families_from_columns
    from ltpred.thresholds import pa_thresholds

    g, (fam_id, role, status, age) = _simulate_pa_calibration()
    lo, hi, ki, kp = pa_thresholds(status, age, pop_prev=0.1)

    mix = families_from_columns(fam_id, role, lo, hi, K_i=ki, K_pop=kp)
    est = estimate_liability(mix, h2=0.5, method="pa", use_mixture=True).est["genetic"]
    interval = families_from_columns(fam_id, role, lo, hi)   # same bounds, no mixture
    est0 = estimate_liability(interval, h2=0.5, method="pa",
                              use_mixture=False).est["genetic"]

    slope = np.cov(g, est)[0, 1] / np.var(est)
    slope0 = np.cov(g, est0)[0, 1] / np.var(est0)
    assert np.corrcoef(g, est)[0, 1] > 0.50            # not degraded (pre-fix: ~0.47)
    assert abs(float(np.mean(est - g))) < 0.45         # no blow-up (pre-fix bias ~+0.84)
    assert abs(slope - slope0) < 0.04, (slope, slope0)  # calibrated like the exact model

    # the mixture on age bounds must closely reproduce the direct interval encoding --
    # the split is at the lifetime threshold, so the age bound only flags
    # censoring. Pre-fix these diverged (corr ~0.98, mean off by ~0.6).
    assert np.corrcoef(est, est0)[0, 1] > 0.999
    assert abs(np.mean(est) - np.mean(est0)) < 0.05


def test_method_dispatch_and_multi_trait_error():
    t = float(stats.norm.isf(0.05))
    fam = Family("f", [Member("o", t, np.inf)])
    for m in ("pa", "pearson-aitken", "AITKEN"):
        res = estimate_liability([fam], h2=0.5, method=m, out=("genetic",))
        assert res.est["genetic"][0] > 0
    with pytest.raises(NotImplementedError):
        estimate_liability([fam], h2=[0.5, 0.5], method="pa",
                           genetic_corrmat=np.eye(2), full_corrmat=np.eye(2))
    with pytest.raises(ValueError):
        estimate_liability([fam], h2=0.5, method="bogus")


def test_mixture_rejects_K_on_pinned_rows():
    # a pinned observation needs no censoring mixture; K's there can invert
    # the mixture interval and produce NaN moments
    t = float(stats.norm.isf(0.10))
    pin = 1.3
    cov = np.array([[0.5, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="pinned"):
        pa_algorithm(cov, [-np.inf, pin], [np.inf, pin], target=0,
                     K_i=[np.nan, 0.02], K_pop=[np.nan, 0.10])
    with pytest.raises(ValueError, match="pinned"):
        pa_estimate_batched(cov, np.array([[-np.inf, pin]]),
                            np.array([[np.inf, pin]]), target=0,
                            K_is=np.array([[np.nan, 0.02]]),
                            K_pops=np.array([[np.nan, 0.10]]))
    fam = Family("f", [Member("o", t, np.inf),
                       Member("m", pin, pin, K_i=0.02, K_pop=0.10)])
    with pytest.raises(ValueError, match="pinned"):
        _estimate_liability_pa([fam], h2=0.5, use_mixture=True)
    from ltpred.estimate import estimate_liability_pa_arrays
    with pytest.raises(ValueError, match="pinned"):
        estimate_liability_pa_arrays(
            ["o", "m"], np.array([[-np.inf, pin]]), np.array([[np.inf, pin]]),
            h2=0.5,
            K_i=np.array([[np.nan, 0.02]]), K_pop=np.array([[np.nan, 0.10]]),
            use_mixture=True)


def test_mixture_allows_pinned_rows_with_nan_K():
    # the legitimate case: an onset-pinned case (NaN/NaN) alongside a
    # genuinely censored control still runs
    t = float(stats.norm.isf(0.10))
    pin = 1.3
    fam = Family("f", [Member("o", pin, pin),
                       Member("s1", -np.inf, t, K_i=0.02, K_pop=0.10)])
    res = _estimate_liability_pa([fam], h2=0.5, use_mixture=True)
    assert np.isfinite(res.est["genetic"][0])


def test_pa_entry_points_validate_bounds():
    # unlike rtmvnorm_gibbs the public PA entries never called validate_bounds:
    # NaN bounds propagated silently, and a reversed pair narrower than 1e-3 was
    # "sorted" by the narrow-interval quadrature
    h2 = 0.5
    cov = np.array([[h2, h2], [h2, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        pa_algorithm(cov, [-np.inf, np.nan], [np.inf, np.inf])
    with pytest.raises(ValueError, match="reversed bounds"):
        pa_algorithm(cov, [-np.inf, 1.0], [np.inf, 0.9995])
    with pytest.raises(ValueError, match="NaN"):
        pa_estimate_batched(cov, [[-np.inf, np.nan]], [[np.inf, np.inf]])
    with pytest.raises(ValueError, match="reversed bounds"):
        pa_estimate_batched(cov, [[-np.inf, 1.0]], [[np.inf, 0.9995]])


@pytest.mark.parametrize(
    "cov,match",
    [(np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
     (np.array([[np.inf, 0.0], [0.0, 1.0]]), "finite"),
     (np.array([[1.0, 0.9], [0.1, 1.0]]), "symmetric"),
     (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive-semidefinite"),
     (np.diag([1.0, 0.0]), "strictly positive"),
     (np.zeros((2, 3)), "square")],
)
def test_pa_entry_points_validate_covariance(cov, match):
    # Unlike Gibbs, PA accepts singular PSD matrices, but an indefinite matrix is
    # not a covariance and zero marginal variance is unsupported by truncation.
    lo, hi = [-np.inf, 1.0], [np.inf, np.inf]
    with pytest.raises(ValueError, match=match):
        pa_algorithm(cov, lo, hi)
    with pytest.raises(ValueError, match=match):
        pa_estimate_batched(cov, [lo], [hi])


def test_pa_accepts_merely_psd_covariance():
    # A singular PSD covariance with positive marginal variances remains valid.
    psd = np.ones((2, 2))
    est, var = pa_algorithm(psd, [-np.inf, 1.0], [np.inf, np.inf])
    assert np.isfinite(est) and np.isfinite(var)
    est, var = pa_estimate_batched(psd, [[-np.inf, 1.0]], [[np.inf, np.inf]])
    assert np.isfinite(est[0]) and np.isfinite(var[0])


@pytest.mark.parametrize("fn", ["pa_algorithm", "pa_estimate_batched"])
@pytest.mark.parametrize("delta", [-2, +2])
def test_pa_entry_points_check_bounds_width(fn, delta):
    # Both public entry points fancy-index the bounds by a length-d
    # permutation. Before this check an over-wide bounds vector had its tail
    # silently dropped -- a confident wrong estimate on exactly the path where
    # a mis-packed column matrix is most likely -- and a short one raised a
    # bare IndexError. The F4 fix reached only the private role-array helpers.
    cov = np.eye(4) * 1.0
    cov[0, 1] = cov[1, 0] = 0.4
    d = cov.shape[0] + delta
    lo = np.full(d, -np.inf)
    hi = np.full(d, np.inf)
    with pytest.raises(ValueError, match=r"shape"):
        if fn == "pa_algorithm":
            pa_algorithm(cov, lo, hi)
        else:
            pa_estimate_batched(cov, lo[None, :], hi[None, :])


@pytest.mark.parametrize("target", [-1, 4, 99])
def test_pa_entry_points_check_target_range(target):
    # An out-of-range target used to reach np.delete/np.ix_ and fail obscurely.
    cov = np.eye(4)
    lo, hi = np.full(4, -np.inf), np.full(4, np.inf)
    with pytest.raises(ValueError, match="target"):
        pa_algorithm(cov, lo, hi, target=target)
    with pytest.raises(ValueError, match="target"):
        pa_estimate_batched(cov, lo[None, :], hi[None, :], target=target)


def test_pa_batched_rejects_1d_bounds():
    # (F,) instead of (F, d) is the natural single-family slip; it previously
    # produced an IndexError from inside the kernel.
    cov = np.eye(3)
    lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
    with pytest.raises(ValueError, match=r"shape \(F, 3\)"):
        pa_estimate_batched(cov, lo, hi)
