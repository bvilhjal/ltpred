"""Age / prevalence / liability conversions."""

import numpy as np
import pytest
from scipy import stats

from ltpred.thresholds import (convert_age_to_cir, _convert_cir_to_age,
                               convert_age_to_thresh, convert_liability_to_aoo,
                               prevalence_thresholds, age_thresholds,
                               liability_threshold, pa_thresholds,
                               thresholds_from_cip)


def test_cir_monotone_and_bounded():
    ages = np.arange(0, 120)
    cir = convert_age_to_cir(ages, pop_prev=0.1)
    assert np.all(np.diff(cir) >= 0)          # non-decreasing in age
    assert cir[0] >= 0 and cir[-1] <= 0.1     # bounded by prevalence
    # half prevalence at the mid point
    assert convert_age_to_cir(60, pop_prev=0.1, mid_point=60) == pytest.approx(0.05)


def test_cir_age_round_trip():
    ages = np.array([30.0, 45.0, 60.0, 75.0])
    cir = convert_age_to_cir(ages, pop_prev=0.1)
    back = _convert_cir_to_age(cir, pop_prev=0.1)
    assert np.allclose(back, ages, atol=1e-8)


def test_cir_to_age_nan_above_prevalence():
    assert np.isnan(_convert_cir_to_age(0.2, pop_prev=0.1))


def test_age_to_thresh_logistic_formula():
    age = 50.0
    cir = convert_age_to_cir(age, pop_prev=0.1)
    assert convert_age_to_thresh(age, pop_prev=0.1) == pytest.approx(stats.norm.isf(cir))


def test_thresh_of_aoo_recovers_liability():
    # The onset-to-threshold map shared by LT-FH++ and ADuLT is invertible:
    # thresh(aoo(liab)) == liab for a case liability above the base threshold.
    liab = np.array([1.5, 2.0, 2.7, 3.2])
    aoo = convert_liability_to_aoo(liab, pop_prev=0.1)
    recovered = convert_age_to_thresh(aoo, pop_prev=0.1)
    assert np.allclose(recovered, liab, atol=1e-8)


def test_prevalence_thresholds():
    status = np.array([True, False, True, False])
    lower, upper = prevalence_thresholds(status, pop_prev=0.05)
    t = float(liability_threshold(0.05))
    assert np.array_equal(lower, np.where(status, t, -np.inf))
    assert np.array_equal(upper, np.where(status, np.inf, t))


@pytest.mark.parametrize(
    "helper,args",
    [
        (convert_age_to_cir, ([50.0],)),
        (convert_age_to_thresh, ([50.0],)),
        (convert_liability_to_aoo, ([2.0],)),
        (prevalence_thresholds, ([0, 1],)),
        (age_thresholds, ([0, 1], [40.0, 70.0])),
        (pa_thresholds, ([0, 1], [40.0, 70.0])),
    ],
)
def test_public_threshold_helpers_require_explicit_prevalence(helper, args):
    with pytest.raises(TypeError, match="pop_prev"):
        helper(*args)


@pytest.mark.parametrize(
    "status",
    [np.array([False, True]), np.array([0, 1]), np.array([0.0, 1.0])],
)
def test_threshold_builders_accept_boolean_or_exact_binary_status(status):
    age = np.array([40.0, 70.0])
    cip_ages = np.array([0.0, 100.0])
    cip_values = np.array([0.0, 0.2])
    expected = np.array([False, True])

    lower, _ = prevalence_thresholds(status, pop_prev=0.1)
    assert np.array_equal(np.isfinite(lower), expected)
    lower, _ = age_thresholds(status, age, pop_prev=0.1)
    assert np.array_equal(np.isfinite(lower), expected)
    lower, _, _, _ = pa_thresholds(status, age, pop_prev=0.1)
    assert np.array_equal(np.isfinite(lower), expected)
    lower, _, _, _ = thresholds_from_cip(
        status, age, cip_ages, cip_values)
    assert np.array_equal(np.isfinite(lower), expected)


@pytest.mark.parametrize(
    "status",
    [np.array([0.0, np.nan]), [-9, 0], [0, 2], [[0, 1]], 1, ["0", "1"]],
)
def test_threshold_builders_reject_invalid_status_codes_and_shapes(status):
    n = max(1, np.asarray(status).size)
    age = np.linspace(40.0, 70.0, n)
    builders = (
        lambda: prevalence_thresholds(status, pop_prev=0.1),
        lambda: age_thresholds(status, age, pop_prev=0.1),
        lambda: pa_thresholds(status, age, pop_prev=0.1),
        lambda: thresholds_from_cip(
            status, age, [0.0, 100.0], [0.0, 0.2]),
    )
    for build in builders:
        with pytest.raises(ValueError, match="status"):
            build()


def test_thresholds_from_cip_matches_manual():
    # an empirical CIP curve; interpolate and threshold
    cip_ages = np.array([0, 40, 80])
    cip_values = np.array([0.0, 0.05, 0.10])          # k_pop = 0.10
    status = np.array([1, 0])
    age = np.array([40, 40])
    lo, up, ki, kp = thresholds_from_cip(status, age, cip_ages, cip_values,
                                         case_mode="interval")
    # CIP at age 40 is 0.05 -> threshold Phi^-1(1 - 0.05)
    t40 = stats.norm.isf(0.05)
    assert up[1] == pytest.approx(t40)                # control upper
    assert lo[0] == pytest.approx(t40)                # case lower (interval)
    assert up[0] == np.inf
    assert np.isnan(ki[0]) and ki[1] == pytest.approx(0.05)
    assert kp[1] == pytest.approx(0.10)


def test_thresholds_from_cip_pin_mode_and_validation():
    # Onset-pinned cases (used by LT-FH++ and ADuLT) are the default.
    lo, up, _, _ = thresholds_from_cip([1], [50], [0, 100], [0.0, 0.2])
    assert lo[0] == up[0]                              # pinned case
    with pytest.raises(ValueError, match="strictly increasing"):
        thresholds_from_cip([1], [50], [100, 0], [0.2, 0.0])   # unsorted ages
    with pytest.raises(ValueError):
        thresholds_from_cip([1], [50], [0, 100], [0.0, 0.2], case_mode="bogus")
    with pytest.raises(ValueError):                    # decreasing CIP
        thresholds_from_cip([1], [50], [0, 100], [0.2, 0.1])
    with pytest.raises(ValueError, match="below max"):
        thresholds_from_cip([0], [50], [0, 100], [0.0, 0.2], k_pop=0.1)


@pytest.mark.parametrize(
    "status,age,cip_ages,cip_values,k_pop,min_cip,match",
    [
        ([0, 1], [50], [0, 100], [0.0, 0.2], None, 1e-5, "equal length"),
        ([0], [50], [0, 50], [0.0], None, 1e-5, "equal length"),
        ([0], [50], [0, 0], [0.0, 0.1], None, 1e-5, "strictly increasing"),
        ([0], [np.nan], [0, 100], [0.0, 0.2], None, 1e-5, "finite"),
        ([0], [50], [0, 100], [0.0, np.nan], None, 1e-5, "finite"),
        ([0], [50], [0, 100], [-0.1, 0.2], None, 1e-5, r"\[0, 1\)"),
        ([0], [50], [0, 100], [0.0, 1.0], None, 1e-5, r"\[0, 1\)"),
        ([0], [50], [0, 100], [0.0, 0.2], 1.0, 1e-5, r"\(0, 1\)"),
        ([0], [50], [0, 100], [0.0, 0.2], None, 0.0, r"\(0, 1\)"),
        ([0], [50], [0, 100], [0.0, 0.2], 0.2, 0.3, "must not exceed"),
    ],
)
def test_thresholds_from_cip_rejects_invalid_inputs(
        status, age, cip_ages, cip_values, k_pop, min_cip, match):
    with pytest.raises(ValueError, match=match):
        thresholds_from_cip(status, age, cip_ages, cip_values,
                            k_pop=k_pop, min_cip=min_cip)


def test_age_thresholds_case_pinned_control_open():
    status = np.array([True, False])
    age = np.array([40.0, 70.0])
    lower, upper = age_thresholds(status, age, pop_prev=0.1)
    # case pinned (lower == upper), control open below its age threshold
    assert lower[0] == upper[0]
    assert lower[1] == -np.inf and np.isfinite(upper[1])


def test_age_thresholds_broadcast_scalar_age_to_status_shape():
    lower, upper = age_thresholds([1, 0], 50.0, pop_prev=0.1)
    threshold = float(convert_age_to_thresh(50.0, pop_prev=0.1))
    assert lower.shape == upper.shape == (2,)
    assert np.array_equal(upper, [threshold, threshold])
    assert upper.flags.writeable


@pytest.mark.parametrize("prev", [0.0, 1.0, -0.1, 1.5])
def test_simple_threshold_helpers_reject_degenerate_prevalence(prev):
    # liability_threshold(0) = +inf and (1) = -inf flowed silently into bounds
    # (a K=0 "case" pin at (inf, inf) reaches the sampler); the logistic age
    # helpers turned out-of-range prevalences into silent NaN thresholds
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        liability_threshold(prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        convert_age_to_cir(50, pop_prev=prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        convert_age_to_thresh(50, pop_prev=prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        convert_liability_to_aoo(1.5, pop_prev=prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        prevalence_thresholds([1, 0], pop_prev=prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        age_thresholds([1, 0], [40, 70], pop_prev=prev)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        pa_thresholds([1, 0], [40, 70], pop_prev=prev)


def test_liability_threshold_validates_elementwise():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        liability_threshold(np.array([0.05, 1.5]))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        liability_threshold(np.array([-0.2, 0.05]))
    # valid arrays still map elementwise
    t = liability_threshold(np.array([0.05, 0.10]))
    assert np.allclose(t, [stats.norm.isf(0.05), stats.norm.isf(0.10)])


@pytest.mark.parametrize("K", [1e-8, 1e-12, 1e-16, 1e-17, 1e-100, 1e-300])
def test_liability_threshold_survives_the_far_tail(K):
    # liability_threshold used to compute norm_ppf(1.0 - K); that subtraction
    # discards the tail below ~1e-16 and returns +inf for K <= 1.1e-16, even
    # though _validate_pop_prev admits the whole open interval (0, 1). The
    # identity Phi^-1(1-K) = -Phi^-1(K) is exact and cancellation-free.
    got = float(liability_threshold(K))
    assert np.isfinite(got)
    assert got == pytest.approx(float(stats.norm.isf(K)), rel=1e-12)


def test_liability_threshold_unchanged_in_the_ordinary_range():
    # ...and the rewrite must not move any value users actually see.
    for K in (0.5, 0.1, 0.05, 0.01, 0.001, 1e-4):
        assert float(liability_threshold(K)) == pytest.approx(
            float(stats.norm.isf(K)), rel=1e-15)


def test_age_thresholds_finite_for_young_ages_at_legal_parameters():
    # pop_prev=0.1 with slope=1.0 is a legal combination, but the old
    # 1.0 - cir cancellation made the implied CIR underflow at young ages, so
    # a case pinned at age 20 got the degenerate bound pair (inf, inf).
    ages = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    thr = convert_age_to_thresh(ages, pop_prev=0.1, slope=1.0)
    assert np.all(np.isfinite(thr))
    assert np.all(np.diff(thr) < 0)          # older -> lower threshold

    lower, upper = age_thresholds(np.array([1, 0]), np.array([20.0, 20.0]),
                                  pop_prev=0.1, slope=1.0)
    assert np.all(np.isfinite(lower[:1]))    # the case pin
    assert np.isfinite(upper[0]) and np.isfinite(upper[1])
