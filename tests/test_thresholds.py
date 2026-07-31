"""Age / prevalence / liability conversions."""

import numpy as np
import pytest
from scipy import stats

from ltpred.thresholds import (convert_age_to_cir, _convert_cir_to_age,
                               convert_age_to_thresh, convert_liability_to_aoo,
                               prevalence_thresholds, age_thresholds,
                               liability_threshold, thresholds_from_cip)


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
