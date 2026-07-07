"""Age / prevalence / liability conversions."""

import numpy as np
import pytest
from scipy import stats

from ltpred.thresholds import (convert_age_to_cir, convert_cir_to_age,
                               convert_age_to_thresh, convert_liability_to_aoo,
                               truncated_normal_cdf,
                               convert_observed_to_liability_scale,
                               prevalence_thresholds, age_thresholds,
                               liability_threshold)


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
    back = convert_cir_to_age(cir, pop_prev=0.1)
    assert np.allclose(back, ages, atol=1e-8)


def test_cir_to_age_nan_above_prevalence():
    assert np.isnan(convert_cir_to_age(0.2, pop_prev=0.1))


def test_age_to_thresh_logistic_formula():
    age = 50.0
    cir = convert_age_to_cir(age, pop_prev=0.1)
    assert convert_age_to_thresh(age, pop_prev=0.1) == pytest.approx(stats.norm.isf(cir))


def test_thresh_of_aoo_recovers_liability():
    # convert_age_to_thresh and convert_liability_to_aoo are inverses (ADuLT):
    # thresh(aoo(liab)) == liab for a case liability above the base threshold.
    liab = np.array([1.5, 2.0, 2.7, 3.2])
    aoo = convert_liability_to_aoo(liab, pop_prev=0.1)
    recovered = convert_age_to_thresh(aoo, pop_prev=0.1)
    assert np.allclose(recovered, liab, atol=1e-8)


def test_truncated_normal_cdf_at_bounds():
    lo = stats.norm.isf(0.05)
    assert truncated_normal_cdf(lo, lower=lo, upper=np.inf) == pytest.approx(0.0)
    hi = 3.0
    assert truncated_normal_cdf(hi, lower=lo, upper=hi) == pytest.approx(1.0)


def test_observed_to_liability_matches_lee():
    obs_h2, k, p = 0.2, 0.01, 0.5
    z = stats.norm.pdf(stats.norm.isf(k))
    expected = obs_h2 * (k * (1 - k) / z ** 2) * (k * (1 - k)) / (p * (1 - p))
    assert convert_observed_to_liability_scale(obs_h2, k, p) == pytest.approx(expected)


def test_observed_to_liability_no_ascertainment():
    obs_h2, k = 0.2, 0.05
    z = stats.norm.pdf(stats.norm.isf(k))
    expected = obs_h2 * (k * (1 - k) / z ** 2)
    assert convert_observed_to_liability_scale(obs_h2, k, None) == pytest.approx(expected)


def test_prevalence_thresholds():
    status = np.array([True, False, True, False])
    lower, upper = prevalence_thresholds(status, pop_prev=0.05)
    t = float(liability_threshold(0.05))
    assert np.array_equal(lower, np.where(status, t, -np.inf))
    assert np.array_equal(upper, np.where(status, np.inf, t))


def test_age_thresholds_case_pinned_control_open():
    status = np.array([True, False])
    age = np.array([40.0, 70.0])
    lower, upper = age_thresholds(status, age, pop_prev=0.1)
    # case pinned (lower == upper), control open below its age threshold
    assert lower[0] == upper[0]
    assert lower[1] == -np.inf and np.isfinite(upper[1])
