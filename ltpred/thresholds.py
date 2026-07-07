"""Turning case/control status, age and prevalence into liability thresholds.

Under the liability-threshold model a person is a case when their full liability
exceeds a threshold ``T``. With a single population prevalence ``K`` that
threshold is ``T = Phi^-1(1 - K)``. LT-FH++ makes it *age-dependent*: the
cumulative incidence rises with age along a logistic curve, so each age maps to
its own threshold. A case is pinned at the threshold matching its age of onset
(the ADuLT model); a control's liability lies below the threshold for its current
age. This module ports LTFHPlus's ``convert_*`` helpers and adds two convenience
builders, :func:`prevalence_thresholds` (classic LT-FH) and
:func:`age_thresholds` (LT-FH++/ADuLT), that produce the ``(lower, upper)`` bounds
the estimator consumes.
"""

from __future__ import annotations

import numpy as np

from ._mathfun import norm_cdf, norm_ppf

__all__ = ["convert_age_to_cir", "convert_cir_to_age", "convert_age_to_thresh",
           "convert_liability_to_aoo", "truncated_normal_cdf",
           "convert_observed_to_liability_scale", "prevalence_thresholds",
           "age_thresholds", "liability_threshold", "pa_thresholds"]


def liability_threshold(pop_prev):
    """Single-prevalence liability threshold ``T = Phi^-1(1 - K)``."""
    return norm_ppf(1.0 - np.asarray(pop_prev, dtype=float))


def convert_age_to_cir(age, pop_prev=0.1, mid_point=60.0, slope=1.0 / 8.0):
    """Cumulative incidence rate at a given age (logistic curve).

    ``cir(age) = pop_prev / (1 + exp((mid_point - age) * slope))`` -- incidence
    climbs from ~0 at young ages toward ``pop_prev`` at old ages, passing half-way
    at ``mid_point``. Vectorised over ``age``."""
    age = np.asarray(age, dtype=float)
    return pop_prev / (1.0 + np.exp((mid_point - age) * slope))


def convert_cir_to_age(cir, pop_prev=0.1, mid_point=60.0, slope=1.0 / 8.0):
    """Invert :func:`convert_age_to_cir`: the age at a cumulative incidence ``cir``.

    ``mid_point - log(pop_prev/cir - 1) / slope``, clamped at 0. Returns ``nan``
    where ``cir >= pop_prev`` (that incidence is never reached). Vectorised."""
    cir = np.asarray(cir, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        age = mid_point - np.log(pop_prev / cir - 1.0) / slope
    age = np.where(cir >= pop_prev, np.nan, age)
    return np.maximum(age, 0.0)


def convert_age_to_thresh(age, dist="logistic", pop_prev=0.1, mid_point=60.0,
                          slope=1.0 / 8.0, min_age=10.0, max_age=90.0,
                          lower=None, upper=np.inf):
    """Liability threshold implied by an age (or age of onset).

    With ``dist="logistic"`` the threshold is ``Phi^-1(1 - cir(age))`` where
    ``cir`` is :func:`convert_age_to_cir`, so a younger onset (lower incidence)
    gives a higher threshold, i.e. a more extreme liability. With ``dist="normal"``
    the threshold instead interpolates linearly in age between the truncated-normal
    bounds ``lower``/``upper``. Vectorised over ``age``. Port of
    LTFHPlus::convert_age_to_thresh."""
    age = np.asarray(age, dtype=float)
    if lower is None:
        lower = norm_ppf(0.95)
    if dist == "logistic":
        cir = convert_age_to_cir(age, pop_prev=pop_prev, mid_point=mid_point,
                                 slope=slope)
        return norm_ppf(1.0 - cir)
    if dist == "normal":
        frac = (1.0 - (age - min_age) / max_age)
        return norm_ppf(frac * (norm_cdf(upper) - norm_cdf(lower)) + norm_cdf(lower))
    raise ValueError("dist must be 'logistic' or 'normal'")


def truncated_normal_cdf(liability, lower=None, upper=np.inf):
    """CDF of a standard normal truncated to ``(lower, upper)``, at ``liability``.

    ``(Phi(l) - Phi(lower)) / (Phi(upper) - Phi(lower))`` -- the probability a
    truncated-normal draw falls at or below ``liability``. Vectorised over
    ``liability``. Port of LTFHPlus::truncated_normal_cdf."""
    if lower is None:
        lower = norm_ppf(0.95)
    liability = np.asarray(liability, dtype=float)
    return (norm_cdf(liability) - norm_cdf(lower)) / (norm_cdf(upper) - norm_cdf(lower))


def convert_liability_to_aoo(liability, dist="logistic", pop_prev=0.1,
                             mid_point=60.0, slope=1.0 / 8.0, min_aoo=10.0,
                             max_aoo=90.0, lower=None, upper=np.inf):
    """Age of onset implied by a case's true liability.

    Higher liability -> earlier onset. With ``dist="logistic"`` this is
    :func:`convert_cir_to_age` applied to the incidence ``1 - Phi(liability)``;
    with ``dist="normal"`` it maps the truncated-normal CDF linearly onto
    ``[min_aoo, min_aoo + max_aoo]``. Vectorised. Port of
    LTFHPlus::convert_liability_to_aoo."""
    liability = np.asarray(liability, dtype=float)
    if lower is None:
        lower = norm_ppf(0.95)
    if dist == "logistic":
        cir = 1.0 - norm_cdf(liability)
        return convert_cir_to_age(cir, pop_prev=pop_prev, mid_point=mid_point,
                                  slope=slope)
    if dist == "normal":
        return (1.0 - truncated_normal_cdf(liability, lower=lower, upper=upper)) * max_aoo + min_aoo
    raise ValueError("dist must be 'logistic' or 'normal'")


def convert_observed_to_liability_scale(obs_h2=0.5, pop_prev=0.05, prop_cases=0.5):
    """Rescale an observed-scale heritability to the liability scale (Lee et al. 2011).

    Multiplies by ``K(1-K)/z^2`` where ``z = phi(Phi^-1(1-K))``; when ``prop_cases``
    is given (ascertained case/control study) an extra ``K(1-K)/(P(1-P))`` factor
    corrects for the case oversampling. Vectorised over the inputs. Port of
    LTFHPlus::convert_observed_to_liability_scale."""
    obs_h2 = np.asarray(obs_h2, dtype=float)
    pop_prev = np.asarray(pop_prev, dtype=float)
    t = norm_ppf(1.0 - pop_prev)
    z = np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)  # phi(t)
    factor = pop_prev * (1.0 - pop_prev) / (z * z)
    if prop_cases is None:
        return obs_h2 * factor
    prop_cases = np.asarray(prop_cases, dtype=float)
    return obs_h2 * factor * (pop_prev * (1.0 - pop_prev)) / (prop_cases * (1.0 - prop_cases))


def prevalence_thresholds(status, pop_prev=0.1):
    """Classic LT-FH bounds from binary status and a single prevalence.

    Cases get ``(T, inf)`` and controls ``(-inf, T)`` with ``T = Phi^-1(1 - K)``.
    ``status`` is a boolean/0-1 array. Returns ``(lower, upper)`` arrays -- no age
    information, the special case ``LT-FH`` before ``++``."""
    status = np.asarray(status, dtype=bool)
    t = float(liability_threshold(pop_prev))
    lower = np.where(status, t, -np.inf)
    upper = np.where(status, np.inf, t)
    return lower, upper


def age_thresholds(status, age, pop_prev=0.1, mid_point=60.0, slope=1.0 / 8.0):
    """LT-FH++/ADuLT bounds from status and age (of onset for cases).

    A case is pinned at its onset threshold (``lower = upper = thresh(age_of_onset)``,
    a point mass); a control lies below the threshold for its current age
    (``lower = -inf``, ``upper = thresh(current_age)``). ``age`` is the age of onset
    for cases and the current/censoring age for controls. Returns ``(lower, upper)``
    arrays ready for :func:`ltpred.estimate.estimate_liability`. This is the
    construction LTFHPlus builds internally in ``construct_thresholds``."""
    status = np.asarray(status, dtype=bool)
    thr = convert_age_to_thresh(age, dist="logistic", pop_prev=pop_prev,
                                mid_point=mid_point, slope=slope)
    thr = np.asarray(thr, dtype=float)
    lower = np.where(status, thr, -np.inf)
    upper = np.where(status, thr, thr)
    return lower, upper


def pa_thresholds(status, age, pop_prev=0.1, mid_point=60.0, slope=1.0 / 8.0):
    """PA-FGRS inputs from status and age: ``(lower, upper, K_i, K_pop)``.

    A case gets ``(thresh(age_of_onset), inf)`` and no mixture (``K_i = K_pop =
    nan``). An age-censored control gets ``(-inf, thresh(current_age))`` together
    with its cumulative incidence ``K_i = cir(current_age)`` and the lifetime
    prevalence ``K_pop = pop_prev`` -- the two numbers the Pearson-Aitken mixture
    uses to correct for the control not yet having passed through their full risk
    period. Feed the result to :func:`ltpred.estimate.estimate_liability` with
    ``method="pearson-aitken"``."""
    status = np.asarray(status, dtype=bool)
    age = np.asarray(age, dtype=float)
    thr = np.asarray(convert_age_to_thresh(age, dist="logistic", pop_prev=pop_prev,
                                           mid_point=mid_point, slope=slope), dtype=float)
    cir = np.asarray(convert_age_to_cir(age, pop_prev=pop_prev, mid_point=mid_point,
                                        slope=slope), dtype=float)
    lower = np.where(status, thr, -np.inf)
    upper = np.where(status, np.inf, thr)
    K_i = np.where(status, np.nan, cir)
    K_pop = np.where(status, np.nan, float(pop_prev))
    return lower, upper, K_i, K_pop
