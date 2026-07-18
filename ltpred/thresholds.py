"""Turning case/control status, age and prevalence into liability thresholds.

Under the liability-threshold model a person is a case when their full liability
exceeds a threshold ``T``. With a single population prevalence ``K`` that
threshold is ``T = Phi^-1(1 - K)``. Personalised models instead derive a person's
threshold from cumulative incidence at their age, birth year and sex. A case can
be pinned at the threshold matching their age of onset; a control's liability is
bounded above by the threshold at their current age.

Those bounds do not, by themselves, name the model. They form **LT-FH++** when
the proband and relatives are conditioned on together, and **ADuLT** when only
the proband is used (no family history). This module ports LTFHPlus's ``convert_*``
helpers and provides convenience builders that produce the ``(lower, upper)``
bounds consumed by either inference engine.
"""

from __future__ import annotations

import numpy as np

from ._mathfun import norm_cdf, norm_ppf

__all__ = ["convert_age_to_cir", "convert_cir_to_age", "convert_age_to_thresh",
           "convert_liability_to_aoo", "truncated_normal_cdf",
           "convert_observed_to_liability_scale", "prevalence_thresholds",
           "age_thresholds", "liability_threshold", "pa_thresholds",
           "thresholds_from_cip"]


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
    """Age-dependent, onset-pinned bounds from status and age.

    A case is pinned at its onset threshold (``lower = upper = thresh(age_of_onset)``,
    a point mass); a control lies below the threshold for its current age
    (``lower = -inf``, ``upper = thresh(current_age)``). ``age`` is the age of onset
    for cases and the current/censoring age for controls. Returns ``(lower, upper)``
    arrays ready for :func:`ltpred.estimate.estimate_liability`.

    This helper uses one logistic CIP curve and is mainly for simulation and
    tutorials. For full LT-FH++, use :func:`thresholds_from_cip` with age-, birth-
    year- and sex-specific curves and include relatives. The same personalised
    construction with proband rows only is ADuLT."""
    status = np.asarray(status, dtype=bool)
    thr = convert_age_to_thresh(age, dist="logistic", pop_prev=pop_prev,
                                mid_point=mid_point, slope=slope)
    thr = np.asarray(thr, dtype=float)
    lower = np.where(status, thr, -np.inf)
    upper = np.where(status, thr, thr)
    return lower, upper


def pa_thresholds(status, age, pop_prev=0.1, mid_point=60.0, slope=1.0 / 8.0):
    """PA-FGRS inputs from status and age: ``(lower, upper, K_i, K_pop)``.

    The name refers to the PA-FGRS observation model, not to selecting the
    Pearson-Aitken inference engine. To infer onset-pinned LT-FH++ or ADuLT with
    Pearson-Aitken, use :func:`age_thresholds` or
    :func:`thresholds_from_cip` with ``case_mode="pin"`` and pass the resulting
    families to the default estimator.

    A case gets ``(thresh(age_of_onset), inf)`` and no mixture (``K_i = K_pop =
    nan``). An age-censored control gets ``(-inf, thresh(current_age))`` together
    with its cumulative incidence ``K_i = cir(current_age)`` and the lifetime
    prevalence ``K_pop = pop_prev``. Feed the result to
    :func:`ltpred.estimate.estimate_liability` with ``method="pearson-aitken"``.

    The control ``upper`` is the *age-specific* threshold ``Phi^-1(1 - K_i)``, and
    how the estimator reads it depends on ``use_mixture``:

    * ``use_mixture=False`` -- the supplied intervals are used directly. Controls
      have the same age-specific upper bound as :func:`age_thresholds`, but cases
      remain intervals rather than the point pins used by LT-FH++ and ADuLT; the
      encodings are not equivalent.
    * ``use_mixture=True`` -- the PA-FGRS censored-control correction switches on. It
      does the age adjustment itself, from ``K_i``/``K_pop``, by splitting the
      control's liability at the *lifetime* threshold ``Phi^-1(1 - K_pop)`` into a
      genuine-control and a not-yet-onset-case component (Krebs et al. 2024, eqs.
      S3-S5). The age-specific ``upper`` is then only a censored-vs-observed flag --
      its value does not re-enter, so the mixture does **not** double-count the
      censoring already implicit in the age bound.

    Either way the result is well calibrated; the two are near-identical estimators
    of the same generative model."""
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


def thresholds_from_cip(status, age, cip_ages, cip_values, k_pop=None,
                        case_mode="pin", min_cip=1e-5):
    """Liability bounds (+ ``K_i``, ``K_pop``) from an *empirical* CIP curve.

    The production alternative to the logistic ``age_thresholds`` /
    ``pa_thresholds``: instead of the built-in logistic incidence, pass a
    population-representative cumulative-incidence curve as ``cip_ages`` (ascending)
    and ``cip_values`` (CIP at each age). Call once **per stratum** (sex, birth
    year, ancestry, ...) with that stratum's curve. Each person's CIP is linearly
    interpolated at their ``age`` and clipped to ``[min_cip, k_pop]`` before being
    turned into a threshold ``Phi^-1(1 - CIP)``. ``numpy.interp`` holds the first
    or last CIP constant outside the supplied age grid; it does not extrapolate
    incidence, so the grid should cover all analysed onset/follow-up ages.

    ``k_pop`` is the lifetime prevalence for the stratum. It defaults to
    ``max(cip_values)``, which is appropriate only when the curve reaches the
    intended lifetime horizon; otherwise pass a separately justified value.
    ``case_mode`` sets the case encoding: ``"pin"`` (the
    default) pins a case at ``thresh(age_of_onset)`` (the encoding used by LT-FH++
    with family history and ADuLT without it), while ``"interval"`` uses
    ``(thresh(age_of_onset), inf)`` (the PA-FGRS encoding). Controls are always
    ``(-inf, thresh(current_age))`` and carry ``K_i`` / ``K_pop`` for the PA
    censored-control mixture. Returns ``(lower, upper, K_i, K_pop)``."""
    status = np.asarray(status, dtype=bool)
    age = np.asarray(age, dtype=float)
    cip_ages = np.asarray(cip_ages, dtype=float)
    cip_values = np.asarray(cip_values, dtype=float)
    if status.ndim != 1 or age.ndim != 1 or status.shape != age.shape:
        raise ValueError("status and age must be one-dimensional arrays of equal length")
    if cip_ages.ndim != 1 or cip_values.ndim != 1 or cip_ages.size == 0:
        raise ValueError("cip_ages and cip_values must be non-empty one-dimensional arrays")
    if cip_ages.shape != cip_values.shape:
        raise ValueError("cip_ages and cip_values must have equal length")
    if not np.all(np.isfinite(age)) or not np.all(np.isfinite(cip_ages)):
        raise ValueError("age and cip_ages must contain only finite values")
    if not np.all(np.isfinite(cip_values)):
        raise ValueError("cip_values must contain only finite values")
    if np.any(np.diff(cip_ages) <= 0):
        raise ValueError("cip_ages must be strictly increasing")
    if np.any(np.diff(cip_values) < 0):
        raise ValueError("cip_values must be non-decreasing (a cumulative incidence)")
    if np.any((cip_values < 0.0) | (cip_values >= 1.0)):
        raise ValueError("cip_values must lie in [0, 1)")
    if case_mode not in ("pin", "interval"):
        raise ValueError("case_mode must be 'pin' or 'interval'")
    if not np.isfinite(min_cip) or not 0.0 < min_cip < 1.0:
        raise ValueError("min_cip must lie in (0, 1)")
    cip_max = float(np.max(cip_values))
    kpop = cip_max if k_pop is None else float(k_pop)
    if not np.isfinite(kpop) or not 0.0 < kpop < 1.0:
        raise ValueError("k_pop must lie in (0, 1)")
    if min_cip > kpop:
        raise ValueError("min_cip must not exceed k_pop")
    if kpop < cip_max:
        raise ValueError(
            f"k_pop ({kpop}) is below max(cip_values) ({cip_max}); lifetime "
            "prevalence must be at least the largest cumulative incidence")

    cip = np.interp(age, cip_ages, cip_values)          # CIP at each person's age
    cip = np.clip(cip, min_cip, kpop)
    thr = norm_ppf(1.0 - cip)

    lower = np.where(status, thr, -np.inf)
    if case_mode == "pin":
        upper = np.where(status, thr, thr)
    else:
        upper = np.where(status, np.inf, thr)
    K_i = np.where(status, np.nan, cip)
    K_pop = np.where(status, np.nan, kpop)
    return lower, upper, K_i, K_pop
