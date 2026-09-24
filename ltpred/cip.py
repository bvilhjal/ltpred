"""Cumulative-incidence (CIP) estimation from registry-style health data.

The personalised thresholds of LT-FH++ / ADuLT / PA-FGRS consume a
cumulative-incidence curve per stratum (sex x birth cohort) through
`ltpred.thresholds.thresholds_from_cip`. This module estimates that curve
from individual follow-up records: per person an entry age (start of
follow-up), an exit age, and what happened at exit.

Two estimators, both handling **left truncation** (delayed entry: register
starts mid-life, immigration) and **right censoring** (emigration, end of
follow-up). Delayed entry must be independent of the event process conditional
on the modelled strata, with overlapping risk-set support; right censoring
must likewise be conditionally non-informative:

* `kaplan_meier_cip` -- the product-limit estimator ``CIP = 1 - S`` for
  the single-event setting. Under independent censoring it estimates net risk:
  incidence in a hypothetical world without the censoring process. Death must
  not be treated as censoring when the target is the crude diagnosed
  proportion, even if death and diagnosis are independent. Greenwood standard
  errors.
* `aalen_johansen_cip` -- the Aalen--Johansen estimator of the
  cause-specific cumulative incidence with competing risks (diagnosis vs
  death without diagnosis). This is the estimator LT-FH++ used for its CIPs
  (Pedersen et al. 2022: Aalen-Johansen with death and emigration as
  competing events, one curve per sex x birth-year stratum) and the standard
  counting-process practice of the Danish register literature (Andersen,
  Borgan, Gill & Keiding 1993; e.g. Beck et al. 2024, *Acta Psychiatr
  Scand*). Pointwise SEs use the finite-risk-set, tie-correct Aalen (1978)
  variance (the ``cmprsk::cuminc`` convention).

Risk-set convention (counting processes; Andersen, Borgan, Gill & Keiding,
*Statistical Models Based on Counting Processes*, 1993): person ``i`` is at
risk at age ``t`` when ``age_entry_i < t <= age_exit_i``. Events and censorings
occur at ``age_exit``; zero-length follow-ups (``exit == entry``) never enter a
risk set. Ages may be continuous; ties are handled by grouping on unique exit
ages. Stratification (e.g. by sex x birth-year bands) is done by calling an
estimator once per stratum -- see the worked example in the docstring of each
estimator and ``benchmarks/bench_cip_estimation.py``.

The returned `CipCurve` carries ascending ``ages``, non-decreasing
``values`` in ``[0, 1]``, pointwise standard errors and event counts.
`ltpred.thresholds.thresholds_from_cip` consumes a non-degenerate curve
only (all values below one). If a terminal event exhausts the empirical risk
set, the valid estimate is exactly one but cannot define a finite probit
threshold; choose a scientifically justified earlier/lifetime horizon rather
than silently clipping it. ``thresholds_from_cip`` takes ``k_pop``
separately: pass the curve's value at the intended lifetime horizon, or a
separately justified estimate.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import validate_binary

__all__ = ["CipCurve", "kaplan_meier_cip", "aalen_johansen_cip"]


@dataclass
class CipCurve:
    """An estimated cumulative-incidence curve.

    ``ages`` are ascending and ``values`` non-decreasing in ``[0, 1]``.
    Values strictly below one feed
    `ltpred.thresholds.thresholds_from_cip`; an exact terminal one is a
    valid estimate but has no finite probit threshold. ``se`` is the
    pointwise standard error (Greenwood for Kaplan-Meier, the finite-risk-set,
    tie-correct Aalen (1978) variance for Aalen-Johansen); ``n_events`` counts
    events of interest; ``n_entered`` the number of persons contributing any
    follow-up; ``estimator`` records which estimator produced the curve."""
    ages: np.ndarray
    values: np.ndarray
    se: np.ndarray
    n_events: int
    n_entered: int
    estimator: str


def _validate_followup(age_entry, age_exit):
    age_entry = np.asarray(age_entry, dtype=float)
    age_exit = np.asarray(age_exit, dtype=float)
    if age_entry.ndim != 1 or age_exit.ndim != 1 or age_entry.shape != age_exit.shape:
        raise ValueError("age_entry and age_exit must be 1-D arrays of equal length")
    if age_entry.size == 0:
        raise ValueError("need at least one person")
    if not np.all(np.isfinite(age_entry)) or not np.all(np.isfinite(age_exit)):
        raise ValueError("ages must contain only finite values")
    if np.any(age_entry < 0):
        raise ValueError("age_entry must be nonnegative")
    if np.any(age_exit < age_entry):
        raise ValueError("age_exit must not precede age_entry")
    return age_entry, age_exit


def _risk_sets(age_entry, age_exit, grid):
    """At-risk counts Y(t) = #{entry < t <= exit} on a sorted grid."""
    se = np.sort(age_entry)
    sx = np.sort(age_exit)
    entered = np.searchsorted(se, grid, side="left")   # count(entry < t)
    gone = np.searchsorted(sx, grid, side="left")      # count(exit < t)
    return entered - gone


def _count_at(sorted_vals, grid):
    """Number of values exactly on each grid point (both sorted)."""
    lo = np.searchsorted(sorted_vals, grid, side="left")
    hi = np.searchsorted(sorted_vals, grid, side="right")
    return hi - lo


def kaplan_meier_cip(age_entry: ArrayLike, age_exit: ArrayLike,
                     is_event: ArrayLike) -> CipCurve:
    """Kaplan-Meier cumulative incidence ``1 - S(t)`` with left truncation.

    Parameters
    ----------
    age_entry, age_exit
        Per-person ages at start and end of follow-up (1-D, equal length,
        ``exit >= entry``).
    is_event
        Boolean (or 0/1) per person: True if follow-up ended with the event of
        interest (diagnosis), False if censored (emigration, administrative
        end, death -- but see the note).

    Returns
    -------
    CipCurve
        ``CIP = 1 - S`` at each unique event age, with Greenwood pointwise
        standard errors.

    Notes
    -----
    Under independent censoring this estimates net risk: incidence in a
    hypothetical world without the censoring process. When the target is the
    proportion actually diagnosed, death before diagnosis is a competing
    event rather than censoring, irrespective of whether its time is
    statistically independent of diagnosis. Use `aalen_johansen_cip`
    with death as the competing event; the two estimands are compared in
    ``benchmarks/bench_cip_estimation.py``.
    """
    age_entry, age_exit = _validate_followup(age_entry, age_exit)
    is_event_raw = np.asarray(is_event)
    if is_event_raw.shape != age_entry.shape:
        raise ValueError("is_event must match the follow-up arrays' length")
    is_event = validate_binary(is_event_raw, name="is_event", ndim=1)
    if np.any(is_event & (age_exit == age_entry)):
        raise ValueError("zero-length follow-up cannot carry an event")

    grid = np.unique(age_exit[is_event])
    if grid.size == 0:
        raise ValueError("no events of interest in the data")
    Y = _risk_sets(age_entry, age_exit, grid)
    d = _count_at(np.sort(age_exit[is_event]), grid).astype(float)
    if np.any(Y <= 0):
        raise ValueError("empty risk set at an event age (check follow-up data)")
    if np.any(d > Y):
        raise ValueError(
            "event count exceeds the risk set at an event age "
            "(zero-length follow-up cannot carry an event)")

    surv = np.cumprod(1.0 - d / Y)
    cip = 1.0 - surv
    # Greenwood: var(S) = S^2 * sum d / (Y (Y - d)); where Y == d the remaining
    # survival is 0 and so is the variance.
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(Y > d, d / (Y * (Y - d)), 0.0)
    var = surv ** 2 * np.cumsum(contrib)
    return CipCurve(ages=grid, values=cip, se=np.sqrt(var),
                    n_events=int(d.sum()),
                    n_entered=int(np.sum(age_exit > age_entry)),
                    estimator="kaplan-meier")


def aalen_johansen_cip(age_entry: ArrayLike, age_exit: ArrayLike,
                       event_type: ArrayLike, cause: int = 1) -> CipCurve:
    """Aalen-Johansen cumulative incidence with competing risks.

    Parameters
    ----------
    age_entry, age_exit
        Per-person follow-up ages (as in `kaplan_meier_cip`).
    event_type
        Integer code per person: 0 = censored, ``cause`` = event of interest
        (default 1, diagnosis), any other positive code = competing event
        (e.g. 2 = death without diagnosis).
    cause
        The event code to estimate the cumulative incidence for.

    Returns
    -------
    CipCurve
        Cause-specific cumulative incidence at each unique event age (any
        cause), with pointwise standard errors from the finite-risk-set,
        tie-correct **Aalen (1978) variance** (the estimator reported by
        ``cmprsk::cuminc``; Andersen, Borgan, Gill & Keiding 1993, sec. IV.4).

    Notes
    -----
    The estimator is ``F_cause(t) = sum S(t_i-) * d_cause(t_i) / Y(t_i)``,
    where ``S`` is the overall survival (any event). The competing events are
    not censorable in the Kaplan-Meier sense; comparing against
    `kaplan_meier_cip` (which treats them as censoring) quantifies the
    estimand difference. This is the estimator LT-FH++ used for its CIPs
    (Pedersen et al. 2022: Aalen-Johansen with death and emigration as
    competing events, sex x birth-year strata). The variance includes the
    finite-risk-set and tied-event corrections from Aalen (1978) and ABGK
    (1993, sec. IV.4), evaluated in the algebraically equivalent recurrence
    used by ``cmprsk::cuminc`` so it remains finite when the final risk set is
    exhausted.
    """
    age_entry, age_exit = _validate_followup(age_entry, age_exit)
    event_type = np.asarray(event_type)
    if event_type.shape != age_entry.shape:
        raise ValueError("event_type must match the follow-up arrays' length")
    if not np.issubdtype(event_type.dtype, np.integer):
        if (event_type.dtype.kind == "f" and np.all(np.isfinite(event_type))
                and np.all(event_type == np.floor(event_type))):
            event_type = event_type.astype(np.int64)
        else:
            raise ValueError("event_type must be an integer code array")
    if np.any(event_type < 0):
        raise ValueError("event_type must be nonnegative")
    if isinstance(cause, (bool, np.bool_)):
        raise TypeError("cause must be a positive integer event code, not bool")
    try:
        cause = operator.index(cause)
    except TypeError:
        raise TypeError("cause must be a positive integer event code") from None
    if cause <= 0:
        raise ValueError("cause must be a positive event code; 0 denotes censoring")
    if np.any((event_type > 0) & (age_exit == age_entry)):
        raise ValueError("zero-length follow-up cannot carry an event")
    if not np.any(event_type == cause):
        raise ValueError(f"no events of cause {cause} in the data")

    grid = np.unique(age_exit[event_type > 0])
    F, pieces = _aj_on_grid(age_entry, age_exit, event_type, cause, grid)

    se = np.sqrt(np.clip(_aalen_variance(F, pieces), 0.0, None))
    return CipCurve(ages=grid, values=F, se=se,
                    n_events=int(np.sum(event_type == cause)),
                    n_entered=int(np.sum(age_exit > age_entry)),
                    estimator="aalen-johansen")


def _aj_on_grid(age_entry, age_exit, event_type, cause, grid):
    """Aalen-Johansen CIF for ``cause`` on a fixed grid, plus variance pieces.

    An empty risk set (``Y == 0``) at a grid age leaves the increment undefined:
    there is no one left to have had the event, yet the grid says one did.
    Raises, matching `kaplan_meier_cip`."""
    Y = _risk_sets(age_entry, age_exit, grid).astype(float)
    d_all = _count_at(np.sort(age_exit[event_type > 0]), grid).astype(float)
    d_cause = _count_at(np.sort(age_exit[event_type == cause]), grid).astype(float)
    if np.any(Y <= 0):
        raise ValueError(
            "empty risk set at an event age (check follow-up data)")
    if np.any(d_all > Y):
        raise ValueError(
            "event count exceeds the risk set at an event age "
            "(zero-length follow-up cannot carry an event)")
    surv = np.cumprod(1.0 - d_all / Y)
    surv_prev = np.concatenate(([1.0], surv[:-1]))       # S(t-)
    incr = surv_prev * d_cause / Y
    pieces = (surv_prev, d_cause, d_all, Y)
    return np.cumsum(incr), pieces


def _aalen_variance(F, pieces):
    """Finite-risk-set, tie-correct Aalen variance of the CIF.

    This is the recurrence used by ``cmprsk::cuminc``. It is algebraically
    equivalent to the closed form in ABGK (1993, sec. IV.4) while the
    post-event survival is positive. The recurrence also defines the boundary
    case where an event exhausts the risk set; evaluating the closed form
    directly would produce removable ``0 / 0`` terms there.
    """
    surv_prev, d_cause, d_all, Y = pieces
    d_other = d_all - d_cause
    surv_after = surv_prev * (1.0 - d_all / Y)
    var = np.empty(F.size, dtype=float)

    # ``v1 + F^2 v3 - 2 F v2`` is the influence-function variance. Separate
    # cause/other contributions are essential when causes are tied: replacing
    # their finite-population factors with d / Y**2 is only a large-risk-set
    # approximation.
    v1 = v2 = v3 = 0.0
    for j in range(F.size):
        if d_other[j] > 0 and surv_after[j] > 0:
            tie = 1.0 if d_other[j] == 1 else (
                1.0 - (d_other[j] - 1.0) / (Y[j] - 1.0)
            )
            increment = (
                surv_prev[j] ** 2 * tie * d_other[j] / Y[j] ** 2
            )
            inv_surv = 1.0 / surv_after[j]
            scaled_f = F[j] * inv_surv
            v1 += scaled_f ** 2 * increment
            v2 += inv_surv * scaled_f * increment
            v3 += inv_surv ** 2 * increment

        if d_cause[j] > 0:
            # For Y == d_cause == 1 the finite-population fraction is the
            # removable 0 / 0 boundary. ``cmprsk`` assigns tie=1, giving the
            # finite last-event contribution rather than NaN.
            tie = 1.0 if d_cause[j] == 1 else (
                1.0 - (d_cause[j] - 1.0) / (Y[j] - 1.0)
            )
            increment = (
                surv_prev[j] ** 2 * tie * d_cause[j] / Y[j] ** 2
            )
            inv_surv = 0.0 if surv_after[j] <= 0 else 1.0 / surv_after[j]
            scaled_f = 1.0 + F[j] * inv_surv
            v1 += scaled_f ** 2 * increment
            v2 += inv_surv * scaled_f * increment
            v3 += inv_surv ** 2 * increment

        var[j] = v1 + F[j] ** 2 * v3 - 2.0 * F[j] * v2

    return var
