"""Cumulative incidence from follow-up records with delayed entry.

`kaplan_meier_cip` estimates net incidence with Greenwood SEs;
`aalen_johansen_cip` estimates cause-specific crude incidence with competing
risks and finite-risk-set, tie-correct Aalen SEs. A person is at risk at ``t``
when ``age_entry < t <= age_exit``; zero-length follow-up cannot carry an
event. Both estimators handle right censoring and tied event ages.

The returned `CipCurve` feeds `thresholds_from_cip` per population stratum.
Curves reaching one cannot define a finite probit threshold; choose a justified
lifetime horizon and ``k_pop``. See ``docs/cip-estimation.md`` for estimand
choice and examples.
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
    """Ascending event ages, cumulative incidence, pointwise SEs and counts."""
    ages: np.ndarray
    values: np.ndarray
    se: np.ndarray
    n_events: int
    n_entered: int
    estimator: str


def _validate_followup(age_entry, age_exit):
    """Float 1-D follow-up arrays: finite, ``0 <= entry <= exit``, non-empty."""
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
    """Estimate net cumulative incidence ``1 - S(t)`` by Kaplan--Meier.

    ``age_entry`` and ``age_exit`` are aligned follow-up ages;
    ``is_event`` is boolean or exact 0/1. Delayed entry and right censoring
    use the risk convention ``entry < t <= exit``. Returns event ages,
    cumulative incidence and Greenwood pointwise SEs in a `CipCurve`.

    For the proportion actually diagnosed before death, death is a competing
    event rather than censoring: use `aalen_johansen_cip` instead.
    """
    age_entry, age_exit = _validate_followup(age_entry, age_exit)
    is_event_raw = np.asarray(is_event)
    if is_event_raw.shape != age_entry.shape:
        raise ValueError("is_event must match the follow-up arrays' length")
    is_event = validate_binary(is_event_raw, name="is_event", ndim=1)
    if np.any(is_event & (age_exit == age_entry)):
        raise ValueError("zero-length follow-up cannot carry an event")

    grid, d = np.unique(age_exit[is_event], return_counts=True)
    if grid.size == 0:
        raise ValueError("no events of interest in the data")
    Y = _risk_sets(age_entry, age_exit, grid)
    d = d.astype(float)
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
    """Estimate cause-specific cumulative incidence by Aalen--Johansen.

    ``event_type`` uses 0 for censoring, ``cause`` for the event of interest
    and other positive codes for competing events. Follow-up arrays match
    `kaplan_meier_cip`. Returns every event age with
    ``sum S(t-) * d_cause / Y`` and finite-risk-set, tie-correct Aalen SEs.
    The variance recurrence remains defined when the final risk set is
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
    """Aalen-Johansen CIF for ``cause`` on ``grid`` (the unique event ages), plus variance pieces.

    An empty risk set (``Y == 0``) at a grid age leaves the increment undefined:
    there is no one left to have had the event, yet the grid says one did.
    Raises, matching `kaplan_meier_cip`."""
    Y = _risk_sets(age_entry, age_exit, grid).astype(float)
    # grid is np.unique of these exits at the caller, so counts align with it.
    _, d_all = np.unique(age_exit[event_type > 0], return_counts=True)
    d_all = d_all.astype(float)
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
    """Evaluate the tie-correct Aalen variance as three cumulative sums.

    This ``cmprsk::cuminc``-equivalent recurrence avoids removable 0/0 terms
    when an event exhausts the risk set.
    """
    surv_prev, d_cause, d_all, Y = pieces
    d_other = d_all - d_cause
    surv_after = surv_prev * (1.0 - d_all / Y)

    # ``v1 + F^2 v3 - 2 F v2`` is the Aalen variance (docs/cip-estimation.md,
    # Equation 4). Separate cause/other contributions are essential when causes
    # are tied: replacing their finite-population factors with d / Y**2 is only
    # a large-risk-set approximation. Each contribution at age j depends only
    # on quantities at j (never on the running v's), so the recurrence is
    # three plain cumulative sums -- vectorised below, no Python loop over
    # what may be a 1e5-age register grid.
    def _contrib(d, active, scaled):
        """Per-age (v1, v2, v3) increments for one event class."""
        with np.errstate(divide="ignore", invalid="ignore"):
            # tie = 1 at d == 1 (incl. the removable Y == d == 1 boundary,
            # where cmprsk likewise assigns 1), else 1 - (d-1)/(Y-1).
            tie = np.where(d == 1, 1.0, 1.0 - (d - 1.0) / (Y - 1.0))
            increment = surv_prev ** 2 * tie * d / Y ** 2
            inv_surv = np.where(surv_after > 0, 1.0 / surv_after, 0.0)
        s = scaled(F, inv_surv)
        inc = np.where(active, increment, 0.0)
        return s ** 2 * inc, inv_surv * s * inc, inv_surv ** 2 * inc

    o1, o2, o3 = _contrib(d_other, (d_other > 0) & (surv_after > 0),
                          lambda f, inv: f * inv)
    c1, c2, c3 = _contrib(d_cause, d_cause > 0,
                          lambda f, inv: 1.0 + f * inv)
    v1 = np.cumsum(o1 + c1); v2 = np.cumsum(o2 + c2); v3 = np.cumsum(o3 + c3)
    return v1 + F ** 2 * v3 - 2.0 * F * v2
