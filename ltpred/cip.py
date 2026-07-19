"""Cumulative-incidence (CIP) estimation from registry-style health data.

The personalised thresholds of LT-FH++ / ADuLT / PA-FGRS consume a
cumulative-incidence curve per stratum (sex x birth cohort) through
:func:`ltpred.thresholds.thresholds_from_cip`. This module estimates that curve
from individual follow-up records: per person an entry age (start of
follow-up), an exit age, and what happened at exit.

Two estimators, both handling **left truncation** (delayed entry: register
starts mid-life, immigration) and **right censoring** (emigration, end of
follow-up):

* :func:`kaplan_meier_cip` -- the product-limit estimator ``CIP = 1 - S`` for
  the single-event setting. Correct when censoring (including death) is
  *independent* of the event process; when death is a real competing event it
  overestimates incidence (treating the dead as if they could still be
  diagnosed). Greenwood standard errors.
* :func:`aalen_johansen_cip` -- the Aalen--Johansen estimator of the
  cause-specific cumulative incidence with competing risks (diagnosis vs
  death without diagnosis). This is the estimator LT-FH++ used for its CIPs
  (Pedersen et al. 2022: Aalen-Johansen with death and emigration as
  competing events, one curve per sex x birth-year stratum) and the standard
  counting-process practice of the Danish register literature (Andersen,
  Borgan, Gill & Keiding 1993; e.g. Beck et al. 2024, *Acta Psychiatr
  Scand*). Pointwise SEs by the closed-form Aalen (1978) variance (the
  ``cmprsk::cuminc`` convention) by default, with a person-level bootstrap
  as an optional cross-check.

Risk-set convention (counting processes; Andersen, Borgan, Gill & Keiding,
*Statistical Models Based on Counting Processes*, 1993): person ``i`` is at
risk at age ``t`` when ``age_entry_i < t <= age_exit_i``. Events and censorings
occur at ``age_exit``; zero-length follow-ups (``exit == entry``) never enter a
risk set. Ages may be continuous; ties are handled by grouping on unique exit
ages. Stratification (e.g. by sex x birth-year bands) is done by calling an
estimator once per stratum -- see the worked example in the docstring of each
estimator and ``benchmarks/bench_cip_estimation.py``.

The returned :class:`CipCurve` carries ``ages`` and ``values`` in exactly the
form :func:`~ltpred.thresholds.thresholds_from_cip` consumes (ascending ages,
non-decreasing values in ``[0, 1)``), plus pointwise standard errors and event
counts. ``thresholds_from_cip`` takes the lifetime prevalence ``k_pop``
separately: pass the curve's value at the intended lifetime horizon, or a
separately justified estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CipCurve", "kaplan_meier_cip", "aalen_johansen_cip"]


@dataclass
class CipCurve:
    """An estimated cumulative-incidence curve.

    ``ages`` (ascending) and ``values`` (non-decreasing, in ``[0, 1)``) feed
    :func:`ltpred.thresholds.thresholds_from_cip` directly. ``se`` is the
    pointwise standard error (Greenwood for Kaplan-Meier, bootstrap for
    Aalen-Johansen); ``n_events`` counts events of interest;
    ``n_entered`` the number of persons contributing any follow-up;
    ``estimator`` records which estimator produced the curve."""
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


def kaplan_meier_cip(age_entry, age_exit, is_event):
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
    Treating death as independent censoring is the classic Kaplan-Meier
    assumption. When death before diagnosis is common (elderly onset
    disorders), it biases incidence upward -- use
    :func:`aalen_johansen_cip` with death as the competing event instead; the
    two are compared in ``benchmarks/bench_cip_estimation.py``.
    """
    age_entry, age_exit = _validate_followup(age_entry, age_exit)
    is_event = np.asarray(is_event, dtype=bool)
    if is_event.shape != age_entry.shape:
        raise ValueError("is_event must match the follow-up arrays' length")

    grid = np.unique(age_exit[is_event])
    if grid.size == 0:
        raise ValueError("no events of interest in the data")
    Y = _risk_sets(age_entry, age_exit, grid)
    d = _count_at(np.sort(age_exit[is_event]), grid).astype(float)
    if np.any(Y <= 0):
        raise ValueError("empty risk set at an event age (check follow-up data)")

    surv = np.cumprod(1.0 - d / Y)
    cip = 1.0 - surv
    # Greenwood: var(S) = S^2 * sum d / (Y (Y - d)); where Y == d the remaining
    # survival is 0 and so is the variance.
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(Y > d, d / (Y * (Y - d)), 0.0)
    var = surv ** 2 * np.cumsum(contrib)
    return CipCurve(ages=grid, values=cip, se=np.sqrt(var),
                    n_events=int(d.sum()), n_entered=int(age_entry.size),
                    estimator="kaplan-meier")


def aalen_johansen_cip(age_entry, age_exit, event_type, cause=1, *,
                       n_boot=None, seed=None):
    """Aalen-Johansen cumulative incidence with competing risks.

    Parameters
    ----------
    age_entry, age_exit
        Per-person follow-up ages (as in :func:`kaplan_meier_cip`).
    event_type
        Integer code per person: 0 = censored, ``cause`` = event of interest
        (default 1, diagnosis), any other positive code = competing event
        (e.g. 2 = death without diagnosis).
    cause
        The event code to estimate the cumulative incidence for.
    n_boot, seed
        ``None`` (default) computes the closed-form **Aalen (1978) variance**
        (the estimator reported by ``cmprsk::cuminc``; Andersen, Borgan, Gill
        & Keiding 1993, sec. IV.4). An integer ``n_boot > 0`` instead uses a
        person-level nonparametric bootstrap (a useful cross-check), and
        ``n_boot=0`` skips SEs (returns NaNs).

    Returns
    -------
    CipCurve
        Cause-specific cumulative incidence at each unique event age (any
        cause), with pointwise standard errors.

    Notes
    -----
    The estimator is ``F_cause(t) = sum S(t_i-) * d_cause(t_i) / Y(t_i)``,
    where ``S`` is the overall survival (any event). The competing events are
    not censorable in the Kaplan-Meier sense; comparing against
    :func:`kaplan_meier_cip` (which treats them as censoring) quantifies the
    competing-risks bias. This is the estimator LT-FH++ used for its CIPs
    (Pedersen et al. 2022: Aalen-Johansen with death and emigration as
    competing events, sex x birth-year strata). The closed-form variance is

        Var F_k(t) = sum_j [S(t_j-)]^2 d_kj/Y_j^2
                   + sum_j (F_k(t) - F_k(t_j))^2 d_j/Y_j^2
                   - 2 sum_j (F_k(t) - F_k(t_j)) S(t_j-) d_kj/Y_j^2

    (Aalen 1978, *Ann. Statist.* 6:534-545; ABGK 1993 eq. IV.4).
    """
    age_entry, age_exit = _validate_followup(age_entry, age_exit)
    event_type = np.asarray(event_type)
    if event_type.shape != age_entry.shape:
        raise ValueError("event_type must match the follow-up arrays' length")
    if not np.issubdtype(event_type.dtype, np.integer):
        raise ValueError("event_type must be an integer code array")
    if np.any(event_type < 0):
        raise ValueError("event_type must be nonnegative")
    if not np.any(event_type == cause):
        raise ValueError(f"no events of cause {cause} in the data")

    grid = np.unique(age_exit[event_type > 0])
    F, pieces = _aj_on_grid(age_entry, age_exit, event_type, cause, grid)

    if n_boot is None:
        se = np.sqrt(np.clip(_aalen_variance(F, pieces), 0.0, None))
    elif int(n_boot) > 0:
        rng = np.random.default_rng(seed)
        n = age_entry.size
        boot = np.empty((int(n_boot), grid.size))
        idx_all = np.arange(n)
        for b in range(int(n_boot)):
            idx = rng.choice(idx_all, size=n, replace=True)
            boot[b] = _aj_on_grid(age_entry[idx], age_exit[idx],
                                  event_type[idx], cause, grid)[0]
        se = boot.std(axis=0, ddof=1)
    else:
        se = np.full(grid.size, np.nan)
    return CipCurve(ages=grid, values=F, se=se,
                    n_events=int(np.sum(event_type == cause)),
                    n_entered=int(age_entry.size),
                    estimator="aalen-johansen")


def _aj_on_grid(age_entry, age_exit, event_type, cause, grid):
    """Aalen-Johansen CIF for ``cause`` on a fixed grid, plus variance pieces."""
    Y = _risk_sets(age_entry, age_exit, grid).astype(float)
    d_all = _count_at(np.sort(age_exit[event_type > 0]), grid).astype(float)
    d_cause = _count_at(np.sort(age_exit[event_type == cause]), grid).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1.0 - np.where(Y > 0, d_all / Y, 0.0))
    surv_prev = np.concatenate(([1.0], surv[:-1]))       # S(t-)
    with np.errstate(divide="ignore", invalid="ignore"):
        incr = np.where(Y > 0, surv_prev * d_cause / Y, 0.0)
    pieces = (surv_prev, d_cause, d_all, Y)
    return np.cumsum(incr), pieces


def _aalen_variance(F, pieces):
    """Aalen (1978) variance of the CIF (ABGK 1993, sec. IV.4)."""
    surv_prev, d_cause, d_all, Y = pieces
    with np.errstate(divide="ignore", invalid="ignore"):
        a = surv_prev ** 2 * d_cause / Y ** 2
        b = d_all / Y ** 2
        c = surv_prev * d_cause / Y ** 2
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    c = np.nan_to_num(c)
    # Var_i = sum_{j<=i} [ a_j + (F_i - F_j)^2 b_j - 2 (F_i - F_j) c_j ]
    dF = F[:, None] - F[None, :]                      # (i, j): F_i - F_j
    tri = np.tril(np.ones_like(dF, dtype=bool))       # j <= i
    var = (tri @ a) + np.sum(tri * (dF ** 2) * b[None, :], axis=1) \
        - 2.0 * np.sum(tri * dF * c[None, :], axis=1)
    return var
