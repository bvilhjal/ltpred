"""Simulate families under the liability-threshold model.

Draws each family's liabilities from the liability-threshold family covariance (genetic ``g``, full
``o``, and relatives jointly multivariate normal), assigns case/control status by
thresholding the full liabilities, and packages the per-member truncation bounds
as ready-to-estimate :class:`~ltpred.family.Family` objects. Two flavours:

* ``use_age=False`` -- classic LT-FH: a single prevalence threshold ``T``, cases
  ``(T, inf)`` and controls ``(-inf, T)``. Self-consistent, handy for validating
  the estimator against the simulated truth.
* ``use_age=True`` -- age-aware simulation with generation-consistent current
  ages. Status is ``onset <= current age`` (a young person with high liability
  is a censored control, not a case). ``onset_model`` chooses how onset is
  generated; ``case_encoding`` chooses the bound written for observed cases.

The returned :class:`Simulation` also keeps the true liabilities so downstream
tests can check that the estimated genetic liability tracks the simulated one.
Port of LTFHPlus::simulate_under_LTM(_single), with coherent follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .covariance import construct_covmat_single
from .family import Family, Member
from .thresholds import (liability_threshold, convert_liability_to_aoo,
                         convert_age_to_thresh, _convert_cir_to_age)

__all__ = ["Simulation", "simulate_under_LTM_single"]

# Mean years older than the proband. Used only when use_age=True, so parents
# are not drawn independently younger than their children.
_AGE_GAP = {
    "o": 0.0, "s": 0.0, "mhs": 0.0, "phs": 0.0,
    "m": 29.0, "f": 31.0, "mau": 29.0, "pau": 31.0,
    "mgm": 56.0, "mgf": 58.0, "pgm": 56.0, "pgf": 58.0,
    "c": -28.0,
}

# Fallback ranges for `_age_range` (role-stem lookup; tests pin the child bug).
_AGE_RANGES = {
    "o": (10, 60), "s": (10, 60), "mhs": (10, 60), "phs": (10, 60),
    "m": (40, 80), "f": (40, 80), "mau": (40, 80), "pau": (40, 80),
    "mgm": (65, 95), "mgf": (65, 95), "pgm": (65, 95), "pgf": (65, 95),
    "c": (0, 30),
}

_ONSET_MODELS = ("threshold_crossing", "stochastic")
_CASE_ENCODINGS = ("pin", "interval", "lifetime")


def _role_stem(role):
    # Strip the numbering to get the role stem. Children carry a two-part index
    # (``c1.2``), so the dot has to go too -- ``"c1.2".rstrip("0123456789")``
    # stops at the dot and yields ``"c1."``, which never matched the ``"c"`` key
    # and silently gave simulated children the 10-60 default.
    return role.rstrip("0123456789").rstrip(".").rstrip("0123456789")


def _age_range(role):
    return _AGE_RANGES.get(_role_stem(role), (10, 60))


def _draw_ages(roles, n, rng):
    """Generation-consistent current ages: parents older than the proband."""
    age_o = rng.uniform(20.0, 55.0, size=n)
    ages = {}
    for role in roles:
        stem = _role_stem(role)
        gap = _AGE_GAP.get(stem, 0.0)
        sd = 0.0 if stem == "o" else (4.0 if abs(gap) >= 20.0 else 3.0)
        ages[role] = np.clip(age_o + gap + rng.normal(0.0, sd, n), 1.0, 110.0)
    return ages


@dataclass
class Simulation:
    """Output of :func:`simulate_under_LTM_single`.

    ``liabilities`` is ``(n_sim, d)`` with columns in ``roles`` order (``g``, ``o``,
    relatives); ``genetic``/``full`` are its first two columns for convenience.
    ``status`` maps each non-``g`` role to its boolean *observed* case array.
    ``ages`` / ``onset`` are per-role arrays when ``use_age=True`` (onset is
    ``inf`` for never-affected people). ``families`` is the list of one-proband
    families (bounds only) to pass to the estimator."""
    roles: list
    covmat: np.ndarray
    liabilities: np.ndarray
    status: dict
    families: list
    pop_prev: float
    ages: dict = None
    onset: dict = None
    onset_model: str = None
    case_encoding: str = None

    @property
    def genetic(self):
        return self.liabilities[:, self.roles.index("g")]

    @property
    def full(self):
        return self.liabilities[:, self.roles.index("o")]


def _record_onset(aoo, onset_resolution):
    """The onset age a register would *record*, at ``onset_resolution`` years.

    Registers record onset on a grid (a year, a quarter), not to the instant, and
    every case bound is derived from that recorded value -- so the same rounding
    must apply to whichever ``case_encoding`` is in use, otherwise ``pin`` and
    ``interval`` would silently describe different observation processes.
    ``onset_resolution=None`` records the exact simulated onset, which is what
    makes ``pin`` reproduce the true liability exactly."""
    if onset_resolution is None:
        return aoo
    return round(aoo / onset_resolution) * onset_resolution


def _validate_onset_resolution(onset_resolution):
    if onset_resolution is None:
        return None
    if isinstance(onset_resolution, (bool, np.bool_)):
        raise TypeError("onset_resolution must be a positive number or None, "
                        "not bool")
    try:
        onset_resolution = float(onset_resolution)
    except (TypeError, ValueError):
        raise TypeError(
            "onset_resolution must be a positive number or None") from None
    if not np.isfinite(onset_resolution) or onset_resolution <= 0.0:
        raise ValueError("onset_resolution must be finite and > 0, or None")
    return onset_resolution


def _resolve_age_options(use_age, onset_model, case_encoding):
    if not use_age:
        if onset_model is not None or case_encoding is not None:
            raise ValueError(
                "onset_model and case_encoding apply only when use_age=True")
        return None, None
    if onset_model is None:
        onset_model = "threshold_crossing"
    if onset_model not in _ONSET_MODELS:
        raise ValueError(
            f"onset_model must be one of {_ONSET_MODELS}, got {onset_model!r}")
    if case_encoding is None:
        case_encoding = "pin" if onset_model == "threshold_crossing" else "lifetime"
    if case_encoding not in _CASE_ENCODINGS:
        raise ValueError(
            f"case_encoding must be one of {_CASE_ENCODINGS}, got {case_encoding!r}")
    return onset_model, case_encoding


def _onset_times(liab, pop_prev, mid_point, slope, onset_model, lifetime_t, rng):
    """Per-person onset age; ``inf`` means never affected in this lifetime."""
    n = liab.shape[0]
    onset = np.full(n, np.inf)
    if onset_model == "threshold_crossing":
        aoo = convert_liability_to_aoo(liab, pop_prev=pop_prev,
                                       mid_point=mid_point, slope=slope)
        finite = np.isfinite(aoo)
        onset[finite] = np.maximum(aoo[finite], 0.0)
        return onset
    # Stochastic: lifetime case if l > T; onset ~ CIP among lifetime cases,
    # independent of the value of l given that. Pinning at T(onset) is then
    # *not* the true liability — the point of this mode.
    lifetime_case = liab > lifetime_t
    if not np.any(lifetime_case):
        return onset
    u = rng.random(int(lifetime_case.sum()))
    cir = np.clip(u * pop_prev, 1e-12, pop_prev * (1.0 - 1e-12))
    onset[lifetime_case] = _convert_cir_to_age(
        cir, pop_prev=pop_prev, mid_point=mid_point, slope=slope)
    return onset


def simulate_under_LTM_single(fam_vec: Sequence[str] | None = ("m", "f", "s1", "mgm",
                                                              "mgf", "pgm", "pgf"),
                              n_fam: Mapping[str, int] | None = None,
                              add_ind: bool = True, h2: float = 0.5,
                              n_sim: int = 1000, pop_prev: float = 0.1,
                              use_age: bool = False, mid_point: float = 60.0,
                              slope: float = 1.0 / 8.0, seed: int | None = None,
                              onset_model: str | None = None,
                              case_encoding: str | None = None,
                              onset_resolution: float | None = 1.0) -> Simulation:
    """Simulate ``n_sim`` families for a single trait.

    Builds the covariance from ``fam_vec``/``n_fam`` (``g``/``o`` prepended when
    ``add_ind``), draws liabilities, and returns a :class:`Simulation`.

    With ``use_age=False`` (default) the bounds are classic LT-FH: one lifetime
    threshold, status ``l > T``.

    With ``use_age=True`` current ages are generation-consistent, and a person
    is an observed case only if their onset age is at most their current age.
    ``onset_model`` is ``"threshold_crossing"`` (default: onset is the CIP
    inverse of true liability — the LT-FH++ convention) or ``"stochastic"``
    (lifetime status at ``T``, onset drawn
    from the CIP independently of ``l`` given being a lifetime case).
    ``case_encoding`` is ``"pin"`` (default for threshold-crossing),
    ``"interval"`` (age-specific ``[T(onset), inf)``), or ``"lifetime"``
    (default for stochastic: ``[T, inf)``). This helper has one logistic CIP
    and does not simulate the full sex/birth-cohort personalisation of
    LT-FH++.

    ``onset_resolution`` is the grid a register is taken to **record** onset on,
    in years (default ``1.0``, whole years; ``None`` records the exact simulated
    onset). Case bounds are built from the recorded value under both ``"pin"``
    and ``"interval"``, so the two encodings describe the same observation
    process. This matters when the simulation is used as an oracle: under
    ``threshold_crossing`` a pin reproduces the true liability **exactly only
    with** ``onset_resolution=None``. At the default one-year grid the pinned
    bound sits within roughly 0.02 of the simulated liability (about 2% of its
    SD), which is a floor on any measured recovery — realistic, but not zero."""
    onset_resolution = _validate_onset_resolution(onset_resolution)
    onset_model, case_encoding = _resolve_age_options(
        use_age, onset_model, case_encoding)
    cov_obj = construct_covmat_single(fam_vec=fam_vec, n_fam=n_fam,
                                      add_ind=add_ind, h2=h2)
    roles = cov_obj.roles
    if "g" not in roles or "o" not in roles:
        raise ValueError("simulation requires add_ind=True (needs g and o)")
    d = len(roles)
    rng = np.random.default_rng(seed)
    liab = rng.multivariate_normal(np.zeros(d), cov_obj.matrix, size=n_sim)

    t = float(liability_threshold(pop_prev))
    non_g = [r for r in roles if r != "g"]

    if use_age:
        ages = _draw_ages(non_g, n_sim, rng)
        onset = {
            r: _onset_times(liab[:, roles.index(r)], pop_prev, mid_point, slope,
                            onset_model, t, rng)
            for r in non_g
        }
        status = {r: onset[r] <= ages[r] for r in non_g}
    else:
        ages = onset = None
        status = {r: liab[:, roles.index(r)] > t for r in non_g}

    families = []
    for i in range(n_sim):
        members = []
        for r in non_g:
            is_case = bool(status[r][i])
            if not use_age:
                lower, upper = (t, np.inf) if is_case else (-np.inf, t)
            elif is_case:
                aoo = float(onset[r][i])
                if case_encoding == "lifetime":
                    lower, upper = t, np.inf
                else:
                    # An observed case has a finite onset at or before their
                    # current age; the guard covers only a direct call with a
                    # hand-built onset array.
                    aoo_r = (0.0 if not np.isfinite(aoo)
                             else _record_onset(aoo, onset_resolution))
                    thr = float(convert_age_to_thresh(
                        aoo_r, pop_prev=pop_prev, mid_point=mid_point,
                        slope=slope))
                    lower, upper = (thr, np.inf) if case_encoding == "interval" \
                        else (thr, thr)
            else:
                thr = float(convert_age_to_thresh(
                    ages[r][i], pop_prev=pop_prev, mid_point=mid_point,
                    slope=slope))
                lower, upper = -np.inf, thr
            members.append(Member(role=r, lower=lower, upper=upper,
                                   pid=f"fam_{i}_{r}"))
        families.append(Family(fam_id=f"fam_{i}", members=members))

    return Simulation(roles=roles, covmat=cov_obj.matrix, liabilities=liab,
                      status=status, families=families, pop_prev=pop_prev,
                      ages=ages, onset=onset, onset_model=onset_model,
                      case_encoding=case_encoding)
