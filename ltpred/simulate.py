"""Simulate families under the liability-threshold model.

Draws each family's liabilities from the LT-FH++ covariance (genetic ``g``, full
``o``, and relatives jointly multivariate normal), assigns case/control status by
thresholding the full liabilities, and packages the per-member truncation bounds
as ready-to-estimate :class:`~ltpred.family.Family` objects. Two flavours:

* ``use_age=False`` -- classic LT-FH: a single prevalence threshold ``T``, cases
  ``(T, inf)`` and controls ``(-inf, T)``. Self-consistent, handy for validating
  the estimator against the simulated truth.
* ``use_age=True`` -- LT-FH++/ADuLT: each member gets an age; a case is pinned at
  the threshold of its (rounded) age of onset, a control lies below the threshold
  for its current age.

The returned :class:`Simulation` also keeps the true liabilities so downstream
tests can check that the estimated genetic liability tracks the simulated one.
Port of LTFHPlus::simulate_under_LTM(_single).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import construct_covmat_single
from .family import Family, Member
from .thresholds import (liability_threshold, convert_liability_to_aoo,
                         convert_age_to_thresh)

__all__ = ["Simulation", "simulate_under_LTM_single", "simulate_under_LTM"]

# plausible current-age ranges per role, used only when use_age=True (the exact
# model affects realism, not the method); (low, high) inclusive-exclusive.
_AGE_RANGES = {
    "o": (10, 60), "s": (10, 60), "mhs": (10, 60), "phs": (10, 60),
    "m": (40, 80), "f": (40, 80), "mau": (40, 80), "pau": (40, 80),
    "mgm": (65, 95), "mgf": (65, 95), "pgm": (65, 95), "pgf": (65, 95),
    "c": (0, 30),
}


def _age_range(role):
    stem = role.rstrip("0123456789")
    return _AGE_RANGES.get(stem, (10, 60))


@dataclass
class Simulation:
    """Output of :func:`simulate_under_LTM_single`.

    ``liabilities`` is ``(n_sim, d)`` with columns in ``roles`` order (``g``, ``o``,
    relatives); ``genetic``/``full`` are its first two columns for convenience.
    ``status`` maps each non-``g`` role to its boolean case array. ``families`` is
    the list of one-proband families (bounds only) to pass to the estimator."""
    roles: list
    covmat: np.ndarray
    liabilities: np.ndarray
    status: dict
    families: list
    pop_prev: float

    @property
    def genetic(self):
        return self.liabilities[:, self.roles.index("g")]

    @property
    def full(self):
        return self.liabilities[:, self.roles.index("o")]


def simulate_under_LTM_single(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                              n_fam=None, add_ind=True, h2=0.5, n_sim=1000,
                              pop_prev=0.1, use_age=False, mid_point=60.0,
                              slope=1.0 / 8.0, seed=None):
    """Simulate ``n_sim`` families for a single trait.

    Builds the covariance from ``fam_vec``/``n_fam`` (``g``/``o`` prepended when
    ``add_ind``), draws liabilities, thresholds them at prevalence ``pop_prev``, and
    returns a :class:`Simulation`. With ``use_age`` the bounds follow the ADuLT
    age-of-onset construction; otherwise the classic case/control bounds."""
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
    status = {r: liab[:, roles.index(r)] > t for r in non_g}

    if use_age:
        ages = {}
        for r in non_g:
            lo, hi = _age_range(r)
            ages[r] = rng.integers(lo, hi, size=n_sim)
    else:
        ages = None

    families = []
    for i in range(n_sim):
        members = []
        for r in non_g:
            col = roles.index(r)
            is_case = bool(status[r][i])
            if use_age:
                if is_case:
                    aoo = convert_liability_to_aoo(liab[i, col], pop_prev=pop_prev,
                                                   mid_point=mid_point, slope=slope)
                    aoo = 0.0 if not np.isfinite(aoo) else round(float(aoo))
                    thr = float(convert_age_to_thresh(aoo, pop_prev=pop_prev,
                                                      mid_point=mid_point, slope=slope))
                    lower, upper = thr, thr
                else:
                    thr = float(convert_age_to_thresh(ages[r][i], pop_prev=pop_prev,
                                                      mid_point=mid_point, slope=slope))
                    lower, upper = -np.inf, thr
            else:
                lower, upper = (t, np.inf) if is_case else (-np.inf, t)
            members.append(Member(role=r, lower=lower, upper=upper,
                                   pid=f"fam_{i}_{r}"))
        families.append(Family(fam_id=f"fam_{i}", members=members))

    return Simulation(roles=roles, covmat=cov_obj.matrix, liabilities=liab,
                      status=status, families=families, pop_prev=pop_prev)


# alias mirroring the R dispatcher name
simulate_under_LTM = simulate_under_LTM_single
