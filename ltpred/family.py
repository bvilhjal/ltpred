"""Input containers: a proband, their relatives, and each one's liability bounds.

The family-history models condition on a proband plus zero or more relatives,
each with a ``role`` (``o`` = the proband's own status, ``m``/``f``/``s1``/... =
relatives) and a liability interval ``(lower, upper)``. A :class:`Member` holds
one such observed-person record; a :class:`Family` groups records for one proband. The
genetic-liability row ``g`` is added automatically by the estimator.

Model identity depends on both the bounds and the rows supplied: personalised
age/sex/cohort bounds with relatives are LT-FH++; the same bounds with only role
``o`` are family-free ADuLT. Classic single-prevalence bounds with relatives are
LT-FH.

For a single trait ``lower``/``upper`` are scalars. For the multi-trait model they
are length-``n_pheno`` sequences (one interval per phenotype, in ``phen_names``
order). :func:`families_from_columns` builds the list from flat, tibble-style
columns -- the shape the R package's ``.tbl`` input uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ._validation import validate_own_status

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Member", "Family", "families_from_columns"]


@dataclass
class Member:
    """One family member's role and liability truncation bounds.

    ``lower``/``upper`` are scalars for a single trait or length-``n_pheno``
    sequences for the multi-trait model. They may encode one-sided case/control
    truncation, an onset-pinned case (``lower == upper``), or an uninformative
    interval. ``pid`` is an optional personal identifier.

    ``K_i``/``K_pop`` are only used by the Pearson-Aitken estimator's censored
    -control mixture: ``K_i`` is the individual's (age/sex-stratified) cumulative
    incidence and ``K_pop`` the lifetime population prevalence. Leave them ``None``
    (the default) for cases and for the non-mixture model.

    ``aod`` is the member's age at diagnosis (cases) or age at last follow-up
    (controls) -- only used by the onset-age-structured genetic-correlation fit
    (``research.advanced_fitting.fit_genetic_correlation_decay``)."""
    role: str
    lower: object
    upper: object
    pid: Optional[object] = None
    K_i: Optional[object] = None
    K_pop: Optional[object] = None
    aod: Optional[object] = None


@dataclass
class Family:
    """A proband and their relatives, keyed by ``fam_id``.

    ``members`` should include the proband's own status as role ``o`` plus any
    relatives; the estimator inserts the missing ``g`` (and ``o``, if absent)
    with uninformative ``(-inf, inf)`` bounds."""
    fam_id: object
    members: list = field(default_factory=list)


# Textual stand-ins for a missing id, as emitted by pandas/R CSV writers.
_MISSING_IDS = frozenset({"", ".", "na", "n/a", "nan", "none", "null", "<na>"})


def _is_missing_id(x: object) -> bool:
    """True when ``x`` cannot serve as a grouping key."""
    if x is None:
        return True
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return True
    if isinstance(x, bytes):
        x = x.decode("utf-8", "replace")
    if isinstance(x, str):
        return x.strip().lower() in _MISSING_IDS
    return False


def families_from_columns(fam_id: ArrayLike, role: ArrayLike, lower: ArrayLike,
                          upper: ArrayLike, pid: ArrayLike | None = None,
                          K_i: ArrayLike | None = None,
                          K_pop: ArrayLike | None = None,
                          aod: ArrayLike | None = None,
                          own_status: Literal["in", "out"] = "in") -> list[Family]:
    """Group flat, column-oriented threshold data into a list of families.

    Mirrors the R ``.tbl`` input (columns ``fam_id``, ``role``, ``lower``,
    ``upper`` and optionally ``pid``). Records sharing a ``fam_id`` become one
    :class:`Family`; family order follows first appearance. For the multi-trait
    model pass ``lower``/``upper`` as 2-D (rows x phenotypes). ``K_i``/``K_pop`` are
    optional per-row columns for the Pearson-Aitken censored-control mixture.
    Missing or non-finite numeric ``fam_id`` values are rejected: they cannot
    group records and would otherwise fragment silently into one-member families.

    ``own_status=\"out\"`` (use I) rewrites every role-``o`` bound to
    ``(-inf, inf)`` and clears that row's ``K_i``/``K_pop``, without dropping
    the row, so later ``pids`` still come from the proband record. ``\"in\"``
    (default, use II) leaves the bounds alone. Other spellings raise.
    A family that is only role ``o`` with ``own_status=\"out\"`` has nothing
    to condition on and raises, matching an empty family.
    """
    own_status = validate_own_status(own_status)
    fam_id = np.asarray(fam_id)
    role = np.asarray(role, dtype=object)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    pid = np.asarray(pid, dtype=object) if pid is not None else None
    K_i = np.asarray(K_i, dtype=float) if K_i is not None else None
    K_pop = np.asarray(K_pop, dtype=float) if K_pop is not None else None
    aod = np.asarray(aod, dtype=float) if aod is not None else None

    n = len(fam_id)
    if not (len(role) == n == lower.shape[0] == upper.shape[0]):
        raise ValueError("fam_id, role, lower and upper must share length")
    if lower.shape != upper.shape:
        raise ValueError(
            f"lower and upper must have the same shape; got {lower.shape} "
            f"and {upper.shape}")
    # The optional columns are indexed positionally alongside the mandatory
    # ones below, so an over-long column would be silently truncated to the
    # first n rows -- a wrong-but-finite estimate -- and a short one would die
    # with a bare IndexError far from the cause.
    for _name, _col in (("pid", pid), ("K_i", K_i),
                        ("K_pop", K_pop), ("aod", aod)):
        if _col is not None and _col.shape[0] != n:
            raise ValueError(
                f"{_name} must have length {n} to match fam_id; got "
                f"{_col.shape[0]}")
    if fam_id.dtype.kind == "f":
        missing = ~np.isfinite(fam_id)
    elif fam_id.dtype.kind in ("U", "S", "O"):
        # String ids need the textual sentinels a CSV loader produces. These
        # are worse than a float NaN: NaN != NaN fragments records into
        # singletons, whereas every \"\" or \"NA\" compares *equal* and merges
        # unrelated probands into one family.
        missing = np.array([_is_missing_id(x) for x in fam_id], dtype=bool)
    else:
        missing = np.zeros(n, dtype=bool)
    if np.any(missing):
        raise ValueError(
            "fam_id must not contain missing values (a non-finite number, None, "
            "or an empty/NA string): a missing id cannot group "
            "records and would silently fragment them into one-member "
            "families")

    order = []
    groups = {}
    for i in range(n):
        key = fam_id[i]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    families = []
    only_o = []
    for key in order:
        members = []
        for i in groups[key]:
            lo_i, hi_i = lower[i], upper[i]
            k_i = None if K_i is None else K_i[i]
            k_pop = None if K_pop is None else K_pop[i]
            role_i = str(role[i])
            if own_status == "out" and role_i == "o":
                lo_i = np.full(np.shape(lo_i), -np.inf) if np.ndim(lo_i) else -np.inf
                hi_i = np.full(np.shape(hi_i), np.inf) if np.ndim(hi_i) else np.inf
                k_i = k_pop = None
            members.append(Member(role=role_i, lower=lo_i, upper=hi_i,
                                   pid=(None if pid is None else pid[i]),
                                   K_i=k_i, K_pop=k_pop,
                                   aod=(None if aod is None else aod[i])))
        if own_status == "out" and members and all(m.role == "o" for m in members):
            only_o.append(key)
        families.append(Family(fam_id=key, members=members))
    if only_o:
        shown = ", ".join(repr(fid) for fid in only_o[:5])
        more = "" if len(only_o) <= 5 else f", ... (+{len(only_o) - 5} more)"
        raise ValueError(
            f"{len(only_o)} of {len(families)} families have only role 'o' "
            f"with own_status='out' ({shown}{more}); with nothing to "
            "condition on the estimate would be the prior mean 0, "
            "indistinguishable from a real score. Include relatives, "
            "or drop those families.")
    return families
