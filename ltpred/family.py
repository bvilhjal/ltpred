"""Family input records and flat-column conversion.

A `Member` stores one person's role and liability bounds; a `Family` groups
members for one proband. Role ``o`` is the proband's own observation and ``g``
is added by the estimator. Bounds are scalar for one trait or one sequence per
trait. Personalised bounds with relatives define LT-FH++; the proband-only
construction defines ADuLT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import sys

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Member", "Family", "families_from_columns"]


@dataclass
class Member:
    """One member's role, liability bounds and optional identifiers.

    Bounds may describe cases, controls, finite onset pins or uninformative
    intervals. ``K_i``/``K_pop`` belong only on age-censored controls for the
    PA mixture, with ``0 <= K_i <= K_pop < 1`` and ``K_pop > 0``; use ``None``
    (NaN in arrays) elsewhere. ``aod`` is used only by research onset-age fits.
    """
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
    # pandas is optional; a caller supplying NA/NaT has already imported it.
    pandas = sys.modules.get("pandas")
    if pandas is not None and (x is pandas.NA or x is pandas.NaT):
        return True
    return False


def _is_missing_parent(pid, index):
    """True for a missing-id marker, or for ``0``/``"0"`` absent from ``index``.

    Zero is a valid parent when it is an explicitly listed id."""
    return _is_missing_id(pid) or (pid in (0, "0") and pid not in index)


def _pid_key(pid):
    """Return a hashable normalised pid, or ``None`` when it is missing."""
    if isinstance(pid, np.generic):
        pid = pid.item()
    if isinstance(pid, bytes):
        pid = pid.decode("utf-8", "replace")
    if isinstance(pid, str):
        pid = pid.strip()
        return None if pid.lower() in _MISSING_IDS else pid
    if _is_missing_id(pid):
        return None
    try:
        hash(pid)
    except TypeError:
        return str(pid)
    return pid


def _member_pid_key(member):
    """`_pid_key` of ``member.pid``, computed once per assigned pid.

    The object estimators and the moment fitters re-validate the same family
    objects call after call (a bootstrap resample revisits every member), so
    the normalization is cached on the instance together with the pid object
    it was computed from; reassigning ``member.pid`` invalidates it."""
    cached = member.__dict__.get("_pid_key_cache")
    if cached is not None and cached[0] is member.pid:
        return cached[1]
    key = _pid_key(member.pid)
    member._pid_key_cache = (member.pid, key)
    return key


def _proband_pid(family):
    """Preserve personal join keys; family ids are only a pid-free fallback."""
    own = next((m for m in family.members if m.role == "o"), None)
    if own is not None and not _is_missing_id(own.pid):
        return own.pid
    if any(not _is_missing_id(m.pid) for m in family.members):
        raise ValueError(
            f"family {family.fam_id!r} supplies member pids but no proband pid; "
            "include role 'o' with its pid. For family-history-only prediction, "
            "retain that row with uninformative (-inf, inf) bounds.")
    return family.fam_id


def families_from_columns(fam_id: ArrayLike, role: ArrayLike, lower: ArrayLike,
                          upper: ArrayLike, pid: ArrayLike | None = None,
                          K_i: ArrayLike | None = None,
                          K_pop: ArrayLike | None = None,
                          aod: ArrayLike | None = None) -> list[Family]:
    """Group aligned flat columns into families in first-appearance order.

    Required columns are ``fam_id``, ``role``, ``lower`` and ``upper``;
    optional ``pid``, ``K_i``, ``K_pop`` and ``aod`` align by row. Multi-trait
    bounds have one row per member and one column per trait. Missing family ids
    and mixed string/non-string ids raise. Retain role ``o`` and its pid even
    when its bounds are uninformative.
    """
    raw_fam_id = fam_id
    fam_id = np.asarray(fam_id)
    if fam_id.dtype.kind in "US" and not (
            isinstance(raw_fam_id, np.ndarray) and raw_fam_id.dtype.kind in "US"):
        # NumPy stringifies a mixed sequence, so 1 and "1" would silently
        # group as one family. Refuse the ambiguity instead of guessing.
        items = np.asarray(raw_fam_id, dtype=object).ravel()
        if not all(isinstance(x, (str, bytes)) or _is_missing_id(x)
                   for x in items):
            raise ValueError(
                "fam_id mixes string and non-string ids (e.g. 1 and '1'), "
                "which would merge distinct families; convert every fam_id "
                "to one type first")
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
        # singletons, whereas every "" or "NA" compares *equal* and merges
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

    groups = {}
    for i in range(n):
        key = fam_id[i]
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    families = []
    for key, indices in groups.items():
        members = []
        for i in indices:
            members.append(Member(role=str(role[i]), lower=lower[i], upper=upper[i],
                                   pid=(None if pid is None else pid[i]),
                                   K_i=(None if K_i is None else K_i[i]),
                                   K_pop=(None if K_pop is None else K_pop[i]),
                                   aod=(None if aod is None else aod[i])))
        families.append(Family(fam_id=key, members=members))
    return families
