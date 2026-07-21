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
from typing import Optional

import numpy as np

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
    (:func:`ltpred.fit.fit_genetic_correlation_decay`)."""
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


def families_from_columns(fam_id, role, lower, upper, pid=None, K_i=None,
                        K_pop=None, aod=None):
    """Group flat, column-oriented threshold data into a list of families.

    Mirrors the R ``.tbl`` input (columns ``fam_id``, ``role``, ``lower``,
    ``upper`` and optionally ``pid``). Records sharing a ``fam_id`` become one
    :class:`Family`; family order follows first appearance. For the multi-trait
    model pass ``lower``/``upper`` as 2-D (rows x phenotypes). ``K_i``/``K_pop`` are
    optional per-row columns for the Pearson-Aitken censored-control mixture."""
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

    order = []
    groups = {}
    for i in range(n):
        key = fam_id[i]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    families = []
    for key in order:
        members = []
        for i in groups[key]:
            members.append(Member(role=str(role[i]), lower=lower[i], upper=upper[i],
                                   pid=(None if pid is None else pid[i]),
                                   K_i=(None if K_i is None else K_i[i]),
                                   K_pop=(None if K_pop is None else K_pop[i]),
                                   aod=(None if aod is None else aod[i])))
        families.append(Family(fam_id=key, members=members))
    return families
