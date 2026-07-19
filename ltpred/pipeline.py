"""End-to-end register pipeline: trio records -> pedigrees -> CIP thresholds
-> per-proband genetic-liability scores.

The pieces exist separately -- :mod:`ltpred.pedigree` discovers relatives,
:mod:`ltpred.cip` estimates the incidence curve,
:func:`~ltpred.thresholds.thresholds_from_cip` turns it into bounds, and
:func:`~ltpred.estimate.estimate_liability_from_kinship` scores one pedigree.
:func:`estimate_liabilities` chains them for a register-scale run: for each
proband it extracts the pedigree (up to ``max_degree`` relationship-degrees,
with the ancestral closure), builds each member's personalised bounds from the
CIP of their stratum, and estimates the proband's posterior mean genetic
liability with the Pearson-Aitken engine.

Two observation designs, per the package's leakage conventions:

* **GWAS phenotype** (default): the proband's own status is conditioning
  information and is included.
* **Prospective prediction** (``index_age`` given): the proband's own bound is
  uninformative ``(-inf, inf)``, and **familywise censoring** is applied --
  every relative's events after the proband-specific index age are censored at
  that age (a later case becomes a censored control at the index age), so no
  post-index information enters the score (the LT-FGRS pipeline's censoring
  step, Pedersen et al. 2026).

Scale note: estimation is per proband with a small dense kinship covariance
(register neighbourhoods are tens of people), which is the right architecture
at population scale; grouping probands by identical pedigree structure to
share one ``A`` is a future optimization, not required for correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import kinship_from_pedigree
from .estimate import estimate_liability_from_kinship
from .pedigree import build_parent_graph, extract_pedigree
from .thresholds import thresholds_from_cip

__all__ = ["PopulationScores", "estimate_liabilities"]


@dataclass
class PopulationScores:
    """Per-proband pipeline output.

    ``probands`` are the proband ids (input order); ``est`` the posterior mean
    genetic-liability estimates; ``var`` the PA conditional-variance
    approximations; ``n_relatives`` the number of pedigree members beyond the
    proband per proband (0 = no recorded relatives within ``max_degree`` --
    the estimate is then the prior mean 0); ``degree_max`` the largest
    extracted relationship degree per proband."""
    probands: list
    est: np.ndarray
    var: np.ndarray
    n_relatives: np.ndarray
    degree_max: np.ndarray


def estimate_liabilities(ids, father, mother, *, probands, status, age,
                         cip_ages=None, cip_values=None, k_pop=None, h2=0.5,
                         max_degree=3, case_mode="pin", index_age=None,
                         strata=None, cip_by_stratum=None):
    """Estimate per-proband genetic liabilities from register records.

    Parameters
    ----------
    ids, father, mother
        Population trio records (as for
        :func:`~ltpred.pedigree.build_parent_graph`).
    probands
        The ids to estimate for (each must be among ``ids``).
    status, age
        Per-person arrays aligned to ``ids``: ``status`` True for an observed
        case, ``age`` the onset age for cases and the current/exit age
        otherwise.
    cip_ages, cip_values, k_pop
        The cumulative-incidence curve (e.g. from
        :func:`~ltpred.cip.aalen_johansen_cip`) used for everyone's
        thresholds. For stratum-specific curves pass ``strata`` and
        ``cip_by_stratum`` instead: ``strata`` are per-person labels aligned
        to ``ids`` and ``cip_by_stratum`` maps each label to
        ``(cip_ages, cip_values, k_pop)``.
    h2, max_degree, case_mode
        The liability-scale heritability; the relationship-degree limit for
        pedigree extraction (3 includes first cousins); and the case encoding
        (``"pin"`` for LT-FH++/ADuLT pinned onsets, ``"interval"`` for the
        age-dependent interval variant).
    index_age
        Optional per-proband ages (aligned to ``probands``) switching on
        prospective prediction: the proband's bound becomes uninformative and
        relatives are censored at the index age (familywise censoring).

    Returns
    -------
    PopulationScores
    """
    ids = list(ids)
    n = len(ids)
    if not (len(father) == len(mother) == n):
        raise ValueError("ids, father and mother must share length")
    status = np.asarray(status, dtype=bool)
    age = np.asarray(age, dtype=float)
    if status.shape != (n,) or age.shape != (n,):
        raise ValueError("status and age must be 1-D arrays aligned to ids")
    if not np.all(np.isfinite(age)):
        raise ValueError("age must contain only finite values")
    if case_mode not in ("pin", "interval"):
        raise ValueError("case_mode must be 'pin' or 'interval'")
    probands = list(probands)
    if index_age is not None:
        index_age = np.asarray(index_age, dtype=float)
        if index_age.shape != (len(probands),):
            raise ValueError("index_age must be 1-D and aligned to probands")
        if not np.all(np.isfinite(index_age)):
            raise ValueError("index_age must contain only finite values")
    if (strata is None) != (cip_by_stratum is None):
        raise ValueError("strata and cip_by_stratum must be given together")
    if strata is None and (cip_ages is None or cip_values is None):
        raise ValueError("supply either a single CIP (cip_ages, cip_values) "
                         "or strata + cip_by_stratum")
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != (n,):
            raise ValueError("strata must be 1-D and aligned to ids")
        missing = set(strata) - set(cip_by_stratum)
        if missing:
            raise ValueError(f"no CIP supplied for strata: {sorted(missing)}")

    graph = build_parent_graph(ids, father, mother)
    pos = graph.index
    missing_pb = [p for p in probands if p not in pos]
    if missing_pb:
        raise ValueError(f"probands not among ids: {missing_pb[:5]}")

    F = len(probands)
    est = np.empty(F)
    var = np.empty(F)
    n_rel = np.zeros(F, dtype=int)
    deg_max = np.zeros(F, dtype=int)

    for k, proband in enumerate(probands):
        ped = extract_pedigree(graph, proband, max_degree=max_degree)
        m = len(ped.ids)
        midx = np.array([pos[q] for q in ped.ids])
        m_status = status[midx].copy()
        m_age = age[midx].copy()

        if index_age is not None:
            # familywise censoring: every relative's follow-up stops at the
            # proband's index age (row 0 is the proband, handled below)
            ia = float(index_age[k])
            post = m_age > ia
            post[0] = False
            m_status[post] = False
            m_age[post] = ia

        if strata is None:
            lo, hi, _, _ = thresholds_from_cip(
                m_status, m_age, cip_ages, cip_values, k_pop=k_pop,
                case_mode=case_mode)
        else:
            lo = np.empty(m)
            hi = np.empty(m)
            ms = strata[midx]
            for s in set(ms):
                sel = ms == s
                ca, cv, ck = cip_by_stratum[s]
                lo_s, hi_s, _, _ = thresholds_from_cip(
                    m_status[sel], m_age[sel], ca, cv, k_pop=ck,
                    case_mode=case_mode)
                lo[sel] = lo_s
                hi[sel] = hi_s

        if index_age is not None:
            lo[0], hi[0] = -np.inf, np.inf      # uninformative proband bound

        _, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        e, v = estimate_liability_from_kinship(A, lo[None, :], hi[None, :],
                                               h2=h2, target=0)
        est[k] = e[0]
        var[k] = v[0]
        n_rel[k] = m - 1
        deg_max[k] = int(ped.degree.max()) if m else 0

    return PopulationScores(probands=probands, est=est, var=var,
                            n_relatives=n_rel, degree_max=deg_max)
