"""Supported population-register scoring from trio records and empirical CIPs.

The low-level pieces live in `ltpred.pedigree`, `ltpred.thresholds`
and `ltpred.estimate`. This module supplies the small but consequential
glue between them: row alignment, the distinction between pedigree closure and
the observation set, and calendar-time censoring for prospective prediction.

The first public driver is intentionally narrow. It uses pinned-onset LT-FH++
bounds and deterministic Pearson--Aitken inference. Interval-case encodings and
the PA-FGRS censored-control mixture remain available through the lower-level
APIs, but are separate observation models rather than switches hidden in this
register workflow.

The driver also reports on the two input problems that most often make a
register analysis wrong without making any single score wrong: parent
references that failed to resolve (counted per table; an implausibly high
unresolved share warns, and zero resolved references raises), and probands
who were not at risk at their prediction landmark (a per-proband state, with
a warning on prevalent cases).
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import validate_binary
from ._selected_kinship import _SelectedKinship, _covariance_reduction_is_safe
from .covariance import kinship_from_pedigree
from .estimate import estimate_liability_from_kinship
from .pedigree import build_parent_graph, extract_pedigree
from .thresholds import _cip_bounds, _validate_cip_curve
from .family import _is_missing_id, _is_missing_parent
from ._results import _TableExport

__all__ = ["PopulationScores", "estimate_liabilities"]

# Share of records carrying an unresolved non-null parent above which a
# register boundary stops being a plausible explanation on its own.
_UNRESOLVED_PARENT_WARN_FRACTION = 0.5
# A proband's relationships come from a dense A over its extracted pedigree
# when many members are selected (cheaper than per-pair recursion), and from
# bounded, memoized selected-pair recursion when few are or the pedigree is
# too deep for a dense A (1,500 members is 18 MB).
_DENSE_KINSHIP_MAX_MEMBERS = 1500
_DENSE_KINSHIP_COST_PER_MEMBER = 2


@dataclass
class PopulationScores(_TableExport):
    """Per-proband register-pipeline output, aligned to ``probands``.

    ``est`` is the Pearson--Aitken approximation to the posterior mean genetic
    liability and ``var`` its posterior-variance approximation. ``n_relatives``
    counts structural non-proband members reached within ``max_degree``;
    ``n_conditioned`` counts informative diagnosis bounds that actually enter
    the observation set (including the proband under ``use="gwas"``);
    ``n_closure_only`` counts ancestors added solely to preserve exact kinship;
    and ``degree_max`` is the largest non-closure degree present.

    ``frac_records_with_unresolved_parents`` is the fraction of population records with at
    least one non-null parent reference that matched no id; a high value means
    a register boundary (parents born before registration started) or a broken
    id join. ``proband_state`` is set under ``use="prediction"`` only (``None``
    under ``use="gwas"``) and classifies each proband at their landmark:
    ``"prevalent_case"`` (diagnosed at or before the landmark),
    ``"exited_before_index"`` (follow-up ended strictly before it), or
    ``"disease_free_and_followed"``. Only the last belongs in a prospective
    incident-risk evaluation.
    """

    probands: list
    est: np.ndarray
    var: np.ndarray
    n_relatives: np.ndarray
    n_conditioned: np.ndarray
    n_closure_only: np.ndarray
    degree_max: np.ndarray
    frac_records_with_unresolved_parents: float = 0.0
    proband_state: np.ndarray | None = None

    @property
    def se(self):
        """Zero Monte-Carlo error for deterministic PA, not approximation error."""
        return np.zeros_like(self.est)

    def to_dict(self):
        """Copy aligned columns, including pid, est, se, var and cohort diagnostics."""
        columns = {"pid": np.asarray(self.probands, dtype=object).copy(),
                   "se": self.se}
        for name in ("est", "var", "n_relatives", "n_conditioned", "n_closure_only",
                     "degree_max", "proband_state"):
            value = getattr(self, name)
            if value is not None:
                columns[name] = np.asarray(value).copy()
        columns["frac_records_with_unresolved_parents"] = np.full(
            len(self.probands), self.frac_records_with_unresolved_parents)
        return columns


def _validate_cip_inputs(n: int, *, cip_ages: ArrayLike | None,
                         cip_values: ArrayLike | None, k_pop: float | None,
                         strata: ArrayLike | None,
                         cip_by_stratum: Mapping | None
                         ) -> np.ndarray | None:
    """Validate the single-curve versus stratified-curve routing contract."""
    stratified = strata is not None or cip_by_stratum is not None
    if stratified:
        if strata is None or cip_by_stratum is None:
            raise ValueError("strata and cip_by_stratum must be given together")
        if cip_ages is not None or cip_values is not None or k_pop is not None:
            raise ValueError(
                "single-CIP inputs and strata + cip_by_stratum are mutually exclusive")
        if not isinstance(cip_by_stratum, Mapping):
            raise TypeError("cip_by_stratum must be a mapping")
        strata_array = np.asarray(strata)
        if strata_array.shape != (n,):
            raise ValueError("strata must be one-dimensional and aligned to ids")
        try:
            missing = {value for value in strata_array.tolist()
                       if value not in cip_by_stratum}
        except TypeError:
            raise TypeError("strata labels must be hashable") from None
        if missing:
            labels = sorted((repr(value) for value in missing))
            raise ValueError(f"no CIP supplied for strata: {labels}")
        return strata_array

    if cip_ages is None or cip_values is None:
        raise ValueError(
            "supply either cip_ages + cip_values or strata + cip_by_stratum")
    return None


def estimate_liabilities(
    ids: Sequence,
    father: Sequence,
    mother: Sequence,
    *,
    probands: Sequence,
    status: ArrayLike,
    age: ArrayLike,
    use: Literal["gwas", "prediction"],
    cip_ages: ArrayLike | None = None,
    cip_values: ArrayLike | None = None,
    k_pop: float | None = None,
    h2: float,
    max_degree: int = 3,
    birth_time: ArrayLike | None = None,
    index_time: ArrayLike | None = None,
    condition_closure: bool = False,
    strata: ArrayLike | None = None,
    cip_by_stratum: Mapping | None = None,
    kinship_cache_size: int = 100_000,
) -> PopulationScores:
    """Estimate genetic liabilities from population trio and diagnosis records.

    ``ids``/``father``/``mother`` are one population parent-pointer table.
    ``status`` and ``age`` are aligned to ``ids``: ``age`` is attained age at
    diagnosis for a case and attained age at last follow-up/exit for a control.
    Supply either one empirical population CIP (``cip_ages``, ``cip_values``,
    optional ``k_pop``) or per-person ``strata`` plus ``cip_by_stratum``, a mapping from each label
    to ``(cip_ages, cip_values, k_pop)``.

    ``use`` is required because it changes the observation set. ``"gwas"``
    includes the proband's own diagnosis. ``"prediction"`` leaves it
    uninformative and requires ``birth_time`` (one value per ``id``) plus
    ``index_time`` (one value per proband) on a common numeric calendar scale
    whose unit matches ``age``. For member ``j`` at proband index time ``t``,
    the censoring age is ``t - birth_time[j]``. A diagnosis or exit after ``t``
    becomes a control censored at that member-specific age; a member not yet
    born is uninformative. Thus relatives from different generations are never
    assigned the proband's attained age.

    Pedigree ancestors added only for exact kinship are uninformative by
    default, so ``max_degree`` bounds the diagnosis observations even though
    relationships retain their full ancestral contribution. Set
    ``condition_closure=True`` only to deliberately condition on those extra
    diagnoses. Inference is pinned-onset LT-FH++ with deterministic
    Pearson--Aitken; use the lower-level APIs for interval/mixture models.

    ``h2``: Liability-scale additive heritability for this disease. Required: there is no
    disease-independent default, for the same reason ``pop_prev`` has none. See
    the data-preparation guide, "Getting heritability on the liability scale", for choosing between pedigree/twin and
    SNP estimates and for the sensitivity analysis.

    Only target and informative observation relationships are used. They
    come from a dense matrix over the extracted pedigree when most members
    are selected, otherwise from selected pair recursion, where
    ``kinship_cache_size`` bounds the number of ancestor-pair results reused
    across probands (zero disables memoization). The parent graph is checked
    for cycles once. Near covariance singularities the full extracted matrix
    is retained to preserve the existing positive-definite correction.

    Two table/proband boundary conditions are reported rather than silently
    absorbed. A non-null parent reference that matches no id becomes a founder
    (a register boundary: parents born before registration started), and the
    fraction of records with at least one such reference is returned as
    ``PopulationScores.frac_records_with_unresolved_parents``; when **no** non-null
    reference resolves the table cannot be a boundary effect -- an id-format
    or join mismatch is the likely cause -- and the call raises, while an
    unresolved share above 50% warns. Under ``use="prediction"`` each proband
    is classified at their landmark into
    ``PopulationScores.proband_state`` and any ``"prevalent_case"`` warns;
    neither diagnostic alters the scores, and neither is a substitute for
    constructing an eligible incident-risk cohort.
    """
    ids = list(ids)
    father = list(father)
    mother = list(mother)
    n = len(ids)
    if not (len(father) == len(mother) == n):
        raise ValueError("ids, father and mother must share length")

    probands = list(probands)
    if not probands:
        raise ValueError("probands must contain at least one id")
    if any(_is_missing_id(pid) for pid in probands):
        raise ValueError("probands must not contain missing ids")
    if len(set(probands)) != len(probands):
        raise ValueError("probands must be unique; duplicate ids would duplicate scores")
    if h2 is None:
        raise ValueError("h2 must be a numeric liability-scale heritability in (0, 1]")
    if use not in ("gwas", "prediction"):
        raise ValueError("use must be 'gwas' or 'prediction'")
    if not isinstance(condition_closure, (bool, np.bool_)):
        raise TypeError("condition_closure must be boolean")
    condition_closure = bool(condition_closure)
    if isinstance(max_degree, bool) or not isinstance(max_degree, (int, np.integer)):
        raise TypeError("max_degree must be an integer")
    if max_degree < 1:
        raise ValueError("max_degree must be at least 1")

    status_array = validate_binary(status, name="status", ndim=1)
    age_array = np.asarray(age, dtype=float)
    if status_array.shape != (n,) or age_array.shape != (n,):
        raise ValueError("status and age must be one-dimensional and aligned to ids")
    if not np.all(np.isfinite(age_array)) or np.any(age_array < 0.0):
        raise ValueError("age must contain only finite, nonnegative values")

    strata_array = _validate_cip_inputs(
        n, cip_ages=cip_ages, cip_values=cip_values, k_pop=k_pop,
        strata=strata, cip_by_stratum=cip_by_stratum)
    # Validate each curve once, not once per proband: an empirical CIP can
    # carry one point per event age.
    if strata_array is None:
        curves = {None: _validate_cip_curve(cip_ages, cip_values, k_pop)}
    else:
        curves = {}
        for label in set(strata_array.tolist()):
            try:
                curve = cip_by_stratum[label]
                if len(curve) != 3:
                    raise ValueError
                curve_ages, curve_values, curve_k_pop = curve
            except (TypeError, ValueError):
                raise ValueError(
                    "each cip_by_stratum value must be "
                    "(cip_ages, cip_values, k_pop)") from None
            curves[label] = _validate_cip_curve(curve_ages, curve_values, curve_k_pop)

    def bounds(member_status, member_age, member_strata):
        if member_strata is None:
            return _cip_bounds(member_status, member_age, curves[None])[:2]
        lower = np.empty(member_status.shape[0])
        upper = np.empty(member_status.shape[0])
        for label in set(member_strata.tolist()):
            selected = member_strata == label
            lower[selected], upper[selected] = _cip_bounds(
                member_status[selected], member_age[selected], curves[label])[:2]
        return lower, upper

    birth_array = None
    index_array = None
    record_time = None
    if use == "gwas":
        if birth_time is not None or index_time is not None:
            raise ValueError(
                "birth_time/index_time are prediction inputs and must be omitted for use='gwas'")
    else:
        if birth_time is None or index_time is None:
            raise ValueError(
                "use='prediction' requires birth_time and index_time")
        birth_array = np.asarray(birth_time, dtype=float)
        index_array = np.asarray(index_time, dtype=float)
        if birth_array.shape != (n,):
            raise ValueError("birth_time must be one-dimensional and aligned to ids")
        if index_array.shape != (len(probands),):
            raise ValueError("index_time must be one-dimensional and aligned to probands")
        if not np.all(np.isfinite(birth_array)) or not np.all(np.isfinite(index_array)):
            raise ValueError("birth_time and index_time must contain only finite values")
        record_time = birth_array + age_array
        if not np.all(np.isfinite(record_time)):
            raise ValueError("birth_time + age must be finite")

    graph = build_parent_graph(ids, father, mother)
    selected_kinship = _SelectedKinship(graph, kinship_cache_size)
    pos = graph.index

    # An unlisted non-null parent is a founder here -- unavoidable at a
    # register boundary. But the same rule absorbs an id-format mismatch or a
    # failed join, which turns every family-history score into an
    # own-status-only score, so the resolution rate is surfaced instead of
    # silent. Shared missing-id markers and unlisted zero markers never count.
    n_refs = 0
    n_unresolved_records = 0
    for fid, mid in zip(father, mother):
        f_null = _is_missing_parent(fid, pos)
        m_null = _is_missing_parent(mid, pos)
        n_refs += (not f_null) + (not m_null)
        n_unresolved_records += ((not f_null and fid not in pos)
                                 or (not m_null and mid not in pos))
    frac_records_with_unresolved_parents = n_unresolved_records / n if n else 0.0
    if n_refs > 0 and graph.n_unresolved_parents == n_refs:
        raise ValueError(
            f"no non-null father/mother reference resolved against ids "
            f"({n_refs} given, 0 matched). A register boundary cannot explain "
            "this -- within-register parent links would still resolve. Check "
            "id formats and dtypes (e.g. integer ids with string parent "
            "references can never match).")
    if frac_records_with_unresolved_parents > _UNRESOLVED_PARENT_WARN_FRACTION:
        warnings.warn(
            f"{frac_records_with_unresolved_parents:.1%} of records carry a non-null "
            "parent reference that matches no id. Parents born before "
            "registration started are expected founders, but a share this "
            "high also results from an id-format mismatch or a failed join, "
            "which silently reduces every family-history score to the "
            "proband's own status. See PopulationScores."
            "frac_records_with_unresolved_parents.", UserWarning, stacklevel=2)
    missing_probands = [proband for proband in probands if proband not in pos]
    if missing_probands:
        raise ValueError(f"probands not among ids: {missing_probands[:5]}")
    if use == "prediction":
        proband_birth = np.array([birth_array[pos[proband]] for proband in probands])
        before_birth = np.flatnonzero(index_array < proband_birth)
        if before_birth.size:
            raise ValueError(
                "index_time must not precede the corresponding proband's birth_time; "
                f"invalid positions {before_birth.tolist()}")
        # The proband's own row is left uninformative, so a proband already
        # diagnosed -- or already out of follow-up -- at the landmark is scored
        # like any other, and only the caller can exclude them. Name the state
        # instead. A case exactly at the landmark is prevalent (the driver's
        # post-index test is strict); a control whose follow-up ends exactly
        # at the landmark counts as followed.
        proband_rows = np.array([pos[proband] for proband in probands])
        proband_record = record_time[proband_rows]
        proband_case = status_array[proband_rows]
        proband_state = np.where(
            proband_case & (proband_record <= index_array), "prevalent_case",
            np.where(~proband_case & (proband_record < index_array),
                     "exited_before_index", "disease_free_and_followed"))
        n_prevalent = int(np.count_nonzero(proband_state == "prevalent_case"))
        if n_prevalent:
            warnings.warn(
                f"{n_prevalent} of {len(probands)} probands were already "
                "diagnosed at or before their index_time "
                "(proband_state='prevalent_case'). They are not at-risk "
                "probands: including them in a prospective evaluation is "
                "cohort-level leakage and inflates discrimination. Restrict "
                "probands to the disease-free-and-followed cohort or treat "
                "these scores as non-prospective.",
                UserWarning, stacklevel=2)
    else:
        proband_state = None

    n_probands = len(probands)
    est = np.empty(n_probands)
    var = np.empty(n_probands)
    n_relatives = np.empty(n_probands, dtype=int)
    n_conditioned = np.empty(n_probands, dtype=int)
    n_closure_only = np.empty(n_probands, dtype=int)
    degree_max = np.empty(n_probands, dtype=int)

    if use == "gwas":            # every record enters unchanged: bounds once
        gwas_lower, gwas_upper = bounds(status_array, age_array, strata_array)

    for k, proband in enumerate(probands):
        ped = extract_pedigree(graph, proband, max_degree=max_degree)
        m = len(ped.ids)
        member_index = np.fromiter((pos[pid] for pid in ped.ids), dtype=np.intp,
                                   count=m)
        member_status = status_array[member_index].copy()
        member_age = age_array[member_index].copy()
        not_born = np.zeros(m, dtype=bool)

        if use == "prediction":
            member_birth = birth_array[member_index]
            cutoff_age = float(index_array[k]) - member_birth
            # Zero attained age also carries zero observable follow-up. Treat a
            # member born exactly at the landmark like one not yet born.
            not_born = cutoff_age <= 0.0
            post_index = record_time[member_index] > float(index_array[k])
            available_post_index = post_index & ~not_born
            member_status[available_post_index] = False
            member_age[available_post_index] = cutoff_age[available_post_index]
            # Supply a harmless finite placeholder to the threshold builder;
            # these rows become uninformative immediately below.
            member_status[not_born] = False
            member_age[not_born] = 0.0

        if use == "prediction":
            lower, upper = bounds(member_status, member_age,
                                  None if strata_array is None
                                  else strata_array[member_index])
        else:
            lower = gwas_lower[member_index]
            upper = gwas_upper[member_index]

        uninformative = np.zeros(m, dtype=bool)
        if not condition_closure:
            uninformative |= ped.closure_only
        if use == "prediction":
            uninformative |= not_born
            uninformative[0] = True
        lower[uninformative] = -np.inf
        upper[uninformative] = np.inf

        if _covariance_reduction_is_safe(h2, m):
            selected = np.flatnonzero(~uninformative)
            if uninformative[0]:
                selected = np.concatenate(([0], selected))
            n_sel = selected.size
            # Cached pair recursion costs about one lookup per selected pair
            # (ancestors are shared across probands); a dense A over the
            # extracted pedigree -- exact, since it holds every ancestor --
            # costs about _DENSE_KINSHIP_COST_PER_MEMBER lookups per member.
            if (m <= _DENSE_KINSHIP_MAX_MEMBERS
                    and n_sel * (n_sel + 1) // 2 > _DENSE_KINSHIP_COST_PER_MEMBER * m):
                _, full = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
                relationship = full[np.ix_(selected, selected)]
            else:
                relationship = selected_kinship.matrix(member_index[selected])
            inference_lower = lower[selected]
            inference_upper = upper[selected]
        else:
            # Repair depends on the full covariance, not only its selected
            # principal block. Preserve that behavior at h2=1 and nearby.
            _, relationship = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
            inference_lower, inference_upper = lower, upper
        estimate, _, posterior_var = estimate_liability_from_kinship(
            relationship, inference_lower[None, :], inference_upper[None, :], h2=h2,
            target=0, method="pearson-aitken")
        est[k] = estimate[0]
        var[k] = posterior_var[0]

        in_scope = ~ped.closure_only
        n_relatives[k] = int(np.count_nonzero(in_scope) - 1)
        n_conditioned[k] = int(np.count_nonzero(
            ~((lower == -np.inf) & (upper == np.inf))))
        n_closure_only[k] = int(np.count_nonzero(ped.closure_only))
        degree_max[k] = int(np.max(ped.degree[in_scope]))

    return PopulationScores(
        probands=probands,
        est=est,
        var=var,
        n_relatives=n_relatives,
        n_conditioned=n_conditioned,
        n_closure_only=n_closure_only,
        degree_max=degree_max,
        frac_records_with_unresolved_parents=frac_records_with_unresolved_parents,
        proband_state=proband_state,
    )
