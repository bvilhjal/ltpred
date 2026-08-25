"""Object-path PA/Gibbs engines."""

from __future__ import annotations

import numpy as np

from .covariance import construct_covmat_multi, correct_positive_definite
from ._validation import validate_mixture_inputs
from ._estimate_core import (
    LiabilityResult, _OUT_NAMES, _bounds_dtype, _normalise_out,
    _ordered_thresholds,
)
from ._estimate_group import (
    _assert_nonempty_families, _check_unique_roles,
    _validate_multitrait_bounds, _warn_unconverged, _warn_if_corrected,
    _group_by_structure, _base_seeds, _estimate_group,
)
from ._estimate_role_arrays import (
    _stack_object_members, _gibbs_from_role_arrays, _pa_from_role_arrays,
    _scalar_member_bounds,
)

def _estimate_liability_single(families, h2=0.5, out=("genetic",), tol=0.01, n_sim=100_000, burn_in=1000, seed=None, max_rounds=100, dtype=np.float64, c2=None, m2=None):
    """Estimate genetic/full liabilities for one trait, family by family.

    ``families`` is a list of :class:`~ltpred.family.Family` (build one from flat
    columns with :func:`~ltpred.family.families_from_columns`). Families sharing a
    structure are sampled together in the parallel kernel. ``out`` selects
    ``\"genetic\"`` and/or ``\"full\"``. ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) at negligible accuracy
    cost. ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or ``None``.
    Returns a :class:`LiabilityResult` whose arrays line up with ``families``."""
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    names = [_OUT_NAMES[c] for c in out_coords]
    n = len(families)
    seeds = _base_seeds(seed, n, max_rounds)

    est = {name: np.empty(n) for name in names}
    se = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for role_key, idx in _group_by_structure(families):
        # Keep the first family's member order so seeded Gibbs sweeps match
        # the kinship path on the same pedigree (coordinate order is the RNG
        # order). Bounds are stacked by those role names, not by arrival order
        # of later families in the group.
        roles = [m.role for m in families[idx[0]].members]
        lowers, uppers, _ki, _kp, group_pids = _stack_object_members(
            families, idx, roles, dtype)
        g_est, g_se, g_var = _gibbs_from_role_arrays(
            roles, lowers, uppers, h2, out_coords, seeds[idx],
            tol, n_sim, burn_in, max_rounds, c2=c2, m2=m2)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]
                var[name][f] = g_var[slot, c]

    _warn_unconverged(se, names, tol, max_rounds, n)
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)


def _estimate_liability_pa(families, h2=0.5, out=("genetic",), use_mixture=False,
                           dtype=np.float64, c2=None, m2=None):
    """Deterministic Pearson-Aitken liability inference for one trait.

    ``c2``/``m2`` wire sibship (``C``) and couple (``M``) shared-environment
    components into the family covariance (``h2 + c2 + m2 <= 1`` required).

    A deterministic sequential-moment alternative to
    ``_estimate_liability_single``: no sampling, tolerance, or Monte-Carlo
    error. It is exact for a single truncation but approximate for multiple
    sequential truncations. ``out=(\"genetic\",)`` estimates the proband's
    genetic liability ``g`` conditional on the whole family;
    ``\"full\"`` is ``E[l_o | own interval and relatives]`` — the target's own
    bound is applied after the relative fold, matching Gibbs. Unbind ``o``
    (or omit it) for a relatives-only predictor. ``use_mixture=True`` turns
    on the age-censored-control mixture, using each member's ``K_i``/``K_pop``
    (see :func:`ltpred.thresholds.pa_thresholds`).
    ``dtype=np.float32`` halves the per-family bound memory. Returns a
    :class:`LiabilityResult` with ``se = 0`` and PA approximations to conditional
    variances in ``var``."""
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    if use_mixture:
        n_members = sum(len(family.members) for family in families)
        lower = np.empty(n_members)
        upper = np.empty(n_members)
        K_i, K_pop = [], []
        slot = 0
        for family in families:
            for member in family.members:
                lower[slot], upper[slot] = _scalar_member_bounds(
                    member, family.fam_id)
                K_i.append(np.nan if member.K_i is None else member.K_i)
                K_pop.append(np.nan if member.K_pop is None else member.K_pop)
                slot += 1
        validate_mixture_inputs(
            K_i, K_pop, expected_shape=(n_members,), require_pair=True,
            lower=lower, upper=upper,
            context="family mixture inputs")
        del K_i, K_pop, lower, upper
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    names = [_OUT_NAMES[c] for c in out_coords]
    n = len(families)

    est = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for role_key, idx in _group_by_structure(families):
        # PA is a sequential approximation, so its fold order must be a property
        # of the family structure rather than whichever member/family happened
        # to arrive first. `_pa_from_role_arrays` canonicalises via sorted roles.
        roles = list(role_key)
        lowers, uppers, K_is, K_pops, group_pids = _stack_object_members(
            families, idx, roles, dtype, use_mixture=use_mixture)
        group_est, group_var = _pa_from_role_arrays(
            roles, lowers, uppers, h2, out_coords, K_i=K_is, K_pop=K_pops,
            use_mixture=use_mixture, c2=c2, m2=m2,
            mixture_require_pair=False)  # the global gate above already ran
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in zip(out_coords, names):
                est[name][f] = group_est[c][slot]
                var[name][f] = group_var[c][slot]

    se = {name: np.zeros(n) for name in names}   # deterministic: no Monte-Carlo error
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)


def _estimate_liability_multi(families, h2_vec, genetic_corrmat, full_corrmat,
                              phen_names=None, out=("genetic",), tol=0.01,
                              n_sim=100_000, burn_in=1000, seed=None,
                              max_rounds=100, dtype=np.float64):
    """Estimate genetic/full liabilities jointly across several correlated traits.

    Each member's ``lower``/``upper`` must be length-``n_pheno`` sequences (one
    interval per phenotype, in ``phen_names`` order). Builds the phenotype-major
    multi-trait covariance, samples same-structure families together, and returns a
    :class:`LiabilityResult` with one column per (output, phenotype), e.g.
    ``\"genetic_<phen>\"``. ``dtype=np.float32`` halves the per-family bound memory.
    ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or ``None``.
    Port of LTFHPlus::estimate_liability_multi."""
    h2_vec = np.asarray(h2_vec, dtype=float)
    n_pheno = len(h2_vec)
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(n_pheno)]
    elif len(set(phen_names)) != len(phen_names):
        # result dicts are keyed by (output, phenotype) name — a duplicate name
        # would silently collapse two traits' columns onto one key
        raise ValueError(
            f"phen_names contains duplicates {phen_names!r}; each phenotype "
            "needs a distinct name")
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    _validate_multitrait_bounds(families, n_pheno)
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    col_names = [f"{_OUT_NAMES[c]}_{phen_names[p]}"
                 for p in range(n_pheno) for c in out_coords]

    n = len(families)
    seeds = _base_seeds(seed, n, max_rounds)
    est = {name: np.empty(n) for name in col_names}
    se = {name: np.empty(n) for name in col_names}
    var = {name: np.empty(n) for name in col_names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for _key, idx in _group_by_structure(families):
        roles = [m.role for m in families[idx[0]].members]
        cov_obj = construct_covmat_multi(fam_vec=roles, add_ind=True,
                                         genetic_corrmat=genetic_corrmat,
                                         full_corrmat=full_corrmat,
                                         h2_vec=h2_vec, phen_names=phen_names)
        cov, n_corrections = correct_positive_definite(cov_obj.matrix)
        _warn_if_corrected(n_corrections, "Gibbs sampling")
        k_roles = len(cov_obj.roles) // n_pheno
        fam_roles = cov_obj.roles[:k_roles]
        o_pos = fam_roles.index("o") if "o" in fam_roles else None
        gibbs_out = sorted(c + k_roles * p for p in range(n_pheno) for c in out_coords)

        lowers, uppers, group_pids = [], [], []
        for f in idx:
            lo, hi, mpids = _ordered_thresholds(families[f], fam_roles)  # (k_roles, n_pheno)
            lowers.append(lo.T.reshape(-1))   # phenotype-major
            uppers.append(hi.T.reshape(-1))
            group_pids.append(mpids[o_pos] if o_pos is not None and mpids[o_pos] is not None
                              else families[f].fam_id)
        lowers = np.array(lowers, dtype=dtype)
        uppers = np.array(uppers, dtype=dtype)

        g_est, g_se, g_var = _estimate_group(
            cov, gibbs_out, lowers, uppers, seeds[idx], tol, n_sim, burn_in,
            max_rounds)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(col_names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]
                var[name][f] = g_var[slot, c]

    _warn_unconverged(se, col_names, tol, max_rounds, n)
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)
