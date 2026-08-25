"""Role-array covariance alignment and PA/Gibbs helpers."""

from __future__ import annotations

import numpy as np

from .covariance import construct_covmat_single, correct_positive_definite
from .gibbs import as_bounds
from .pearson_aitken import pa_estimate_batched
from ._validation import validate_bounds, validate_mixture_inputs
from ._estimate_group import (
    _check_unique_role_labels, _warn_if_corrected, _estimate_group,
)

def _align_to_cov(roles, cov_roles, columns, defaults):
    """Map ``(F, len(roles))`` column arrays onto ``cov_roles`` order by role name.

    ``roles`` labels the columns of each array in ``columns``; a ``cov_role`` absent
    from ``roles`` (e.g. the auto-added ``g``) is filled with the matching entry of
    ``defaults``. Returns a list of ``(F, d)`` arrays in ``cov_roles`` order.

    Each output keeps **its own** column's dtype, so a float32 ``lower`` paired
    with a float64 ``upper`` does not silently demote the latter."""
    role_to_col = {r: i for i, r in enumerate(roles)}
    F = columns[0].shape[0]
    d = len(cov_roles)
    out = [np.full((F, d), dv, dtype=col.dtype)          # keep each input's dtype
           for col, dv in zip(columns, defaults)]
    for p, cr in enumerate(cov_roles):
        j = role_to_col.get(cr)
        if j is not None:
            for a, col in enumerate(columns):
                out[a][:, p] = col[:, j]
    return out


def _single_trait_cov(roles, h2, c2, m2, engine, *, canonical=False):
    """Single-trait covariance with a PD nudge.

    Pearson-Aitken passes ``canonical=True`` so the sequential fold order is
    a property of the role *set*. Gibbs keeps the caller order: the sweep
    consumes RNG per coordinate, so sorting would change seeded draws."""
    fam_vec = sorted(roles) if canonical else list(roles)
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True,
                                      h2=h2, c2=c2, m2=m2)
    cov, n_corrections = correct_positive_definite(cov_obj.matrix)
    _warn_if_corrected(n_corrections, engine)
    return cov_obj, cov


def _target_index(cov_roles, out_coord):
    """Row of ``g`` (``out_coord==0``) or ``o`` in a single-trait covariance."""
    return cov_roles.index("g") if out_coord == 0 else cov_roles.index("o")


def _pa_from_role_arrays(roles, lower, upper, h2, out_coords, K_i=None,
                         K_pop=None, use_mixture=False, c2=None, m2=None,
                         mixture_require_pair=True):
    """PA estimates for one or more targets on same-structure role arrays.

    ``roles`` labels the columns of ``lower``/``upper`` (no ``g`` -- the
    constructor adds it). Returns ``(est, var)`` dicts keyed by out-coord.
    ``mixture_require_pair=False`` skips the \"at least one valid K pair\" gate
    for callers that already ran it globally over every family: a structure
    group may legitimately contain no mixture pair of its own (e.g. every
    family in the group is all-cases), so re-requiring one per group would
    reject a call the global gate accepted."""
    roles = list(roles)
    _check_unique_role_labels(roles)
    lower = as_bounds(lower)
    upper = as_bounds(upper)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[1] != len(roles):
        raise ValueError(
            f"lower and upper must be (n_families, {len(roles)}) -- one "
            f"column per role {roles}; got {lower.shape} and {upper.shape}")
    validate_bounds(lower, upper, context="array estimator bounds")
    cov_obj, cov = _single_trait_cov(
        roles, h2, c2, m2, "Pearson-Aitken estimation", canonical=True)
    lo, hi = _align_to_cov(roles, cov_obj.roles, (lower, upper),
                           (-np.inf, np.inf))
    ki = kp = None
    if use_mixture:
        K_i, K_pop = validate_mixture_inputs(
            K_i, K_pop, expected_shape=lower.shape,
            require_pair=mixture_require_pair,
            lower=lower, upper=upper,
            context="array estimator mixture inputs")
        ki, kp = _align_to_cov(
            roles, cov_obj.roles, (as_bounds(K_i), as_bounds(K_pop)),
            (np.nan, np.nan))
    est, var = {}, {}
    for coord in out_coords:
        e, v = pa_estimate_batched(
            cov, lo, hi, target=_target_index(cov_obj.roles, coord),
            K_is=ki, K_pops=kp)
        est[coord] = e
        var[coord] = v
    return est, var


def _gibbs_from_role_arrays(roles, lower, upper, h2, out_coords, seeds,
                            tol, n_sim, burn_in, max_rounds, c2=None, m2=None):
    """Gibbs estimates for one or more targets on same-structure role arrays.

    Returns ``(est, se, var)`` exactly as :func:`_estimate_group` does."""
    roles = list(roles)
    _check_unique_role_labels(roles)
    lower = as_bounds(lower)
    upper = as_bounds(upper)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[1] != len(roles):
        raise ValueError(
            f"lower and upper must be (n_families, {len(roles)}) -- one "
            f"column per role {roles}; got {lower.shape} and {upper.shape}")
    validate_bounds(lower, upper, context="array estimator bounds")
    cov_obj, cov = _single_trait_cov(roles, h2, c2, m2, "Gibbs sampling")
    lo, hi = _align_to_cov(roles, cov_obj.roles, (lower, upper),
                           (-np.inf, np.inf))
    targets = [_target_index(cov_obj.roles, c) for c in out_coords]
    return _estimate_group(cov, targets, lo, hi, seeds, tol, n_sim, burn_in,
                           max_rounds)


def _scalar_member_bounds(member, fam_id):
    """Return one member's single-trait bounds with the public shape error."""
    lo = np.asarray(member.lower, dtype=float)
    hi = np.asarray(member.upper, dtype=float)
    if lo.size != 1 or hi.size != 1:
        raise ValueError(
            f"family {fam_id!r} has length-{max(lo.size, hi.size)} bounds; "
            "the single-trait estimator needs scalar bounds per member -- "
            "use estimate_liability's multi-trait model")
    return float(lo.reshape(())), float(hi.reshape(()))


def _stack_object_members(families, idx, roles, dtype, use_mixture=False):
    """Stack one structure group's member scalars into ``(F, len(roles))`` arrays.

    ``roles`` is the group's user-role key (no ``g``). Missing ``o`` is not
    inserted here — :func:`_align_to_cov` fills it as unbounded."""
    F, k = len(idx), len(roles)
    lowers = np.empty((F, k), dtype=dtype)
    uppers = np.empty((F, k), dtype=dtype)
    K_is = np.empty((F, k), dtype=dtype) if use_mixture else None
    K_pops = np.empty((F, k), dtype=dtype) if use_mixture else None
    pids = []
    for slot, f in enumerate(idx):
        fam = families[f]
        by_role = {m.role: m for m in fam.members}
        for j, role in enumerate(roles):
            member = by_role[role]
            lowers[slot, j], uppers[slot, j] = _scalar_member_bounds(
                member, fam.fam_id)
            if use_mixture:
                K_is[slot, j] = (np.nan if member.K_i is None
                                 else float(member.K_i))
                K_pops[slot, j] = (np.nan if member.K_pop is None
                                   else float(member.K_pop))
        o = by_role.get("o")
        pids.append(o.pid if o is not None and o.pid is not None else fam.fam_id)
    return lowers, uppers, K_is, K_pops, pids
