"""Gibbs group sampler and family-structure helpers."""

from __future__ import annotations

import warnings

import numpy as np

from .gibbs import (gibbs_params, gibbs_estimate_batched, _MAX_SEED,
                    _validate_seed, _validate_burn_in)
from ._estimate_core import (
    _validate_mc_se_n_sim, _validate_tol, _validate_max_rounds,
)

def _estimate_group(cov, out_idx, lowers, uppers, base_seeds, tol, n_sim,
                    burn_in, max_rounds):
    """Sample every family in one structure group and iterate to tolerance.

    ``lowers``/``uppers`` are ``(F, d)`` truncation bounds for the ``F`` families that
    share covariance ``cov``; ``out_idx`` are the coordinate indices to estimate.
    The one-time :func:`gibbs_params` factorisation is reused across families and
    rounds; coordinates that are untruncated in every family (the genetic
    rows on the public path) are collapsed out of the sweep. Each round runs
    the parallel kernel over the still-unconverged families and pools their
    batch means (fixed batch size ``b``) so earlier draws are not wasted. Returns ``(est, se, var)`` of shape ``(F, ncols)``, where
    ``var`` is the Monte-Carlo estimate of the target's **posterior** variance
    (pooled over rounds from the streamed sums of squares) and ``se`` the
    batch-means error of ``est``.

    The single choke point every Gibbs estimate path funnels through, so the
    shared sampler controls (``tol``, ``n_sim``, ``burn_in``, ``max_rounds``) are
    validated once here for all of them."""
    n_sim = _validate_mc_se_n_sim(n_sim)
    burn_in = _validate_burn_in(burn_in)
    tol = _validate_tol(tol)
    max_rounds = _validate_max_rounds(max_rounds)
    F = lowers.shape[0]
    ncols = len(out_idx)
    out_idx = np.asarray(out_idx, dtype=np.int64)
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))

    b = max(int(np.floor(np.sqrt(n_sim))), 2)
    nb = n_sim // b

    # streaming batch-means accumulators (no per-family batch-mean arrays):
    tot = np.zeros((F, ncols))          # sum of samples -> mean
    tot_sq = np.zeros((F, ncols))       # sum of squares -> posterior variance
    total_n = np.zeros(F)               # total kept samples
    bm_s1 = np.zeros((F, ncols))        # sum of batch means Y_k across rounds
    bm_s2 = np.zeros((F, ncols))        # sum of Y_k^2 across rounds
    bm_m = np.zeros(F)                  # total number of batches
    est = np.zeros((F, ncols))
    se = np.full((F, ncols), np.inf)
    var = np.zeros((F, ncols))

    active = np.arange(F)
    rnd = 0
    while active.size and rnd < max_rounds:
        # wrap again: the per-round offset can carry a base seed past uint32
        seeds = np.where(base_seeds[active] < 0, -1,
                         (base_seeds[active] + rnd) % (_MAX_SEED + 1))
        ts, tsq, s1, s2 = gibbs_estimate_batched(
            P, sd, sd0, lowers[active], uppers[active],
            out_idx, n_sim, burn_in, b, nb, seeds, cov=cov)
        still = []
        for ai, f in enumerate(active):
            tot[f] += ts[ai]
            tot_sq[f] += tsq[ai]
            total_n[f] += n_sim
            bm_s1[f] += s1[ai]
            bm_s2[f] += s2[ai]
            bm_m[f] += nb
            m = bm_m[f]
            # sum((Y - Ybar)^2) = S2 - S1^2 / M  (pooled over rounds, batch size b)
            ss = bm_s2[f] - bm_s1[f] ** 2 / m
            sigma2 = b * ss / (m - 1)
            # batchmeans convention: the SE denominator is the m*b draws that
            # enter the batches, not the total kept draws (n_sim >= nb*b).
            se[f] = np.sqrt(np.maximum(sigma2, 0.0) / (m * b))
            est[f] = tot[f] / total_n[f]
            # E[x^2] - E[x]^2 over every retained draw; a pinned (fixed)
            # coordinate gives exactly 0, and rounding cannot make a variance
            # negative, so clamp.
            var[f] = np.maximum(tot_sq[f] / total_n[f] - est[f] ** 2, 0.0)
            if not np.all(se[f] <= tol):
                still.append(f)
        active = np.array(still, dtype=int)
        rnd += 1
    return est, se, var


def _assert_nonempty_families(families):
    """Reject a family that carries no observed member at all.

    Such a family has nothing to condition on, so every estimator would
    return the prior mean 0 — indistinguishable in the output from a genuine
    estimate that happens to land near zero. In practice it almost always means
    a join dropped the rows rather than that the proband is truly unobserved.
    A warning used to leave that zero in the GWAS phenotype if it was ignored.
    """
    empty = [fam.fam_id for fam in families if not fam.members]
    if empty:
        shown = ", ".join(repr(fid) for fid in empty[:5])
        more = "" if len(empty) <= 5 else f", ... (+{len(empty) - 5} more)"
        raise ValueError(
            f"{len(empty)} of {len(families)} families have no members "
            f"({shown}{more}); with nothing to condition on the estimate "
            "would be the prior mean 0, indistinguishable from a real score. "
            "Drop those families or fix the join that dropped their member rows.")


def _check_unique_roles(families):
    """Reject a family with a duplicated role (two rows both ``s1``, etc.) or a
    user-supplied ``g`` member.

    Each role names one individual, so a repeat would silently merge two relatives
    into one covariance coordinate. Number repeated relatives instead (``s1``,
    ``s2``). ``g`` is never legitimate user input: the estimator adds the genetic
    coordinate itself, so a supplied ``g`` row would silently condition that
    coordinate on the member's bounds."""
    for fam in families:
        roles = [m.role for m in fam.members]
        if "g" in roles:
            raise ValueError(
                f"family {fam.fam_id!r} has a member with role 'g'; the genetic "
                "liability coordinate is added by the estimator and cannot be "
                "observed — remove that member (the proband's own status is "
                "role 'o').")
        if len(roles) != len(set(roles)):
            dup = sorted({r for r in roles if roles.count(r) > 1})
            raise ValueError(
                f"family {fam.fam_id!r} has duplicate role(s) {dup}; each role "
                "names one individual — number repeated relatives (s1, s2, ...).")


def _check_unique_role_labels(roles):
    """Reject duplicate column labels or a ``g`` column in the array estimators'
    ``roles`` — the same contract as :func:`_check_unique_roles` for the object
    paths: each column identifies a different family member, and ``g`` is added
    by the estimator rather than supplied."""
    if "g" in roles:
        raise ValueError(
            "roles must not contain 'g'; the genetic liability coordinate is "
            "added by the estimator, so a supplied 'g' column would silently "
            "condition it — remove that column")
    if len(roles) != len(set(roles)):
        raise ValueError("roles contains duplicate role labels; each column must "
                         "identify a different family member")


def _validate_multitrait_bounds(families, n_pheno):
    """Require one explicit bound per phenotype on every observed member.

    Broadcasting a scalar across traits silently asserts that the same phenotype
    observation was made for every trait. That is almost never an intentional
    multi-trait input, so the object APIs require exact one-dimensional
    ``(n_pheno,)`` bounds instead.
    """
    for fam in families:
        for member in fam.members:
            for name in ("lower", "upper"):
                value = np.asarray(getattr(member, name))
                if value.ndim != 1 or value.shape[0] != n_pheno:
                    raise ValueError(
                        f"family {fam.fam_id!r} member {member.role!r} {name} "
                        f"must be a length-{n_pheno} one-dimensional sequence "
                        "(one bound per phenotype); scalar bounds are only valid "
                        "for single-trait estimation")


def _warn_unconverged(se, names, tol, max_rounds, n):
    """Warn (once) if any family's batch-means SE is still above ``tol``."""
    unconverged = np.zeros(n, dtype=bool)
    for name in names:
        unconverged |= se[name] > tol
    if unconverged.any():
        warnings.warn(
            f"{int(unconverged.sum())} of {n} families did not reach tol={tol} "
            f"within max_rounds={max_rounds}; their reported se exceed tol — "
            "increase max_rounds or n_sim, or inspect res.se.", stacklevel=3)


def _warn_if_corrected(n_corrections, engine):
    """Warn when the family covariance needed nudging to strict PD.

    Reported by every estimator path (the multi-trait Gibbs path has always
    warned): a (numerically) singular covariance usually marks a degenerate
    model -- e.g. ``h2 = 1`` makes ``g`` and ``o`` perfectly correlated -- and
    silently repairing it would hide that from the user."""
    if n_corrections:
        warnings.warn(
            "The covariance was singular or numerically singular and was nudged "
            f"to strict positive definiteness for {engine}.",
            RuntimeWarning, stacklevel=3)


def _group_by_structure(families):
    """Bucket families by their role *set* so a bucket shares one covariance.

    The key is the **sorted** roles, so families with the same relatives in a
    different row order (``[o, m, f]`` vs ``[o, f, m]``) land in one bucket rather
    than fragmenting into smaller Numba batches — bounds are aligned by role name
    downstream, so order within a family does not matter. Yields
    ``(role_tuple, [global_index, ...])``."""
    groups = {}
    order = []
    for i, fam in enumerate(families):
        key = tuple(sorted(m.role for m in fam.members))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [(key, groups[key]) for key in order]


def _base_seeds(seed, n, max_rounds):
    """Per-family base seeds; family ``i`` owns the block starting at ``i*max_rounds``
    so each round of :func:`_estimate_group` gets its own stream.

    ``-1`` is the kernel's *unseeded* sentinel. It must stay reachable only from
    ``seed=None``: validating here keeps a user's negative seed from silently
    landing on it (which would leave family 0 non-reproducible), and wrapping keeps
    the derived block inside the uint32 range the kernel's RNG accepts."""
    if seed is None:
        return np.full(n, -1, dtype=np.int64)
    seed = _validate_seed(seed)
    return (seed + np.arange(n, dtype=np.int64) * int(max_rounds)) % (_MAX_SEED + 1)
