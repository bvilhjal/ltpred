"""Posterior genetic and full liability from family truncation bounds.

`estimate_liability` handles `Family` objects, the array functions handle one
role set per call, and `estimate_liability_from_kinship` handles arbitrary
pedigrees. Single-trait inference defaults to deterministic Pearson--Aitken
(PA); multi-trait inference uses Gibbs; additive nuclear families can use
quadrature. PA alone implements the ``K_i``/``K_pop`` censoring mixture.

Families are validated and grouped by role set before calling the shared
kernels. ``var`` is posterior variance; ``se`` is Monte-Carlo error only.
See ``docs/estimation.md`` and ``docs/algorithm.md`` for the models and
fold-order details.
"""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .covariance import (construct_covmat_single, construct_covmat_multi,
                         construct_covmat_from_kinship, correct_positive_definite,
                         _PSD_CERTIFIED, _PD_CERTIFIED)
from .gibbs import (gibbs_params, _gibbs_round, _as_bounds, _MAX_SEED,
                    _validate_seed, _validate_burn_in)
from .pearson_aitken import pa_estimate_batched, _tnorm_moments_loc
from ._numba import _jit
from ._validation import validate_bounds, validate_mixture_inputs
from .family import _member_pid_key, _proband_pid
from ._results import _TableExport

__all__ = ["LiabilityResult", "estimate_liability",
           "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
           "estimate_liability_from_kinship"]

_PA_METHODS = {"pa", "pearson-aitken"}


def _warn_unused_sampler_controls(method, tol, n_sim, burn_in, seed, max_rounds):
    """Warn when a deterministic engine receives non-default Gibbs controls."""
    controls = {"tol": (tol, 0.01), "n_sim": (n_sim, 100_000),
                "burn_in": (burn_in, 1000), "seed": (seed, None),
                "max_rounds": (max_rounds, 100)}
    changed = [name for name, (value, default) in controls.items()
               if value != default]
    if changed:
        warnings.warn(
            f"{method} is deterministic and ignores Gibbs controls: "
            + ", ".join(changed) + ". Use method='gibbs' to sample.",
            UserWarning, stacklevel=3)


def _resolve_method(method, default):
    """Normalize a supported engine spelling; model restrictions belong to the caller.

    The single method-name -> engine gate shared by `estimate_liability`
    and `estimate_liability_from_kinship`: ``None`` maps to ``default``,
    the PA alias ``"pa"`` to ``"pearson-aitken"``; ``"gibbs"`` and
    ``"quadrature"`` keep their names.
    Other spellings raise."""
    if method is None:
        method = default
    name = str(method).lower()
    if name in _PA_METHODS:
        return "pearson-aitken"
    if name in {"gibbs", "quadrature"}:
        return name
    raise ValueError(f"unknown method {method!r}; use 'gibbs', 'pearson-aitken' or 'quadrature'")


_OUT_COORDS = {"genetic": 0, "full": 1}
_OUT_NAMES = {0: "genetic", 1: "full"}


def _bounds_dtype(dtype):
    """Validate the per-family bounds dtype: float32 (half memory) or float64."""
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    return dt


def _out_entries(out):
    """Return a non-empty list of requested estimator output names."""
    if isinstance(out, str):
        entries = [out]
    else:
        try:
            entries = list(out)
        except TypeError:
            raise TypeError(
                "out must be 'genetic'/'full' or a non-empty sequence of them"
            ) from None
    if not entries:
        raise ValueError("out must contain at least one output name")
    return entries


def _resolve_out_entry(value):
    """Resolve one output name; only the exact strings ``"genetic"``/``"full"``."""
    if not isinstance(value, str):
        raise TypeError(f"out entry {value!r} must be one of genetic/full, "
                        f"not {type(value).__name__}")
    if value not in _OUT_COORDS:
        raise ValueError(f"out entry {value!r} must be one of genetic/full")
    return _OUT_COORDS[value]


@dataclass
class LiabilityResult(_TableExport):
    """Aligned liability estimates and numerical diagnostics.

    ``est``, ``se`` and ``var`` map output names to per-family arrays. ``var``
    is posterior uncertainty, whereas ``se`` is Monte-Carlo error in ``est``
    (zero for deterministic engines, not proof of exactness). Multi-trait names
    are suffixed by phenotype. ``pids`` contains proband ids, falling back to
    ``fam_id`` only for wholly pid-free families. Quadrature diagnostics record
    refinement change and nodes per active dimension, not an error bound.
    """
    fam_ids: np.ndarray
    pids: object
    est: dict
    se: dict
    var: dict = None
    quadrature_error: dict | None = None
    quadrature_nodes: dict | None = None

    def to_dict(self):
        """Return copied columns, including requested numerical diagnostics."""
        columns = {"fam_id": np.asarray(self.fam_ids, dtype=object).copy(),
                   "pid": np.asarray(self.pids, dtype=object).copy()}
        for prefix, values in (("", self.est), ("se_", self.se), ("var_", self.var),
                               ("quadrature_error_", self.quadrature_error),
                               ("quadrature_nodes_", self.quadrature_nodes)):
            if values is not None:
                columns.update({prefix + name: np.asarray(value).copy()
                                for name, value in values.items()})
        return columns

    @property
    def genetic(self):
        """Return the single-trait ``est['genetic']`` array."""
        if "genetic" not in self.est:
            raise AttributeError(
                "no 'genetic' column; multi-trait results use per-trait keys like "
                "'genetic_<trait>' — index .est directly")
        return self.est["genetic"]


def _normalise_out(out):
    """Resolve ``out`` to sorted unique coordinates (0 = ``g``, 1 = ``o``)."""
    coords = [_resolve_out_entry(value) for value in _out_entries(out)]
    return sorted(set(coords))


def _single_out(out):
    """Resolve ``out`` to one column index for the APIs that return a single array.

    Accepts ``"genetic"``/``"full"`` or a length-1 sequence, so every estimator
    takes the same spellings as `estimate_liability`."""
    entries = _out_entries(out)
    if len(entries) != 1:
        raise ValueError("this API returns a single column; out must be one of "
                         "genetic/full (or a length-1 sequence)")
    return _resolve_out_entry(entries[0])


def _validate_mc_se_n_sim(n_sim):
    """Return an integer draw count large enough for the Gibbs MC-SE rule."""
    if isinstance(n_sim, (bool, np.bool_)):
        raise TypeError("n_sim must be an integer >= 4, not bool")
    try:
        n_sim = operator.index(n_sim)
    except TypeError:
        raise TypeError("n_sim must be an integer >= 4") from None
    if n_sim < 4:
        raise ValueError(
            "n_sim must be at least 4 to form two batches for the Monte-Carlo SE")
    return n_sim


def _validate_tol(tol):
    """Return the Gibbs convergence tolerance as a finite positive float.

    A NaN tolerance never satisfies ``se <= tol``, so the sampler runs to
    ``max_rounds`` yet the unconverged warning (keyed on the same comparison)
    stays silent; a non-positive one can never be reached."""
    if isinstance(tol, (bool, np.bool_)):
        raise TypeError("tol must be a positive real number, not bool")
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        raise TypeError("tol must be a positive real number") from None
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")
    return tol


def _validate_max_rounds(max_rounds):
    """Return ``max_rounds`` as a positive int: with none, the convergence loop
    never runs and every family keeps its all-zero initial estimates."""
    if isinstance(max_rounds, (bool, np.bool_)):
        raise TypeError("max_rounds must be a positive integer, not bool")
    try:
        max_rounds = operator.index(max_rounds)
    except TypeError:
        raise TypeError("max_rounds must be a positive integer") from None
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    return max_rounds


def _batch_means(samples: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Batch-means estimate and Monte-Carlo SE of column means (Jones et al. 2006).

    Splits ``n`` samples into ``a = n // b`` consecutive batches of size
    ``b = floor(sqrt(n))``; ``est`` is the mean of all ``n`` samples and
    ``se = sqrt(b * var(batch means) / (a*b))``. R ``batchmeans::bmmat`` instead
    reports the mean of the batch means and divides by ``n``; the two agree when
    ``b`` divides ``n``. Accepts a 1-D or 2-D ``(n, ncols)`` array and returns
    ``(est, se)`` arrays over columns (length-1 arrays for 1-D input).
    Adapted from R ``batchmeans::bmmat``, the rule LTFHPlus uses to decide the
    Gibbs sampler has converged. (The estimator applies it online to its
    streamed values, keeping the first round's batch size as rounds pool; this
    stays for direct use and tests.) Requires at least 4 samples (two batches),
    the same minimum the estimator enforces on ``n_sim``."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 4:
        raise ValueError(
            "batch_means needs at least 4 samples to form two batches for the "
            "Monte-Carlo SE")
    b = int(np.floor(np.sqrt(n)))
    a = n // b
    used = x[:a * b].reshape(a, b, x.shape[1])
    batch_mean = used.mean(axis=1)             # (a, ncols)
    mu = batch_mean.mean(axis=0)               # (ncols,)
    sigma2 = b * np.sum((batch_mean - mu) ** 2, axis=0) / (a - 1)
    se = np.sqrt(sigma2 / (a * b))
    est = x.mean(axis=0)
    return est, se


def _ordered_thresholds(family, cov_roles):
    """Align a family's member bounds to the covariance's role ordering.

    Builds ``lower``/``upper`` in ``cov_roles`` order, inserting the
    missing genetic row ``g`` -- and ``o`` if the proband gave no own status -- as
    the uninformative interval ``(-inf, inf)``. Mirrors LTFHPlus's
    ``add_missing_roles_for_proband``. Returns ``(lower, upper)`` with
    per-phenotype columns when the inputs are vectors."""
    by_role = {m.role: m for m in family.members}
    n_pheno = 1
    for m in family.members:
        n_pheno = max(n_pheno, np.size(m.lower))

    def bounds(role):
        m = by_role.get(role)
        if m is None:  # g always missing; o missing when no proband status given
            return np.full(n_pheno, -np.inf), np.full(n_pheno, np.inf)
        lo = np.broadcast_to(np.asarray(m.lower, dtype=float), (n_pheno,))
        hi = np.broadcast_to(np.asarray(m.upper, dtype=float), (n_pheno,))
        return lo, hi

    lower, upper = map(np.array, zip(*(bounds(role) for role in cov_roles)))
    validate_bounds(lower, upper, context=f"family {family.fam_id!r} bounds")
    return lower, upper


def _estimate_group(cov, out_idx, lowers, uppers, base_seeds, tol, n_sim,
                    burn_in, max_rounds):
    """Sample one role-set group in rounds until its batch-means SE meets ``tol``.

    Only unconverged families re-enter `_gibbs_round`. Sums, squares and batch
    means are pooled online, so no draw arrays are retained. Returns
    ``(est, se, posterior_var)`` with shape ``(F, n_outputs)``; callers warn
    when ``max_rounds`` is exhausted. This is the single validation point for
    the Gibbs controls.
    """
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
    # Collapse invariants (BLUP map, kept-block factors) depend only on the
    # covariance and each family's unbounded mask, never on the round, so one
    # dict serves every round of this group.
    collapse_cache = {}
    while active.size and rnd < max_rounds:
        # wrap again: the per-round offset can carry a base seed past uint32
        seeds = np.where(base_seeds[active] < 0, -1,
                         (base_seeds[active] + rnd) % (_MAX_SEED + 1))
        ts, tsq, s1, s2 = _gibbs_round(
            P, sd, sd0, lowers[active], uppers[active],
            out_idx, n_sim, burn_in, b, nb, seeds, cov=cov,
            _collapse_cache=collapse_cache)
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
            # the SE denominator is the m*b draws that enter the batches, not the
            # total kept draws (n_sim >= nb*b); R batchmeans divides by all draws.
            se[f] = np.sqrt(np.maximum(sigma2, 0.0) / (m * b))
            est[f] = tot[f] / total_n[f]
            # E[x^2] - E[x]^2 over every retained draw; a pinned (fixed)
            # coordinate gives 0 up to rounding, which can be negative, so clamp.
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
    a join dropped the rows rather than that the proband is truly unobserved,
    so it raises rather than warns: an ignored warning would leave that zero in
    a GWAS phenotype.
    """
    if len(families) == 0:
        raise ValueError("families must contain at least one family")
    empty = [fam.fam_id for fam in families if not fam.members]
    if empty:
        shown = ", ".join(repr(fid) for fid in empty[:5])
        more = "" if len(empty) <= 5 else f", ... (+{len(empty) - 5} more)"
        raise ValueError(
            f"{len(empty)} of {len(families)} families have no members "
            f"({shown}{more}); with nothing to condition on the estimate "
            "would be the prior mean 0, indistinguishable from a real score. "
            "Drop those families or fix the join that dropped their member rows.")


def _check_unique_roles(families, *, check_pids=True, return_groups=False):
    """Validate roles and pids; optionally return role-set groups in one pass."""
    groups = {}
    for family_index, fam in enumerate(families):
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
        if return_groups:
            groups.setdefault(tuple(sorted(roles)), []).append(family_index)
        if not check_pids:  # the fitters validate both within/across families once
            continue
        keys = [key for key in map(_member_pid_key, fam.members)
                if key is not None]
        if len(keys) != len(set(keys)):
            dup = sorted({repr(k) for k in keys if keys.count(k) > 1})
            raise ValueError(
                f"pid(s) {', '.join(dup)} appear more than once in family "
                f"{fam.fam_id!r}, under different roles; one person's record "
                "would count as independent evidence. A person who fills two "
                "roles (consanguinity) needs "
                "the kinship route: estimate_liability_from_kinship or "
                "estimate_liabilities.")
    if return_groups:
        return list(groups.items())


def _check_unique_role_labels(roles):
    """Reject duplicate column labels or a ``g`` column in the array estimators'
    ``roles`` — the same contract as `_check_unique_roles` for the object
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


def _warn_unconverged_array(se, tol, max_rounds):
    """Warn for an ``(F, ncols)`` array of Monte-Carlo SEs."""
    se = np.asarray(se)
    _warn_unconverged({"se": se.max(axis=1)}, ["se"], tol, max_rounds,
                      se.shape[0])


def _warn_if_corrected(n_corrections, engine):
    """Warn when the family covariance needed nudging to strict PD.

    Reported by every estimator path: a (numerically) singular covariance
    usually marks a degenerate model -- e.g. ``h2 = 1`` makes ``g`` and ``o``
    perfectly correlated -- and silently repairing it would hide that from the
    user."""
    if n_corrections:
        warnings.warn(
            "The covariance was singular or numerically singular and was nudged "
            f"to strict positive definiteness for {engine}.",
            RuntimeWarning, stacklevel=3)


def _group_by_structure(families):
    """Bucket families by their role *set* so a bucket shares one covariance.

    Step G1 of Algorithm G (methods report), also used by PA and quadrature.
    The key is the **sorted** roles, so families with the same relatives in a
    different row order (``[o, m, f]`` vs ``[o, f, m]``) land in one bucket
    rather than fragmenting into smaller Numba batches; bounds are then aligned
    by role name. The key fixes the bucket, not the covariance row order (see
    the module docstring). Yields ``(role_tuple, [global_index, ...])`` in
    order of first appearance."""
    groups = {}
    for i, fam in enumerate(families):
        key = tuple(sorted(m.role for m in fam.members))
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return list(groups.items())


#: Odd multiplier of Knuth's multiplicative hash: a bijection on uint32, so
#: distinct user seeds get distinct block starts (blocks can overlap; see `_base_seeds`).
_SEED_STRIDE = 2654435761


def _base_seeds(seed, n, max_rounds, start=0):
    """Return one deterministic seed block per global family index.

    Family ``i`` starts at ``(seed * _SEED_STRIDE + i * max_rounds) % 2**32``;
    round ``r`` adds ``r``. This makes draws independent of scheduling,
    chunking and other families. ``start`` supports streaming without a full
    seed array. ``seed=None`` uses fresh OS entropy per family. Blocks can
    overlap for large cohorts/seed ranges and wrap after 2**32 seeds.
    """
    if seed is None:
        return np.random.default_rng().integers(0, _MAX_SEED + 1, size=n,
                                                dtype=np.int64)
    seed = _validate_seed(seed)
    families = np.arange(start, start + n, dtype=np.int64)   # global indices
    mixed = (seed * _SEED_STRIDE) % (_MAX_SEED + 1)
    return (mixed + families * int(max_rounds)) % (_MAX_SEED + 1)


def _estimate_liability_single(families, h2, out=("genetic",), tol=0.01, n_sim=100_000, burn_in=1000, seed=None, max_rounds=100, dtype=np.float64, c2=None, m2=None):
    """Estimate genetic/full liabilities for one trait, family by family.

    ``families`` is a list of `ltpred.family.Family` (build one from flat
    columns with `ltpred.family.families_from_columns`). Families sharing a
    structure are sampled together in the parallel kernel. ``out`` selects
    ``"genetic"`` and/or ``"full"``. ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) at negligible accuracy
    cost. ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or ``None``;
    family ``i`` (input position) gets seed block ``i`` of `_base_seeds`. Each
    bucket's covariance rows follow its first family's member order.
    Returns a `LiabilityResult` whose arrays line up with ``families``."""
    groups = _check_unique_roles(families, return_groups=True)
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

    for _role_key, idx in groups:
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


def _estimate_liability_pa(families, h2, out=("genetic",), use_mixture=False,
                           dtype=np.float64, c2=None, m2=None):
    """Deterministic PA inference for one trait and `Family` objects.

    Supports the censored-control mixture and C/M components. Returns a
    `LiabilityResult` with zero Monte-Carlo ``se`` and sequential-moment
    posterior variances. ``dtype=np.float32`` halves bound storage.
    """
    groups = _check_unique_roles(families, return_groups=True)
    _assert_nonempty_families(families)
    found_mixture_pair = False
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    names = [_OUT_NAMES[c] for c in out_coords]
    n = len(families)

    est = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for role_key, idx in groups:
        # PA is a sequential approximation, so its fold order must be a property
        # of the family structure rather than whichever member/family happened
        # to arrive first. `_pa_from_role_arrays` canonicalises via sorted roles.
        roles = list(role_key)
        lowers, uppers, K_is, K_pops, group_pids = _stack_object_members(
            families, idx, roles, dtype, use_mixture=use_mixture)
        if use_mixture and np.any(~np.isnan(K_is) & ~np.isnan(K_pops)):
            found_mixture_pair = True
        group_est, group_var = _pa_from_role_arrays(
            roles, lowers, uppers, h2, out_coords, K_i=K_is, K_pop=K_pops,
            use_mixture=use_mixture, c2=c2, m2=m2,
            mixture_require_pair=False)
        for slot, f in enumerate(idx):     # ids may be tuples: no fancy index
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
        for c, name in zip(out_coords, names):
            est[name][idx] = group_est[c]
            var[name][idx] = group_var[c]

    if use_mixture and not found_mixture_pair:
        raise ValueError(
            "family mixture inputs: use_mixture=True requires at "
            "least one active censored-control K_i/K_pop pair")
    se = {name: np.zeros(n) for name in names}   # deterministic: no Monte-Carlo error
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)


def _estimate_liability_quadrature(families, h2, out, dtype, atol, max_nodes):
    """Adapt the nuclear-family quadrature API without hiding diagnostics."""
    from .quadrature import estimate_liability_quadrature_arrays

    groups = _check_unique_roles(families, return_groups=True)
    _assert_nonempty_families(families)
    dtype = _bounds_dtype(dtype)
    names = [_OUT_NAMES[c] for c in _normalise_out(out)]
    n = len(families)
    est = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    error = {name: np.empty(n) for name in names}
    nodes = {name: np.empty(n, dtype=int) for name in names}
    fam_ids, pids = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for role_key, idx in groups:
        roles = list(role_key)
        lo, hi, _, _, group_pids = _stack_object_members(families, idx, roles, dtype)
        for slot, f in enumerate(idx):     # ids may be tuples: no fancy index
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
        for name in names:
            result = estimate_liability_quadrature_arrays(
                roles, lo, hi, h2=h2, out=name, atol=atol, max_nodes=max_nodes)
            est[name][idx], var[name][idx] = result.est, result.var
            error[name][idx], nodes[name][idx] = result.error, result.n_nodes
    return LiabilityResult(fam_ids, pids, est, {name: np.zeros(n) for name in names},
                           var, quadrature_error=error, quadrature_nodes=nodes)


def _estimate_liability_multi(families, h2_vec, genetic_corrmat, full_corrmat,
                              phen_names=None, out=("genetic",), tol=0.01,
                              n_sim=100_000, burn_in=1000, seed=None,
                              max_rounds=100, dtype=np.float64):
    """Gibbs inference for correlated traits with phenotype-major covariance.

    Every member needs one bound per trait. Output columns are named
    ``"genetic_<phen>"`` or ``"full_<phen>"``.
    """
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
    groups = _check_unique_roles(families, return_groups=True)
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

    for _key, idx in groups:
        roles = [m.role for m in families[idx[0]].members]
        cov_obj = construct_covmat_multi(fam_vec=roles, add_ind=True,
                                         genetic_corrmat=genetic_corrmat,
                                         full_corrmat=full_corrmat,
                                         h2_vec=h2_vec, phen_names=phen_names)
        cov, n_corrections = correct_positive_definite(cov_obj.matrix)
        _warn_if_corrected(n_corrections, "Gibbs sampling")
        k_roles = len(cov_obj.roles) // n_pheno
        fam_roles = cov_obj.roles[:k_roles]
        gibbs_out = sorted(c + k_roles * p for p in range(n_pheno) for c in out_coords)

        lowers, uppers, group_pids = [], [], []
        for f in idx:
            lo, hi = _ordered_thresholds(families[f], fam_roles)  # (k_roles, n_pheno)
            lowers.append(lo.T.reshape(-1))   # phenotype-major
            uppers.append(hi.T.reshape(-1))
            group_pids.append(_proband_pid(families[f]))
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


@_jit
def _adult_full_moments(lower, upper):
    """Scalar ADuLT moments without a family covariance or PA allocation."""
    n = len(lower)
    mean, variance = np.empty(n), np.empty(n)
    for i in range(n):
        mean[i], variance[i] = _tnorm_moments_loc(0.0, 1.0, lower[i], upper[i])
    return mean, variance


def _prepare_role_arrays(roles, lower, upper):
    """Validate role columns and bounds; return ``(roles, lower, upper)``."""
    roles = list(roles)
    _check_unique_role_labels(roles)
    lower = _as_bounds(lower)
    upper = _as_bounds(upper)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[1] != len(roles):
        raise ValueError(
            f"lower and upper must be (n_families, {len(roles)}) -- one "
            f"column per role {roles}; got {lower.shape} and {upper.shape}")
    validate_bounds(lower, upper, context="array estimator bounds")
    return roles, lower, upper


def _pa_from_role_arrays(roles, lower, upper, h2, out_coords, K_i=None,
                         K_pop=None, use_mixture=False, c2=None, m2=None,
                         mixture_require_pair=True):
    """PA estimates for one or more targets on same-structure role arrays.

    ``roles`` labels the columns of ``lower``/``upper`` (no ``g`` -- the
    constructor adds it). Returns ``(est, var)`` dicts keyed by out-coord.
    ``mixture_require_pair=False`` skips the "at least one valid K pair" gate
    for callers that run it globally over every family: a structure group may
    legitimately contain no mixture pair of its own (e.g. every family in the
    group is all-cases), so re-requiring one per group would reject a call the
    global gate accepts. Per-entry validation still runs here."""
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    # det(cov[g,o])/trace(cov[g,o]) bounds its smallest eigenvalue. Stay well
    # above the legacy repair threshold; boundary requests retain that path.
    if (roles == ["o"] and not use_mixture and c2 in (None, 0) and m2 in (None, 0)
            and np.ndim(h2) == 0 and 0 < h2 < 1
            and h2 * (1 - h2) / (1 + h2) > 1e-6):
        mean, variance = _adult_full_moments(lower[:, 0], upper[:, 0])
        est, var = {}, {}
        for coord in out_coords:
            est[coord] = h2 * mean if coord == 0 else mean
            var[coord] = h2 * (1 - h2) + h2**2 * variance if coord == 0 else variance
        return est, var
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
            roles, cov_obj.roles, (_as_bounds(K_i), _as_bounds(K_pop)),
            (np.nan, np.nan))
    est, var = {}, {}
    for coord in out_coords:
        e, v = pa_estimate_batched(
            cov, lo, hi, target=_target_index(cov_obj.roles, coord),
            K_is=ki, K_pops=kp, _psd_certified=_PSD_CERTIFIED)
        est[coord] = e
        var[coord] = v
    return est, var


def _gibbs_from_role_arrays(roles, lower, upper, h2, out_coords, seeds,
                            tol, n_sim, burn_in, max_rounds, c2=None, m2=None):
    """Gibbs estimates for one or more targets on same-structure role arrays.

    Builds the covariance in ``g, o, roles`` order (caller order kept, since
    it is the RNG order), aligns the bounds to it and returns
    ``(est, se, var)`` exactly as `_estimate_group` does."""
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    cov_obj, cov = _single_trait_cov(roles, h2, c2, m2, "Gibbs sampling")
    lo, hi = _align_to_cov(roles, cov_obj.roles, (lower, upper),
                           (-np.inf, np.inf))
    targets = [_target_index(cov_obj.roles, c) for c in out_coords]
    return _estimate_group(cov, targets, lo, hi, seeds, tol, n_sim, burn_in,
                           max_rounds)


def _scalar_member_bounds(member, fam_id):
    """Return one member's single-trait bounds with the public shape error."""
    lo, hi = member.lower, member.upper
    if isinstance(lo, (float, int, np.floating, np.integer)) and isinstance(
            hi, (float, int, np.floating, np.integer)):
        return float(lo), float(hi)      # fast path: already scalars
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
    inserted here — `_align_to_cov` fills it as unbounded. One pass over each
    family's members against a role→column map (no per-family role dict, no
    per-cell scalar setitems — the PA object path spent most of its time in
    those). Bounds that are already scalars go straight into the row lists;
    anything else falls back to `_scalar_member_bounds`, so the public shape
    error is unchanged."""
    F, k = len(idx), len(roles)
    lowers = np.empty((F, k), dtype=dtype)
    uppers = np.empty((F, k), dtype=dtype)
    K_is = np.empty((F, k), dtype=dtype) if use_mixture else None
    K_pops = np.empty((F, k), dtype=dtype) if use_mixture else None
    column = {role: j for j, role in enumerate(roles)}
    scalar = (float, int, np.floating, np.integer)
    pids = []
    for slot, f in enumerate(idx):
        fam = families[f]
        lo_row = [0.0] * k
        hi_row = [0.0] * k
        if use_mixture:
            ki_row = [np.nan] * k
            kp_row = [np.nan] * k
        for member in fam.members:
            j = column[member.role]
            lo, hi = member.lower, member.upper
            if isinstance(lo, scalar) and isinstance(hi, scalar):
                lo_row[j], hi_row[j] = float(lo), float(hi)
            else:
                lo_row[j], hi_row[j] = _scalar_member_bounds(member, fam.fam_id)
            if use_mixture:
                if member.K_i is not None:
                    ki_row[j] = float(member.K_i)
                if member.K_pop is not None:
                    kp_row[j] = float(member.K_pop)
        lowers[slot] = lo_row
        uppers[slot] = hi_row
        if use_mixture:
            K_is[slot] = ki_row
            K_pops[slot] = kp_row
        pids.append(_proband_pid(fam))
    return lowers, uppers, K_is, K_pops, pids


def estimate_liability_pa_arrays(roles: Sequence[str], lower: ArrayLike,
                                 upper: ArrayLike, h2: float,
                                 out: str = "genetic",
                                 K_i: ArrayLike | None = None,
                                 K_pop: ArrayLike | None = None,
                                 use_mixture: bool = False,
                                 c2: float | None = None, m2: float | None = None
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """PA inference for same-structure bound arrays.

    ``lower`` and ``upper`` have shape ``(n_families, len(roles))``; ``g`` is
    added internally. Returns ``(est, posterior_var)``. ``out`` selects
    ``"genetic"`` or ``"full"``. C/M components and the PA-only ``K_i``/
    ``K_pop`` mixture are optional. Float32 bounds halve input storage while
    accumulators remain float64.
    """
    coord = _single_out(out)
    est, var = _pa_from_role_arrays(
        roles, lower, upper, h2, [coord], K_i=K_i, K_pop=K_pop,
        use_mixture=use_mixture, c2=c2, m2=m2)
    return est[coord], var[coord]


def estimate_liability_gibbs_arrays(roles: Sequence[str], lower: ArrayLike,
                                    upper: ArrayLike, h2: float,
                                    out: str = "genetic", tol: float = 0.01,
                                    n_sim: int = 100_000, burn_in: int = 1000,
                                    seed: int | None = None,
                                    max_rounds: int = 100,
                                    c2: float | None = None,
                                    m2: float | None = None,
                                    return_var: bool = False
                                    ) -> tuple[np.ndarray, ...]:
    """Gibbs inference for same-structure bound arrays.

    Inputs match `estimate_liability_pa_arrays`. Returns ``(est, se)``, plus
    posterior variance when ``return_var=True``. Fresh chains of ``n_sim``
    retained draws (after ``burn_in``) are pooled until the batch-means SE is
    at most ``tol`` or ``max_rounds`` is reached; exhaustion warns. Seeds are
    uint32 integers or ``None`` and depend on global row index, making chunking
    and thread scheduling irrelevant.
    """
    coord = _single_out(out)
    seeds = _base_seeds(seed, len(lower), max_rounds)
    est, se, var = _gibbs_from_role_arrays(
        roles, lower, upper, h2, [coord], seeds, tol, n_sim, burn_in,
        max_rounds, c2=c2, m2=m2)
    _warn_unconverged_array(se, tol, max_rounds)
    if return_var:
        return est[:, 0], se[:, 0], var[:, 0]
    return est[:, 0], se[:, 0]


def estimate_liability_from_kinship(A: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                                    h2: float, target: int = 0,
                                    out: str = "genetic", tol: float = 0.01,
                                    n_sim: int = 100_000, burn_in: int = 1000,
                                    seed: int | None = None, max_rounds: int = 100,
                                    method: str | None = None,
                                    K_i: ArrayLike | None = None,
                                    K_pop: ArrayLike | None = None,
                                    use_mixture: bool = False, *,
                                    c2: float | None = None,
                                    c_kernel: ArrayLike | None = None,
                                    m2: float | None = None,
                                    m_kernel: ArrayLike | None = None,
                                    _certified_psd: object = None
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a target's liability from an arbitrary relationship matrix.

    ``A`` is an ``n × n`` additive relationship matrix shared by all rows of
    ``lower``/``upper``, whose shapes are ``(n_families, n)`` (one family for
    1-D input). ``target`` selects the proband; their genetic coordinate is
    added internally. Inbred liabilities are standardised to unit marginal
    variance. Returns ``(est, se, posterior_var)``.

    PA is the default; Gibbs is optional. ``out`` selects genetic or full
    liability. The PA-only censoring mixture uses aligned ``K_i``/``K_pop``.
    C/M proportions require aligned PSD, unit-diagonal ``c_kernel``/
    ``m_kernel`` matrices because relatedness alone cannot identify those
    environmental relationships.

    ``_certified_psd`` is internal: pipeline callers use its private sentinels
    to skip redundant matrix validation only when construction guarantees the
    required PSD/PD properties.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be a square (n, n) relationship matrix")
    lower = np.atleast_2d(_as_bounds(lower))
    upper = np.atleast_2d(_as_bounds(upper))
    validate_bounds(lower, upper, context="kinship estimator bounds")
    if lower.shape[1] != n or upper.shape[1] != n:
        raise ValueError(f"lower/upper must have {n} columns (one per pedigree member)")
    out_coord = _single_out(out)
    method_name = _resolve_method(method, "pearson-aitken")
    if method_name == "pearson-aitken":
        _warn_unused_sampler_controls(method_name, tol, n_sim, burn_in, seed, max_rounds)
    if method_name == "quadrature":
        raise NotImplementedError(
            "quadrature requires nuclear-family roles; use estimate_liability "
            "or estimate_liability_quadrature_arrays, not an arbitrary kinship matrix")
    if use_mixture and method_name == "gibbs":
        raise ValueError(
            "use_mixture=True is only supported by Pearson-Aitken; the Gibbs "
            "estimator does not implement the censored-control mixture")

    F = lower.shape[0]
    if (method_name == "pearson-aitken" and not use_mixture
            and h2 is not None and np.ndim(h2) == 0 and 0 < h2 < 1
            and h2 * (1 - h2) / (1 + h2) > 1e-6
            and c2 is None and c_kernel is None and m2 is None and m_kernel is None
            and n == 1 and A[0, 0] == 1.0
            and isinstance(target, (int, np.integer)) and not isinstance(target, bool)
            and target == 0
            and not np.any(lower[:, 0] == upper[:, 0])):
        # Family-free proband, non-inbred (A = [[1.]] exactly), no pins: the
        # 2x2 covariance [[c, c], [c, 1]] with c = h2 / (h2 + 1 - h2) collapses
        # the PA fold to the ADuLT scalar moments — the same identity
        # `_pa_from_role_arrays` dispatches on for roles == ["o"]. The margin
        # gate (as there) keeps the 2x2 matrix safely above the repair
        # threshold. Each expression reproduces the matrix path's arithmetic
        # in its exact operation order, so results are bit-identical.
        residual = 1.0 - h2
        scale = math.sqrt(h2 * A[0, 0] + residual)
        c = h2 * A[0, 0] / (scale * scale)
        mean, variance = _adult_full_moments(lower[:, 0], upper[:, 0])
        if out_coord == 0:
            # est: mu[0] = 0.0 + c * m*; var: cov[0,0] = c + (c * (v* - 1)) * c,
            # returned as sd * sd by the kernel's final infinite-interval call.
            est = 0.0 + c * mean
            sd = np.sqrt(c + (c * (variance - 1.0)) * c)
            var = sd * sd
        else:   # out="full": the target row keeps its own truncated moments
            est, var = mean, variance
        # PA is deterministic: no Monte-Carlo error, so se is exactly zero.
        return est, np.zeros_like(est), var

    cov_obj = construct_covmat_from_kinship(
        A, h2=h2, target=target, add_ind=True,
        c2=c2, c_kernel=c_kernel, m2=m2, m_kernel=m_kernel,
        _certified_psd=_certified_psd)
    if _certified_psd is _PD_CERTIFIED:
        cov, n_corrections = cov_obj.matrix, 0
    else:
        cov, n_corrections = correct_positive_definite(cov_obj.matrix)
    _warn_if_corrected(n_corrections, "liability estimation")
    # prepend the unbounded genetic-liability (g) coordinate
    neg = np.full((F, 1), -np.inf, dtype=lower.dtype)
    pos = np.full((F, 1), np.inf, dtype=upper.dtype)
    lo = np.ascontiguousarray(np.concatenate([neg, lower], axis=1))
    hi = np.ascontiguousarray(np.concatenate([pos, upper], axis=1))
    tgt = 0 if out_coord == 0 else 1 + int(target)     # g row, or the target's o row
    # The repair above (or the caller's _PD_CERTIFIED bound) puts this exact
    # array's min eigenvalue past 1e-8, certifying the PA gate's PSD tolerance.
    pa_gate = _PSD_CERTIFIED

    if method_name == "pearson-aitken":
        if not use_mixture:
            est, var = pa_estimate_batched(cov, lo, hi, target=tgt,
                                           _psd_certified=pa_gate)
        else:
            K_i, K_pop = validate_mixture_inputs(
                K_i, K_pop, expected_shape=lower.shape, require_pair=True,
                lower=lower, upper=upper,
                context="kinship estimator mixture inputs")
            nan_g = np.full((F, 1), np.nan, dtype=_as_bounds(K_i).dtype)
            ki = np.ascontiguousarray(np.concatenate([nan_g, _as_bounds(K_i)], axis=1))
            kp = np.ascontiguousarray(np.concatenate(
                [np.full((F, 1), np.nan, dtype=_as_bounds(K_pop).dtype),
                 _as_bounds(K_pop)], axis=1))
            est, var = pa_estimate_batched(cov, lo, hi, target=tgt,
                                           K_is=ki, K_pops=kp,
                                           _psd_certified=pa_gate)
        # PA is deterministic: no Monte-Carlo error, so se is exactly zero.
        return est, np.zeros_like(est), var

    seeds = _base_seeds(seed, F, max_rounds)
    est, se, var = _estimate_group(cov, [tgt], lo, hi, seeds, tol, n_sim,
                                   burn_in, max_rounds)
    _warn_unconverged_array(se, tol, max_rounds)
    return est[:, 0], se[:, 0], var[:, 0]


def estimate_liability(families: Sequence, h2: ArrayLike, *,
                       method: str | None = None,
                       out: str | Sequence[str] = ("genetic",),
                       tol: float = 0.01, use_mixture: bool = False,
                       genetic_corrmat: ArrayLike | None = None,
                       full_corrmat: ArrayLike | None = None,
                       phen_names: Sequence[str] | None = None,
                       n_sim: int = 100_000, burn_in: int = 1000,
                       seed: int | None = None, max_rounds: int = 100,
                       dtype: object = np.float64, c2: float | None = None,
                       m2: float | None = None,
                       quadrature_atol: float = 1e-8,
                       quadrature_max_nodes: int = 128) -> LiabilityResult:
    """Estimate conditional genetic or full liabilities for `Family` objects.

    ``h2`` is required liability-scale heritability. A scalar selects the
    single-trait model and deterministic PA by default; a vector with
    ``genetic_corrmat`` and ``full_corrmat`` selects multi-trait Gibbs.
    ``method`` can force ``"pa"``, ``"gibbs"`` or nuclear-family
    ``"quadrature"``. The PA-only censoring mixture needs ``use_mixture=True``
    and member ``K_i``/``K_pop`` values.

    ``out`` accepts ``"genetic"``, ``"full"`` or both; multi-trait columns are
    suffixed by ``phen_names``. Single-trait C/M components use ``c2``/``m2``
    with ``h2 + c2 + m2 <= 1``. Float32 bounds halve storage without reducing
    accumulator precision.

    Gibbs controls are ``tol``, ``n_sim``, ``burn_in``, ``seed`` and
    ``max_rounds``. Deterministic engines warn if those controls are changed.
    Quadrature supports only additive, noninbred nuclear families with
    ``0 <= h2 < 1`` and no C/M or mixture; its two controls limit refinement
    change and nodes per dimension. Results contain summaries, not draws; use
    `ltpred.gibbs.rtmvnorm_gibbs` for draws. See ``docs/estimation.md`` for
    observation models, examples and uncertainty semantics.
    """
    if h2 is None:
        raise ValueError("h2 must be numeric: supply the liability-scale heritability for this disease")
    if (np.ndim(h2) > 0 and np.size(h2) == 1 and genetic_corrmat is None
            and full_corrmat is None):
        h2 = float(np.ravel(h2)[0])   # a length-1 h2 is a scalar request
    is_multi = np.ndim(h2) > 0 or genetic_corrmat is not None or full_corrmat is not None

    # default: PA (single trait), Gibbs (multi, PA can't)
    method_name = _resolve_method(method, "gibbs" if is_multi else "pearson-aitken")
    if method_name != "gibbs":
        _warn_unused_sampler_controls(method_name, tol, n_sim, burn_in, seed, max_rounds)
    if method_name == "quadrature":
        if is_multi:
            raise NotImplementedError("quadrature is single-trait; use method='gibbs' for multiple traits")
        if use_mixture or any(value is not None and np.any(np.asarray(value) != 0)
                              for value in (c2, m2)):
            raise ValueError("quadrature supports the additive nuclear-family model without c2/m2 or mixture")
        return _estimate_liability_quadrature(
            families, h2, out, dtype, quadrature_atol, quadrature_max_nodes)
    if method_name == "pearson-aitken":
        if is_multi:
            raise NotImplementedError(
                "Pearson-Aitken estimation is single-trait; use method='gibbs' "
                "for the multi-trait model.")
        return _estimate_liability_pa(families, h2=h2, out=out,
                                      use_mixture=use_mixture, dtype=dtype,
                                      c2=c2, m2=m2)

    if use_mixture:
        raise ValueError(
            "use_mixture=True is only supported by Pearson-Aitken; the Gibbs "
            "estimator does not implement the censored-control mixture")

    if not is_multi:
        return _estimate_liability_single(families, h2=h2, out=out, tol=tol,
                                          n_sim=n_sim, burn_in=burn_in, seed=seed,
                                          max_rounds=max_rounds, dtype=dtype,
                                          c2=c2, m2=m2)
    if genetic_corrmat is None or full_corrmat is None:
        raise ValueError("multi-trait estimation needs genetic_corrmat and full_corrmat")
    if ((c2 is not None and np.any(np.asarray(c2, dtype=float) != 0.0))
            or (m2 is not None and np.any(np.asarray(m2, dtype=float) != 0.0))):
        raise NotImplementedError(
            "c2/m2 shared-environment components are not supported for "
            "multi-trait liability estimation: c2/m2 proportions alone do not specify "
            "cross-trait environmental covariance; use separate single-trait "
            "estimation only if giving up cross-trait borrowing is intended")
    return _estimate_liability_multi(families, h2_vec=h2,
                                     genetic_corrmat=genetic_corrmat,
                                     full_corrmat=full_corrmat, phen_names=phen_names,
                                     out=out, tol=tol, n_sim=n_sim, burn_in=burn_in,
                                     seed=seed, max_rounds=max_rounds, dtype=dtype)
