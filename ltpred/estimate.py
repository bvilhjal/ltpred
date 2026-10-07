"""Conditional liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

For each proband this conditions a Gaussian family covariance on every
member's liability interval and reports, for the proband's genetic liability
``g`` and/or full liability ``o``, the posterior mean ``est``, the posterior
variance ``var`` and the Monte-Carlo SE ``se`` of ``est``. Bounds and relative
rows distinguish LT-FH, LT-FH++ and ADuLT; PA-FGRS also needs the
``K_i``/``K_pop`` censoring mixture, which only Pearson-Aitken (PA) implements.
The genetic estimate is the continuous phenotype fed to a GWAS.

Dispatch. Every route ends in an array kernel over same-covariance families.

* Object API. `estimate_liability` picks the engine (`_resolve_method`): PA by
  default for scalar ``h2``; Gibbs for multi-trait, which only Gibbs supports;
  ``"quadrature"`` on request for additive nuclear families. Its drivers
  bucket families by sorted role set (`_group_by_structure`; methods report,
  Algorithm G, step G1), stack each bucket's bounds and call one kernel per
  bucket.
* Array API. `estimate_liability_pa_arrays`, `estimate_liability_gibbs_arrays`
  and `ltpred.chunked`: one role set per call, bounds ``(F, len(roles))``.
* Kinship API. `estimate_liability_from_kinship`: an arbitrary relationship
  matrix ``A``; the whole call is one group.

Coordinate order. Covariance rows are ``g``, ``o``, then relatives
(multi-trait: that block per phenotype, phenotype-major). `_align_to_cov`
maps role columns by name and fills ``g``, and an absent ``o``, with
``(-inf, inf)``; the kinship route prepends ``g`` to ``A``'s order. Gibbs
keeps the caller's (object API: the bucket's first family's) relative order,
because the sweep consumes RNG in coordinate order; PA sorts the roles so its
sequential fold depends only on the role set.

Engines. PA (`ltpred.pearson_aitken`) is deterministic with ``se = 0``: exact
for pins followed by at most one interval, a sequential two-moment
approximation otherwise. Gibbs runs Algorithm G (`ltpred.gibbs`) via
`_estimate_group`, with per-family seeds from `_base_seeds`. Every covariance
passes `correct_positive_definite`, with a warning if it was nudged.

Identity. ``g`` is never an input role; roles are unique within a family, and
a non-missing ``pid`` may occur once per family (across families is allowed).
Member pids without a proband pid raise; `LiabilityResult` describes the
result ``pids``.
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
                         _PSD_CERTIFIED)
from .gibbs import (gibbs_params, gibbs_estimate_batched, as_bounds, _MAX_SEED,
                    _validate_seed, _validate_burn_in)
from .pearson_aitken import pa_estimate_batched, _tnorm_moments_loc
from ._numba import _jit
from ._validation import validate_bounds, validate_mixture_inputs
from .family import _member_pid_key, _proband_pid
from ._results import _TableExport

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
           "estimate_liability_from_kinship"]

_PA_METHODS = {"pa", "pearson-aitken", "pearson_aitken", "aitken"}


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
    the PA aliases (``"pa"``, ``"pearson_aitken"``, ``"aitken"``) to
    ``"pearson-aitken"``; ``"gibbs"`` and ``"quadrature"`` keep their names.
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
    """Per-family liability estimates and their numerical uncertainty summaries.

    ``est``/``se``/``var`` map a column name to a per-family array (aligned with
    ``fam_ids``). Single-trait columns are ``"genetic"`` / ``"full"``; multi-trait
    columns are suffixed with the phenotype, e.g. ``"genetic_height"``.
    ``pids`` holds each row's join key: the proband's (role ``o``) pid, or the
    ``fam_id`` for a family that carries no pids.

    The two uncertainty fields answer different questions and neither substitutes
    for the other:

    * ``var`` is the target's **posterior** (conditional) variance — how uncertain
      this proband's liability is given their family. All engines report it:
      Gibbs as a Monte-Carlo estimate from its retained draws (Rao--Blackwellised
      for coordinates collapsed out of the sweep, such as ``g``: the variance of
      ``E[g | y]`` over the sampled coordinates ``y`` plus the constant
      ``Var(g | y)``), Pearson-Aitken as its sequential-moment approximation,
      quadrature by numerical integration.
      It does **not** shrink as you sample more.
    * ``se`` describes **Monte-Carlo error** in ``est``. Gibbs reports
      the batch-means Monte-Carlo error, which does shrink with more draws;
      Pearson-Aitken is deterministic and reports ``se = 0``. Zero PA SE means no
      Monte-Carlo error, not zero approximation error — PA's sequential fold stays
      approximate for multiple remaining intervals, in both ``est`` and ``var``.

    Quadrature also reports zero Monte-Carlo ``se``. Its ``quadrature_error``
    dictionary records the largest change in either moment over its last two
    refinements, not a certified error bound; ``quadrature_nodes`` records nodes
    per active factor dimension. Both are ``None`` for the other engines."""
    fam_ids: np.ndarray
    pids: object
    est: dict
    se: dict
    var: dict = None
    quadrature_error: dict | None = None
    quadrature_nodes: dict | None = None

    def to_dict(self):
        """Copy columns: fam_id, pid, estimates and se_name/var_name diagnostics.

        A column mapping preserves row order and repeated pids without silently
        overwriting scores. Diagnostic prefixes avoid collisions between traits
        such as "height" and "height_se". Multi-trait suffixes are retained.
        Quadrature results also get ``quadrature_error_<name>`` and
        ``quadrature_nodes_<name>`` columns.
        """
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
        """Shorthand for the single-trait genetic-liability estimate ``est['genetic']``
        (the usual output). Multi-trait results are keyed per trait, e.g.
        ``genetic_height`` — index ``.est`` directly for those."""
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


def batch_means(samples: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Batch-means estimate and Monte-Carlo SE of column means (Jones et al. 2006).

    Splits ``n`` samples into ``a = n // b`` consecutive batches of size
    ``b = floor(sqrt(n))``, then ``se = sqrt(b * var(batch_means) / (a*b))``
    — the denominator is the ``a*b`` draws that actually enter the batches,
    matching R ``batchmeans::bmmat``. Accepts a
    1-D or 2-D ``(n, ncols)`` array and returns ``(est, se)`` arrays over columns
    (length-1 arrays for 1-D input).
    Port of R ``batchmeans::bmmat`` -- the rule LTFHPlus uses to decide the Gibbs
    sampler has converged. (The estimator computes the same quantity online inside
    the kernel; this stays for direct use and tests.) Requires at least 4
    samples (two batches), the same minimum the estimator enforces on ``n_sim``."""
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
    """Run Algorithm G on one structure group and iterate to tolerance.

    ``lowers``/``uppers`` are ``(F, d)`` bounds, already in the row order of
    ``cov``; ``out_idx`` are the coordinates to estimate. With batch size
    ``b = max(floor(sqrt(n_sim)), 2)`` and ``nb = n_sim // b`` batches, each
    round (methods report, Algorithm G, step G6):

    1. calls `gibbs_estimate_batched` on the active (unconverged) families,
       family ``f`` seeded with ``(base_seeds[f] + round) % 2**32``
       (`_base_seeds` never emits the kernel's ``-1`` unseeded sentinel;
       unseeded calls get fresh entropy seeds). Each call starts fresh chains
       with their own ``burn_in`` and collapses each family's own unbounded
       coordinates (step G2), so the other active families never affect its
       draws;
    2. pools the returned sums over rounds: ``est = sum / N`` and
       ``var = max(sumsq / N - est^2, 0)`` with ``N = rounds * n_sim``, and
       the batch-means ``se = sqrt(b * sum((Y_k - Ybar)^2) / (M - 1) / (M * b))``
       over all ``M = rounds * nb`` batch means ``Y_k``;
    3. drops a family once ``se <= tol`` for every output column.

    ``max_rounds`` caps the rounds (kernel calls) per family; families still
    above ``tol`` keep their last pooled values (callers warn via
    `_warn_unconverged`). Returns ``(est, se, var)`` of shape ``(F, ncols)``:
    ``var`` estimates the target's **posterior** variance, ``se`` the
    Monte-Carlo error of ``est``. The single choke point of every Gibbs
    estimate path, so ``tol``, ``n_sim``, ``burn_in`` and ``max_rounds`` are
    validated here once."""
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
        ts, tsq, s1, s2 = gibbs_estimate_batched(
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


def _check_unique_roles(families, *, check_pids=True):
    """Reject a family with a duplicated role (two rows both ``s1``, etc.) or a
    user-supplied ``g`` member.

    Each role names one individual, so a repeat would silently merge two relatives
    into one covariance coordinate. Number repeated relatives instead (``s1``,
    ``s2``). ``g`` is never legitimate user input: the estimator adds the genetic
    coordinate itself, so a supplied ``g`` row would silently condition that
    coordinate on the member's bounds. Likewise one non-missing ``pid`` under
    two roles would count one person's record twice; the same person in
    different families (a shared register relative) is legitimate."""
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
    """`_warn_unconverged` for the ``(F, ncols)`` SE array the array entry points hold."""
    se = np.asarray(se)
    _warn_unconverged({"se": np.atleast_2d(se.T).T.max(axis=1)}, ["se"], tol,
                      max_rounds, se.shape[0])


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
#: distinct user seeds map to distinct seed blocks (see `_base_seeds`).
_SEED_STRIDE = 2654435761


def _base_seeds(seed, n, max_rounds, start=0):
    """Per-family base seeds: family ``i`` owns the ``max_rounds`` consecutive
    seeds from ``(seed * _SEED_STRIDE + i*max_rounds) % 2**32``.

    `_estimate_group` seeds round ``r`` of that family with
    ``(base + r) % 2**32``, so each family owns a block of ``max_rounds``
    consecutive seeds. Its draws are then fixed by ``seed``, its global index
    ``i``, ``max_rounds``, the sampler settings, the covariance row order and
    its own bounds -- not by thread scheduling, chunking or the other families
    in the call.

    The user seed is multiplied by ``_SEED_STRIDE`` before the block layout:
    with the raw value, seed ``s`` family ``i`` round ``r`` and seed ``s + 1``
    family ``i`` round ``r - 1`` name the *same* stream, so replicate loops
    over consecutive seeds (``for rep in range(R): ... seed=rep``) resample
    each other's draws family by family -- with identical rows and
    ``max_rounds=1`` bit for bit. The stride is odd, hence injective on
    uint32; a wrap can still collide two families' streams when
    ``n * max_rounds`` exceeds 2**32 (at the default ``max_rounds=100`` that
    is beyond ~43M families per call), and even then only for specific seed
    pairs, not every neighbour.

    ``seed=None`` returns fresh per-family entropy seeds instead of the
    kernel's ``-1`` unseeded sentinel. The sentinel draws from the Numba
    worker-thread RNG, which a previous seeded call in this process leaves in
    a deterministic state -- making later unseeded calls replay it exactly --
    and which advances differently under different thread schedules. Fresh
    seeds keep an unseeded run nondeterministic call to call, exactly like
    the seeded path, and scheduling-independent within a call. ``start`` is
    the first family's global index, so a streamed batch builds only its own
    seeds."""
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


def _estimate_liability_pa(families, h2, out=("genetic",), use_mixture=False,
                           dtype=np.float64, c2=None, m2=None):
    """Deterministic Pearson-Aitken liability inference for one trait.

    ``c2``/``m2`` wire sibship (``C``) and couple (``M``) shared-environment
    components into the family covariance (``h2 + c2 + m2 <= 1`` required).

    A deterministic sequential-moment alternative to
    ``_estimate_liability_single``: no sampling, tolerance, or Monte-Carlo
    error. It is exact for a single truncation but approximate for multiple
    sequential truncations. ``out=("genetic",)`` estimates the proband's
    genetic liability ``g`` conditional on the whole family;
    ``"full"`` is ``E[l_o | own interval and relatives]`` — the target's own
    interval bound is applied after the relative fold; pins are conditioned first. Unbind ``o``
    (or omit it) for a relatives-only predictor. ``use_mixture=True`` turns
    on the age-censored-control mixture, using each member's ``K_i``/``K_pop``
    (see `ltpred.thresholds.pa_thresholds`).
    ``dtype=np.float32`` halves the per-family bound memory. Returns a
    `LiabilityResult` with ``se = 0`` and PA approximations to conditional
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
        for slot, f in enumerate(idx):     # ids may be tuples: no fancy index
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
        for c, name in zip(out_coords, names):
            est[name][idx] = group_est[c]
            var[name][idx] = group_var[c]

    se = {name: np.zeros(n) for name in names}   # deterministic: no Monte-Carlo error
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)


def _estimate_liability_quadrature(families, h2, out, dtype, atol, max_nodes):
    """Adapt the nuclear-family quadrature API without hiding diagnostics."""
    from .quadrature import estimate_liability_quadrature_arrays

    _check_unique_roles(families)
    _assert_nonempty_families(families)
    dtype = _bounds_dtype(dtype)
    names = [_OUT_NAMES[c] for c in _normalise_out(out)]
    n = len(families)
    est = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    error = {name: np.empty(n) for name in names}
    nodes = {name: np.empty(n, dtype=int) for name in names}
    fam_ids, pids = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for role_key, idx in _group_by_structure(families):
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
    """Estimate genetic/full liabilities jointly across several correlated traits.

    Each member's ``lower``/``upper`` must be length-``n_pheno`` sequences (one
    interval per phenotype, in ``phen_names`` order). Builds the phenotype-major
    multi-trait covariance, samples same-structure families together, and returns a
    `LiabilityResult` with one column per (output, phenotype), e.g.
    ``"genetic_<phen>"``. ``dtype=np.float32`` halves the per-family bound memory.
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
    lower = as_bounds(lower)
    upper = as_bounds(upper)
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
    for callers that already ran it globally over every family: a structure
    group may legitimately contain no mixture pair of its own (e.g. every
    family in the group is all-cases), so re-requiring one per group would
    reject a call the global gate accepted."""
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
            roles, cov_obj.roles, (as_bounds(K_i), as_bounds(K_pop)),
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
    """Array-level Pearson-Aitken estimator — skips ``Family``/``Member`` objects.

    The production fast path for many same-structure probands: ``roles`` is the
    shared list of member roles (``o`` and relatives; ``g`` is added), and ``lower``
    / ``upper`` are ``(n_families, len(roles))`` bounds aligned to ``roles`` (build
    them straight from your columns, e.g. with a threshold helper). The covariance
    is built once (with the ``c2``/``m2`` sibship and couple shared-environment
    components, ``h2 + c2 + m2 <= 1``). ``h2`` must lie in ``(1e-8, 1]``
    (the samplers' strict-PD floor); ``h2 = 1``
    makes ``g`` and ``o`` collinear, so the covariance is nudged with a warning.
    ``out`` is ``"genetic"`` (target ``g``) or ``"full"``
    (``E[l_o | own interval and relatives]``), or a length-1 sequence of either.
    ``use_mixture`` with ``K_i``/``K_pop`` (same shape, ``NaN`` where unused)
    enables the censored-control mixture; the per-entry rules are those of
    `ltpred.family.Member`. Returns PA sequential-moment approximations
    ``(est, var)`` of length ``n_families``."""
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
    """Array-level Gibbs inference — skips ``Family``/``Member`` objects.

    Same array inputs as `estimate_liability_pa_arrays` (float32 ``lower``/
    ``upper`` halve their memory); the covariance takes the same ``c2``/``m2``
    shared-environment components. Returns ``(est, se)`` (posterior mean and
    batch-means Monte-Carlo SE) of length ``n_families`` for the single target
    selected by ``out``, or ``(est, se, var)`` with ``return_var=True``, where
    ``var`` is the Monte-Carlo estimate of the target's **posterior** variance —
    the comparable quantity to the ``var`` returned by
    `estimate_liability_pa_arrays`, and a different thing from the sampler's
    own error ``se``.

    The sampler runs in rounds, each a fresh chain per family: ``burn_in``
    (a non-negative integer) sweeps are discarded, then ``n_sim`` (an
    integer, at least 4) draws are kept. Rounds are pooled until the batch-means SE of every
    output is ``<= tol`` (finite, > 0) or ``max_rounds`` (>= 1) rounds have
    run; families still above ``tol`` keep their pooled values and a warning
    is issued. ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or
    ``None``; round ``r`` of row ``i`` is seeded with
    ``(mix(seed) + i*max_rounds + r) % 2**32`` for an integer mixing ``mix``
    of ``seed`` (see `_base_seeds`), so results depend on row order and
    ``max_rounds`` as well as ``seed`` (see the `ltpred.gibbs` module
    docstring for Algorithm G). ``seed=None`` draws fresh per-family seeds
    from OS entropy, so unseeded runs differ call to call -- also after a
    seeded call in the same process."""
    coord = _single_out(out)
    lower = as_bounds(lower)
    seeds = _base_seeds(seed, np.asarray(lower).shape[0], max_rounds)
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
    """Estimate a target individual's liability from an **arbitrary pedigree**.

    The kinship-based counterpart of the array estimators: instead of the fixed role
    grammar you pass the additive relationship matrix ``A``
    (``n×n``, e.g. from `ltpred.covariance.kinship_from_pedigree`) shared by a
    batch of families, and the per-individual truncation bounds. ``lower``/``upper``
    are ``(n_families, n)`` (one column per pedigree member, in ``A`` order; a
    1-D ``(n,)`` pair is one family); the genetic-liability row for ``target``
    (an index into ``A``) is added internally and left unbounded. ``h2`` must
    lie in ``(1e-8, 1]``.
    Inbred members are standardised to unit marginal full-liability variance, and
    the target's genetic contribution is scaled by its raw liability SD. Thus the
    supplied standard-normal bounds retain their prevalence interpretation when
    diagonal entries of ``A`` exceed one.

    Optional ``c2``/``m2`` proportions require caller-supplied
    ``c_kernel``/``m_kernel`` relationship matrices (``n x n``, symmetric,
    positive semi-definite, unit diagonal). These bring the role estimator's
    shared-environment model to arbitrary pedigrees without pretending that
    sibships or couples can be recovered from additive relatedness ``A`` alone.

    ``out`` selects ``"genetic"`` (the target's genetic liability — the usual
    family-history GWAS phenotype) or ``"full"`` (``E[l_o | own interval and
    relatives]`` on both engines). ``method=None`` uses the
    deterministic Pearson-Aitken engine, matching the main single-trait default;
    pass ``method="gibbs"`` for reference sampling (method names as in
    `estimate_liability`; ``"quadrature"`` raises ``NotImplementedError``).
    ``tol``, ``n_sim``, ``burn_in``, ``seed`` and ``max_rounds`` are the Gibbs
    controls of `estimate_liability_gibbs_arrays`; PA ignores them and warns
    on non-default values. Returns ``(est, se, var)``:
    ``se`` is the Monte-Carlo SE of ``est`` (exactly zero under PA, which is
    deterministic — that means *no sampling error*, not no approximation error),
    and ``var`` is the posterior (conditional) variance of the target liability
    on both engines.
    ``use_mixture=True`` with per-member ``K_i``/``K_pop`` (same shape as
    ``lower``, ``NaN`` where unused; rules as in `ltpred.family.Member`) runs
    the PA-FGRS censored-control mixture; Gibbs does not implement it and
    raises. Each array has length ``n_families``. The covariance is built by
    `ltpred.covariance.construct_covmat_from_kinship`, so results match the
    role-based estimator whenever the pedigree encodes the same relationships — but
    this also handles half-sibs of any degree, cousins, and inbred pedigrees.

    ``_certified_psd`` is internal; do not pass it. Package callers whose ``A``
    was just built by a pedigree constructor (symmetric and PSD by
    construction) use it to skip re-validating ``A``; the assembled covariance
    still passes through `ltpred.covariance.correct_positive_definite`. A
    family-free non-inbred proband under PA (``A = [[1]]``, no pins, no
    mixture or shared environment) uses the scalar ADuLT moments, as the role
    estimator does for ``roles == ["o"]``; both shortcuts are bit-identical to
    the matrix path."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be a square (n, n) relationship matrix")
    lower = np.atleast_2d(as_bounds(lower))
    upper = np.atleast_2d(as_bounds(upper))
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
    cov, n_corrections = correct_positive_definite(cov_obj.matrix)
    _warn_if_corrected(n_corrections, "liability estimation")
    # prepend the unbounded genetic-liability (g) coordinate
    neg = np.full((F, 1), -np.inf, dtype=lower.dtype)
    pos = np.full((F, 1), np.inf, dtype=upper.dtype)
    lo = np.ascontiguousarray(np.concatenate([neg, lower], axis=1))
    hi = np.ascontiguousarray(np.concatenate([pos, upper], axis=1))
    tgt = 0 if out_coord == 0 else 1 + int(target)     # g row, or the target's o row
    # correct_positive_definite just returned min-eig > 1e-8 on this exact
    # array, which certifies the PA input gate's weaker PSD tolerance.
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
            nan_g = np.full((F, 1), np.nan, dtype=as_bounds(K_i).dtype)
            ki = np.ascontiguousarray(np.concatenate([nan_g, as_bounds(K_i)], axis=1))
            kp = np.ascontiguousarray(np.concatenate(
                [np.full((F, 1), np.nan, dtype=as_bounds(K_pop).dtype),
                 as_bounds(K_pop)], axis=1))
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
    """Estimate conditional liabilities, dispatching on method and trait count.

    ``families`` is a list of `ltpred.family.Family`; an empty list or a family
    without members raises. Bounds and relative rows distinguish LT-FH, LT-FH++
    and ADuLT. PA-FGRS also requires ``K_i``/``K_pop`` (see `ltpred.family.Member`
    for which rows may carry them) and ``use_mixture=True`` and is implemented
    only by the Pearson-Aitken engine.

    ``method="quadrature"`` computes additive nuclear-family moments using at
    most two parental factors (roles o/m/f/s1/s2/..., scalar 0 <= h2 < 1,
    no C/M or mixture). ``quadrature_atol`` (default 1e-8) controls successive
    changes in both moments; ``quadrature_max_nodes`` (64..512, default 128)
    limits nodes per dimension. Nonconvergence raises ``RuntimeError``; the
    family index in its message counts within the failing family's role-set
    group, not within ``families``. Diagnostics are retained
    in ``quadrature_error`` and ``quadrature_nodes``; zero ``se`` means no
    Monte-Carlo error. The ordinary Gibbs controls do not steer this method.

    ``method`` selects the inference engine; the **default** (``None``) picks the
    deterministic **Pearson-Aitken (PA)** estimator for a single trait. On the
    benchmarked no-mixture structures PA and Gibbs posterior means agree
    closely and PA is orders of magnitude faster, and both engines are locked
    against the public R packages (LTFHPlus Gibbs, LTFGRS PA); the measured
    figures are in ``benchmarks/RESULTS.md`` (sections 1, 2 and 30). The
    dispatcher falls back to the **Gibbs**
    sampler for the multi-trait model, which PA does not support.
    That path collapses untruncated genetic coordinates out of the
    sweep and Rao--Blackwellises their posterior means. Pass ``method``
    explicitly to override (names are case-insensitive): ``"pearson-aitken"``
    (aliases ``"pa"``, ``"pearson_aitken"``, ``"aitken"``; single trait only,
    ``use_mixture`` enables the age-censored-control correction) or
    ``"gibbs"`` (the truncated-MVN sampler; needed for multiple traits or a
    Monte-Carlo SE), or ``"quadrature"`` under the nuclear-family restrictions
    above. The result contains posterior-mean estimates and method-specific uncertainty fields,
    not retained draws; call `ltpred.gibbs.rtmvnorm_gibbs` directly when
    draws are required. An explicit ``method="pearson-aitken"``
    with a multi-trait request raises.

    ``out`` is ``"genetic"``, ``"full"`` or a non-empty sequence of them.
    Single-trait columns are named ``"genetic"``/``"full"``; multi-trait columns
    ``"<out>_<phen>"``, with ``phen_names`` naming the traits (default
    ``phenotype1``, ``phenotype2``, ...; names must be distinct).

    ``h2``: Liability-scale additive heritability for this disease, in
    ``(1e-8, 1]`` for PA and Gibbs (each entry, for multi-trait; the
    lower bound is the samplers' strict positive-definiteness floor).
    Required: there is no
    disease-independent default, for the same reason the threshold builders
    take no default prevalence. See
    the data-preparation guide, "Getting heritability on the liability scale", for choosing between pedigree/twin and
    SNP estimates and for the sensitivity analysis.

    Scalar ``h2`` -> single trait; a vector ``h2`` with ``genetic_corrmat`` and
    ``full_corrmat`` -> multi-trait. For single-trait estimation, ``c2``/``m2``
    wire sibship (``C``) and couple (``M``) shared-environment components into
    the family covariance (see
    `ltpred.covariance.construct_covmat_single`; ``h2 + c2 + m2 <= 1``
    required). Nonzero ``c2``/``m2`` are not supported for multi-trait
    estimation: component proportions alone do not specify cross-trait
    environmental covariance, so they raise rather than being discarded.
    ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) — useful at biobank
    scale. ``tol``, ``n_sim``, ``burn_in``, ``seed`` and ``max_rounds`` steer
    the Gibbs sampler as in `estimate_liability_gibbs_arrays`, with family
    ``i`` of ``families`` in place of row ``i``; set ``seed`` for reproducible
    Gibbs runs. Deterministic engines ignore these controls and warn on
    non-default values; explicitly passing a default value is
    indistinguishable from omission."""
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
