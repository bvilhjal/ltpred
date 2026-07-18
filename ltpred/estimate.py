"""Liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

For each proband this conditions a family covariance on every member's liability
interval and estimates the posterior mean of the proband's genetic liability ``g``
(and/or full liability ``o``). Gibbs estimates that target by Monte Carlo;
single-trait inference defaults to the deterministic Pearson-Aitken (PA)
sequential-moment approximation. Gibbs also handles multiple traits. The resulting
genetic-liability estimate is the continuous phenotype fed to a GWAS. The observation
bounds and presence or absence of relatives determine the model; ``method`` only
selects the inference engine. In particular, PA is the single-trait default, but
it does not turn classic LT-FH inputs into LT-FH++ automatically.

Families are independent, so the estimator groups those that share a family
structure (identical roles -> identical covariance) and samples the whole group
in one compiled, ``prange``-parallel kernel when Numba is available (otherwise a
serial Python fallback), accumulating the mean and the
batch-means Monte-Carlo SE online. The sampler is re-run, accumulating draws,
until every requested estimate's SE drops below ``tol`` (LTFHPlus's convergence
rule). :func:`estimate_liability_single` handles one trait,
:func:`estimate_liability_multi` several correlated traits, and
:func:`estimate_liability` dispatches between them.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass

import numpy as np

from .covariance import (construct_covmat_single, construct_covmat_multi,
                         construct_covmat_from_kinship, correct_positive_definite)
from .gibbs import (gibbs_params, gibbs_estimate_batched, as_bounds, _MAX_SEED,
                    _validate_seed)
from .pearson_aitken import pa_estimate_batched
from ._validation import validate_bounds

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_single", "estimate_liability_multi",
           "estimate_liability_pa", "estimate_liability_pa_arrays",
           "estimate_liability_gibbs_arrays", "estimate_liability_from_kinship",
           "SensitivityResult", "liability_sensitivity"]

_PA_METHODS = {"pa", "pearson-aitken", "pearson_aitken", "aitken"}

_OUT_ALIASES = {"genetic": 0, "g": 0, 0: 0, "full": 1, "o": 1, 1: 1}
_OUT_NAMES = {0: "genetic", 1: "full"}


def _bounds_dtype(dtype):
    """Validate the per-family bounds dtype: float32 (half memory) or float64."""
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    return dt


def _out_entries(out):
    """Return a non-empty list of requested estimator output aliases."""
    if np.isscalar(out):
        entries = [out]
    elif isinstance(out, np.ndarray) and out.ndim == 0:
        entries = [out.item()]
    else:
        try:
            entries = list(out)
        except TypeError:
            raise TypeError(
                "out must be an alias or a non-empty sequence of aliases") from None
    if not entries:
        raise ValueError("out must contain at least one output alias")
    return entries


def _resolve_out_entry(value):
    """Resolve one documented string or integer output alias."""
    allowed = "genetic/full/g/o/0/1"
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"out entry {value!r} must be one of {allowed}, not bool")
    if isinstance(value, str):
        if value not in _OUT_ALIASES:
            raise ValueError(f"out entry {value!r} must be one of {allowed}")
        return _OUT_ALIASES[value]
    try:
        index = operator.index(value)
    except TypeError:
        raise TypeError(
            f"out entry {value!r} must be one of {allowed}; numeric aliases must "
            "be integers") from None
    if index not in (0, 1):
        raise ValueError(f"out entry {value!r} must be one of {allowed}")
    return index


@dataclass
class LiabilityResult:
    """Per-family liability estimates and their numerical uncertainty summaries.

    ``est``/``se`` map a column name to a per-family array (aligned with
    ``fam_ids``). Single-trait columns are ``"genetic"`` / ``"full"``; multi-trait
    columns are suffixed with the phenotype, e.g. ``"genetic_height"``. The Gibbs
    method reports ``se`` as the batch-means Monte-Carlo error; the deterministic
    Pearson-Aitken method reports ``se = 0`` and fills ``var`` with its
    sequential-moment approximation to the conditional variance. Zero PA SE means
    no Monte-Carlo error, not zero approximation error."""
    fam_ids: np.ndarray
    pids: object
    est: dict
    se: dict
    var: dict = None

    def column(self, name):
        return self.est[name]

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
    coords = [_resolve_out_entry(value) for value in _out_entries(out)]
    return sorted(set(coords))


def _single_out(out):
    """Resolve ``out`` to one column index for the APIs that return a single array.

    Accepts a scalar (``"genetic"``/``"full"``/``0``/``1``) or a length-1 sequence, so
    every estimator takes the same spellings as :func:`estimate_liability`."""
    entries = _out_entries(out)
    if len(entries) != 1:
        raise ValueError("this API returns a single column; out must be one of "
                         "genetic/full (or a length-1 sequence)")
    return _resolve_out_entry(entries[0])


def batch_means(samples):
    """Batch-means estimate and Monte-Carlo SE of column means (Jones et al. 2006).

    Splits ``n`` samples into ``a = n // b`` consecutive batches of size
    ``b = floor(sqrt(n))``, then ``se = sqrt(b * var(batch_means) / n)``. Accepts a
    1-D or 2-D ``(n, ncols)`` array and returns ``(est, se)`` arrays over columns.
    Port of R ``batchmeans::bmmat`` -- the rule LTFHPlus uses to decide the Gibbs
    sampler has converged. (The estimator computes the same quantity online inside
    the kernel; this stays for direct use and tests.)"""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    b = int(np.floor(np.sqrt(n)))
    a = n // b
    used = x[:a * b].reshape(a, b, x.shape[1])
    batch_mean = used.mean(axis=1)             # (a, ncols)
    mu = batch_mean.mean(axis=0)               # (ncols,)
    sigma2 = b * np.sum((batch_mean - mu) ** 2, axis=0) / (a - 1)
    se = np.sqrt(sigma2 / n)
    est = x.mean(axis=0)
    return est, se


def _ordered_thresholds(family, cov_roles):
    """Align a family's member bounds to the covariance's role ordering.

    Builds ``lower``/``upper`` (and pids) in ``cov_roles`` order, inserting the
    missing genetic row ``g`` -- and ``o`` if the proband gave no own status -- as
    the uninformative interval ``(-inf, inf)``. Mirrors LTFHPlus's
    ``add_missing_roles_for_proband``. Returns ``(lower, upper, pids)`` with
    per-phenotype columns when the inputs are vectors."""
    by_role = {m.role: m for m in family.members}
    n_pheno = 1
    for m in family.members:
        n_pheno = max(n_pheno, np.size(m.lower))

    def bounds(role):
        m = by_role.get(role)
        if m is None:  # g always missing; o missing when no proband status given
            return (np.full(n_pheno, -np.inf), np.full(n_pheno, np.inf), None)
        lo = np.broadcast_to(np.asarray(m.lower, dtype=float), (n_pheno,))
        hi = np.broadcast_to(np.asarray(m.upper, dtype=float), (n_pheno,))
        return lo, hi, m.pid

    lower, upper, pids = [], [], []
    for role in cov_roles:
        lo, hi, pid = bounds(role)
        lower.append(lo)
        upper.append(hi)
        pids.append(pid)
    lower = np.array(lower)
    upper = np.array(upper)
    validate_bounds(lower, upper, context=f"family {family.fam_id!r} bounds")
    return lower, upper, pids


def _estimate_group(cov, out_idx, lowers, uppers, base_seeds, tol, n_sim,
                    burn_in, max_rounds):
    """Sample every family in one structure group and iterate to tolerance.

    ``lowers``/``uppers`` are ``(F, d)`` truncation bounds for the ``F`` families that
    share covariance ``cov``; ``out_idx`` are the coordinate indices to estimate.
    The one-time :func:`gibbs_params` factorisation is reused across families and
    rounds; each round runs the parallel kernel over the still-unconverged
    families and pools their batch means (fixed batch size ``b``) so earlier draws
    are not wasted. Returns ``(est, se)`` of shape ``(F, ncols)``."""
    F = lowers.shape[0]
    ncols = len(out_idx)
    out_idx = np.asarray(out_idx, dtype=np.int64)
    P, sd = gibbs_params(cov)
    sd0 = np.sqrt(np.diag(cov))

    b = max(int(np.floor(np.sqrt(n_sim))), 2)
    nb = n_sim // b

    # streaming batch-means accumulators (no per-family batch-mean arrays):
    tot = np.zeros((F, ncols))          # sum of samples -> mean
    total_n = np.zeros(F)               # total kept samples
    bm_s1 = np.zeros((F, ncols))        # sum of batch means Y_k across rounds
    bm_s2 = np.zeros((F, ncols))        # sum of Y_k^2 across rounds
    bm_m = np.zeros(F)                  # total number of batches
    est = np.zeros((F, ncols))
    se = np.full((F, ncols), np.inf)

    active = np.arange(F)
    rnd = 0
    while active.size and rnd < max_rounds:
        # wrap again: the per-round offset can carry a base seed past uint32
        seeds = np.where(base_seeds[active] < 0, -1,
                         (base_seeds[active] + rnd) % (_MAX_SEED + 1))
        ts, s1, s2 = gibbs_estimate_batched(P, sd, sd0, lowers[active], uppers[active],
                                            out_idx, n_sim, burn_in, b, nb, seeds)
        still = []
        for ai, f in enumerate(active):
            tot[f] += ts[ai]
            total_n[f] += n_sim
            bm_s1[f] += s1[ai]
            bm_s2[f] += s2[ai]
            bm_m[f] += nb
            m = bm_m[f]
            # sum((Y - Ybar)^2) = S2 - S1^2 / M  (pooled over rounds, batch size b)
            ss = bm_s2[f] - bm_s1[f] ** 2 / m
            sigma2 = b * ss / (m - 1)
            se[f] = np.sqrt(np.maximum(sigma2, 0.0) / total_n[f])
            est[f] = tot[f] / total_n[f]
            if not np.all(se[f] <= tol):
                still.append(f)
        active = np.array(still, dtype=int)
        rnd += 1
    return est, se


def _check_unique_roles(families):
    """Reject a family with a duplicated role (two rows both ``s1``, etc.).

    Each role names one individual, so a repeat would silently merge two relatives
    into one covariance coordinate. Number repeated relatives instead (``s1``,
    ``s2``)."""
    for fam in families:
        roles = [m.role for m in fam.members]
        if len(roles) != len(set(roles)):
            dup = sorted({r for r in roles if roles.count(r) > 1})
            raise ValueError(
                f"family {fam.fam_id!r} has duplicate role(s) {dup}; each role "
                "names one individual — number repeated relatives (s1, s2, ...).")


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


def estimate_liability_single(families, h2=0.5, out=("genetic",), tol=0.01,
                              n_sim=100_000, burn_in=1000, seed=None,
                              max_rounds=100, dtype=np.float64):
    """Estimate genetic/full liabilities for one trait, family by family.

    ``families`` is a list of :class:`~ltpred.family.Family` (build one from flat
    columns with :func:`~ltpred.family.families_from_columns`). Families sharing a
    structure are sampled together in the parallel kernel. ``out`` selects
    ``"genetic"`` and/or ``"full"``. ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) at negligible accuracy
    cost. ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or ``None``.
    Returns a :class:`LiabilityResult` whose arrays line up with ``families``."""
    _check_unique_roles(families)
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    names = [_OUT_NAMES[c] for c in out_coords]
    n = len(families)
    seeds = _base_seeds(seed, n, max_rounds)

    est = {name: np.empty(n) for name in names}
    se = {name: np.empty(n) for name in names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for _key, idx in _group_by_structure(families):
        roles = [m.role for m in families[idx[0]].members]
        cov_obj = construct_covmat_single(fam_vec=roles, add_ind=True, h2=h2)
        cov, _ = correct_positive_definite(cov_obj.matrix)
        cov_roles = cov_obj.roles
        o_pos = cov_roles.index("o") if "o" in cov_roles else None

        lowers, uppers, group_pids = [], [], []
        for f in idx:
            lo, hi, mpids = _ordered_thresholds(families[f], cov_roles)
            lowers.append(lo[:, 0])
            uppers.append(hi[:, 0])
            group_pids.append(mpids[o_pos] if o_pos is not None and mpids[o_pos] is not None
                              else families[f].fam_id)
        lowers = np.array(lowers, dtype=dtype)
        uppers = np.array(uppers, dtype=dtype)

        g_est, g_se = _estimate_group(cov, out_coords, lowers, uppers,
                                      seeds[idx], tol, n_sim, burn_in, max_rounds)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]

    _warn_unconverged(se, names, tol, max_rounds, n)
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se)


def _ordered_bounds_pa(family, cov_roles):
    """Single-trait bounds + mixture inputs aligned to ``cov_roles`` for PA.

    Like :func:`_ordered_thresholds` but also gathers per-member ``K_i``/``K_pop``
    (``nan`` when absent) and returns scalars per row. The missing genetic row ``g``
    (and ``o`` if absent) becomes the uninformative ``(-inf, inf)`` with ``nan``
    mixture inputs, which the PA sweep treats as a no-op."""
    by_role = {m.role: m for m in family.members}
    lower, upper, pids, K_i, K_pop = [], [], [], [], []
    for role in cov_roles:
        m = by_role.get(role)
        if m is None:
            lower.append(-np.inf); upper.append(np.inf); pids.append(None)
            K_i.append(np.nan); K_pop.append(np.nan)
        else:
            lower.append(float(m.lower)); upper.append(float(m.upper)); pids.append(m.pid)
            K_i.append(np.nan if m.K_i is None else float(m.K_i))
            K_pop.append(np.nan if m.K_pop is None else float(m.K_pop))
    lower = np.array(lower)
    upper = np.array(upper)
    validate_bounds(lower, upper, context=f"family {family.fam_id!r} bounds")
    return lower, upper, pids, np.array(K_i), np.array(K_pop)


def estimate_liability_pa(families, h2=0.5, out=("genetic",), use_mixture=False,
                          dtype=np.float64):
    """Deterministic Pearson-Aitken liability inference for one trait.

    A deterministic sequential-moment alternative to
    :func:`estimate_liability_single`: no sampling, tolerance, or Monte-Carlo
    error. It is exact for a single truncation but approximate for multiple
    sequential truncations. ``out=("genetic",)`` estimates the proband's
    genetic liability ``g`` conditional on the whole family;
    ``"full"`` targets ``o`` and predicts the proband's full liability from the
    *relatives* (its own status is the target and so is not conditioned on).
    ``use_mixture=True`` turns on the age-censored-control mixture, using each
    member's ``K_i``/``K_pop`` (see :func:`ltpred.thresholds.pa_thresholds`).
    ``dtype=np.float32`` halves the per-family bound memory. Returns a
    :class:`LiabilityResult` with ``se = 0`` and PA approximations to conditional
    variances in ``var``."""
    _check_unique_roles(families)
    if use_mixture and not any(
            m.K_i is not None and np.isfinite(np.asarray(m.K_i, dtype=float)).any()
            for fam in families for m in fam.members):
        raise ValueError(
            "use_mixture=True but no family member carries a K_i/K_pop, so the "
            "censored-control mixture has nothing to act on. Build bounds with "
            "pa_thresholds or thresholds_from_cip (which emit K_i/K_pop for controls), "
            "or set use_mixture=False.")
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
        # to arrive first. Bounds below are realigned by role name.
        roles = list(role_key)
        cov_obj = construct_covmat_single(fam_vec=roles, add_ind=True, h2=h2)
        cov, _ = correct_positive_definite(cov_obj.matrix)
        cov_roles = cov_obj.roles
        o_pos = cov_roles.index("o") if "o" in cov_roles else None

        F, d = len(idx), len(cov_roles)
        lowers = np.empty((F, d), dtype=dtype); uppers = np.empty((F, d), dtype=dtype)
        K_is = np.empty((F, d), dtype=dtype) if use_mixture else None
        K_pops = np.empty((F, d), dtype=dtype) if use_mixture else None
        group_pids = []
        for slot, f in enumerate(idx):
            lo, hi, mpids, ki, kp = _ordered_bounds_pa(families[f], cov_roles)
            lowers[slot] = lo; uppers[slot] = hi
            if use_mixture:
                K_is[slot] = ki; K_pops[slot] = kp
            group_pids.append(mpids[o_pos] if o_pos is not None and mpids[o_pos] is not None
                              else families[f].fam_id)

        for c, name in zip(out_coords, names):
            target = cov_roles.index("g") if c == 0 else cov_roles.index("o")
            g_est, g_var = pa_estimate_batched(cov, lowers, uppers, target=target,
                                               K_is=K_is, K_pops=K_pops)
            for slot, f in enumerate(idx):
                est[name][f] = g_est[slot]
                var[name][f] = g_var[slot]
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]

    se = {name: np.zeros(n) for name in names}   # deterministic: no Monte-Carlo error
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se, var=var)


def estimate_liability_multi(families, h2_vec, genetic_corrmat, full_corrmat,
                             phen_names=None, out=("genetic",), tol=0.01,
                             n_sim=100_000, burn_in=1000, seed=None,
                             max_rounds=100, dtype=np.float64):
    """Estimate genetic/full liabilities jointly across several correlated traits.

    Each member's ``lower``/``upper`` must be length-``n_pheno`` sequences (one
    interval per phenotype, in ``phen_names`` order). Builds the phenotype-major
    multi-trait covariance, samples same-structure families together, and returns a
    :class:`LiabilityResult` with one column per (output, phenotype), e.g.
    ``"genetic_<phen>"``. ``dtype=np.float32`` halves the per-family bound memory.
    ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or ``None``.
    Port of LTFHPlus::estimate_liability_multi."""
    h2_vec = np.asarray(h2_vec, dtype=float)
    n_pheno = len(h2_vec)
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(n_pheno)]
    _check_unique_roles(families)
    _validate_multitrait_bounds(families, n_pheno)
    dtype = _bounds_dtype(dtype)
    out_coords = _normalise_out(out)
    col_names = [f"{_OUT_NAMES[c]}_{phen_names[p]}"
                 for p in range(n_pheno) for c in out_coords]

    n = len(families)
    seeds = _base_seeds(seed, n, max_rounds)
    est = {name: np.empty(n) for name in col_names}
    se = {name: np.empty(n) for name in col_names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for _key, idx in _group_by_structure(families):
        roles = [m.role for m in families[idx[0]].members]
        cov_obj = construct_covmat_multi(fam_vec=roles, add_ind=True,
                                         genetic_corrmat=genetic_corrmat,
                                         full_corrmat=full_corrmat,
                                         h2_vec=h2_vec, phen_names=phen_names)
        cov, n_corrections = correct_positive_definite(cov_obj.matrix)
        if n_corrections:
            warnings.warn(
                "The coherent multi-trait covariance was singular or numerically "
                "singular and was nudged to strict positive definiteness for Gibbs "
                "sampling.", RuntimeWarning, stacklevel=2)
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

        g_est, g_se = _estimate_group(cov, gibbs_out, lowers, uppers,
                                      seeds[idx], tol, n_sim, burn_in, max_rounds)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(col_names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]

    _warn_unconverged(se, col_names, tol, max_rounds, n)
    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se)


def _align_to_cov(roles, cov_roles, columns, defaults):
    """Map ``(F, len(roles))`` column arrays onto ``cov_roles`` order by role name.

    ``roles`` labels the columns of each array in ``columns``; a ``cov_role`` absent
    from ``roles`` (e.g. the auto-added ``g``) is filled with the matching entry of
    ``defaults``. Returns a list of ``(F, d)`` arrays in ``cov_roles`` order."""
    role_to_col = {r: i for i, r in enumerate(roles)}
    F = columns[0].shape[0]
    d = len(cov_roles)
    out = [np.full((F, d), dv, dtype=columns[0].dtype) for dv in defaults]  # keep input dtype
    for p, cr in enumerate(cov_roles):
        j = role_to_col.get(cr)
        if j is not None:
            for a, col in enumerate(columns):
                out[a][:, p] = col[:, j]
    return out


def estimate_liability_pa_arrays(roles, lower, upper, h2=0.5, out="genetic",
                                 K_i=None, K_pop=None, use_mixture=False):
    """Array-level Pearson-Aitken estimator — skips ``Family``/``Member`` objects.

    The production fast path for many same-structure probands: ``roles`` is the
    shared list of member roles (``o`` and relatives; ``g`` is added), and ``lower``
    / ``upper`` are ``(n_families, len(roles))`` bounds aligned to ``roles`` (build
    them straight from your columns, e.g. with a threshold helper). The covariance
    is built once. ``out`` is ``"genetic"`` (target ``g``) or ``"full"`` (target
    ``o``). ``use_mixture`` with ``K_i``/``K_pop`` (same shape) enables the
    censored-control mixture. Returns PA sequential-moment approximations
    ``(est, var)`` of length ``n_families``."""
    roles = list(roles)
    if len(roles) != len(set(roles)):
        raise ValueError("roles contains duplicate role labels; each column must "
                         "identify a different family member")
    lower = as_bounds(lower)                # keeps float32 if given, else float64
    upper = as_bounds(upper)
    validate_bounds(lower, upper, context="array estimator bounds")
    # The PA fold is sequential. Canonicalise its covariance order while retaining
    # ``roles`` as the column labels used to realign every caller-supplied array.
    cov_obj = construct_covmat_single(fam_vec=sorted(roles), add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    cov_roles = cov_obj.roles
    target = cov_roles.index("g") if _single_out(out) == 0 else cov_roles.index("o")

    lo, hi = _align_to_cov(roles, cov_roles, (lower, upper), (-np.inf, np.inf))
    if use_mixture:
        K_i = as_bounds(K_i)
        K_pop = as_bounds(K_pop)
        ki, kp = _align_to_cov(roles, cov_roles, (K_i, K_pop), (np.nan, np.nan))
        return pa_estimate_batched(cov, lo, hi, target=target, K_is=ki, K_pops=kp)
    return pa_estimate_batched(cov, lo, hi, target=target)


def estimate_liability_gibbs_arrays(roles, lower, upper, h2=0.5, out="genetic",
                                    tol=0.01, n_sim=100_000, burn_in=1000,
                                    seed=None, max_rounds=100):
    """Array-level Gibbs inference — skips ``Family``/``Member`` objects.

    Same array inputs as :func:`estimate_liability_pa_arrays` (float32 ``lower``/
    ``upper`` halve their memory). Returns ``(est, se)`` (posterior mean and
    batch-means Monte-Carlo SE) of length ``n_families`` for the single target
    selected by ``out``. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``."""
    roles = list(roles)
    if len(roles) != len(set(roles)):
        raise ValueError("roles contains duplicate role labels; each column must "
                         "identify a different family member")
    lower = as_bounds(lower)
    upper = as_bounds(upper)
    validate_bounds(lower, upper, context="array estimator bounds")
    cov_obj = construct_covmat_single(fam_vec=roles, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    cov_roles = cov_obj.roles
    target = cov_roles.index("g") if _single_out(out) == 0 else cov_roles.index("o")

    lo, hi = _align_to_cov(roles, cov_roles, (lower, upper), (-np.inf, np.inf))
    seeds = _base_seeds(seed, lo.shape[0], max_rounds)
    est, se = _estimate_group(cov, [target], lo, hi, seeds, tol, n_sim, burn_in,
                              max_rounds)
    return est[:, 0], se[:, 0]


def estimate_liability_from_kinship(A, lower, upper, h2=0.5, target=0, out="genetic",
                                    tol=0.01, n_sim=100_000, burn_in=1000, seed=None,
                                    max_rounds=100, method=None):
    """Estimate a target individual's liability from an **arbitrary pedigree**.

    The kinship-based counterpart of the array estimators: instead of the fixed role
    grammar you pass the additive relationship matrix ``A``
    (``n×n``, e.g. from :func:`~ltpred.covariance.kinship_from_pedigree`) shared by a
    batch of families, and the per-individual truncation bounds. ``lower``/``upper``
    are ``(n_families, n)`` (one column per pedigree member, in ``A`` order); the
    genetic-liability row for ``target`` is added internally and left unbounded.
    Inbred members are standardised to unit marginal full-liability variance, and
    the target's genetic contribution is scaled by its raw liability SD. Thus the
    supplied standard-normal bounds retain their prevalence interpretation when
    diagonal entries of ``A`` exceed one.

    ``out`` selects ``"genetic"`` (the target's genetic liability — the usual
    family-history GWAS phenotype) or ``"full"`` (its full liability). ``method=None`` uses the
    deterministic Pearson-Aitken engine, matching the main single-trait default;
    pass ``method="gibbs"`` for reference sampling. Returns ``(est, uncertainty)``,
    where the second array is PA's approximate conditional variance or the Gibbs
    Monte-Carlo SE, respectively.
    As in the object PA API, PA ``out="full"`` predicts the target's full liability
    from the other members without conditioning on its own bound; Gibbs conditions
    on all supplied bounds. Each array has length ``n_families``. The covariance is built by
    :func:`~ltpred.covariance.construct_covmat_from_kinship`, so results match the
    role-based estimator whenever the pedigree encodes the same relationships — but
    this also handles half-sibs of any degree, cousins, and inbred pedigrees.
    For Gibbs, ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or
    ``None``; the PA branch ignores it."""
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

    cov_obj = construct_covmat_from_kinship(A, h2=h2, target=target, add_ind=True)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    # prepend the unbounded genetic-liability (g) coordinate
    F = lower.shape[0]
    neg = np.full((F, 1), -np.inf, dtype=lower.dtype)
    pos = np.full((F, 1), np.inf, dtype=upper.dtype)
    lo = np.ascontiguousarray(np.concatenate([neg, lower], axis=1))
    hi = np.ascontiguousarray(np.concatenate([pos, upper], axis=1))
    tgt = 0 if out_coord == 0 else 1 + int(target)     # g row, or the target's o row

    if method is None:
        method = "pearson-aitken"
    if str(method).lower() in _PA_METHODS:
        return pa_estimate_batched(cov, lo, hi, target=tgt)
    if str(method).lower() != "gibbs":
        raise ValueError(f"unknown method {method!r}; use 'gibbs' or 'pearson-aitken'")

    seeds = _base_seeds(seed, F, max_rounds)
    est, se = _estimate_group(cov, [tgt], lo, hi, seeds, tol, n_sim, burn_in,
                              max_rounds)
    return est[:, 0], se[:, 0]


def estimate_liability(families, h2=0.5, *, method=None, out=("genetic",),
                       tol=0.01, use_mixture=False, genetic_corrmat=None,
                       full_corrmat=None, phen_names=None, n_sim=100_000,
                       burn_in=1000, seed=None, max_rounds=100, dtype=np.float64):
    """Estimate conditional liabilities, dispatching on method and trait count.

    The bounds in ``families`` and whether relative rows are present determine the
    model (LT-FH, LT-FH++, ADuLT or PA-FGRS). ``method`` selects only the numerical
    inference engine.

    ``method`` selects the inference engine; the **default** (``None``) picks the
    deterministic **Pearson-Aitken (PA)** estimator for a single trait. On the
    benchmark's tested family structures it matched the Gibbs estimate to about
    ``1e-2`` and ran 315–510x faster on the benchmark hardware. The dispatcher
    falls back to the **Gibbs**
    sampler for the multi-trait model, which PA does not support. Pass ``method``
    explicitly to override: ``"pearson-aitken"`` (aliases ``"pa"``, ``"aitken"``;
    single trait only, ``use_mixture`` enables the age-censored-control correction) or
    ``"gibbs"`` (the truncated-MVN sampler; needed for multiple traits or a
    Monte-Carlo SE). The high-level result contains Gibbs estimates or PA
    approximations to posterior means, plus the method-specific uncertainty fields,
    not retained draws; call :func:`~ltpred.gibbs.rtmvnorm_gibbs` directly when
    draws are required. An explicit ``method="pearson-aitken"``
    with a multi-trait request raises.

    Scalar ``h2`` -> single trait; a vector ``h2`` with ``genetic_corrmat`` and
    ``full_corrmat`` -> multi-trait. ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) — useful at biobank
    scale. For Gibbs, ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``; PA ignores it."""
    is_multi = np.ndim(h2) > 0 or genetic_corrmat is not None or full_corrmat is not None

    if method is None:                       # default: PA (single trait), Gibbs (multi, PA can't)
        method = "gibbs" if is_multi else "pearson-aitken"

    if str(method).lower() in _PA_METHODS:
        if is_multi:
            raise NotImplementedError(
                "Pearson-Aitken estimation is single-trait; use method='gibbs' "
                "for the multi-trait model.")
        return estimate_liability_pa(families, h2=h2, out=out,
                                     use_mixture=use_mixture, dtype=dtype)

    if str(method).lower() != "gibbs":
        raise ValueError(f"unknown method {method!r}; use 'gibbs' or 'pearson-aitken'")

    if not is_multi:
        return estimate_liability_single(families, h2=h2, out=out, tol=tol,
                                         n_sim=n_sim, burn_in=burn_in, seed=seed,
                                         max_rounds=max_rounds, dtype=dtype)
    if genetic_corrmat is None or full_corrmat is None:
        raise ValueError("multi-trait estimation needs genetic_corrmat and full_corrmat")
    return estimate_liability_multi(families, h2_vec=h2,
                                    genetic_corrmat=genetic_corrmat,
                                    full_corrmat=full_corrmat, phen_names=phen_names,
                                    out=out, tol=tol, n_sim=n_sim, burn_in=burn_in,
                                    seed=seed, max_rounds=max_rounds, dtype=dtype)


@dataclass
class SensitivityResult:
    """Result of :func:`liability_sensitivity`.

    ``h2_values`` is the grid that was swept; ``estimates`` is the ``(n_settings,
    n_families)`` matrix of per-setting liability estimates; ``corr`` their
    ``(n_settings, n_settings)`` cross-setting Pearson correlation. ``mean`` / ``sd``
    summarise each setting's estimates, and ``min_corr`` is the smallest off-diagonal
    correlation — the weakest linear agreement across the grid (near 1 means the
    scores differ mostly by near-linear rescaling; it is not a rank-correlation
    measure). ``out``
    labels which liability (``"genetic"`` / ``"full"``) was tracked."""
    h2_values: np.ndarray
    estimates: np.ndarray
    corr: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    min_corr: float
    out: str


def liability_sensitivity(families, h2_values, *, method=None, out="genetic",
                          seed=None, **est_kwargs):
    """Sweep the assumed ``h2`` and report how much the liability estimate moves.

    Turns the "run a sensitivity analysis over plausible ``h2``" advice into one
    call: it re-estimates the target liability for every ``h2`` in ``h2_values`` and
    reports the **cross-setting correlation** of the scores. Because the estimate is
    used as a GWAS phenotype — where only its correlation with the true value matters
    — a high correlation across the grid (``min_corr`` near 1) means the exact ``h2``
    is low-stakes; a low one means it matters and should be pinned down (e.g. with
    :func:`~ltpred.fit.fit_heritability`).

    ``families`` is a single family list; ``method`` / ``out`` and any extra keyword
    arguments (``n_sim``, ``tol``, ``use_mixture``, …) are passed through to
    :func:`estimate_liability`. Returns a :class:`SensitivityResult`.

    To also probe **prevalence / CIP** sensitivity — which changes the truncation
    bounds, not the covariance — rebuild the families under each prevalence and call
    this once per prevalence, comparing the results (the bounds live in the
    families, so they cannot be varied from ``h2`` alone)."""
    h2_values = np.asarray(list(h2_values), dtype=float)
    if h2_values.ndim != 1 or h2_values.size < 2:
        raise ValueError("h2_values must be a 1-D grid of at least 2 values")
    if np.any((h2_values < 0) | (h2_values > 1)):
        raise ValueError("all h2 values must be in [0, 1]")
    if isinstance(out, (list, tuple)):
        out = out[0]
    name = _OUT_NAMES[_single_out(out)]

    rows = []
    for h2 in h2_values:
        res = estimate_liability(families, h2=float(h2), method=method,
                                 out=(name,), seed=seed, **est_kwargs)
        rows.append(np.asarray(res.est[name], dtype=float))
    E = np.vstack(rows)                                 # (n_settings, n_families)
    corr = np.corrcoef(E) if E.shape[0] > 1 else np.ones((1, 1))
    S = h2_values.size
    off = corr[~np.eye(S, dtype=bool)]
    return SensitivityResult(h2_values=h2_values, estimates=E, corr=corr,
                             mean=E.mean(axis=1), sd=E.std(axis=1),
                             min_corr=float(off.min()), out=name)
