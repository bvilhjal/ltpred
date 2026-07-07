"""Posterior mean liabilities -- the LT-FH++ phenotype.

For each proband this runs the truncated-MVN Gibbs sampler over the family
covariance, conditioning every member on their liability interval, and returns
the posterior mean of the proband's genetic liability ``g`` (and/or full liability
``o``). That posterior mean is the continuous phenotype LT-FH++ feeds to a GWAS:
it uses a case's relatives and age to sharpen the estimate of their genetic
value, recovering power a plain case/control label throws away.

Families are independent, so the estimator groups those that share a family
structure (identical roles -> identical covariance) and samples the whole group
in one compiled, ``prange``-parallel kernel, accumulating the mean and the
batch-means Monte-Carlo SE online. The sampler is re-run, accumulating draws,
until every requested estimate's SE drops below ``tol`` (LTFHPlus's convergence
rule). :func:`estimate_liability_single` handles one trait,
:func:`estimate_liability_multi` several correlated traits, and
:func:`estimate_liability` dispatches between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import construct_covmat_single, construct_covmat_multi, correct_positive_definite
from .gibbs import gibbs_params, gibbs_estimate_batched
from .pearson_aitken import pa_estimate_batched

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_single", "estimate_liability_multi",
           "estimate_liability_pa"]

_PA_METHODS = {"pa", "pearson-aitken", "pearson_aitken", "aitken", "pa-fgrs"}

_OUT_ALIASES = {"genetic": 0, "g": 0, 0: 0, "full": 1, "o": 1, 1: 1}
_OUT_NAMES = {0: "genetic", 1: "full"}


@dataclass
class LiabilityResult:
    """Per-family posterior liability estimates and their Monte-Carlo SEs.

    ``est``/``se`` map a column name to a per-family array (aligned with
    ``fam_ids``). Single-trait columns are ``"genetic"`` / ``"full"``; multi-trait
    columns are suffixed with the phenotype, e.g. ``"genetic_height"``. The Gibbs
    method reports ``se`` as the batch-means Monte-Carlo error; the deterministic
    Pearson-Aitken method reports ``se = 0`` and fills ``var`` with the posterior
    variance of each estimate."""
    fam_ids: np.ndarray
    pids: object
    est: dict
    se: dict
    var: dict = None

    def column(self, name):
        return self.est[name]


def _normalise_out(out):
    coords = []
    for o in out:
        if o not in _OUT_ALIASES:
            raise ValueError(f"out entry {o!r} must be one of genetic/full/g/o/0/1")
        coords.append(_OUT_ALIASES[o])
    coords = sorted(set(coords))
    return coords or [0]


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
    return np.array(lower), np.array(upper), pids


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

    tot = np.zeros((F, ncols))
    total_n = np.zeros(F)
    bm_parts = [None] * F
    est = np.zeros((F, ncols))
    se = np.full((F, ncols), np.inf)

    active = np.arange(F)
    rnd = 0
    while active.size and rnd < max_rounds:
        seeds = np.where(base_seeds[active] < 0, -1, base_seeds[active] + rnd)
        ts, bms = gibbs_estimate_batched(P, sd, sd0, lowers[active], uppers[active],
                                         out_idx, n_sim, burn_in, b, nb, seeds)
        still = []
        for ai, f in enumerate(active):
            tot[f] += ts[ai]
            total_n[f] += n_sim
            bm_parts[f] = bms[ai] if bm_parts[f] is None else np.concatenate(
                [bm_parts[f], bms[ai]], axis=1)
            allbm = bm_parts[f]
            M = allbm.shape[1]
            muhat = allbm.mean(axis=1)
            sigma2 = b * np.sum((allbm - muhat[:, None]) ** 2, axis=1) / (M - 1)
            se[f] = np.sqrt(sigma2 / total_n[f])
            est[f] = tot[f] / total_n[f]
            if not np.all(se[f] <= tol):
                still.append(f)
        active = np.array(still, dtype=int)
        rnd += 1
    return est, se


def _group_by_structure(families):
    """Bucket families by their role sequence so a bucket shares one covariance.

    Yields ``(role_tuple, [global_index, ...])``; families with identical member
    roles (the common case: an entire cohort of trios) land in one bucket and are
    sampled together."""
    groups = {}
    order = []
    for i, fam in enumerate(families):
        key = tuple(m.role for m in fam.members)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [(key, groups[key]) for key in order]


def _base_seeds(seed, n, max_rounds):
    if seed is None:
        return np.full(n, -1, dtype=np.int64)
    return (seed + np.arange(n, dtype=np.int64) * max_rounds)


def estimate_liability_single(families, h2=0.5, out=("genetic",), tol=0.01,
                              n_sim=100_000, burn_in=1000, seed=None,
                              max_rounds=100):
    """Estimate genetic/full liabilities for one trait, family by family.

    ``families`` is a list of :class:`~ltpred.family.Family` (build one from flat
    columns with :func:`~ltpred.family.families_from_columns`). Families sharing a
    structure are sampled together in the parallel kernel. ``out`` selects
    ``"genetic"`` and/or ``"full"``. Returns a :class:`LiabilityResult` whose arrays
    line up with ``families``."""
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
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        g_est, g_se = _estimate_group(cov, out_coords, lowers, uppers,
                                      seeds[idx], tol, n_sim, burn_in, max_rounds)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]

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
    return (np.array(lower), np.array(upper), pids,
            np.array(K_i), np.array(K_pop))


def estimate_liability_pa(families, h2=0.5, out=("genetic",), use_mixture=False):
    """Deterministic Pearson-Aitken (PA-FGRS) liability estimation, one trait.

    A closed-form alternative to :func:`estimate_liability_single`: no sampling, no
    tolerance, no Monte-Carlo error. ``out=("genetic",)`` estimates the proband's
    genetic liability ``g`` conditional on the whole family (the PA-FGRS score);
    ``"full"`` targets ``o`` and predicts the proband's full liability from the
    *relatives* (its own status is the target and so is not conditioned on).
    ``use_mixture=True`` turns on the age-censored-control mixture, using each
    member's ``K_i``/``K_pop`` (see :func:`ltpred.thresholds.pa_thresholds`). Returns
    a :class:`LiabilityResult` with ``se = 0`` and posterior variances in ``var``."""
    out_coords = _normalise_out(out)
    names = [_OUT_NAMES[c] for c in out_coords]
    n = len(families)

    est = {name: np.empty(n) for name in names}
    var = {name: np.empty(n) for name in names}
    fam_ids = np.empty(n, dtype=object)
    pids = np.empty(n, dtype=object)

    for _key, idx in _group_by_structure(families):
        roles = [m.role for m in families[idx[0]].members]
        cov_obj = construct_covmat_single(fam_vec=roles, add_ind=True, h2=h2)
        cov, _ = correct_positive_definite(cov_obj.matrix)
        cov_roles = cov_obj.roles
        o_pos = cov_roles.index("o") if "o" in cov_roles else None

        lowers, uppers, K_is, K_pops, group_pids = [], [], [], [], []
        for f in idx:
            lo, hi, mpids, ki, kp = _ordered_bounds_pa(families[f], cov_roles)
            lowers.append(lo); uppers.append(hi); K_is.append(ki); K_pops.append(kp)
            group_pids.append(mpids[o_pos] if o_pos is not None and mpids[o_pos] is not None
                              else families[f].fam_id)
        lowers = np.array(lowers); uppers = np.array(uppers)
        K_is = np.array(K_is); K_pops = np.array(K_pops)

        for c, name in zip(out_coords, names):
            target = cov_roles.index("g") if c == 0 else cov_roles.index("o")
            g_est, g_var = pa_estimate_batched(
                cov, lowers, uppers, target=target,
                K_is=(K_is if use_mixture else None),
                K_pops=(K_pops if use_mixture else None))
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
                             max_rounds=100):
    """Estimate genetic/full liabilities jointly across several correlated traits.

    Each member's ``lower``/``upper`` must be length-``n_pheno`` sequences (one
    interval per phenotype, in ``phen_names`` order). Builds the phenotype-major
    multi-trait covariance, samples same-structure families together, and returns a
    :class:`LiabilityResult` with one column per (output, phenotype), e.g.
    ``"genetic_<phen>"``. Port of LTFHPlus::estimate_liability_multi."""
    h2_vec = np.asarray(h2_vec, dtype=float)
    n_pheno = len(h2_vec)
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(n_pheno)]
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
        cov, _ = correct_positive_definite(cov_obj.matrix)
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
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        g_est, g_se = _estimate_group(cov, gibbs_out, lowers, uppers,
                                      seeds[idx], tol, n_sim, burn_in, max_rounds)
        for slot, f in enumerate(idx):
            fam_ids[f] = families[f].fam_id
            pids[f] = group_pids[slot]
            for c, name in enumerate(col_names):
                est[name][f] = g_est[slot, c]
                se[name][f] = g_se[slot, c]

    return LiabilityResult(fam_ids=fam_ids, pids=pids, est=est, se=se)


def estimate_liability(families, h2=0.5, *, method="gibbs", out=("genetic",),
                       tol=0.01, use_mixture=False, genetic_corrmat=None,
                       full_corrmat=None, phen_names=None, n_sim=100_000,
                       burn_in=1000, seed=None, max_rounds=100):
    """Estimate posterior liabilities, dispatching on method and trait count.

    ``method="gibbs"`` (default) runs the truncated-MVN Gibbs sampler;
    ``method="pearson-aitken"`` (aliases ``"pa"``, ``"pa-fgrs"``) runs the
    deterministic PA-FGRS estimator (single trait only; ``use_mixture`` enables the
    age-censored-control correction). Scalar ``h2`` -> single trait; a vector ``h2``
    with ``genetic_corrmat`` and ``full_corrmat`` -> multi-trait (Gibbs only)."""
    is_multi = np.ndim(h2) > 0 or genetic_corrmat is not None or full_corrmat is not None

    if str(method).lower() in _PA_METHODS:
        if is_multi:
            raise NotImplementedError(
                "Pearson-Aitken estimation is single-trait; use method='gibbs' "
                "for the multi-trait model.")
        return estimate_liability_pa(families, h2=h2, out=out, use_mixture=use_mixture)

    if str(method).lower() != "gibbs":
        raise ValueError(f"unknown method {method!r}; use 'gibbs' or 'pearson-aitken'")

    if not is_multi:
        return estimate_liability_single(families, h2=h2, out=out, tol=tol,
                                         n_sim=n_sim, burn_in=burn_in, seed=seed,
                                         max_rounds=max_rounds)
    if genetic_corrmat is None or full_corrmat is None:
        raise ValueError("multi-trait estimation needs genetic_corrmat and full_corrmat")
    return estimate_liability_multi(families, h2_vec=h2,
                                    genetic_corrmat=genetic_corrmat,
                                    full_corrmat=full_corrmat, phen_names=phen_names,
                                    out=out, tol=tol, n_sim=n_sim, burn_in=burn_in,
                                    seed=seed, max_rounds=max_rounds)
