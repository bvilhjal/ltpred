"""Fit liability-scale heritability from family data (data-augmentation Gibbs).

Everywhere else in ltpred the family covariance is *given* — you supply ``h2`` and
the estimator conditions on it. This module instead **fits** it: it estimates the
liability-scale heritability ``h2`` from the pattern of case/control (and
age-of-onset) statuses across relatives, by treating the latent liabilities as
missing data.

The sampler is inspired by bipred's joint effect/parameter Gibbs: each sweep it
(1) **augments** the latent liabilities — one persistent truncated-MVN draw per
family under the current covariance — and (2) **updates** the covariance parameter
from those draws with a damped moment step. The update is a Haseman–Elston-style
regression of the sampled liability cross-products on the additive relationship,

    h2_hat = sum_pairs A_ij * l_i l_j  /  sum_pairs A_ij^2 ,

pooled over all related pairs in all families, then damped
``h2 <- (1 - damp) h2 + damp h2_hat`` for cross-sweep stability. Because the
liabilities are drawn conditional on each family's observed intervals, pooling
their cross-products reconstructs the model moments, so the chain settles at the
``h2`` consistent with the observed familial resemblance — a threshold-model
variance-component estimate from pedigree affection data.

:func:`fit_heritability` fits the single additive component. :func:`fit_variance_components`
generalises the same data-augmentation to several components via a **multiple**
Haseman-Elston regression (regressing the sampled cross-products on more than one
relationship matrix at once), fitting additive ``A`` and common-environment ``C``
together. Both reuse the well-mixing collapsed truncated-MVN draw and are
validated unbiased; a separate dominance component would need contrasting
relative types (MZ vs DZ twins) and is not offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ._mathfun import norm_cdf, norm_ppf
from .covariance import get_relatedness, correct_positive_definite
from .gibbs import gibbs_params, gibbs_advance, _seed_rng
from .estimate import _group_by_structure, batch_means

__all__ = ["FitResult", "fit_heritability", "VarCompResult",
           "fit_variance_components"]

_SIBSHIP = re.compile(r"o|s\d*")           # proband + full sibs (one sib-ship)
_PARENT = re.compile(r"[mf]")
_GRAND = re.compile(r"[mp]g[mf]")
_HALFSIB = re.compile(r"[mp]hs\d*")
_AVUNC = re.compile(r"[mp]au\d*")


def _pair_type(a, b):
    """Canonical relationship label for a role pair, or ``None`` if unrelated.

    Distinguishes the relative *kinds* the single-``h2`` model lumps together —
    notably ``full_sib`` and ``parent_offspring`` (both relatedness 0.5). Falls back
    to grouping by relatedness (``rel_<A>``) for pairs outside the common set."""
    A = get_relatedness(a, b, 1.0)
    if A <= 0:
        return None

    def full(p, x):
        return p.fullmatch(x) is not None

    def one_each(p, q):
        return (full(p, a) and full(q, b)) or (full(p, b) and full(q, a))

    if full(_SIBSHIP, a) and full(_SIBSHIP, b):
        return "full_sib"
    if one_each(_PARENT, _AVUNC):                 # a parent and their sib
        return "full_sib"
    if one_each(_SIBSHIP, _PARENT) or one_each(_PARENT, _GRAND) \
            or one_each(_GRAND, _AVUNC):
        return "parent_offspring"
    if one_each(_SIBSHIP, _GRAND):
        return "grandparent"
    if one_each(_SIBSHIP, _HALFSIB):
        return "half_sib"
    if one_each(_SIBSHIP, _AVUNC):
        return "avuncular"
    return f"rel_{A:g}"


@dataclass
class FitResult:
    """Result of :func:`fit_heritability`.

    ``h2`` is the posterior-mean liability-scale heritability (mean of the
    post-burn-in trace); ``h2_se`` its batch-means Monte-Carlo standard error.

    **Caveat:** ``h2_se`` is the *within-dataset* Monte-Carlo error of this one fit,
    not the sampling variability of ``h2`` across datasets — in the benchmarks it
    under-states the true SD by ~20-30x. Do **not** use it as a confidence
    interval; bootstrap over families for that. ``samples`` is the post-burn-in
    ``h2`` trace and ``trace`` the full one (for convergence diagnostics)."""
    h2: float
    h2_se: float
    samples: np.ndarray
    trace: np.ndarray
    n_iter: int
    burn_in: int


def _init_x(lowers, uppers):
    """Start each coordinate at the median of its marginal truncated normal
    (sd = 1); pinned coords collapse to their value, unbounded ones to 0."""
    p0 = (norm_cdf(lowers) + norm_cdf(uppers)) / 2.0
    x = norm_ppf(p0)
    return np.where(np.isfinite(x), x, 0.0)


def _prepare_group(families, idx):
    """Per-structure precompute: relationship matrix ``A``, related-pair list,
    per-family bounds/fixed mask, and the initial chain state ``x``."""
    roles = [m.role for m in families[idx[0]].members]
    k = len(roles)
    A = np.array([[get_relatedness(ri, rj, h2=1.0) for rj in roles] for ri in roles])
    A, _ = correct_positive_definite(A)               # ensure PSD (usually a no-op)
    pairs = [(i, j, A[i, j]) for i in range(k) for j in range(i + 1, k)
             if abs(A[i, j]) > 1e-12]
    F = len(idx)
    lowers = np.empty((F, k))
    uppers = np.empty((F, k))
    for slot, f in enumerate(idx):
        for c, m in enumerate(families[f].members):
            lowers[slot, c] = float(m.lower)
            uppers[slot, c] = float(m.upper)
    fixed = np.ascontiguousarray((uppers - lowers) < 1e-8)
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lowers[slot], uppers[slot])
    return dict(A=np.ascontiguousarray(A), pairs=pairs, k=k, F=F,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_heritability(families, *, h2_init=0.5, n_iter=1500, burn_in=500,
                     inner_sweeps=5, damp=0.2, seed=None, eps=1e-4):
    """Estimate liability-scale ``h2`` from family case/control (+age) statuses.

    ``families`` is a list of :class:`~ltpred.family.Family` whose members carry
    liability bounds (from a threshold builder). Runs a data-augmentation Gibbs
    sampler that fits ``h2`` from the familial resemblance among the latent
    liabilities (see the module docstring). ``inner_sweeps`` truncated-MVN sweeps
    are taken per outer iteration; ``damp`` controls the moment-update stability.
    Returns a :class:`FitResult`.

    Needs relatives (at least one related pair); a set of lone probands carries no
    information about ``h2`` and raises."""
    groups = [_prepare_group(families, idx) for _key, idx in _group_by_structure(families)]
    sxx = sum(sum(aij * aij for (_i, _j, aij) in g["pairs"]) * g["F"] for g in groups)
    if sxx <= 0:
        raise ValueError("no related pairs in the families — cannot fit h2 "
                         "(need relatives, not lone probands).")

    if seed is not None:
        _seed_rng(int(seed))

    h2 = float(h2_init)
    trace = np.empty(int(n_iter))
    for it in range(int(n_iter)):
        sxy = 0.0
        for g in groups:
            k = g["k"]
            sigma = (1.0 - h2) * np.eye(k) + h2 * g["A"]   # diag stays 1
            P, sd = gibbs_params(sigma)
            gibbs_advance(P, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(inner_sweeps))
            x = g["x"]
            for (i, j, aij) in g["pairs"]:
                sxy += aij * float(x[:, i] @ x[:, j])
        h2_hat = min(max(sxy / sxx, eps), 1.0 - eps)
        h2 = (1.0 - damp) * h2 + damp * h2_hat
        trace[it] = h2

    samples = trace[int(burn_in):].copy()
    est, se = batch_means(samples)
    return FitResult(h2=float(est[0]), h2_se=float(se[0]), samples=samples,
                     trace=trace, n_iter=int(n_iter), burn_in=int(burn_in))


_COMPONENT_OFFDIAG = {
    # additive relationship (2*kinship)
    "A": lambda a, b: get_relatedness(a, b, 1.0),
    # common (sibship) environment: shared by full sibs
    "C": lambda a, b: 1.0 if _pair_type(a, b) == "full_sib" else 0.0,
}
# Dominance ("D") is deliberately not offered: from sib-only pedigrees it is
# identified only through the small full-sib excess beyond additive, so the
# non-negativity constraint biases it upward (real D over-estimated, and a
# spurious D appears on purely-additive data). It needs contrasting relative
# types (MZ vs DZ twins) to estimate honestly -- out of scope for the fixed
# role grammar here.


@dataclass
class VarCompResult:
    """Result of :func:`fit_variance_components`.

    ``components`` maps each fitted component (``"A"`` additive, ``"C"`` common
    environment) to its estimated **proportion** of the liability variance;
    ``residual`` is the remaining ``e2``. So ``A`` is the (narrow-sense)
    heritability. ``se`` is the within-dataset Monte-Carlo error per component
    (same caveat as :class:`FitResult` — it is the MC error of this one fit, not
    the across-dataset sampling SD, so it under-states the real uncertainty;
    bootstrap families for a genuine CI). ``traces`` are the post-burn-in
    proportion traces per component."""
    components: dict
    residual: float
    se: dict
    traces: dict
    n_iter: int
    burn_in: int


def _component_matrix(roles, comp):
    """Relationship matrix ``K`` for one variance component (diagonal 1)."""
    off = _COMPONENT_OFFDIAG[comp]
    k = len(roles)
    K = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            K[i, j] = K[j, i] = off(roles[i], roles[j])
    return K


def _prepare_group_vc(families, idx, comps):
    """Per-structure precompute for the multi-component HE regression: each
    component's relationship matrix ``K_c``, the list of related pairs with their
    ``(K_c[i,j])_c`` predictor rows, per-family bounds / fixed mask, and the
    initial chain state ``x``."""
    roles = [m.role for m in families[idx[0]].members]
    k = len(roles)
    F = len(idx)
    K = {c: correct_positive_definite(_component_matrix(roles, c))[0] for c in comps}
    pairs = []                                    # (i, j, predictor row of K_c[i,j])
    for i in range(k):
        for j in range(i + 1, k):
            row = np.array([K[c][i, j] for c in comps])
            if np.any(np.abs(row) > 1e-12):
                pairs.append((i, j, row))
    lowers = np.empty((F, k))
    uppers = np.empty((F, k))
    for slot, f in enumerate(idx):
        for c, m in enumerate(families[f].members):
            lowers[slot, c] = float(m.lower)
            uppers[slot, c] = float(m.upper)
    fixed = np.ascontiguousarray((uppers - lowers) < 1e-8)
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lowers[slot], uppers[slot])
    return dict(roles=roles, k=k, F=F, K=K, pairs=pairs,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_variance_components(families, components=("A", "C"), *, n_iter=1500,
                            burn_in=500, inner_sweeps=5, damp=0.2, seed=None,
                            eps=1e-4):
    """Fit liability-scale variance components by a multiple Haseman-Elston regression.

    Generalises :func:`fit_heritability` from one component to several. Each sweep
    it (1) draws the latent liabilities from the **full family truncated-MVN**
    ``N(0, sum_c h2_c K_c + e2 I)`` (the well-mixing collapsed data-augmentation
    step, shared with :func:`fit_heritability`), then (2) updates all proportions
    at once by regressing the sampled cross-products on the component relationship
    matrices over every related pair,

        [h2_c] = (X'X)^-1 X'y ,   X[p, c] = K_c[i, j] ,   y[p] = l_i l_j ,

    damped across sweeps for stability. Unlike a single-``h2`` fit this separates
    relative *kinds*: ``A`` is pinned by the parent-offspring / grandparent /
    avuncular relatednesses while ``C`` is pinned by the full-sib excess, so the
    two do not trade off. The thresholds stay fixed (total liability variance 1);
    components are returned as **proportions**, with ``residual = 1 - sum``.

    ``components``: ``"A"`` additive (its proportion is the narrow-sense
    heritability) and ``"C"`` common (sibship) environment. ``C`` is identified
    only from **full-sib pairs**; without them the design is singular and this
    raises. (Dominance ``"D"`` is intentionally unsupported — see the note by
    ``_COMPONENT_OFFDIAG``; it needs twin contrasts to estimate honestly.)

    Runs a data-augmentation sweep of ``inner_sweeps`` truncated-MVN sweeps per
    outer iteration; ``damp`` controls the moment-update stability. Returns a
    :class:`VarCompResult`. Validated unbiased for ``A`` and ``A+C`` across family
    structures; as with :func:`fit_heritability`, ``se`` under-states the true
    across-dataset SD, so bootstrap families for a confidence interval."""
    comps = list(components)
    for c in comps:
        if c not in _COMPONENT_OFFDIAG:
            avail = ", ".join(_COMPONENT_OFFDIAG)
            raise ValueError(f"unknown component {c!r}; choose from {avail} "
                             "(dominance 'D' is not supported — needs twin data)")
    if len(set(comps)) != len(comps):
        raise ValueError(f"duplicate components in {components!r}")
    if int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be < n_iter ({n_iter})")
    C = len(comps)
    groups = [_prepare_group_vc(families, idx, comps)
              for _key, idx in _group_by_structure(families)]

    # X'X is fixed across sweeps (depends only on the K_c and family counts); the
    # sampled liabilities enter only through X'y. Precompute and factor it once.
    XtX = np.zeros((C, C))
    for g in groups:
        for (_i, _j, row) in g["pairs"]:
            XtX += g["F"] * np.outer(row, row)
    if np.linalg.matrix_rank(XtX, tol=1e-8) < C:
        raise ValueError(
            "variance components not identified from these families — the "
            "relationship design is rank-deficient (e.g. fitting 'C' with no "
            "full-sib pairs, or no related pairs at all).")
    XtX_reg = XtX + 1e-10 * np.eye(C)

    if seed is not None:
        _seed_rng(int(seed))

    h2 = np.full(C, 0.5 / C)
    trace = np.empty((int(n_iter), C))
    for it in range(int(n_iter)):
        e2 = max(1.0 - h2.sum(), eps)
        Xty = np.zeros(C)
        for g in groups:
            k = g["k"]
            sigma = e2 * np.eye(k) + sum(h2[ci] * g["K"][c]
                                         for ci, c in enumerate(comps))
            sigma, _ = correct_positive_definite(sigma)   # diagonal stays 1
            P, sd = gibbs_params(sigma)
            gibbs_advance(P, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(inner_sweeps))
            x = g["x"]
            for (i, j, row) in g["pairs"]:
                Xty += row * float(x[:, i] @ x[:, j])
        h2_hat = np.linalg.solve(XtX_reg, Xty)
        h2_hat = np.clip(h2_hat, eps, 1.0 - eps)
        if h2_hat.sum() > 1.0 - eps:                      # keep e2 > 0
            h2_hat *= (1.0 - eps) / h2_hat.sum()
        h2 = (1.0 - damp) * h2 + damp * h2_hat
        trace[it] = h2

    samples = trace[int(burn_in):]
    est, se = batch_means(samples)
    components_out = {c: float(est[ci]) for ci, c in enumerate(comps)}
    traces = {c: np.ascontiguousarray(samples[:, ci]) for ci, c in enumerate(comps)}
    return VarCompResult(components=components_out,
                         residual=float(1.0 - est.sum()),
                         se={c: float(se[ci]) for ci, c in enumerate(comps)},
                         traces=traces, n_iter=int(n_iter), burn_in=int(burn_in))
