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

Scope: additive heritability (one variance component). The same machinery extends
to a shared-environment / maternal component by regressing on extra relationship
matrices, but those need contrasting relative types (e.g. MZ vs DZ, or
parent-offspring vs sib) to be identifiable, so only ``h2`` is fit here.
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
    # dominance: 1/4 IBD-2 for full sibs, 0 for everything else (no twins here)
    "D": lambda a, b: 0.25 if _pair_type(a, b) == "full_sib" else 0.0,
    # common (sibship) environment: shared by full sibs
    "C": lambda a, b: 1.0 if _pair_type(a, b) == "full_sib" else 0.0,
}


@dataclass
class VarCompResult:
    """Result of :func:`fit_variance_components`.

    ``components`` maps each fitted component (``"A"`` additive, ``"C"`` common
    environment, ``"D"`` dominance) to its posterior-mean **proportion** of the
    liability variance; ``residual`` is the remaining ``e2``. So ``A`` is the
    (narrow-sense) heritability. ``se`` is the within-dataset Monte-Carlo error per
    component (same caveat as :class:`FitResult` — bootstrap families for a real
    CI); ``traces`` are the full proportion traces."""
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
    """Per-structure precompute for the animal-model Gibbs: component matrices and
    their inverses, the standardized bounds / pinned mask, and initial state."""
    roles = [m.role for m in families[idx[0]].members]
    k = len(roles)
    F = len(idx)
    lo = np.empty((F, k))
    hi = np.empty((F, k))
    for slot, f in enumerate(idx):
        for c, m in enumerate(families[f].members):
            lo[slot, c] = float(m.lower)
            hi[slot, c] = float(m.upper)
    pinned = np.ascontiguousarray((hi - lo) < 1e-8)
    K, Kinv, M = {}, {}, {}
    for comp in comps:
        Kc = correct_positive_definite(_component_matrix(roles, comp))[0]
        K[comp] = Kc
        Kinv[comp] = np.linalg.inv(Kc)
        M[comp] = None                          # posterior-map cache, rebuilt per sweep
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lo[slot], hi[slot])
    return dict(roles=roles, k=k, F=F, lo=np.ascontiguousarray(lo),
                hi=np.ascontiguousarray(hi), pinned=pinned, K=K, Kinv=Kinv,
                eye=np.eye(k), l=np.ascontiguousarray(x),
                u={comp: np.zeros((F, k)) for comp in comps})


def fit_variance_components(families, components=("A",), *, n_iter=1200,
                            burn_in=300, inner_sweeps=5, seed=None,
                            prior_df=1.0, prior_scale=0.1, he_init=True):
    """Fit liability-scale variance components with a Bayesian animal-model Gibbs.

    **Experimental.** The additive-only proportion agrees with
    :func:`fit_heritability`, and a real common-environment / dominance component
    is recovered, but the single-site threshold sampler still mixes slowly (the
    trace wanders even with the HE start), so estimates from short runs can be off
    — treat this as a work in progress and validate against
    :func:`fit_heritability` for the additive case. A parameter-expanded / blocked
    sampler (Sorensen & Gianola, *Likelihood, Bayesian and MCMC Methods in
    Quantitative Genetics*) is the intended fix.

    The rigorous alternative to a moment estimator: a data-augmentation Gibbs in
    the Sorensen–Gianola threshold-model / animal-model tradition. Each sweep it

    1. samples the truncated **liabilities from the full family MVN** ``N(0,
       sum_k sigma2_k K_k + I)`` (integrating the random effects out — a
       partially-collapsed step that mixes far better than a coordinate-wise one);
    2. samples each structured random effect ``u_k ~ N(0, sigma2_k K_k)`` given the
       liabilities — the relationship matrix ``K_k`` enforces the correct joint
       structure, so, unlike free per-relative-type correlations, it does **not**
       manufacture a spurious sib excess;
    3. draws each variance from its conjugate scaled-inverse-χ² posterior, and
       records the **Rao-Blackwellised** conditional mean ``E[sigma2_k | u_k]``
       (lower variance than the draw).

    The residual variance is fixed to 1 for identification (thresholds rescaled by
    ``sqrt(total variance)`` each sweep); components are returned as **proportions**
    of the liability variance. ``he_init`` seeds ``sigma2_A`` from the fast
    Haseman-Elston fit (:func:`fit_heritability`) so the chain starts centred.

    ``components``: ``"A"`` additive (its proportion is the narrow-sense
    heritability), ``"C"`` common (sibship) environment, ``"D"`` dominance.
    **Identifiability:** ``C`` and ``D`` share the full-sib structure and are
    confounded without contrasting relative types (e.g. MZ vs DZ twins); fit at
    most one of them unless the pedigrees separate them. Returns a
    :class:`VarCompResult`."""
    comps = list(components)
    for c in comps:
        if c not in _COMPONENT_OFFDIAG:
            raise ValueError(f"unknown component {c!r}; choose from A, C, D")
    groups = [_prepare_group_vc(families, idx, comps)
              for _key, idx in _group_by_structure(families)]

    # start centred: sigma2_A from the fast HE fit (sigma2 = h2/(1-h2), residual=1)
    sigma2 = {c: 0.1 for c in comps}
    if he_init and "A" in comps:
        h2 = min(max(fit_heritability(families, n_iter=400, burn_in=120,
                                      inner_sweeps=inner_sweeps, seed=seed).h2, 0.02), 0.9)
        sigma2["A"] = h2 / (1.0 - h2)

    if seed is not None:
        _seed_rng(int(seed))
    rng = np.random.default_rng(None if seed is None else seed + 777)
    rb_trace = {c: np.empty(int(n_iter)) for c in comps}   # Rao-Blackwellised

    for it in range(int(n_iter)):
        V = 1.0 + sum(sigma2.values())
        sqrtV = np.sqrt(V)
        ss = {c: 0.0 for c in comps}
        ndim = {c: 0 for c in comps}
        for g in groups:
            k, F, eye = g["k"], g["F"], g["eye"]
            # 1. sample latent liabilities l | u  (truncated N(s, 1), bounds x sqrtV)
            s = sum(g["u"][c] for c in comps)
            lo, hi = g["lo"] * sqrtV, g["hi"] * sqrtV
            a_ = norm_cdf(lo - s)
            b_ = norm_cdf(hi - s)
            uu = np.clip(a_ + rng.random((F, k)) * (b_ - a_), 1e-15, 1 - 1e-15)
            L = s + norm_ppf(uu)
            L[g["pinned"]] = lo[g["pinned"]]           # pinned cases fixed at threshold
            # 2. sample each random effect u_c | L (single-site over components)
            for _ in range(1 if len(comps) == 1 else 2):
                for c in comps:
                    r = L - sum(g["u"][d] for d in comps if d != c)
                    SigK = sigma2[c] * g["K"][c]
                    Mc = SigK @ np.linalg.inv(SigK + eye)   # post mean-map = post cov
                    chol = np.linalg.cholesky(Mc + 1e-10 * eye)
                    g["u"][c] = r @ Mc.T + rng.standard_normal((F, k)) @ chol.T
            for c in comps:
                ss[c] += float(np.einsum("fi,ij,fj->", g["u"][c], g["Kinv"][c], g["u"][c]))
                ndim[c] += F * k
        # 3. draw sigma2 (dynamics) + record the Rao-Blackwellised conditional mean
        rb_sigma = {}
        for c in comps:
            num = ss[c] + prior_df * prior_scale
            sigma2[c] = num / rng.chisquare(ndim[c] + prior_df)      # posterior draw
            rb_sigma[c] = num / (ndim[c] + prior_df - 2.0)           # E[sigma2 | u]
        Vrb = 1.0 + sum(rb_sigma.values())
        for c in comps:
            rb_trace[c][it] = rb_sigma[c] / Vrb

    comp_est, se = {}, {}
    for c in comps:
        est, s = batch_means(rb_trace[c][int(burn_in):])
        comp_est[c] = float(est[0])
        se[c] = float(s[0])
    residual = 1.0 - sum(comp_est.values())
    return VarCompResult(components=comp_est, residual=residual, se=se,
                         traces=rb_trace, n_iter=int(n_iter), burn_in=int(burn_in))
