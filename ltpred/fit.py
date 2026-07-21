"""Fit liability-scale heritability from family data (data-augmentation fixed point).

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
relationship matrix at once), fitting additive ``A`` alongside a **bank of
relationship-specific shared-environment components** — ``C`` (full-sib / sibship
environment) and ``M`` (couple / spousal environment) — chosen from
``_COMPONENT_OFFDIAG``. The shipped environment components are equivalence-class
partitions of the pedigree (groups that fully share one environmental deviation),
so their relationship matrices are positive-semidefinite by construction;
different components load on **different relationship contrasts** (``C`` on the
full-sib excess, ``M`` on the resemblance between genetically-unrelated mates), so
a multi-generational pedigree can identify several at once when those contrasts are
linearly independent. All reuse the collapsed truncated-MVN draw. Repository
benchmarks found small bias relative to across-dataset variability in the tested
designs. A dominance component would require an explicit dominance kernel and a
richer relationship design; it is not offered.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

from ._mathfun import norm_cdf, norm_ppf
from ._validation import validate_bounds
from .covariance import (get_relatedness, correct_positive_definite,
                        _is_full_sib, _is_mates)
from .gibbs import (gibbs_params, gibbs_advance, gibbs_advance_moment,
                    _std_tnorm_quantile,
                    _offset_seed, _seed_rng)
from .estimate import (_group_by_structure, _validate_multitrait_bounds,
                       batch_means)
from .family import Family, Member

__all__ = ["FitResult", "fit_heritability", "VarCompResult",
           "fit_variance_components", "GenCorrResult", "fit_genetic_correlation",
           "DecayGenCorrResult", "fit_genetic_correlation_decay",
           "FactorResult", "fit_genetic_factor",
           "BootstrapResult", "bootstrap_fit", "SignificanceTest",
           "test_variance_component", "test_genetic_correlation"]

@dataclass
class FitResult:
    """Result of :func:`fit_heritability`.

    ``h2`` is the liability-scale heritability — the post-burn-in average of the
    fixed-point ``h2`` trace (a stochastic-approximation estimate, **not** a
    posterior mean). ``h2_se`` is its batch-means Monte-Carlo variability.

    **Caveat:** ``h2_se`` is a *within-dataset* Monte-Carlo **diagnostic** of the
    fixed point, not an inferential standard error and not the sampling variability
    of ``h2`` across datasets; it can substantially understate that variability.
    Do **not** use it as a confidence interval. A family-cluster bootstrap is one
    sampling-uncertainty option when its assumptions hold. ``samples`` is the
    post-burn-in ``h2`` trace and ``trace`` the full one
    (for convergence diagnostics; despite the name they are fixed-point iterates,
    not posterior draws)."""
    h2: float
    h2_se: float
    samples: np.ndarray
    trace: np.ndarray
    n_iter: int
    burn_in: int


def _init_x(lowers, uppers, fixed):
    """Initial chain state for the fit paths (unit marginal SD).

    Pinned coordinates start at their exact pin value -- they are never
    resampled, so their start is permanent. Free coordinates start at the
    median of their marginal truncated normal, computed with the sampler's
    tail-stable :func:`ltpred.gibbs._std_tnorm_quantile` rather than
    ``norm_ppf((Phi(lo) + Phi(hi)) / 2)``: beyond |z| ~ 8.2 the plain CDF
    average saturates to 0/1 and the ppf returns +-inf, and a finite-or-0
    fallback then started even an *extreme pinned* coordinate at 0.0 --
    conditioning the whole family on a wrong value forever."""
    x = np.empty(lowers.shape[0], dtype=np.float64)
    for j in range(lowers.shape[0]):
        if fixed[j]:
            x[j] = lowers[j]                    # lower == upper: the exact pin
        else:
            xj = _std_tnorm_quantile(lowers[j], uppers[j], 0.5)
            x[j] = xj if np.isfinite(xj) else 0.0
    return x


def _prepare_group(families, idx):
    """Per-structure precompute: relationship matrix ``A``, related-pair list,
    per-family bounds/fixed mask, and the initial chain state ``x``."""
    roles = sorted(m.role for m in families[idx[0]].members)
    k = len(roles)
    A = np.array([[get_relatedness(ri, rj, h2=1.0) for rj in roles] for ri in roles])
    A, _ = correct_positive_definite(A)               # ensure PSD (usually a no-op)
    pairs = [(i, j, A[i, j]) for i in range(k) for j in range(i + 1, k)
             if abs(A[i, j]) > 1e-12]
    F = len(idx)
    lowers = np.empty((F, k))
    uppers = np.empty((F, k))
    for slot, f in enumerate(idx):
        members = sorted(families[f].members, key=lambda member: member.role)
        for c, m in enumerate(members):
            lowers[slot, c] = float(m.lower)
            uppers[slot, c] = float(m.upper)
    validate_bounds(lowers, uppers, context="heritability fit bounds")
    fixed = np.ascontiguousarray((uppers - lowers) < 1e-8)
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lowers[slot], uppers[slot], fixed[slot])
    return dict(roles=roles, A=np.ascontiguousarray(A), pairs=pairs, k=k, F=F,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_heritability(families, *, h2_init=0.5, n_iter=1500, burn_in=500,
                     inner_sweeps=5, damp=0.2, seed=None, eps=1e-4):
    """Estimate liability-scale ``h2`` from family case/control (+age) statuses.

    ``families`` is a list of :class:`~ltpred.family.Family` whose members carry
    liability bounds (from a threshold builder). Alternates a Gibbs augmentation of
    the latent liabilities with a damped Haseman–Elston update of ``h2`` — a
    **stochastic-approximation fixed point** (not posterior sampling of ``h2``) that
    settles at the value consistent with the familial resemblance (see the module
    docstring). ``inner_sweeps`` truncated-MVN sweeps are taken per outer iteration;
    ``damp`` controls the moment-update stability. Returns a :class:`FitResult`.

    Needs relatives (at least one related pair); a set of lone probands carries no
    information about ``h2`` and raises. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``, ``h2_init`` must lie in [0, 1], and ``burn_in``
    must be smaller than ``n_iter``."""
    if int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be < n_iter ({n_iter})")
    if not 0.0 <= float(h2_init) <= 1.0:
        raise ValueError("h2_init must be in [0, 1]")

    groups = [_prepare_group(families, idx) for _key, idx in _group_by_structure(families)]
    sxx = sum(sum(aij * aij for (_i, _j, aij) in g["pairs"]) * g["F"] for g in groups)
    if sxx <= 0:
        raise ValueError("no related pairs in the families — cannot fit h2 "
                         "(need relatives, not lone probands).")

    if seed is not None:
        _seed_rng(seed)

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
    "C": lambda a, b: 1.0 if _is_full_sib(a, b) else 0.0,
    # couple (spousal) environment: shared by genetically-unrelated mates
    "M": lambda a, b: 1.0 if _is_mates(a, b) else 0.0,
}
# The shipped environment components ``C`` and ``M`` are each an
# **equivalence-class partition**: a set of relatives who fully share one
# environmental deviation (sib-ship for ``C``, couple for ``M``), so their kernels
# are positive-semidefinite. A future component need not be a partition, but its
# kernel must be substantively meaningful, symmetric, and PSD. A *vertical* /
# parent-offspring "shared environment" is deliberately
# not offered because it is not an equivalence relation (parent-offspring
# cohabitation chains across generations), so its indicator matrix is not PSD and
# would be a mis-specified component -- :func:`_component_matrix` guards against it.
#
# Dominance ("D") is likewise not offered. It requires an explicit dominance
# relationship kernel and contrasts linearly independent of the additive and
# sibship kernels; MZ/DZ observations can contribute in a richer design but cannot
# identify A, C, and D by themselves.


@dataclass
class VarCompResult:
    """Result of :func:`fit_variance_components`.

    ``components`` maps each fitted component (``"A"`` additive, ``"C"`` sibship
    common environment, ``"M"`` couple/spousal environment) to its estimated
    **proportion** of the liability variance; ``residual`` is the remaining ``e2``.
    So ``A`` is the (narrow-sense) heritability. For ``method="he"``, ``se`` is a
    within-dataset Monte-Carlo diagnostic (same caveat as :class:`FitResult`), not
    across-dataset sampling uncertainty. For ``method="mcem"``, it is an
    approximate OPG/BHHH information SE with Monte-Carlo, finite-iteration,
    iid-family and model-correctness assumptions. Use an appropriately designed
    family-cluster bootstrap for sampling uncertainty. ``traces`` are the
    post-burn-in proportion traces per component.

    ``loglik`` / ``aic`` are populated only by the ``method="mcem"`` fit: the
    Monte-Carlo (GHK) observed-data log-likelihood and ``AIC = 2·(#components) −
    2·loglik`` (both Monte-Carlo estimates), for comparing nested models (e.g.
    ``A`` vs ``A+C``). They are
    ``None`` for the moment (``"he"``) fit and for pinned/degenerate bounds. As
    always for variance components, AIC can be unreliable near a parameter
    boundary, so use it only as a descriptive comparison. The conditional
    parametric-bootstrap :func:`test_variance_component` is the package's component
    test when its model and sampling assumptions hold."""
    components: dict
    residual: float
    se: dict
    traces: dict
    n_iter: int
    burn_in: int
    loglik: float = None
    aic: float = None


def _component_matrix(roles, comp):
    """Relationship matrix ``K`` for one variance component (diagonal 1).

    A valid variance component has a positive-semidefinite ``K`` (the additive
    relationship is PSD for any consistent pedigree; an equivalence-class partition
    is one sufficient construction for a shared-environment kernel). A non-PSD ``K`` — e.g. a
    vertical parent-offspring "environment" whose sharing chains across generations —
    is not a proper component and would be silently distorted downstream by
    :func:`~ltpred.covariance.correct_positive_definite`, so it is rejected here."""
    off = _COMPONENT_OFFDIAG[comp]
    k = len(roles)
    K = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            K[i, j] = K[j, i] = off(roles[i], roles[j])
    if np.min(np.linalg.eigvalsh(K)) < -1e-8:
        raise ValueError(
            f"component {comp!r} does not yield a positive-semidefinite relationship "
            "matrix for these roles — it is not a proper variance component (the "
            "kernel must be symmetric and positive-semidefinite; the naive "
            "vertical parent-offspring indicator is not).")
    return K


def _prepare_group_vc(families, idx, comps):
    """Per-structure precompute for the multi-component HE regression: each
    component's relationship matrix ``K_c``, the list of related pairs with their
    ``(K_c[i,j])_c`` predictor rows, per-family bounds / fixed mask, and the
    initial chain state ``x``."""
    roles = sorted(m.role for m in families[idx[0]].members)
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
        members = sorted(families[f].members, key=lambda member: member.role)
        for c, m in enumerate(members):
            lowers[slot, c] = float(m.lower)
            uppers[slot, c] = float(m.upper)
    validate_bounds(lowers, uppers, context="variance-component fit bounds")
    fixed = np.ascontiguousarray((uppers - lowers) < 1e-8)
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lowers[slot], uppers[slot], fixed[slot])
    return dict(roles=roles, k=k, F=F, K=K, pairs=pairs,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_variance_components(families, components=("A", "C"), *, method="he",
                            n_iter=1500, burn_in=500, inner_sweeps=5, damp=0.2,
                            seed=None, eps=1e-4):
    """Fit liability-scale variance components by a multiple Haseman-Elston regression.

    ``method="he"`` (default) is the moment fit described below. ``method="mcem"``
    instead runs the package's **finite-iteration, fixed-damping Monte-Carlo
    EM-style likelihood fit** (see :func:`_fit_vc_reml`). Each sweep's M-step
    maximises the Gaussian likelihood of the imputed liabilities rather than
    regressing moments. It returns an approximate OPG/BHHH information SE rather
    than the HE within-dataset Monte-Carlo diagnostic. This is not restricted ML;
    ``method="reml"``/``"ml"`` are aliases retained for backward compatibility.

    Generalises :func:`fit_heritability` from one component to several. Each sweep
    it (1) draws the latent liabilities from the **full family truncated-MVN**
    ``N(0, sum_c h2_c K_c + e2 I)`` (the well-mixing collapsed data-augmentation
    step, shared with :func:`fit_heritability`), then (2) updates all proportions
    at once by regressing the sampled cross-products on the component relationship
    matrices over every related pair,

        [h2_c] = (X'X)^-1 X'y ,   X[p, c] = K_c[i, j] ,   y[p] = l_i l_j ,

    damped across sweeps for stability. Unlike a single-``h2`` fit this separates
    relative *kinds* when the supplied pedigrees yield linearly independent
    relationship contrasts: ``A`` by the parent-offspring /
    grandparent / avuncular relatednesses, ``C`` by the full-sib excess, ``M`` by
    the resemblance between genetically-unrelated mates. The thresholds stay fixed
    (total liability variance 1); components are returned as **proportions**, with
    ``residual = 1 - sum``.

    ``components`` is any subset of the **variance-component bank**:

    - ``"A"`` — additive genetic (its proportion is the narrow-sense heritability);
    - ``"C"`` — sibship common environment, identified only from **full-sib pairs**;
    - ``"M"`` — couple/spousal shared environment, identified only from **mate
      pairs** (the proband's parents, and grandparent couples). Because mates are
      genetically unrelated, ``M`` captures spousal resemblance from *any* source —
      shared adult environment or assortative mating, which parent data alone cannot
      separate. Note that, unlike ``C``, leaving ``M`` unmodelled biases ``A``
      **much less**: mates have ``A_ab = 0`` so they carry little direct weight in
      the additive regression; ``M`` matters when the spousal resemblance is itself
      of interest, or to test/report it.

    A component whose identifying pairs are absent (``C`` with no full-sib pairs,
    ``M`` with no mate pairs, or no related pairs at all) leaves the design singular
    and this raises. (Dominance ``"D"`` is intentionally unsupported — see the note
    by ``_COMPONENT_OFFDIAG``; it needs an explicit dominance kernel and an
    independently informative relationship design.)

    Runs a data-augmentation sweep of ``inner_sweeps`` truncated-MVN sweeps per
    outer iteration; ``damp`` controls the moment-update stability. Returns a
    :class:`VarCompResult`. Simulation benchmarks recover ``A`` and ``A+C`` with
    small bias relative to their across-dataset SD. For ``method="he"``, ``se`` is
    only a within-fit Monte-Carlo diagnostic; the MCEM interpretation is described
    above. Use family resampling for a sampling interval when clusters are
    independent and representative. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``."""
    comps = list(components)
    for c in comps:
        if c not in _COMPONENT_OFFDIAG:
            avail = ", ".join(_COMPONENT_OFFDIAG)
            raise ValueError(f"unknown component {c!r}; choose from {avail} "
                             "(dominance 'D' is not supported)")
    if len(set(comps)) != len(comps):
        raise ValueError(f"duplicate components in {components!r}")
    if int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be < n_iter ({n_iter})")
    if method not in ("he", "mcem", "ml", "reml"):
        raise ValueError(f"unknown method {method!r}; use 'he' or 'mcem' "
                         "('ml'/'reml' are aliases)")
    if method in ("mcem", "ml", "reml"):
        return _fit_vc_reml(families, comps, n_iter=n_iter, burn_in=burn_in,
                            inner_sweeps=inner_sweeps, damp=damp, seed=seed, eps=eps)
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
        _seed_rng(seed)

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


def _reml_sigma(h2, Kmats, k, eps):
    """Sigma(h2) = sum_c h2_c K_c + (1 - sum h2) I  (unit diagonal by construction)."""
    e2 = max(1.0 - float(np.sum(h2)), eps)
    S = e2 * np.eye(k)
    for ci in range(len(h2)):
        S += h2[ci] * Kmats[ci]
    return S


def _mstep_reml(h2, stats, eps):
    """M-step: minimise sum_g F_g [log|Sigma_g| + tr(Sigma_g^-1 S_g)] over the
    proportions ``h2`` (the Gaussian ML for the imputed liabilities), on the simplex
    ``h2_c >= eps, sum h2 <= 1 - eps``. ``stats`` is a list of ``(S_g, F_g, Kmats)``."""
    from scipy.optimize import minimize
    C = len(h2)

    def obj_grad(x):
        obj = 0.0
        grad = np.zeros(C)
        for (S, F, Kmats) in stats:
            k = S.shape[0]
            Sig = _reml_sigma(x, Kmats, k, eps)
            Sinv = np.linalg.inv(Sig)
            _sign, logdet = np.linalg.slogdet(Sig)
            SinvS = Sinv @ S
            obj += F * (logdet + np.trace(SinvS))
            for ci in range(C):
                Dc = Kmats[ci] - np.eye(k)                 # dSigma/dh2_c
                M = Sinv @ Dc
                grad[ci] += F * (np.trace(M) - np.trace(M @ SinvS))
        return obj, grad

    import warnings
    with warnings.catch_warnings():
        # SLSQP occasionally line-searches a hair outside the box and clips — benign
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        res = minimize(obj_grad, np.clip(h2, eps, 1 - eps), jac=True, method="SLSQP",
                       bounds=[(eps, 1.0 - eps)] * C,
                       constraints=[{"type": "ineq", "fun": lambda x: 1.0 - eps - x.sum()}])
    x = np.clip(res.x, eps, 1.0 - eps)
    if x.sum() > 1.0 - eps:
        x *= (1.0 - eps) / x.sum()
    return x


def _reml_observed_se(groups, comps, h2, eps, n_score=60, sweeps=1):
    """Model-based SE from the **observed information** (outer product of per-family
    observed-data scores; Fisher's identity + BHHH). At the estimate, draw liability
    samples per family, average each family's complete-data score over them to get
    its observed-data score, and sum their outer products; the inverse is the
    estimate's covariance."""
    C = len(comps)
    info = np.zeros((C, C))
    for g in groups:
        k = g["k"]
        Kmats = [g["K"][c] for c in comps]
        Sig = correct_positive_definite(_reml_sigma(h2, Kmats, k, eps))[0]
        P, sd = gibbs_params(Sig)
        Sinv = np.linalg.inv(Sig)
        D = [Sinv @ (Kmats[ci] - np.eye(k)) @ Sinv for ci in range(C)]   # Sinv Dc Sinv
        tr = np.array([np.trace(Sinv @ (Kmats[ci] - np.eye(k))) for ci in range(C)])
        F = g["F"]
        acc = np.zeros((F, C))                              # summed per-family score
        for _ in range(int(n_score)):
            gibbs_advance(P, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(sweeps))
            x = g["x"]                                      # (F, k)
            for ci in range(C):
                # complete-data score_c per family = 0.5 (x' D x - tr(Sinv Dc))
                q = np.einsum("fi,ij,fj->f", x, D[ci], x)
                acc[:, ci] += 0.5 * (q - tr[ci])
        sbar = acc / float(n_score)                        # observed-data score / family
        info += sbar.T @ sbar                              # BHHH outer product
    cov = np.linalg.inv(info + 1e-10 * np.eye(C))
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _reml_loglik(groups, comps, h2, eps, rng, n_draw=200):
    """GHK Monte-Carlo estimate of the observed-data log-likelihood
    ``sum_i log P(L_i in truncation rectangle)`` with ``L_i ~ N(0, Sigma(h2))``.

    Each family's rectangle probability is estimated by the GHK simulator:
    Cholesky-transform, then draw the coordinates sequentially from their truncated
    conditionals, accumulating the product of interval masses. Returns ``None`` if
    any coordinate is pinned/degenerate (then the likelihood is a density, not a
    rectangle probability)."""
    total = 0.0
    for g in groups:
        k, F = g["k"], g["F"]
        lo, hi = g["lowers"], g["uppers"]
        if np.any((hi - lo) < 1e-8):
            return None
        Kmats = [g["K"][c] for c in comps]
        L = np.linalg.cholesky(correct_positive_definite(_reml_sigma(h2, Kmats, k, eps))[0])
        probs = np.empty((int(n_draw), F))
        for r in range(int(n_draw)):
            z = np.zeros((F, k))
            pr = np.ones(F)
            for j in range(k):
                partial = z[:, :j] @ L[j, :j] if j else np.zeros(F)
                a = norm_cdf((lo[:, j] - partial) / L[j, j])
                b = norm_cdf((hi[:, j] - partial) / L[j, j])
                pr *= np.clip(b - a, 0.0, None)
                u = np.clip(a + rng.random(F) * (b - a), 1e-15, 1 - 1e-15)
                z[:, j] = norm_ppf(u)
            probs[r] = pr
        total += float(np.sum(np.log(np.clip(probs.mean(axis=0), 1e-300, None))))
    return total


def _fit_vc_reml(families, comps, *, n_iter, burn_in, inner_sweeps, damp, seed, eps):
    """Approximate fixed-damping Monte-Carlo EM-style variance components.

    E-step: draw the liabilities from the truncated family MVN under the current
    ``Sigma(h2)`` (the same augmentation as the HE fit). M-step: set ``h2`` to the
    Gaussian likelihood optimum of those liabilities (:func:`_mstep_reml`) rather
    than the moment regression. The implementation uses fixed damping and reports
    the post-burn-in average; it does not apply an increasing Monte-Carlo E-step or
    stop on an observed-likelihood convergence criterion.
    The SE is an **approximate** observed-information SE (:func:`_reml_observed_se`,
    an OPG/BHHH outer-product estimate that is itself subject to Monte-Carlo error),
    accounting approximately for the information lost to thresholding. The SE,
    GHK likelihood and AIC retain Monte-Carlo and finite-iteration error and require
    validation for the target design. Use :func:`bootstrap_fit` for an iid-family
    cluster percentile interval when its sampling assumptions hold."""
    C = len(comps)
    groups = [_prepare_group_vc(families, idx, comps)
              for _key, idx in _group_by_structure(families)]
    XtX = np.zeros((C, C))
    for g in groups:
        for (_i, _j, row) in g["pairs"]:
            XtX += g["F"] * np.outer(row, row)
    if np.linalg.matrix_rank(XtX, tol=1e-8) < C:
        raise ValueError(
            "variance components not identified from these families — the "
            "relationship design is rank-deficient (e.g. fitting 'C' with no "
            "full-sib pairs, or no related pairs at all).")
    if seed is not None:
        _seed_rng(seed)

    h2 = np.full(C, 0.5 / C)
    trace = np.empty((int(n_iter), C))
    for it in range(int(n_iter)):
        stats = []
        for g in groups:
            k = g["k"]
            Kmats = [g["K"][c] for c in comps]
            Sig = correct_positive_definite(_reml_sigma(h2, Kmats, k, eps))[0]
            P, sd = gibbs_params(Sig)
            gibbs_advance(P, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(inner_sweeps))
            S = (g["x"].T @ g["x"]) / g["F"]
            stats.append((S, g["F"], Kmats))
        h2_hat = _mstep_reml(h2, stats, eps)
        h2 = (1.0 - damp) * h2 + damp * h2_hat
        trace[it] = h2

    samples = trace[int(burn_in):]
    est = samples.mean(axis=0)
    rng = np.random.default_rng(_offset_seed(seed, 999))
    se = _reml_observed_se(groups, comps, est, eps)
    ll = _reml_loglik(groups, comps, est, eps, rng)
    aic = None if ll is None else 2.0 * C - 2.0 * ll
    return VarCompResult(
        components={c: float(est[ci]) for ci, c in enumerate(comps)},
        residual=float(1.0 - est.sum()),
        se={c: float(se[ci]) for ci, c in enumerate(comps)},
        traces={c: np.ascontiguousarray(samples[:, ci]) for ci, c in enumerate(comps)},
        n_iter=int(n_iter), burn_in=int(burn_in),
        loglik=ll, aic=aic)


@dataclass
class GenCorrResult:
    """Result of :func:`fit_genetic_correlation`.

    ``h2`` is the ``(P,)`` vector of per-trait liability-scale heritabilities;
    ``rg`` the ``(P, P)`` **genetic correlation** matrix (diagonal 1), the headline
    output; ``re`` the ``(P, P)`` **environmental correlation** (the phenotypic
    correlation not explained by shared genetics); ``rp`` the ``(P, P)`` phenotypic
    correlation of the full liabilities (same individual, across traits).
    ``genetic_cov`` and ``env_cov`` are the ``(P, P)`` genetic ``G`` and
    environmental ``E`` covariances, so ``rp = G + E`` (``G`` has diagonal ``h2``,
    ``E`` diagonal ``e2 = 1 - h2``). ``se`` holds within-dataset Monte-Carlo
    errors (``"h2"``, ``"rg"``, ``"re"``, ``"rp"``), not sampling uncertainty.
    ``phen_names`` labels the
    traits; ``traces`` are the post-burn-in traces (``"h2"``/``"rg"``/``"re"``/``"rp"``)."""
    h2: np.ndarray
    rg: np.ndarray
    re: np.ndarray
    rp: np.ndarray
    genetic_cov: np.ndarray
    env_cov: np.ndarray
    se: dict
    phen_names: list
    traces: dict
    n_iter: int
    burn_in: int


@dataclass
class DecayGenCorrResult(GenCorrResult):
    """Result of :func:`fit_genetic_correlation_decay` — a GenCorrResult plus
    onset-age decay-rate estimates.

    ``lambda_within`` is the ``(P,)`` vector of within-trait decay rates (how
    fast the same trait's genetic covariance decays with onset-age
    difference); ``lambda_cross`` the ``(P, P)`` matrix of cross-trait decay
    rates (0 on the diagonal). ``kernel`` records the kernel used. A rate of 0
    means no onset-age dependence (the scalar :func:`fit_genetic_correlation`
    model); larger means correlation dies faster with onset-age distance.
    ``negq`` is the per-EM-iteration trace of the negative expected
    complete-data log-likelihood (the M-step objective; it should fall and
    plateau as the EM converges), and ``converged`` reports whether it
    stabilised over the run -- treat estimates with ``converged == False``
    as under-iterated (raise ``n_em``). ``shared_env_cov`` is the fitted
    shared-family environmental covariance ``C`` (zeros unless ``shared_env``
    was enabled); ``env_cov`` is the *total* environment ``C + E``, so
    ``rp == genetic_cov + env_cov`` still holds."""
    lambda_within: np.ndarray
    lambda_cross: np.ndarray
    kernel: str
    converged: bool
    negq: np.ndarray
    shared_env_cov: np.ndarray = None


def _decay_kernel(delta, lam, kernel):
    """Decay kernel ``rho(|delta|; lam)``: ``ou`` = ``exp(-lam |d|)``,
    ``gauss`` = ``exp(-0.5 (lam d)^2)``, ``tent`` = ``(1 - lam |d|)+``."""
    delta = np.abs(delta)
    if kernel == "ou":
        return np.exp(-lam * delta)
    if kernel == "gauss":
        return np.exp(-0.5 * (lam * delta) ** 2)
    if kernel == "tent":
        return np.clip(1.0 - lam * delta, 0.0, None)
    raise ValueError(f"unknown decay kernel {kernel!r} (use 'ou', 'gauss', or 'tent')")


def _prepare_group_decay(families, idx, n_pheno):
    """Like :func:`_prepare_group_multi` but also gathers per-member onset ages."""
    roles = sorted(m.role for m in families[idx[0]].members)
    k = len(roles)
    F = len(idx)
    A = np.array([[get_relatedness(ri, rj, 1.0) for rj in roles] for ri in roles])
    A, _ = correct_positive_definite(A)
    W = np.triu(A, 1)                                 # a<b relatedness weights
    sA2 = float(np.sum(W * W)) * F
    lo = np.empty((F, k, n_pheno))
    hi = np.empty((F, k, n_pheno))
    aod = np.empty((F, k, n_pheno))
    for slot, f in enumerate(idx):
        members = sorted(families[f].members, key=lambda member: member.role)
        for c, m in enumerate(members):
            lo[slot, c] = np.broadcast_to(np.asarray(m.lower, float), (n_pheno,))
            hi[slot, c] = np.broadcast_to(np.asarray(m.upper, float), (n_pheno,))
            if m.aod is None:
                raise ValueError(
                    f"member {m.role!r} of family {families[f].fam_id!r} has no "
                    "aod -- fit_genetic_correlation_decay needs an onset age "
                    "(cases) or censoring age (controls) per member per trait")
            aod[slot, c] = np.broadcast_to(np.asarray(m.aod, float), (n_pheno,))
    if not np.all(np.isfinite(aod)):
        raise ValueError("aod must be finite for every member and trait")
    lo_pm = np.ascontiguousarray(lo.transpose(0, 2, 1).reshape(F, k * n_pheno))
    hi_pm = np.ascontiguousarray(hi.transpose(0, 2, 1).reshape(F, k * n_pheno))
    validate_bounds(lo_pm, hi_pm, context="decay genetic-correlation fit bounds")
    fixed = np.ascontiguousarray((hi_pm - lo_pm) < 1e-8)
    x = np.empty((F, k * n_pheno))
    for slot in range(F):
        x[slot] = _init_x(lo_pm[slot], hi_pm[slot], fixed[slot])
    return dict(roles=roles, k=k, F=F, A=A, W=W, sA2=sA2,
                lowers=lo_pm, uppers=hi_pm, fixed=fixed,
                x=np.ascontiguousarray(x), aod=aod)


def _decay_cov(A, h2, G, rp, E, aod_f, lam_w, lam_x, kernel, C=None):
    """Per-family ``(kP, kP)`` liability covariance with the onset-age kernel.

    Block ``(p, q)``, element ``(i, j)``: ``A_ij * (h2[p] if p==q else G[p,q]) *
    K(|aod_i,p - aod_j,q|)``. Same-trait diagonal is 1; cross-trait same-person
    covariance is ``G[p,q] * K(|aod_i,p - aod_i,q|) + E[p,q]`` (the genetic part
    decays with the person's two onset ages, the environmental part does not).
    ``C`` is the optional shared-family environmental covariance: ``C[p,p]``
    adds to every cross-relative same-trait pair, ``C[p,q]`` to every
    cross-trait pair (including same-person).
    """
    P = len(h2)
    k = A.shape[0]
    Joff = 1.0 - np.eye(k)
    S = np.empty((k * P, k * P))
    for p in range(P):
        for q in range(P):
            if p == q:
                K = _decay_kernel(aod_f[:, p, None] - aod_f[None, :, q],
                                  lam_w[p], kernel)
                block = A * h2[p] * K
                if C is not None:
                    block = block + C[p, p] * Joff
                np.fill_diagonal(block, 1.0)
            else:
                K = _decay_kernel(aod_f[:, p, None] - aod_f[None, :, q],
                                  lam_x[p, q], kernel)
                block = A * G[p, q] * K
                k_self = _decay_kernel(aod_f[:, p] - aod_f[:, q],
                                       lam_x[p, q], kernel)
                cq = C[p, q] if C is not None else 0.0
                block = block + cq
                np.fill_diagonal(block, G[p, q] * k_self + cq + E[p, q])
            S[p * k:(p + 1) * k, q * k:(q + 1) * k] = block
    return S


def _decay_kernel_deriv(delta, lam, kernel):
    """``d/d lam`` of the decay kernel (see :func:`_decay_kernel`)."""
    delta = np.abs(delta)
    if kernel == "ou":
        return -delta * np.exp(-lam * delta)
    if kernel == "gauss":
        return -lam * delta ** 2 * np.exp(-0.5 * (lam * delta) ** 2)
    if kernel == "tent":
        return -delta * ((1.0 - lam * delta) > 0.0)
    raise ValueError(f"unknown decay kernel {kernel!r} (use 'ou', 'gauss', or 'tent')")


def _decay_cov_batch(A, dself, dcross, h2, G, E, lam_w, lam_x, pairs, kernel,
                     C=None):
    """Vectorised :func:`_decay_cov` over ``F`` families sharing structure ``A``.

    ``dself`` is the ``(F, k, k, P)`` same-trait age-difference tensor and
    ``dcross`` a list of ``(F, k, k)`` cross-trait age-difference matrices (one
    per pair); both are precomputed once since ages are fixed. Returns the
    ``(F, kP, kP)`` liability covariances. Identical to looping
    :func:`_decay_cov` per family, but the linear-algebra-heavy fit paths
    (inverse, determinant, score) can then be batched rather than Python-looped.

    ``C`` is the optional shared-family environmental covariance (a family-level
    random effect, constant across relatives): ``C[p,p]`` adds to every
    cross-relative pair of trait ``p`` and ``C[p,q]`` to every pair (including
    same-person) of traits ``p,q``. The same-trait diagonal stays 1 (the unique
    environment absorbs it, ``E[p,p] = 1 - h2 - C[p,p]``).
    """
    P = len(h2)
    k = A.shape[0]
    F = dself.shape[0]
    kP = k * P
    idx = np.arange(k)
    Joff = 1.0 - np.eye(k)
    Sig = np.zeros((F, kP, kP))
    for p in range(P):
        K = _decay_kernel(dself[..., p], lam_w[p], kernel)
        block = A[None] * h2[p] * K
        if C is not None:
            block = block + C[p, p] * Joff[None]
        block[:, idx, idx] = 1.0
        Sig[:, p * k:(p + 1) * k, p * k:(p + 1) * k] = block
    for i, (p, q) in enumerate(pairs):
        KX = _decay_kernel(dcross[i], lam_x[p, q], kernel)
        block = A[None] * G[p, q] * KX
        cq = C[p, q] if C is not None else 0.0
        block = block + cq
        block[:, idx, idx] = G[p, q] * KX[:, idx, idx] + cq + E[p, q]
        Sig[:, p * k:(p + 1) * k, q * k:(q + 1) * k] = block
        Sig[:, q * k:(q + 1) * k, p * k:(p + 1) * k] = block.transpose(0, 2, 1)
    return Sig


def _decay_unpack(theta, P, pairs, lam_max, eps, shared_lambda=False,
                  shared_env=False):
    """Split the decay-fit parameter vector into named covariance pieces.

    Default layout: ``[h2 (P), lam_within (P), G pairs, lam_cross pairs, E pairs]``.
    With ``shared_lambda`` a single decay rate governs every block, so the layout
    is ``[h2 (P), lam (1), G pairs, E pairs]`` and ``lam_w``/``lam_x`` are that one
    rate broadcast -- fewer parameters, always PSD, and less amplitude-decay ridge.
    With ``shared_env`` the vector continues ``[..., c2 (P), C pairs]``: a
    shared-family environmental variance per trait (``c2``) and cross-trait
    covariance (``C``), constant across relatives. ``E`` is then the *unique*
    (within-person) environment with diagonal ``1 - h2 - c2``, and the reported
    total environment is ``C + E``. Returns ``(h2, lam_w, G, lam_x, E, rp, C)``;
    ``rp = G + C + E`` (diagonal 1), ``C is None`` when ``shared_env`` is False.
    Clips ``h2`` to ``(eps, 1-eps)`` and the rates to ``[0, lam_max]``."""
    theta = np.asarray(theta, float)
    h2 = np.clip(theta[:P], eps, 1.0 - eps)
    npairs = len(pairs)
    G = np.diag(h2).astype(float)
    E = np.zeros((P, P))
    lam_x = np.zeros((P, P))
    if shared_lambda:
        lam_w = np.full(P, float(np.clip(theta[P], 0.0, lam_max)))
        lam_x[:] = lam_w[0]
        g0 = P + 1           # layout [h2, lam, G, E]: E is npairs past G
        e_off = npairs
    else:
        lam_w = np.clip(theta[P:2 * P], 0.0, lam_max)
        for i, (p, q) in enumerate(pairs):
            lam_x[p, q] = lam_x[q, p] = np.clip(theta[2 * P + npairs + i],
                                                0.0, lam_max)
        g0 = 2 * P           # layout [h2, lam_w, G, lam_x, E]: E is 2*npairs past G
        e_off = 2 * npairs
    for i, (p, q) in enumerate(pairs):
        G[p, q] = G[q, p] = theta[g0 + i]
        E[p, q] = E[q, p] = theta[g0 + e_off + i]
    if shared_env:
        c0 = g0 + e_off + npairs
        C = np.diag(np.clip(theta[c0:c0 + P], 0.0, 1.0))
        for i, (p, q) in enumerate(pairs):
            C[p, q] = C[q, p] = theta[c0 + P + i]
        np.fill_diagonal(E, 1.0 - h2 - np.diag(C))
    else:
        C = None
        np.fill_diagonal(E, 1.0 - h2)
    rp = G + E + (C if C is not None else 0.0)
    np.fill_diagonal(rp, 1.0)
    return h2, lam_w, G, lam_x, E, rp, C


def _decay_negq_grad(theta, P, M_groups, groups, pairs, lam_max, kernel, eps,
                     shared_lambda=False, shared_env=False):
    """Negative expected complete-data log-likelihood and its analytic gradient.

    ``negQ = 1/2 sum_f [ log|Sig_f| + tr(Sig_f^-1 M_f) ]`` over every family's
    age-structured covariance (``M_f`` the imputed second moment), plus
    ``d negQ / d theta``. This is the L-BFGS objective for the decay M-step. It
    is module-level (not a closure) so the gradient can be unit-tested against
    finite differences -- a wrongly symmetrised cross-trait block once corrupted
    it, which only a finite-difference test catches. With ``shared_lambda`` the
    gradient is summed over the tied decay rates; with ``shared_env`` the
    shared-family environmental variance (``c2``) and cross-trait covariance
    (``C``) gradients are appended."""
    h2, lam_w, G, lam_x, E, rp, C = _decay_unpack(theta, P, pairs, lam_max, eps,
                                                  shared_lambda, shared_env)
    npairs = len(pairs)
    npar = 2 * P + 3 * npairs + (P + npairs if shared_env else 0)
    negQ = 0.0
    grad = np.zeros(npar)
    for g, M in zip(groups, M_groups):
        k, F = g["k"], g["F"]
        Sig = _decay_cov_batch(g["A"], g["dself"], g["dcross"], h2, G, E,
                               lam_w, lam_x, pairs, kernel, C)
        minev = float(np.linalg.eigvalsh(Sig).min())
        if minev <= 1e-9:
            return 1e9 + 1e9 * abs(minev), np.zeros(len(theta))
        Si = np.linalg.inv(Sig)
        logdet = np.linalg.slogdet(Sig)[1]
        negQ += 0.5 * float(np.sum(logdet) + np.einsum("fii->", Si @ M))
        Psi = Si - Si @ M @ Si
        mask = 1.0 - np.eye(k)
        for p in range(P):
            Pp = Psi[:, p * k:(p + 1) * k, p * k:(p + 1) * k]
            KW = _decay_kernel(g["dself"][..., p], lam_w[p], kernel)
            grad[p] += 0.5 * float(
                np.einsum("fij,fji->", Pp, g["A"] * KW * mask))
            dKW = _decay_kernel_deriv(g["dself"][..., p], lam_w[p], kernel)
            grad[P + p] += 0.5 * float(
                np.einsum("fij,fji->", Pp, h2[p] * g["A"] * dKW * mask))
            if shared_env:
                # d Sigma / d c2[p] = off-diagonal ones on block (p,p) == mask
                grad[2 * P + 3 * npairs + p] += 0.5 * float(
                    np.einsum("fij,fji->", Pp, np.broadcast_to(mask, (F, k, k))))
        for i, (p, q) in enumerate(pairs):
            Pq = Psi[:, p * k:(p + 1) * k, q * k:(q + 1) * k]
            KX = _decay_kernel(g["dcross"][i], lam_x[p, q], kernel)
            # 0.5 * tr(Psi D); the (p,q) and (q,p) blocks contribute equally
            # (both einsum(Pq, blk)), so this is einsum(Pq, blk) -- NOT
            # einsum(Pq, blk + blk^T): blk = A*KX is asymmetric, and
            # symmetrising it corrupts the cross-trait gradient.
            blk = g["A"] * KX
            grad[2 * P + i] += float(np.einsum("fij,fij->", Pq, blk))
            dKX = _decay_kernel_deriv(g["dcross"][i], lam_x[p, q], kernel)
            blk = G[p, q] * g["A"] * dKX
            grad[2 * P + npairs + i] += float(np.einsum("fij,fij->", Pq, blk))
            grad[2 * P + 2 * npairs + i] += float(
                np.einsum("fij,fij->", Pq,
                          np.broadcast_to(np.eye(k), (F, k, k))))
            if shared_env:
                # d Sigma / d C[p,q] = all-ones on blocks (p,q),(q,p), so the
                # (two equal) block contributions give grad = sum of Pq.
                grad[2 * P + 3 * npairs + P + i] += float(Pq.sum())
    if shared_lambda:
        # map per-block gradient to the shared layout [h2 (P), lam (1), G, E]:
        # the single rate's gradient is the sum of all tied-rate gradients.
        lam_grad = grad[P:2 * P].sum() + grad[2 * P + npairs:2 * P + 2 * npairs].sum()
        grad = np.concatenate([grad[:P], [lam_grad],
                               grad[2 * P:2 * P + npairs],
                               grad[2 * P + 2 * npairs:]])
    return negQ, grad


def fit_genetic_correlation_decay(families, *, kernel="ou", lam_max=None,
                                  shared_lambda=False, shared_env=False,
                                  n_em=40, n_draw=100, burn=40, m_iter=100,
                                  n_starts=1, seed=None, eps=1e-4,
                                  phen_names=None):
    """Genetic correlation with an onset-age decay (structured ``r_g``).

    Estimates the two-trait (or multi-trait) genetic correlation when the
    genetic covariance between relatives diagnosed at ages ``a1`` and ``a2``
    decays with their onset-age difference:

    ``Cov(g_i^p(a1), g_j^q(a2)) = A_ij * sqrt(h2_p h2_q) * rho_g * K(|a1-a2|; lam)``

    with a decay-rate scalar ``lam`` (the age-difference importance parameter):
    ``lam = 0`` recovers the scalar :func:`fit_genetic_correlation` model and
    larger ``lam`` means correlation dies faster with onset-age distance.
    ``kernel`` selects ``K``: ``"ou"`` (``exp(-lam |d|)``, the default),
    ``"gauss"`` (``exp(-0.5 (lam d)^2)``), or ``"tent"`` (``(1 - lam |d|)+``).
    Each member must carry ``aod`` (onset age for cases, censoring age for
    controls), per trait for the multi-trait model. ``lam_max`` bounds the rate;
    by default it is derived from the observed onset-age span. ``shared_lambda``
    ties every block's decay rate to a single scalar (the user's original
    one-parameter form): fewer parameters, a guaranteed positive-definite
    covariance, and less amplitude-decay ridge -- the recommended setting unless
    there is a specific reason to let the rates differ. ``shared_env`` adds a
    shared-family environmental component ``C`` (a family-level random effect,
    constant across relatives) so the model is an onset-age ACE decomposition
    rather than genetics + within-person environment only; without it, real
    household environment is wrongly attributed to genetics and attenuates
    ``r_g``. ``C`` is only identifiable with enough related pairs and onset-age
    spread, so enable it on data-rich, extended-pedigree designs. ``n_starts``
    re-runs the EM from that many perturbed initialisations and keeps the
    best-fitting (lowest final objective) -- a guard against the multi-modal
    likelihood at small samples (each start costs a full EM run).

    **Why a likelihood M-step, not moments.** A Haseman-Elston regression of the
    augmented cross-products on ``A * K`` (the natural analogue of
    :func:`fit_genetic_correlation`) is *confounded* here: case/control
    ascertainment truncates the liabilities, and the truncation inflation is
    itself age-dependent (closely related, similar-onset pairs are more often
    jointly affected), so it masquerades as a steeply decaying genetic signal
    and drives ``lam`` to its bound. This fit therefore uses a Monte-Carlo **EM**
    whose M-step fully maximises the expected complete-data Gaussian
    log-likelihood ``Q = -1/2 sum_f [ log|Sig_f| + tr(Sig_f^-1 M_f) ]`` (``M_f``
    the imputed second moment, averaged over ``n_draw`` Gibbs draws) by L-BFGS
    with the analytic score. The likelihood separates the genetic decay from the
    ascertainment geometry; the moment regression cannot.

    **Identifiability (read this).** The amplitude (``rho_g``) and the decay
    rate (``lam``) trade off along a likelihood ridge, and the cross-trait
    genetic signal competes with the environmental correlation ``re``: at small
    samples the fit can collapse ``rho_g`` toward 0 or inflate it. The model is
    *identifiable in principle* — the cross-relative cross-trait covariance
    ``A * G * K`` is purely genetic in this model (environment is not shared
    across relatives) — but only **data-rich** designs pin it down: the
    repository kill-test needs on the order of **thousands of families** and
    several dozen EM iterations before ``rho_g`` and ``lam`` climb to their true
    values, with residual attenuation of ``rho_g`` (environment absorbing
    genetic correlation). With few families or little onset-age spread, expect
    noisy, ridge-dominated estimates; prefer the scalar model there. ``n_em``,
    ``n_draw`` and ``m_iter`` control the EM iterations, the per-E-step Gibbs
    draws and the L-BFGS work per M-step.

    Returns a :class:`DecayGenCorrResult`: the usual ``rg``/``re``/``rp``/
    ``h2``/covariances plus ``lambda_within`` (per-trait decay rates) and
    ``lambda_cross`` (cross-trait rates), averaged over the converged tail of
    the EM run. ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or
    ``None``."""
    from scipy.optimize import minimize

    if not families:
        raise ValueError("no families provided")
    _require_member_rows(families, context="decay genetic-correlation fit")
    first_lower = np.asarray(families[0].members[0].lower)
    P = int(first_lower.size) if first_lower.ndim else 1
    if P < 1:
        raise ValueError("need at least one trait")
    _validate_multitrait_bounds(families, P)
    if int(n_em) < 8:
        raise ValueError(f"n_em ({n_em}) must be >= 8 so the converged tail has "
                         "enough points for a Monte-Carlo SE")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(P)]
    elif len(phen_names) != P:
        raise ValueError("phen_names length must match number of traits")

    groups = [_prepare_group_decay(families, idx, P)
              for _key, idx in _group_by_structure(families)]
    sA2 = sum(g["sA2"] for g in groups)
    if sA2 <= 0:
        raise ValueError("no related pairs in the families -- cannot fit the "
                         "decay model (need relatives with onset ages).")
    all_aod = np.concatenate([g["aod"].ravel() for g in groups])
    if lam_max is None:
        span = max(float(all_aod.max() - all_aod.min()), 1.0)
        lam_max = 4.0 / span
    lam_max = float(lam_max)

    if seed is not None:
        _seed_rng(seed)

    pairs = [(p, q) for p in range(P) for q in range(p + 1, P)]
    npairs = len(pairs)
    # theta layout: [h2 (P), lam_within (P), G pairs, lam_cross pairs, E pairs]
    for g in groups:
        aod = g["aod"]                                        # (F, k, P)
        g["dself"] = np.abs(aod[:, :, None, :] - aod[:, None, :, :])   # (F,k,k,P)
        g["dcross"] = [np.abs(aod[:, :, None, p] - aod[:, None, :, q])
                       for (p, q) in pairs]                            # list (F,k,k)

    def pack(h2, lam_w, G, lam_x, E, C):
        parts = [np.asarray(h2, float).ravel(),
                 np.atleast_1d(lam_w[0] if shared_lambda else lam_w).ravel(),
                 np.array([G[p, q] for (p, q) in pairs])]
        if not shared_lambda:
            parts.append(np.array([lam_x[p, q] for (p, q) in pairs]))
        parts.append(np.array([E[p, q] for (p, q) in pairs]))
        if shared_env:
            parts.append(np.diag(C).ravel())
            parts.append(np.array([C[p, q] for (p, q) in pairs]))
        return np.concatenate(parts)

    def unpack(theta):
        return _decay_unpack(theta, P, pairs, lam_max, eps, shared_lambda,
                             shared_env)

    def compute_M(h2, lam_w, G, lam_x, E, rp, C):
        M_groups = []
        for g in groups:
            k, F = g["k"], g["F"]
            kP = k * P
            M = np.empty((F, kP, kP))
            for slot in range(F):
                S = _decay_cov(g["A"], h2, G, rp, E, g["aod"][slot],
                               lam_w, lam_x, kernel, C)
                S, _ = correct_positive_definite(S)
                Pm, sd = gibbs_params(S)
                lo_s = g["lowers"][slot:slot + 1]
                up_s = g["uppers"][slot:slot + 1]
                fx_s = g["fixed"][slot:slot + 1]
                x_s = g["x"][slot:slot + 1]
                gibbs_advance(Pm, sd, lo_s, up_s, fx_s, x_s, int(burn))
                M[slot] = gibbs_advance_moment(Pm, sd, lo_s, up_s, fx_s, x_s,
                                               int(n_draw))[0]
            M_groups.append(M)
        return M_groups

    def qgrad(theta, M_groups):
        return _decay_negq_grad(theta, P, M_groups, groups, pairs, lam_max,
                                kernel, eps, shared_lambda, shared_env)

    c_bounds = ([(0.0, 1.0 - eps)] * P + [(-0.99, 0.99)] * npairs
                if shared_env else [])
    if shared_lambda:
        bounds = ([(eps, 1.0 - eps)] * P + [(0.0, lam_max)]
                  + [(-0.99, 0.99)] * npairs + [(-0.99, 0.99)] * npairs
                  + c_bounds)
    else:
        bounds = ([(eps, 1.0 - eps)] * P + [(0.0, lam_max)] * P
                  + [(-0.99, 0.99)] * npairs + [(0.0, lam_max)] * npairs
                  + [(-0.99, 0.99)] * npairs + c_bounds)

    def _init_theta(rng):
        h2i = np.full(P, 0.4) if rng is None else rng.uniform(0.2, 0.6, P)
        Gi = np.diag(h2i).astype(float)
        Ei = np.diag(1.0 - h2i)
        Ci = np.zeros((P, P))
        lwi = np.zeros(P)
        lxi = np.zeros((P, P))
        for (p, q) in pairs:
            Gi[p, q] = Gi[q, p] = 0.1 if rng is None else rng.uniform(-0.3, 0.3)
            Ei[p, q] = Ei[q, p] = 0.1 if rng is None else rng.uniform(-0.2, 0.2)
            if not shared_lambda:
                lxi[p, q] = lxi[q, p] = (0.0 if rng is None
                                         else rng.uniform(0.0, 0.5 * lam_max))
            if shared_env:
                Ci[p, p] = 0.0 if rng is None else rng.uniform(0.0, 0.2)
                Ci[p, q] = Ci[q, p] = 0.0 if rng is None else rng.uniform(-0.1, 0.1)
        if rng is not None:
            lwi[:] = rng.uniform(0.0, 0.5 * lam_max, P)
        return pack(h2i, lwi, Gi, lxi, Ei, Ci)

    def _run_em(theta):
        tr_h2 = np.empty((int(n_em), P))
        tr_lamw = np.empty((int(n_em), P))
        tr_lamx = np.empty((int(n_em), P, P))
        tr_rg = np.empty((int(n_em), P, P))
        tr_re = np.empty((int(n_em), P, P))
        tr_rp = np.empty((int(n_em), P, P))
        tr_G = np.empty((int(n_em), P, P))
        tr_E = np.empty((int(n_em), P, P))
        tr_C = np.empty((int(n_em), P, P))
        tr_negq = np.empty(int(n_em))
        for it in range(int(n_em)):
            h2, lam_w, G, lam_x, E, rp, C = unpack(theta)
            M_groups = compute_M(h2, lam_w, G, lam_x, E, rp, C)
            res = minimize(qgrad, theta, args=(M_groups,), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": int(m_iter)})
            theta = res.x
            # coherent PSD split (safety net; L-BFGS is bounded but not PD-aware)
            h2, lam_w, G, lam_x, E, rp, C = unpack(theta)
            G, _ = _project_covariance(np.where(np.eye(P, dtype=bool), h2, G), h2,
                                       eps=eps)
            h2 = np.clip(np.diag(G), eps, 1.0 - eps)
            if shared_env:
                C, _ = _project_covariance(C, np.maximum(np.diag(C), 0.0), eps=eps)
                c2 = np.clip(np.diag(C), 0.0, 1.0 - eps - h2)
                np.fill_diagonal(C, c2)
            else:
                C = np.zeros((P, P))
                c2 = np.zeros(P)
            e2 = np.clip(1.0 - h2 - c2, eps, None)
            E, _ = _project_covariance(np.where(np.eye(P, dtype=bool), e2, E), e2,
                                       eps=eps)
            np.fill_diagonal(E, e2)
            env_total = C + E
            rp = G + env_total
            np.fill_diagonal(rp, 1.0)
            theta = pack(h2, lam_w, G, lam_x, E, C)

            rg = _cov_to_corr(G, h2, eps)
            re = _cov_to_corr(env_total, 1.0 - h2, eps)
            tr_h2[it] = h2
            tr_lamw[it] = lam_w
            tr_lamx[it] = lam_x
            tr_rg[it] = rg
            tr_re[it] = re
            tr_rp[it] = rp
            tr_G[it] = G
            tr_E[it] = E
            tr_C[it] = C
            tr_negq[it] = float(res.fun)
        return theta, dict(h2=tr_h2, lamw=tr_lamw, lamx=tr_lamx, rg=tr_rg,
                           re=tr_re, rp=tr_rp, G=tr_G, E=tr_E, C=tr_C,
                           negq=tr_negq)

    best = None
    for start in range(int(n_starts)):
        rng_start = (None if start == 0 else np.random.default_rng(
            None if seed is None else int(seed) + 7919 * start))
        theta, traces = _run_em(_init_theta(rng_start))
        negq_sel = float(traces["negq"][int(n_em) // 2:].mean())
        if best is None or negq_sel < best[0]:
            best = (negq_sel, traces)
    _sel, traces = best
    tr_h2 = traces["h2"]
    tr_lamw = traces["lamw"]
    tr_lamx = traces["lamx"]
    tr_rg = traces["rg"]
    tr_re = traces["re"]
    tr_rp = traces["rp"]
    tr_G = traces["G"]
    tr_E = traces["E"]
    tr_C = traces["C"]
    tr_negq = traces["negq"]

    # average the converged tail (second half) to damp MC-EM jitter
    tail = slice(int(n_em) // 2, int(n_em))
    G_est = tr_G[tail].mean(axis=0)
    E_est = tr_E[tail].mean(axis=0)      # unique (within-person) environment
    C_est = tr_C[tail].mean(axis=0)      # shared-family environment
    env_est = E_est + C_est              # total environment (env_cov)
    lamw_est = tr_lamw[tail].mean(axis=0)
    lamx_est = tr_lamx[tail].mean(axis=0)
    h2_est = np.diag(G_est).copy()
    rg = _cov_to_corr(G_est, h2_est, eps)
    re = _cov_to_corr(env_est, 1.0 - h2_est, eps)
    rp = G_est + env_est
    np.fill_diagonal(rp, 1.0)
    _, h2_se = batch_means(tr_h2[tail])
    _, rg_se = batch_means(tr_rg[tail].reshape(-1, P * P))
    _, re_se = batch_means(tr_re[tail].reshape(-1, P * P))
    _, rp_se = batch_means(tr_rp[tail].reshape(-1, P * P))

    # convergence: the parameter estimates should stop drifting over the
    # converged tail (the negQ trace is shown in traces but is not monotone in
    # MC-EM, so it is a poor criterion). Compare the 3rd vs 4th quarter means.
    q3 = slice(int(n_em) // 2, 3 * int(n_em) // 4)
    q4 = slice(3 * int(n_em) // 4, int(n_em))

    def _drift(tr, tol):
        a = tr[q3].mean(axis=0)
        b = tr[q4].mean(axis=0)
        return float(np.max(np.abs(b - a))) < tol

    converged = bool(_drift(tr_rg, 0.05) and _drift(tr_re, 0.05)
                     and _drift(tr_h2, 0.03) and _drift(tr_lamw, 0.01)
                     and _drift(tr_lamx, 0.01))

    return DecayGenCorrResult(h2=h2_est, rg=rg, re=re, rp=rp,
                              genetic_cov=G_est, env_cov=env_est,
                              se={"h2": h2_se, "rg": rg_se, "re": re_se,
                                  "rp": rp_se},
                              phen_names=list(phen_names),
                              traces={"h2": tr_h2[tail], "rg": tr_rg[tail],
                                      "re": tr_re[tail], "rp": tr_rp[tail],
                                      "lambda_within": tr_lamw[tail],
                                      "lambda_cross": tr_lamx[tail],
                                      "shared_env": tr_C[tail],
                                      "negq": tr_negq},
                              n_iter=int(n_em), burn_in=int(n_em) // 2,
                              lambda_within=lamw_est,
                              lambda_cross=lamx_est,
                              kernel=kernel,
                              converged=converged,
                              negq=tr_negq,
                              shared_env_cov=C_est)



def _multi_cov(A, h2, G, rp):
    """Assemble the phenotype-major ``(kP, kP)`` covariance of the observed members'
    liabilities across ``P`` traits, given the additive relationship ``A`` (``k×k``,
    diagonal 1), per-trait ``h2``, genetic covariance ``G`` and phenotypic
    correlation ``rp``. Block ``(p, q)`` is ``A * (h2_p if p==q else G[p,q])`` with
    its diagonal set to 1 (same trait) or ``rp[p,q]`` (cross trait)."""
    P = len(h2)
    k = A.shape[0]
    S = np.empty((k * P, k * P))
    for p in range(P):
        for q in range(P):
            block = A * (h2[p] if p == q else G[p, q])
            np.fill_diagonal(block, 1.0 if p == q else rp[p, q])
            S[p * k:(p + 1) * k, q * k:(q + 1) * k] = block
    return S


def _project_correlation(matrix, *, zero_pairs=(), eps=1e-8):
    """Project a symmetric matrix to a positive-definite correlation matrix.

    ``zero_pairs`` adds affine constraints used by the pairwise genetic-correlation
    null. Alternating PSD/affine projections keep those entries exactly zero; a
    final shrink towards identity makes the result strictly positive definite
    without changing either the diagonal or constrained zeros.
    """
    corr = np.asarray(matrix, dtype=float)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    pairs = tuple((min(int(i), int(j)), max(int(i), int(j)))
                  for i, j in zero_pairs)
    for i, j in pairs:
        corr[i, j] = corr[j, i] = 0.0

    if pairs:
        # Higham/Dykstra alternating projection: PSD cone, then the affine set
        # (unit diagonal plus the requested zero entries).
        correction = np.zeros_like(corr)
        for _ in range(200):
            residual = corr - correction
            values, vectors = np.linalg.eigh((residual + residual.T) / 2.0)
            psd = (vectors * np.maximum(values, 0.0)) @ vectors.T
            correction = psd - residual
            previous = corr
            corr = (psd + psd.T) / 2.0
            np.fill_diagonal(corr, 1.0)
            for i, j in pairs:
                corr[i, j] = corr[j, i] = 0.0
            if (np.max(np.abs(corr - previous)) < 1e-12
                    and np.min(np.linalg.eigvalsh(corr)) >= -1e-10):
                break
    else:
        values, vectors = np.linalg.eigh(corr)
        corr = (vectors * np.maximum(values, eps)) @ vectors.T
        scale = np.sqrt(np.clip(np.diag(corr), eps, None))
        corr /= np.outer(scale, scale)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)

    smallest = float(np.min(np.linalg.eigvalsh(corr)))
    if smallest < eps:
        shrink = (eps - smallest) / (1.0 - smallest)
        corr = (1.0 - shrink) * corr + shrink * np.eye(corr.shape[0])
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    for i, j in pairs:
        corr[i, j] = corr[j, i] = 0.0
    return corr


def _project_covariance(matrix, variances, *, zero_pairs=(), eps=1e-8):
    """Return a PSD covariance with fixed ``variances`` and its correlation."""
    variances = np.asarray(variances, dtype=float)
    sd = np.sqrt(np.clip(variances, eps, None))
    raw_corr = np.asarray(matrix, dtype=float) / np.outer(sd, sd)
    corr = _project_correlation(raw_corr, zero_pairs=zero_pairs, eps=eps)
    cov = corr * np.outer(sd, sd)
    np.fill_diagonal(cov, variances)
    return cov, corr


def _cov_to_corr(cov, variances, eps=1e-8):
    """Standardise an already-PSD covariance without elementwise clipping."""
    sd = np.sqrt(np.clip(np.asarray(variances, dtype=float), eps, None))
    corr = np.asarray(cov, dtype=float) / np.outer(sd, sd)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _prepare_group_multi(families, idx, n_pheno):
    """Per-structure precompute for the r_g fit: relationship matrix ``A`` and its
    upper-triangle weights, phenotype-major bounds ``(F, kP)`` and initial state."""
    roles = sorted(m.role for m in families[idx[0]].members)
    k = len(roles)
    F = len(idx)
    A = np.array([[get_relatedness(ri, rj, 1.0) for rj in roles] for ri in roles])
    A, _ = correct_positive_definite(A)
    W = np.triu(A, 1)                                 # a<b relatedness weights
    sA2 = float(np.sum(W * W)) * F                    # sum_{a<b} A_ab^2, pooled over F
    lo = np.empty((F, k, n_pheno))
    hi = np.empty((F, k, n_pheno))
    for slot, f in enumerate(idx):
        members = sorted(families[f].members, key=lambda member: member.role)
        for c, m in enumerate(members):
            lo[slot, c] = np.broadcast_to(np.asarray(m.lower, float), (n_pheno,))
            hi[slot, c] = np.broadcast_to(np.asarray(m.upper, float), (n_pheno,))
    # phenotype-major: coordinate p*k + a
    lo_pm = np.ascontiguousarray(lo.transpose(0, 2, 1).reshape(F, k * n_pheno))
    hi_pm = np.ascontiguousarray(hi.transpose(0, 2, 1).reshape(F, k * n_pheno))
    validate_bounds(lo_pm, hi_pm, context="genetic-correlation fit bounds")
    fixed = np.ascontiguousarray((hi_pm - lo_pm) < 1e-8)
    x = np.empty((F, k * n_pheno))
    for slot in range(F):
        x[slot] = _init_x(lo_pm[slot], hi_pm[slot], fixed[slot])
    return dict(roles=roles, k=k, F=F, A=A, W=W, sA2=sA2,
                lowers=lo_pm, uppers=hi_pm,
                fixed=fixed, x=np.ascontiguousarray(x))


def _require_member_rows(families, *, context):
    """Reject memberless families where a fit needs observed phenotype rows."""
    for family in families:
        if not family.members:
            raise ValueError(
                f"{context}: family {family.fam_id!r} has no members")


def fit_genetic_correlation(families, *, n_iter=1500, burn_in=500, inner_sweeps=5,
                            damp=0.2, seed=None, eps=1e-4, phen_names=None):
    """Estimate the **genetic correlation** between traits from family data.

    The multi-trait generalisation of :func:`fit_heritability`: a **cross-trait**
    Haseman–Elston regression. Each member must carry one case/control interval per
    trait (``lower``/``upper`` are length-``P`` sequences, as for
    :func:`~ltpred.estimate.estimate_liability_multi`). Each sweep draws the
    members' ``P``-trait liabilities from the full truncated-MVN under the current
    parameters, then updates by regressing the sampled cross-products on the
    additive relationship ``A``:

    - **same trait, different relatives** → ``h2_p = sum A_ij l_ip l_jp / sum A_ij^2``;
    - **different trait, different relatives** → the genetic covariance
      ``G[p,q] = sum A_ij (l_ip l_jq + l_iq l_jp) / (2 sum A_ij^2)``;
    - **same individual, different trait** → the phenotypic correlation ``rp[p,q]``.

    The genetic correlation is ``rg[p,q] = G[p,q] / sqrt(h2_p h2_q)``. The phenotypic
    correlation splits into genetic and environmental parts, ``rp = G + E``, so the
    **environmental correlation** ``re[p,q] = (rp[p,q] - G[p,q]) / sqrt(e2_p e2_q)``
    (``e2 = 1 - h2``) is returned too. Damped across sweeps; needs related pairs
    (raises otherwise). Repository benchmarks found approximately unbiased estimates
    near the null and mild attenuation at large ``|rg|`` in the tested designs.

    Returns a :class:`GenCorrResult` (``rg``, ``re``, ``rp``, per-trait ``h2``, the
    ``genetic_cov``/``env_cov`` matrices). As with :func:`fit_heritability`, the
    reported ``se`` is a within-dataset Monte-Carlo error. Use a family-cluster
    bootstrap for sampling uncertainty when families are independent and
    representative. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``."""
    if not families:
        raise ValueError("no families provided")
    _require_member_rows(families, context="genetic-correlation fit")
    first_lower = np.asarray(families[0].members[0].lower)
    P = int(first_lower.size)
    if P < 2:
        raise ValueError("fit_genetic_correlation needs >= 2 traits — each member's "
                         "lower/upper must be length-n_pheno (see estimate_liability_multi)")
    _validate_multitrait_bounds(families, P)
    if int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be < n_iter ({n_iter})")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(P)]
    elif len(phen_names) != P:
        raise ValueError("phen_names length must match number of traits")

    groups = [_prepare_group_multi(families, idx, P)
              for _key, idx in _group_by_structure(families)]
    sA2 = sum(g["sA2"] for g in groups)
    if sA2 <= 0:
        raise ValueError("no related pairs in the families — cannot fit genetic "
                         "correlation (need relatives, not lone probands).")
    n_members = sum(g["F"] * g["k"] for g in groups)

    if seed is not None:
        _seed_rng(seed)

    h2 = np.full(P, 0.4)
    G = np.diag(h2).astype(float)
    E = np.diag(1.0 - h2)
    tr_h2 = np.empty((int(n_iter), P))
    tr_rg = np.empty((int(n_iter), P, P))
    tr_re = np.empty((int(n_iter), P, P))
    tr_rp = np.empty((int(n_iter), P, P))
    G_sum = np.zeros((P, P))
    E_sum = np.zeros((P, P))
    for it in range(int(n_iter)):
        numG = np.zeros((P, P))
        numRp = np.zeros((P, P))
        rp = G + E
        for g in groups:
            k = g["k"]
            sigma = _multi_cov(g["A"], h2, G, rp)
            Pm, sd = gibbs_params(sigma)
            gibbs_advance(Pm, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(inner_sweeps))
            gram = g["x"].T @ g["x"]                   # (kP, kP)
            for p in range(P):
                for q in range(P):
                    block = gram[p * k:(p + 1) * k, q * k:(q + 1) * k]
                    numG[p, q] += float(np.sum(g["W"] * block))    # a<b relatedness
                    numRp[p, q] += float(np.trace(block))          # same individual
        h2_hat = np.clip(np.array([numG[p, p] / sA2 for p in range(P)]), eps, 1 - eps)
        G_raw = np.diag(h2_hat).astype(float)
        for p in range(P):
            for q in range(p + 1, P):
                G_raw[p, q] = G_raw[q, p] = (
                    numG[p, q] + numG[q, p]) / (2 * sA2)
        G_hat, _ = _project_covariance(G_raw, h2_hat, eps=eps)
        rp_hat = numRp / n_members
        drp = np.sqrt(np.clip(np.diag(rp_hat), eps, None))
        rp_hat = rp_hat / np.outer(drp, drp)
        np.fill_diagonal(rp_hat, 1.0)
        e2_hat = 1.0 - h2_hat
        E_hat, _ = _project_covariance(rp_hat - G_hat, e2_hat, eps=eps)

        G = (1 - damp) * G + damp * G_hat
        E = (1 - damp) * E + damp * E_hat
        h2 = np.diag(G).copy()
        np.fill_diagonal(E, 1.0 - h2)
        rp = G + E
        np.fill_diagonal(rp, 1.0)

        rg = _cov_to_corr(G, h2, eps)
        re = _cov_to_corr(E, 1.0 - h2, eps)
        tr_h2[it] = h2
        tr_rg[it] = rg
        tr_re[it] = re
        tr_rp[it] = rp
        if it >= int(burn_in):
            G_sum += G
            E_sum += E

    sl = slice(int(burn_in), int(n_iter))
    _, h2_se = batch_means(tr_h2[sl])
    _, rg_se = batch_means(tr_rg[sl].reshape(-1, P * P))
    _, re_se = batch_means(tr_re[sl].reshape(-1, P * P))
    _, rp_se = batch_means(tr_rp[sl].reshape(-1, P * P))
    # Average covariance states (a convex, hence PSD, operation) and derive every
    # reported correlation from those same two matrices. Ratio-averaging rg/re
    # separately would generally break rp == G + E after burn-in.
    n_post = int(n_iter) - int(burn_in)
    G_est = G_sum / n_post
    E_est = E_sum / n_post
    h2_est = np.diag(G_est).copy()
    rg_est = _cov_to_corr(G_est, h2_est, eps)
    re_est = _cov_to_corr(E_est, 1.0 - h2_est, eps)
    rp_est = G_est + E_est
    np.fill_diagonal(rp_est, 1.0)
    return GenCorrResult(
        h2=h2_est, rg=rg_est, re=re_est, rp=rp_est,
        genetic_cov=G_est, env_cov=E_est,
        se=dict(h2=h2_se, rg=rg_se.reshape(P, P), re=re_se.reshape(P, P),
                rp=rp_se.reshape(P, P)),
        phen_names=list(phen_names),
        traces=dict(h2=tr_h2[sl].copy(), rg=tr_rg[sl].copy(),
                    re=tr_re[sl].copy(), rp=tr_rp[sl].copy()),
        n_iter=int(n_iter), burn_in=int(burn_in))


@dataclass
class FactorResult:
    """Result of :func:`fit_genetic_factor` — a common-factor decomposition of the
    genetic correlation matrix, ``r_g ≈ Λ Λ' + Ψ`` (a Genomic-SEM-style common-factor
    model on the correlation scale).

    ``loadings`` is the ``(P, n_factors)`` matrix ``Λ`` of standardised factor
    loadings (each trait's correlation with a latent genetic factor). ``communality``
    is the per-trait proportion of *genetic* variance explained by the common
    factor(s) — the row sums of ``Λ²``, constrained to ``[0, 1]`` — and
    ``uniqueness = 1 − communality`` the trait-specific genetic residual. ``fitted`` is the
    model-implied correlation ``Λ Λ' + diag(Ψ)`` (diagonal 1) and ``residual`` the
    misfit ``r_g − fitted``, whose **off-diagonal** is what the fit targets.

    ``srmr`` is the standardised root-mean-square of those off-diagonal residuals.
    Values around 0.05–0.08 are sometimes used as informal descriptive heuristics;
    they are not a calibrated test that a given number of factors suffices.
    ``prop_explained`` is the fraction of the off-diagonal genetic-correlation
    structure the factor(s) capture. ``df = ½((P − m)² − (P + m))`` is the model
    nominal degrees of freedom. At ``df = 0`` (e.g. one factor on three traits), an
    admissible solution is just-identified, but incompatible correlation signs or a
    Heywood solution can put the optimum on the communality boundary and leave
    non-zero residual misfit. ``P ≥ 4`` makes a one-factor model over-identified,
    but this point-matrix MINRES fit still provides no calibrated factor-number test.

    For ``n_factors > 1`` the ``loadings`` are the unrotated (MINRES) orientation:
    ``communality``, ``fitted`` and ``srmr`` are rotation-invariant, but the
    individual loadings are only defined up to an orthogonal rotation. Align
    bootstrap loading matrices by sign, permutation and rotation before elementwise
    intervals, or use rotation-invariant summaries. ``input_correlation`` records
    whether the supplied matrix already had a unit
    diagonal (a correlation) or was standardised from a covariance."""
    loadings: np.ndarray
    uniqueness: np.ndarray
    communality: np.ndarray
    n_factors: int
    fitted: np.ndarray
    residual: np.ndarray
    srmr: float
    prop_explained: float
    df: int
    phen_names: list
    input_correlation: bool


def _minres_loadings(R, m, W, max_iter, tol):
    """MINRES common-factor loadings: minimise the (optionally weighted) sum of
    squared **off-diagonal** residuals of ``R − Λ Λ'`` over the ``(P, m)`` loading
    matrix ``Λ`` — the diagonal is excluded because the uniquenesses absorb it, so the
    factors explain the *correlations*, not each trait's own variance. Warm-started
    from the top-``m`` eigenpairs of ``R`` (principal factors) and polished by L-BFGS-B
    with the analytic gradient ``−2 (W∘Rres) Λ``. ``W`` is an optional ``(P, P)``
    inverse-variance weight matrix (its diagonal is ignored)."""
    from scipy.optimize import minimize
    P = R.shape[0]
    Wm = np.ones((P, P)) if W is None else np.array(W, dtype=float)
    Wm = 0.5 * (Wm + Wm.T)
    np.fill_diagonal(Wm, 0.0)                       # off-diagonal objective only
    w, V = np.linalg.eigh(R)                        # principal-factor warm start
    order = np.argsort(w)[::-1][:m]
    L0 = V[:, order] * np.sqrt(np.clip(w[order], 0.0, None))

    def obj_grad(vec):
        L = vec.reshape(P, m)
        Rres = R - L @ L.T
        WR = Wm * Rres
        f = 0.5 * float(np.sum(WR * Rres))
        grad = -2.0 * (WR @ L)
        return f, grad.ravel()

    res = minimize(obj_grad, L0.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=int(max_iter), gtol=float(tol), ftol=1e-14))
    L = res.x.reshape(P, m)
    # Unconstrained MINRES can obtain an exact off-diagonal fit only by assigning a
    # trait more than 100% common variance (a Heywood solution), or can run to
    # enormous loadings for an incompatible-sign three-trait matrix. In that case,
    # refit on the admissible set ||L_i||² <= 1. SLSQP handles the row-wise quadratic
    # constraints; ordinary admissible fits keep the faster L-BFGS-B solution above.
    if np.any(np.sum(L * L, axis=1) > 1.0 + max(float(tol), 1e-8)):
        def constraints(vec):
            rows = vec.reshape(P, m)
            return 1.0 - np.sum(rows * rows, axis=1)

        def constraints_jac(vec):
            rows = vec.reshape(P, m)
            jac = np.zeros((P, P * m))
            for i in range(P):
                jac[i, i * m:(i + 1) * m] = -2.0 * rows[i]
            return jac

        start = np.array(L, copy=True)
        norms = np.sqrt(np.sum(start * start, axis=1))
        outside = norms > 1.0
        start[outside] /= norms[outside, None]
        constrained = minimize(
            obj_grad, start.ravel(), jac=True, method="SLSQP",
            bounds=[(-1.0, 1.0)] * (P * m),
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraints_jac},
            options=dict(maxiter=int(max_iter), ftol=float(tol)))
        if not constrained.success:
            raise RuntimeError("constrained MINRES factor fit did not converge: "
                               f"{constrained.message}")
        L = constrained.x.reshape(P, m)

    # Remove tiny feasibility violations left by the numerical optimizer. This is
    # only a round-off projection; substantive inadmissibility was handled above.
    norms = np.sqrt(np.sum(L * L, axis=1))
    outside = norms > 1.0
    L[outside] /= norms[outside, None]
    for k in range(m):                             # deterministic sign per factor
        if L[np.argmax(np.abs(L[:, k])), k] < 0:
            L[:, k] *= -1.0
    return L


def fit_genetic_factor(genetic, n_factors=1, *, phen_names=None, weights=None,
                       max_iter=500, tol=1e-8):
    """Fit a genetic **common-factor** model ``r_g ≈ Λ Λ' + Ψ`` (Genomic-SEM-lite).

    Given the genetic correlations among ``P`` traits — typically from
    :func:`fit_genetic_correlation` — this asks whether a few latent genetic factors
    reproduce them: does *one* genetic factor explain the pairwise ``r_g`` (a general
    genetic axis shared across the traits), or are several needed? It is the
    pedigree-scale analogue of the Genomic-SEM common-factor model fit to an
    LD-score-regression genetic covariance (Grotzinger et al. 2019).

    ``genetic`` is either a :class:`GenCorrResult` (its ``rg`` matrix and
    ``phen_names`` are used) or a ``(P, P)`` genetic correlation / covariance array; a
    covariance is standardised to a correlation first, so ``loadings`` are always on
    the correlation scale. The fit is **MINRES** common-factor analysis: choose ``Λ``
    (``P × n_factors``) to minimise the sum of squared **off-diagonal** residuals of
    ``r_g − Λ Λ'``, letting the trait-specific uniquenesses ``Ψ`` soak up the
    diagonal — so the factor(s) explain the *cross-trait* genetic correlations rather
    than each trait's own heritable variance.

    A single factor needs ``P ≥ 3`` traits (three correlations pin one set of
    loadings), and ``n_factors`` must leave the model (over-)identified,
    ``df = ½((P − n_factors)² − (P + n_factors)) ≥ 0``. Returns a
    :class:`FactorResult` with the loadings, per-trait communalities (genetic variance
    explained by the factor[s]) and an ``srmr`` fit index. Comparisons by ``srmr`` /
    ``prop_explained`` are descriptive in-sample comparisons, not calibrated tests.

    ``weights`` optionally supplies a ``(P, P)`` inverse-variance weight matrix for a
    diagonally-weighted (DWLS) fit — e.g. ``1 / se²`` of each ``r_g`` — instead of the
    unweighted (ULS) default. Because the within-dataset ``se`` from
    :func:`fit_genetic_correlation` understates the true sampling variability, prefer
    weights (and uncertainty on the loadings) from bootstrapping the whole
    ``fit_genetic_correlation`` → ``fit_genetic_factor`` pipeline over families
    (:func:`bootstrap_fit`) to relying on that ``se``. This is a descriptive
    decomposition of a *point-estimate* correlation matrix; it carries no inference
    of its own. For multiple factors, align bootstrap signs, permutations and
    rotations before elementwise loading intervals, or use rotation-invariant
    summaries."""
    if isinstance(genetic, GenCorrResult):
        M = np.asarray(genetic.rg, dtype=float)
        if phen_names is None:
            phen_names = list(genetic.phen_names)
    else:
        M = np.asarray(genetic, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("genetic must be a square (P, P) matrix or a GenCorrResult")
    if not np.all(np.isfinite(M)):
        raise ValueError("genetic matrix must contain only finite values")
    if not np.allclose(M, M.T, rtol=1e-7, atol=1e-10):
        raise ValueError("genetic matrix must be symmetric")
    if np.any(np.diag(M) <= 0.0):
        raise ValueError("genetic matrix diagonal must be strictly positive")
    eig = np.linalg.eigvalsh(M)
    psd_tol = 1e-8 * max(1.0, float(np.max(np.abs(eig))))
    if eig[0] < -psd_tol:
        raise ValueError("genetic matrix must be positive-semidefinite")
    P = M.shape[0]
    m = int(n_factors)
    if m < 1:
        raise ValueError("n_factors must be >= 1")
    if P < 3:
        raise ValueError("need >= 3 traits for a common-factor model "
                         "(a single factor is unidentified for P < 3)")
    df = ((P - m) ** 2 - (P + m)) // 2
    if df < 0:
        raise ValueError(f"{m} factors are not identified from {P} traits "
                         f"(model df = {df} < 0); use fewer factors")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(P)]
    elif len(phen_names) != P:
        raise ValueError("phen_names length must match number of traits")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (P, P):
            raise ValueError("weights must be a (P, P) matrix")
        if (not np.all(np.isfinite(weights)) or
                not np.allclose(weights, weights.T, rtol=1e-7, atol=1e-10) or
                np.any(weights < 0.0)):
            raise ValueError("weights must be finite, symmetric, and non-negative")

    M = 0.5 * (M + M.T)                             # symmetrise, then standardise
    input_correlation = bool(np.allclose(np.diag(M), 1.0, atol=1e-6))
    d = np.sqrt(np.diag(M))
    R = M / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    if np.linalg.eigvalsh(R)[0] < -1e-8:
        raise ValueError("standardised genetic matrix must be positive-semidefinite")

    L = _minres_loadings(R, m, weights, max_iter, tol)
    communality = np.sum(L * L, axis=1)            # admissible row sums of Λ²
    uniqueness = 1.0 - communality
    fitted = L @ L.T + np.diag(uniqueness)
    residual = R - fitted
    iu = np.triu_indices(P, 1)
    off = residual[iu]
    srmr = float(np.sqrt(np.mean(off ** 2))) if off.size else 0.0
    ss_tot = float(np.sum(R[iu] ** 2))
    prop_explained = (float(1.0 - np.sum(off ** 2) / ss_tot)
                      if ss_tot > 1e-12 else 0.0)
    return FactorResult(
        loadings=L, uniqueness=uniqueness, communality=communality,
        n_factors=m, fitted=fitted, residual=residual, srmr=srmr,
        prop_explained=prop_explained, df=int(df),
        phen_names=list(phen_names), input_correlation=input_correlation)


@dataclass
class BootstrapResult:
    """Result of :func:`bootstrap_fit`.

    ``estimate`` is the point estimate from the full data; ``se`` the bootstrap
    standard error (SD of the resampled estimates); ``ci_low`` / ``ci_high`` the
    percentile confidence interval at ``ci_level``; ``samples`` the ``(n_boot, …)``
    array of per-resample estimates. Shapes follow whatever the estimator returns
    (scalar → 0-d arrays; vector/matrix → that shape)."""
    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_level: float
    n_boot: int
    samples: np.ndarray


def bootstrap_fit(families, estimator, *, n_boot=100, seed=None, ci_level=0.95):
    """Approximate percentile uncertainty by **resampling family clusters**.

    The HE ``se`` reported by :func:`fit_heritability`,
    :func:`fit_variance_components` and :func:`fit_genetic_correlation` is a
    *within-dataset* Monte-Carlo diagnostic and can substantially understate
    sampling variability. This helper resamples families with replacement
    ``n_boot`` times and reports the refit SD and percentile interval. Its sampling
    interpretation assumes independent, non-overlapping, representative family
    clusters and a compatible ascertainment/model; it is not bias-corrected,
    studentized, or automatically calibrated at boundaries.

    ``estimator`` is a callable ``families -> value`` returning the quantity of
    interest as a float or array, e.g.::

        bootstrap_fit(fams, lambda f: fit_heritability(f, seed=1).h2)
        bootstrap_fit(fams, lambda f: fit_variance_components(f, ("A", "C"),
                                          seed=1).components["C"])
        bootstrap_fit(fams, lambda f: fit_genetic_correlation(f, seed=1).rg[0, 1])

    Fix the estimator's internal ``seed`` so each refit is deterministic given its
    resample — then the bootstrap spread reflects family sampling, not the sampler's
    own Monte-Carlo noise. Returns a :class:`BootstrapResult`. Cost is ``n_boot + 1``
    fits, so this is deliberately expensive; lower ``n_boot`` for a quick check.

    ``estimator`` must return a stable scalar/array shape. ``n_boot`` is an integer
    at least 2; substantially more than the default 100 may be needed for stable
    interval endpoints. ``seed`` follows NumPy ``default_rng`` semantics and seeds
    only the resampling; ``ci_level`` sets the percentile interval."""
    families = list(families)
    n = len(families)
    if n < 2:
        raise ValueError("need at least 2 families to bootstrap")
    if isinstance(n_boot, (bool, np.bool_)):
        raise TypeError("n_boot must be an integer >= 2, not bool")
    try:
        n_boot = operator.index(n_boot)
    except TypeError:
        raise TypeError("n_boot must be an integer >= 2") from None
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    point = np.asarray(estimator(families), dtype=float)
    samples = np.empty((n_boot,) + point.shape, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = np.asarray(estimator([families[i] for i in idx]), dtype=float)
        if sample.shape != point.shape:
            raise ValueError(
                f"estimator returned shape {sample.shape} for bootstrap resample "
                f"{b}; expected stable shape {point.shape}")
        samples[b] = sample
    alpha = (1.0 - ci_level) / 2.0
    return BootstrapResult(
        estimate=point,
        se=samples.std(axis=0, ddof=1),
        ci_low=np.quantile(samples, alpha, axis=0),
        ci_high=np.quantile(samples, 1.0 - alpha, axis=0),
        ci_level=float(ci_level), n_boot=n_boot, samples=samples)


# --------------------------------------------------------------------------- #
#  Significance testing (parametric bootstrap = nested-model comparison)       #
# --------------------------------------------------------------------------- #
@dataclass
class SignificanceTest:
    """Result of a parametric-bootstrap significance test.

    ``estimate`` is the observed statistic (a component's variance proportion, or a
    genetic correlation); ``p_value`` is the Monte-Carlo p-value from refitting the
    full model on ``n_boot`` datasets simulated under the null; ``null`` is that
    conditional plug-in null distribution of the statistic. ``label`` names the
    hypothesis. Validity depends on the fitted null model, independent family
    clusters, and thresholds determined independently of the observed outcome; the
    result is not a general guarantee of finite-sample calibration."""
    estimate: float
    p_value: float
    null: np.ndarray
    n_boot: int
    label: str


def _recover_thresholds(lo, hi):
    """Per-coordinate threshold from **case/control-style** bounds: cases ``(t, inf)``,
    controls ``(-inf, t)`` (case iff liability > t, so ``t`` is the finite endpoint);
    fully-unbounded ``(-inf, inf)`` coordinates are uninformative. Raises for pinned
    or two-sided-finite (age-of-onset) bounds, which the parametric bootstrap would
    have to re-encode. ``lo``/``hi`` broadcast to any shape; returns ``(T, informative)``."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    both_inf = np.isneginf(lo) & np.isposinf(hi)
    control = np.isneginf(lo) & np.isfinite(hi)
    case = np.isfinite(lo) & np.isposinf(hi)
    if not np.all(both_inf | control | case):
        raise NotImplementedError(
            "significance test supports case/control-style bounds only — cases "
            "(t, inf), controls (-inf, t); pinned or interval age-of-onset bounds "
            "are not supported.")
    T = np.where(control, hi, np.where(case, lo, np.nan))
    return T, ~both_inf


def _assert_case_control_bounds(families, thresholds_are_status_independent=False):
    """Validate bounds that the parametric bootstrap can coherently re-simulate.

    A common threshold for each trait is unambiguously reusable after status is
    simulated. Individualised thresholds are accepted only after the caller
    explicitly confirms that they came from baseline variables independent of the
    observed outcome (for example sex or birth cohort), never age at onset."""
    if not isinstance(thresholds_are_status_independent, (bool, np.bool_)):
        raise TypeError("thresholds_are_status_independent must be bool")
    if not families:
        return
    _require_member_rows(families, context="significance test")
    P = int(np.size(families[0].members[0].lower))
    thresholds = [[] for _ in range(P)]
    for fam in families:
        for m in fam.members:
            lo = np.broadcast_to(np.asarray(m.lower, dtype=float), (P,))
            hi = np.broadcast_to(np.asarray(m.upper, dtype=float), (P,))
            T, informative = _recover_thresholds(lo, hi)
            for p in range(P):
                if informative[p]:
                    thresholds[p].append(float(T[p]))
    varying = any(values and not np.allclose(values, values[0], rtol=1e-10,
                                              atol=1e-12)
                  for values in thresholds)
    if varying and not thresholds_are_status_independent:
        raise ValueError(
            "individualised thresholds cannot be reused under the null unless "
            "they were fixed independently of observed status; pass "
            "thresholds_are_status_independent=True only for thresholds derived "
            "from baseline covariates, never age at onset")


def _threshold_status(members, L_kP, k, P, fam_id):
    """Rebuild a family from simulated liabilities ``L_kP`` (phenotype-major length
    ``k*P``, coord ``p*k + a``) by re-thresholding each observed coordinate."""
    out = []
    for a, m in enumerate(members):
        T, inform = _recover_thresholds(np.broadcast_to(np.asarray(m.lower, float), (P,)),
                                        np.broadcast_to(np.asarray(m.upper, float), (P,)))
        los, his = [], []
        for p in range(P):
            if not inform[p]:
                los.append(-np.inf); his.append(np.inf)
            else:
                is_case = L_kP[p * k + a] > T[p]
                los.append(T[p] if is_case else -np.inf)
                his.append(np.inf if is_case else T[p])
        if P == 1:
            out.append(Member(m.role, los[0], his[0]))
        else:
            out.append(Member(m.role, los, his))
    return Family(fam_id, out)


def _simulate_null(families, h2_vec, G, rp, rng):
    """One null dataset on the same pedigrees + thresholds. ``G`` is the genetic
    covariance under the requested null and ``rp = G + E`` its phenotypic
    correlation."""
    P = len(h2_vec)
    out = [None] * len(families)
    for _key, idx in _group_by_structure(families):
        roles = sorted(mm.role for mm in families[idx[0]].members)
        k = len(roles)
        A = correct_positive_definite(_component_matrix(roles, "A"))[0]
        sig = correct_positive_definite(_multi_cov(A, h2_vec, G, rp))[0]
        L = rng.multivariate_normal(np.zeros(k * P), sig, size=len(idx))
        for slot, f in enumerate(idx):
            members = sorted(families[f].members, key=lambda member: member.role)
            out[f] = _threshold_status(members, L[slot], k, P, families[f].fam_id)
    return out


def test_variance_component(families, component="C", *, n_boot=200, seed=None,
                            thresholds_are_status_independent=False, **fit_kwargs):
    """Test whether a variance component is needed (its proportion > 0).

    The frequentist analog of the SEM likelihood-ratio test "is `C` in the model?"
    — a **parametric bootstrap**. It fits the full ``A + component`` model (observed
    statistic), fits the null ``A``-only model, then simulates ``n_boot`` datasets
    under that null *on the same pedigrees and thresholds*, refits the full model on
    each, and returns the one-sided p-value ``P(estimate >= observed | H0)``.

    ``component`` is the non-additive component to test — ``"C"`` (sibship common
    environment) or ``"M"`` (couple/spousal environment); the full model is
    ``A + component`` and the null is ``A``-only (``"A"`` itself and the unsupported
    ``"D"`` are rejected). Extra keyword args pass through to
    :func:`fit_variance_components` (use a smaller ``n_iter`` to keep the ``n_boot``
    refits affordable). Bounds must be case/control-style. A threshold that varies
    across people is rejected unless ``thresholds_are_status_independent=True``;
    make that assertion only for externally determined baseline thresholds, never
    thresholds derived from age at onset. This is a conditional plug-in bootstrap,
    not a universal calibration guarantee. Returns a :class:`SignificanceTest`."""
    if component == "A" or component not in _COMPONENT_OFFDIAG:
        avail = ", ".join(c for c in _COMPONENT_OFFDIAG if c != "A")
        raise ValueError(f"component must be a non-additive component ({avail})")
    _assert_case_control_bounds(families, thresholds_are_status_independent)
    comps = ("A", component)
    obs = fit_variance_components(families, comps, seed=seed, **fit_kwargs).components[component]
    h2A = float(fit_variance_components(families, ("A",), seed=seed, **fit_kwargs).components["A"])
    h2_vec = np.array([h2A])
    G = np.array([[h2A]])
    rp = np.array([[1.0]])
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        sim = _simulate_null(families, h2_vec, G, rp, rng)
        s = _offset_seed(seed, b + 1)
        null[b] = fit_variance_components(sim, comps, seed=s, **fit_kwargs).components[component]
    p = (1 + int(np.sum(null >= obs))) / (1 + int(n_boot))
    return SignificanceTest(estimate=float(obs), p_value=float(p), null=null,
                            n_boot=int(n_boot), label=f"{component} proportion > 0")


def test_genetic_correlation(families, i=0, j=1, *, n_boot=200, seed=None,
                             thresholds_are_status_independent=False, **fit_kwargs):
    """Test whether the genetic correlation between two traits is non-zero.

    Parametric-bootstrap analog of the SEM test "is the cross-trait genetic path
    zero?". Fits the full model (observed ``rg[i,j]``), then simulates ``n_boot``
    datasets under a **pair-specific null** — ``rg[i,j] = 0`` while preserving each
    trait's ``h2``, the environmental covariance, and all nuisance genetic
    correlations compatible with a coherent PSD null — on the same pedigrees and
    thresholds, refits, and returns the two-sided p-value
    ``P(|rg| >= |observed| | H0)``. Extra keyword args pass to
    :func:`fit_genetic_correlation`. Bounds must be case/control-style. A threshold
    that varies across people is rejected unless
    ``thresholds_are_status_independent=True``; make that assertion only for
    externally determined baseline thresholds, never thresholds derived from age at
    onset. This is a conditional plug-in bootstrap, not a universal calibration
    guarantee. Returns a :class:`SignificanceTest`."""
    def trait_index(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer trait index, not bool")
        try:
            return operator.index(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer trait index") from None

    i = trait_index(i, "i")
    j = trait_index(j, "j")
    if not families:
        raise ValueError("no families provided")
    _require_member_rows(families, context="genetic-correlation test")
    P = int(np.size(families[0].members[0].lower))
    if not (0 <= i < P and 0 <= j < P) or i == j:
        raise ValueError(f"i and j must be distinct trait indices in [0, {P})")
    _assert_case_control_bounds(families, thresholds_are_status_independent)
    full = fit_genetic_correlation(families, seed=seed, **fit_kwargs)
    obs = float(full.rg[i, j])
    h2_vec = np.asarray(full.h2, dtype=float)
    # Zero only the tested path. If that edit makes G indefinite, project to the
    # nearest correlation matrix under the fixed-zero constraint; nuisance paths
    # move only as much as coherence requires. Keep environmental covariance exact.
    G0, _ = _project_covariance(full.genetic_cov, h2_vec,
                                zero_pairs=((i, j),), eps=1e-8)
    rp_null = G0 + np.asarray(full.env_cov, dtype=float)
    np.fill_diagonal(rp_null, 1.0)
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        sim = _simulate_null(families, h2_vec, G0, rp_null, rng)
        s = _offset_seed(seed, b + 1)
        null[b] = fit_genetic_correlation(sim, seed=s, **fit_kwargs).rg[i, j]
    p = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (1 + int(n_boot))
    return SignificanceTest(estimate=obs, p_value=float(p), null=null,
                            n_boot=int(n_boot), label=f"r_g[{i},{j}] != 0")
