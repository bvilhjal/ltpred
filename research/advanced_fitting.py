"""Inferential fitting machinery, split out of the lean ``ltpred.fit`` core.

**Unsupported research code** — benchmarked explorations, not part of the
supported package API. Importable as ``research.advanced_fitting`` from a
repository checkout (the repo root must be on ``sys.path``); the API may change
without notice, and the estimators' own caveats (in the docstrings below) apply
in full. Nothing here feeds back into ``ltpred``.

Contents:

- :func:`fit_variance_components_mcem` — the finite-iteration, fixed-damping
  Monte-Carlo EM-style likelihood fit of the variance components (the former
  ``fit_variance_components(method="mcem")`` route), with its approximate
  OPG/BHHH information SE and GHK Monte-Carlo log-likelihood/AIC
  (:class:`MCEMVarCompResult`);
- :func:`fit_genetic_correlation` — cross-trait Haseman-Elston genetic
  correlation (:class:`GenCorrResult`);
- :func:`fit_genetic_correlation_decay` — onset-age-structured genetic
  correlation by Monte-Carlo EM (:class:`DecayGenCorrResult`);
- :func:`fit_genetic_factor` — MINRES common-factor decomposition of a
  genetic correlation matrix (:class:`FactorResult`);
- :func:`test_variance_component` / :func:`test_genetic_correlation` —
  conditional parametric-bootstrap significance tests
  (:class:`SignificanceTest`);
- :func:`fit_nurture` — closed-form direct/indirect (genetic-nurture)
  moment fit (:class:`NurtureFit`).

The module reuses the lean core's private helpers (``ltpred.fit._prepare_group_vc``,
``ltpred.fit._component_matrix``, ``ltpred.gibbs._init_chain`` and the
``ltpred.estimate`` / ``ltpred.gibbs`` internals) — importing ``ltpred`` private
names from research code is deliberate and accepted.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

from ltpred._mathfun import norm_cdf, norm_ppf
from ltpred._validation import validate_bounds
from ltpred.covariance import get_relatedness, correct_positive_definite
from ltpred.estimate import (_group_by_structure, _validate_multitrait_bounds,
                             _check_unique_roles, _assert_nonempty_families,
                             batch_means)
from ltpred.family import Family, Member
from ltpred.fit import (fit_variance_components, _COMPONENT_OFFDIAG,
                        _component_matrix, _prepare_group_vc,
                        _validate_population_sampling, _assert_common_thresholds,
                        _assert_population_case_rate, _assert_nonoverlapping_pids,
                        _validate_iteration_controls, _validate_update_controls,
                        _assert_observed_identification)
from ltpred.gibbs import (gibbs_params, gibbs_advance, gibbs_advance_moment,
                          _init_chain, _FIXED_TOL, _offset_seed, _seed_rng)

__all__ = ["MCEMVarCompResult", "fit_variance_components_mcem",
           "GenCorrResult", "fit_genetic_correlation",
           "DecayGenCorrResult", "fit_genetic_correlation_decay",
           "FactorResult", "fit_genetic_factor",
           "SignificanceTest", "test_variance_component",
           "test_genetic_correlation", "NurtureFit", "fit_nurture"]


@dataclass
class MCEMVarCompResult:
    """Result of :func:`fit_variance_components_mcem`.

    ``components`` maps each fitted component (``"A"`` additive, ``"C"`` sibship
    common environment, ``"M"`` couple/spousal environment) to its estimated
    **proportion** of the liability variance; ``residual`` is the remaining ``e2``.
    So ``A`` is the (narrow-sense) heritability. ``se`` is an approximate
    OPG/BHHH information SE with Monte-Carlo, finite-iteration, iid-family and
    model-correctness assumptions — not a guarantee; use an appropriately
    designed family-cluster bootstrap for sampling uncertainty. ``traces`` are
    the post-burn-in proportion traces per component.

    ``loglik`` / ``aic`` are the Monte-Carlo (GHK) observed-data log-likelihood
    and ``AIC = 2·(#components) − 2·loglik`` (both Monte-Carlo estimates), for
    comparing nested models (e.g. ``A`` vs ``A+C``). They are ``None`` for
    pinned/degenerate bounds. As always for variance components, AIC can be
    unreliable near a parameter boundary, so use it only as a descriptive
    comparison. The conditional parametric-bootstrap :func:`test_variance_component`
    is the component test when its model and sampling assumptions hold."""
    components: dict
    residual: float
    se: dict
    traces: dict
    n_iter: int
    burn_in: int
    loglik: float = None
    aic: float = None


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


def fit_variance_components_mcem(families, components=("A", "C"), *,
                                 n_iter=1500, burn_in=500, inner_sweeps=5,
                                 damp=0.2, seed=None, eps=1e-4, sampling=None):
    """Approximate fixed-damping Monte-Carlo EM-style variance components.

    **Sampling contract:** like the core moment fitters, this supports
    independent, non-overlapping, unascertained population-sampled families only.
    Pass ``sampling="population"`` to acknowledge that contract; omitting it
    warns, and any other value raises.

    The likelihood counterpart of the Haseman-Elston
    :func:`ltpred.fit.fit_variance_components` moment fit (this is the route the
    old ``fit_variance_components(method="mcem")`` selected; ``method="reml"`` /
    ``"ml"`` were aliases of it). ``components`` is the same
    variance-component bank (``"A"``, ``"C"``, ``"M"``).

    E-step: draw the liabilities from the truncated family MVN under the current
    ``Sigma(h2)`` (the same augmentation as the HE fit). M-step: set ``h2`` to the
    Gaussian likelihood optimum of those liabilities (:func:`_mstep_reml`) rather
    than the moment regression. The implementation uses fixed damping and reports
    the post-burn-in average; it does not apply an increasing Monte-Carlo E-step or
    stop on an observed-likelihood convergence criterion. This is not restricted ML.
    The SE is an **approximate** observed-information SE (:func:`_reml_observed_se`,
    an OPG/BHHH outer-product estimate that is itself subject to Monte-Carlo error),
    accounting approximately for the information lost to thresholding. The SE,
    GHK likelihood and AIC retain Monte-Carlo and finite-iteration error and require
    validation for the target design. Use :func:`ltpred.fit.bootstrap_fit` for an
    iid-family cluster percentile interval when its sampling assumptions hold.

    Returns a :class:`MCEMVarCompResult`. ``seed`` must be a non-boolean integer
    in ``[0, 2**32 - 1]`` or ``None``.

    **Requires a common case/control threshold per trait**: sharing
    :func:`fit_variance_components`' pooled-moment augmentation, it inherits the
    same personalised-threshold bias and **rejects** personalised/onset-pinned
    LT-FH++ bounds rather than fitting them to the boundary. Standard NaN and
    interval-order validation still applies."""
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps)
    damp, eps = _validate_update_controls(damp, eps)
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    _assert_nonoverlapping_pids(families, "fit_variance_components_mcem")
    comps = list(components)
    if not comps:
        raise ValueError("components must contain at least one component")
    for c in comps:
        if c not in _COMPONENT_OFFDIAG:
            avail = ", ".join(_COMPONENT_OFFDIAG)
            raise ValueError(f"unknown component {c!r}; choose from {avail} "
                             "(dominance 'D' is not supported)")
    if len(set(comps)) != len(comps):
        raise ValueError(f"duplicate components in {components!r}")
    if int(burn_in) < 0 or int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be non-negative and "
                         f"< n_iter ({n_iter})")
    _validate_population_sampling(sampling, "fit_variance_components_mcem")
    # Shares the pooled-moment augmentation, so it has fit_variance_components'
    # personalised-threshold failure mode; reject those bounds the same way.
    _assert_common_thresholds(families, 1, context="fit_variance_components_mcem")
    _assert_population_case_rate(families, 1,
                                 context="fit_variance_components_mcem")
    C = len(comps)
    groups = [_prepare_group_vc(families, idx, comps)
              for _key, idx in _group_by_structure(families)]
    _assert_observed_identification(groups, comps, context="MCEM fit")
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
    return MCEMVarCompResult(
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
    fixed = np.ascontiguousarray((hi_pm - lo_pm) < _FIXED_TOL)
    ones = np.ones(k * n_pheno)      # unit marginal SD: pre-standardised bounds
    x = np.empty((F, k * n_pheno))
    for slot in range(F):
        x[slot] = _init_chain(lo_pm[slot], hi_pm[slot], ones)
    return dict(roles=roles, k=k, F=F, A=A, W=W, sA2=sA2,
                lowers=lo_pm, uppers=hi_pm, fixed=fixed,
                x=np.ascontiguousarray(x), aod=aod)


def _decay_cov(A, h2, G, E, aod_f, lam_w, lam_x, kernel, C=None):
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
            # The zero gradient makes this penalty region a stationary point for
            # L-BFGS-B. Safe by design, not by accident: every M-step starts
            # from the previous PSD-projected iterate (``_run_em`` re-projects
            # after each L-BFGS call), so the optimizer never *begins* inside
            # the region — the rising 1e9 penalty only pushes its line searches
            # back toward feasibility.
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
                                  phen_names=None, sampling=None):
    """Genetic correlation with an onset-age decay (structured ``r_g``).

    **Sampling contract:** like the core moment fitters, this supports
    independent, non-overlapping, unascertained population-sampled families only.
    Pass ``sampling="population"`` to acknowledge that contract; omitting it
    warns, and any other value raises.

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
    must be at least 1; it re-runs the EM from that many perturbed
    initialisations and keeps the best-fitting (lowest final objective) -- a
    guard against the multi-modal likelihood at small samples (each start costs
    a full EM run).

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
    if int(n_starts) < 1:
        raise ValueError(f"n_starts ({n_starts}) must be >= 1")
    if int(burn) < 0:
        raise ValueError(f"burn ({burn}) must be non-negative")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(P)]
    elif len(phen_names) != P:
        raise ValueError("phen_names length must match number of traits")
    _validate_population_sampling(sampling, "fit_genetic_correlation_decay")

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

    def compute_M(h2, lam_w, G, lam_x, E, C):
        M_groups = []
        for g in groups:
            k, F = g["k"], g["F"]
            kP = k * P
            M = np.empty((F, kP, kP))
            for slot in range(F):
                S = _decay_cov(g["A"], h2, G, E, g["aod"][slot],
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
                Ci[p, q] = Ci[q, p] = 0.0 if rng is None else rng.uniform(-0.1, 0.1)
        if shared_env and rng is not None:
            # per trait, not per pair -- inside the pair loop the last trait's
            # c2 never gets a random start (and for P = 1 there are no pairs)
            np.fill_diagonal(Ci, rng.uniform(0.0, 0.2, P))
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
            M_groups = compute_M(h2, lam_w, G, lam_x, E, C)
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
                c2 = np.clip(np.diag(C), 0.0, 1.0 - eps - h2)
                # Project with the capped variances. Projecting first and then
                # shrinking only the diagonal can make C indefinite again.
                C, _ = _project_covariance(C, c2, eps=eps)
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
                              # reshape to (P, P) like fit_genetic_correlation:
                              # GenCorrResult.se["rg"] is a matrix, and callers
                              # index it as se["rg"][i, j].
                              se={"h2": h2_se,
                                  "rg": rg_se.reshape(P, P),
                                  "re": re_se.reshape(P, P),
                                  "rp": rp_se.reshape(P, P)},
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
    if np.any(variances < 0.0):
        raise ValueError("variances must be non-negative")
    sd = np.sqrt(variances)
    active = variances > 0.0
    safe_sd = np.where(active, sd, 1.0)
    raw_corr = np.asarray(matrix, dtype=float) / np.outer(safe_sd, safe_sd)
    raw_corr[~active, :] = 0.0
    raw_corr[:, ~active] = 0.0
    np.fill_diagonal(raw_corr, 1.0)
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
    fixed = np.ascontiguousarray((hi_pm - lo_pm) < _FIXED_TOL)
    ones = np.ones(k * n_pheno)      # unit marginal SD: pre-standardised bounds
    x = np.empty((F, k * n_pheno))
    for slot in range(F):
        x[slot] = _init_chain(lo_pm[slot], hi_pm[slot], ones)
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
                            damp=0.2, seed=None, eps=1e-4, phen_names=None,
                            sampling=None):
    """Estimate the **genetic correlation** between traits from family data.

    **Sampling contract:** like the core moment fitters, this supports
    independent, non-overlapping, unascertained population-sampled families only.
    Pass ``sampling="population"`` to acknowledge that contract; omitting it
    warns, and any other value raises.

    The multi-trait generalisation of :func:`ltpred.fit.fit_heritability`: a
    **cross-trait** Haseman–Elston regression. Each member must carry one
    case/control interval per
    trait (``lower``/``upper`` are length-``P`` sequences, as for the multi-trait
    ``ltpred.estimate._estimate_liability_multi``). Each sweep draws the
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
    ``genetic_cov``/``env_cov`` matrices). As with :func:`ltpred.fit.fit_heritability`,
    the reported ``se`` is a within-dataset Monte-Carlo error. Use a family-cluster
    bootstrap for sampling uncertainty when families are independent and
    representative. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``.

    **Requires a common case/control threshold per trait**: the cross-trait
    Haseman-Elston regression shares the pooled-moment augmentation and inherits
    its personalised-threshold bias, so personalised/onset-pinned LT-FH++ bounds
    are **rejected** (use :func:`fit_genetic_correlation_decay` for onset-age
    structure). Standard shape, NaN, and interval-order validation still
    applies."""
    if not families:
        raise ValueError("no families provided")
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps)
    damp, eps = _validate_update_controls(damp, eps)
    _check_unique_roles(families)
    _assert_nonoverlapping_pids(families, "fit_genetic_correlation")
    _require_member_rows(families, context="genetic-correlation fit")
    first_lower = np.asarray(families[0].members[0].lower)
    P = int(first_lower.size)
    if P < 2:
        raise ValueError("fit_genetic_correlation needs >= 2 traits — each member's "
                         "lower/upper must be length-n_pheno (see "
                         "estimate._estimate_liability_multi)")
    _validate_multitrait_bounds(families, P)
    if int(burn_in) < 0 or int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be non-negative and "
                         f"< n_iter ({n_iter})")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(P)]
    elif len(phen_names) != P:
        raise ValueError("phen_names length must match number of traits")
    _validate_population_sampling(sampling, "fit_genetic_correlation")
    # Cross-trait HE shares the pooled-moment augmentation; personalised
    # per-person thresholds bias it the same way, so reject them (the onset-age
    # decay model, fit_genetic_correlation_decay, is the route for age structure).
    _assert_common_thresholds(families, P, context="fit_genetic_correlation")
    _assert_population_case_rate(families, P, context="fit_genetic_correlation")
    groups = [_prepare_group_multi(families, idx, P)
              for _key, idx in _group_by_structure(families)]
    sA2 = sum(g["sA2"] for g in groups)
    if sA2 <= 0:
        raise ValueError("no related pairs in the families — cannot fit genetic "
                         "correlation (need relatives, not lone probands).")
    _assert_observed_identification(groups, ("A",), P,
                                    context="genetic-correlation fit")
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
    unweighted (ULS) default. Only the off-diagonal is used (and validated): the
    ``r_g`` diagonal is the constant 1, so its ``se`` is exactly 0 and ``1 / se²``
    is ``+inf`` there, which is accepted and ignored. Because the within-dataset ``se`` from
    :func:`fit_genetic_correlation` understates the true sampling variability, prefer
    weights (and uncertainty on the loadings) from bootstrapping the whole
    ``fit_genetic_correlation`` → ``fit_genetic_factor`` pipeline over families
    (:func:`ltpred.fit.bootstrap_fit`) to relying on that ``se``. This is a
    descriptive decomposition of a *point-estimate* correlation matrix; it carries no
    inference of its own. For multiple factors, align bootstrap signs, permutations and
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
        # Check the off-diagonal only: `_minres_loadings` zeroes the diagonal
        # (the objective is off-diagonal), and the natural DWLS recipe
        # `1 / se**2` is +inf on the diagonal because `se["rg"]` is exactly 0
        # there -- the r_g diagonal is the constant 1 across the whole trace.
        # Rejecting on entries the fit discards made the documented recipe
        # unusable as written.
        off = ~np.eye(P, dtype=bool)
        if (not np.all(np.isfinite(weights[off])) or
                not np.allclose(weights, weights.T, rtol=1e-7, atol=1e-10,
                                equal_nan=True) or
                np.any(weights[off] < 0.0)):
            raise ValueError("weights must be finite, symmetric, and "
                             "non-negative off the diagonal")

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
    :func:`ltpred.fit.fit_variance_components` (use a smaller ``n_iter`` to keep
    the ``n_boot``
    refits affordable). Pass ``sampling="population"`` only for independent,
    non-overlapping, unascertained population-sampled families; this bootstrap
    does not repair selection bias. Bounds must be case/control-style. A threshold that varies
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


@dataclass
class NurtureFit:
    """Result of :func:`fit_nurture`.

    ``h2`` is the **direct** additive heritability and ``nurture`` the indirect
    (genetic-nurture) coefficient of
    ``research.covariance_extensions.construct_covmat_nurture``.

    ``h2_additive_po`` and ``h2_additive_sib`` are what a *nurture-blind*
    additive model would report from each relative type on its own, i.e.
    ``2 * cov``. Their **disagreement is the diagnostic**: under a purely
    additive model both estimate the same ``h2``, so a gap means the two
    relative types cannot be reconciled without an indirect path.
    ``disagreement`` is ``h2_additive_sib - h2_additive_po``, which is zero
    exactly when ``nurture`` is zero.
    """
    h2: float
    nurture: float
    h2_additive_po: float
    h2_additive_sib: float
    disagreement: float


def fit_nurture(cov_parent_offspring, cov_sib_sib):
    """Closed-form method-of-moments fit of the direct/indirect model.

    The nurture covariance of
    ``research.covariance_extensions.construct_covmat_nurture`` implies

    .. code-block:: text

        cov_parent_offspring = h2*(1 + 2n)/2
        cov_sib_sib          = h2*(1 + 2n)^2/2

    Two equations in two unknowns, so the ratio isolates the indirect path and
    the estimates are exact rather than iterative::

        1 + 2n = cov_sib_sib / cov_parent_offspring
        n      = (cov_sib_sib / cov_parent_offspring - 1) / 2
        h2     = 2 * cov_parent_offspring^2 / cov_sib_sib

    Both inputs are **liability-scale** covariances between the two relative
    types. From binary case/control data, obtain them with
    :mod:`ltpred.tetrachoric` rather than from observed-scale correlations.

    This is what makes ``nurture`` a fitted quantity rather than a supplied one.
    It is a moment estimator: no standard errors, and it inherits whatever bias
    the input covariances carry. Wrap it in
    :func:`ltpred.fit.bootstrap_fit`-style family-cluster resampling of those
    covariances if sampling uncertainty is needed.

    A sibling covariance *below* the parent-offspring one implies a negative
    ``nurture`` (a contrast effect); that is returned rather than clipped, since
    silently flooring it at zero would hide a real signal. A strong enough
    contrast (``nurture < -0.5``) even drives the parent-offspring covariance
    itself negative -- ``construct_covmat_nurture`` still emits a valid
    standardised (PSD) model there, and this fitter is its exact inverse, so a
    negative ``po`` is accepted too. Only the genuinely degenerate ``nurture =
    -0.5`` corner (``po = 0`` and ``ss = 0``, where the ratio is undefined) and
    covariances no standardised model can reproduce (implied ``h2`` outside
    ``(0, 1]`` or a negative residual) are rejected -- see the raised errors.
    """
    po = float(cov_parent_offspring)
    ss = float(cov_sib_sib)
    # ss = h2*(1+2n)^2/2 >= 0 for any real model; ss = 0 (and then po = 0) is the
    # degenerate nurture = -0.5 corner where 1 + 2n vanishes and the ratio is
    # undefined. po = 0 alone is the same corner. Everything else -- including a
    # negative po from a strong contrast (nurture < -0.5) -- is invertible, and
    # the h2/residual checks below reject inputs no standardised model produces.
    if abs(po) < 1e-15:
        raise ValueError(
            "cov_parent_offspring is (numerically) zero: the degenerate "
            "nurture = -0.5 corner, where the direct/indirect model is not "
            f"identified (got {po})")
    if abs(ss) < 1e-15:
        raise ValueError(
            "cov_sib_sib is (numerically) zero: the degenerate nurture = -0.5 "
            f"corner leaves h2 = 2*po^2/ss undefined (got {ss})")

    nurture = (ss / po - 1.0) / 2.0
    h2 = 2.0 * po * po / ss
    if not (0.0 < h2 <= 1.0):
        raise ValueError(
            f"the implied direct h2 is {h2:.4f}, outside (0, 1]. These two "
            "covariances are not jointly reproducible by the direct/indirect "
            "model; check that both are liability-scale and estimated on the "
            "same population")
    resid = 1.0 - h2 - 2.0 * nurture * nurture * h2 - 2.0 * nurture * h2
    if resid < -1e-12:
        raise ValueError(
            f"the implied (h2={h2:.4f}, nurture={nurture:.4f}) leave a negative "
            f"residual variance ({resid:.4f}), so no standardised liability "
            "model reproduces these covariances")

    h2_po = 2.0 * po                      # what an additive model reads off each
    h2_ss = 2.0 * ss                      # relative type, taken alone
    return NurtureFit(h2=h2, nurture=nurture, h2_additive_po=h2_po,
                      h2_additive_sib=h2_ss, disagreement=h2_ss - h2_po)
