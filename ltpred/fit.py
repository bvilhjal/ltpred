"""Fit liability-scale variance components from family data (data-augmentation fixed point).

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
``h2 <- (1 - damp) h2 + damp h2_hat`` for cross-sweep stability. Conditional
cross-products reconstruct population moments only for independent,
unascertained population-sampled families. Case/control enrichment, selection
on family history, and overlapping pedigrees change those moments; this module
does not model that sampling process.

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

This module is the lean fitting core that feeds back into prediction. The
inferential machinery built on top of it — the Monte-Carlo EM likelihood
variance-component fit, the multi-trait genetic-correlation and onset-age-decay
fits, the common-factor model, the parametric-bootstrap significance tests, and
the genetic-nurture moment fit — lives in ``research/advanced_fitting.py`` as
unsupported research code. :func:`bootstrap_fit` here is the family-cluster
resampling helper shared by those fits and these.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass

import numpy as np

from ._validation import validate_bounds
from .covariance import get_relatedness, _is_full_sib, _is_mates
from .gibbs import (gibbs_params, gibbs_advance,
                    _init_chain, _FIXED_TOL, _seed_rng)
from .estimate import _group_by_structure, batch_means

__all__ = ["FitResult", "fit_heritability", "VarCompResult",
           "fit_variance_components", "BootstrapResult", "bootstrap_fit"]


def _validate_population_sampling(sampling, context):
    """Make the supported sampling contract explicit while preserving old calls."""
    if sampling is None:
        warnings.warn(
            f"{context} assumes independent, non-overlapping, unascertained "
            "population-sampled families. Case/control or family-history "
            "ascertainment can drive moment estimates to the boundary; it is not "
            "corrected by this fitter or by bootstrap_fit. Pass "
            "sampling='population' only after verifying that contract.",
            RuntimeWarning, stacklevel=3)
        return
    if sampling != "population":
        raise ValueError(
            f"{context} supports only sampling='population'; ascertained or "
            "overlapping-family designs require an estimator that models their "
            "sampling process")


def _validate_update_controls(damp, eps):
    """Validate and normalise the stochastic fixed-point controls."""
    if isinstance(damp, (bool, np.bool_)):
        raise TypeError("damp must be a real number, not bool")
    if isinstance(eps, (bool, np.bool_)):
        raise TypeError("eps must be a real number, not bool")
    try:
        damp = float(damp)
        eps = float(eps)
    except (TypeError, ValueError):
        raise TypeError("damp and eps must be real numbers") from None
    if not np.isfinite(damp) or not 0.0 < damp <= 1.0:
        raise ValueError("damp must lie in (0, 1]")
    if not np.isfinite(eps) or not 1e-8 <= eps < 0.5:
        raise ValueError("eps must lie in [1e-8, 0.5)")
    return damp, eps


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


def _prepare_group(families, idx):
    """Per-structure precompute: relationship matrix ``A``, related-pair list,
    per-family bounds/fixed mask, and the initial chain state ``x``."""
    roles = sorted(m.role for m in families[idx[0]].members)
    k = len(roles)
    A = np.array([[get_relatedness(ri, rj, h2=1.0) for rj in roles] for ri in roles])
    min_eig = float(np.linalg.eigvalsh(A).min()) if k else 0.0
    if min_eig < -1e-8:
        raise ValueError(
            "the additive relationship kernel is not positive-semidefinite "
            f"for these roles (minimum eigenvalue {min_eig:.3g}); check the "
            "pedigree rather than repairing the theoretical relationships")
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
    fixed = np.ascontiguousarray((uppers - lowers) < _FIXED_TOL)
    ones = np.ones(k)                # unit marginal SD: bounds are pre-standardised
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_chain(lowers[slot], uppers[slot], ones)
    return dict(roles=roles, A=np.ascontiguousarray(A), pairs=pairs, k=k, F=F,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_heritability(families, *, h2_init=0.5, n_iter=1500, burn_in=500,
                     inner_sweeps=5, damp=0.2, seed=None, eps=1e-4,
                     sampling=None):
    """Estimate liability-scale ``h2`` from family case/control (+age) statuses.

    **Sampling contract:** this moment fitter supports independent,
    non-overlapping, unascertained population-sampled families only. Pass
    ``sampling="population"`` to acknowledge that contract. Omitting ``sampling``
    currently emits a compatibility warning; any other value raises. The fitter
    has no ascertainment likelihood or sampling weights, so case/control
    enrichment or selection on family history can produce severe boundary bias.

    ``families`` is a list of :class:`~ltpred.family.Family` whose members carry
    liability bounds (from a threshold builder). Alternates a Gibbs augmentation of
    the latent liabilities with a damped Haseman–Elston update of ``h2`` — a
    **stochastic-approximation fixed point** (not posterior sampling of ``h2``) that
    settles at the value consistent with the familial resemblance (see the module
    docstring). ``inner_sweeps`` truncated-MVN sweeps are taken per outer
    iteration; ``damp`` in ``(0, 1]`` controls the moment-update stability and
    ``eps`` in ``[1e-8, 0.5)`` keeps the covariance away from a singular boundary.
    Returns a :class:`FitResult`.

    Needs relatives (at least one related pair); a set of lone probands carries no
    information about ``h2`` and raises. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``, ``h2_init`` must lie in [0, 1], and ``burn_in``
    must be smaller than ``n_iter``.

    Common and person-specific one-sided, two-sided, and pinned rectangles are
    accepted. Their geometry cannot establish whether the observation model is
    scientifically coherent, so callers must ensure the supplied bounds represent
    the observation process intended for fitting. Standard NaN and interval-order
    validation still applies."""
    if int(burn_in) >= int(n_iter):
        raise ValueError(f"burn_in ({burn_in}) must be < n_iter ({n_iter})")
    if not 0.0 <= float(h2_init) <= 1.0:
        raise ValueError("h2_init must be in [0, 1]")
    damp, eps = _validate_update_controls(damp, eps)
    _validate_population_sampling(sampling, "fit_heritability")
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
    So ``A`` is the (narrow-sense) heritability. ``se`` is a within-dataset
    Monte-Carlo diagnostic (same caveat as :class:`FitResult`), not
    across-dataset sampling uncertainty. Use an appropriately designed
    family-cluster bootstrap for sampling uncertainty. ``traces`` are the
    post-burn-in proportion traces per component."""
    components: dict
    residual: float
    se: dict
    traces: dict
    n_iter: int
    burn_in: int


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
    # A component kernel may legitimately be singular: an exact shared-class
    # block is rank one. Keep it exact so fitting and prediction use the same
    # kernel; the assembled liability covariance is made strictly PD below.
    K = {c: _component_matrix(roles, c) for c in comps}
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
    fixed = np.ascontiguousarray((uppers - lowers) < _FIXED_TOL)
    ones = np.ones(k)                # unit marginal SD: bounds are pre-standardised
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_chain(lowers[slot], uppers[slot], ones)
    return dict(roles=roles, k=k, F=F, K=K, pairs=pairs,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_variance_components(families, components=("A", "C"), *,
                            n_iter=1500, burn_in=500, inner_sweeps=5, damp=0.2,
                            seed=None, eps=1e-4, sampling=None):
    """Fit liability-scale variance components by a multiple Haseman-Elston regression.

    **Sampling contract:** like :func:`fit_heritability`, this supports
    independent, non-overlapping, unascertained population-sampled families only.
    Pass ``sampling="population"`` to acknowledge that contract. The fitter does
    not correct case/control or family-history ascertainment.

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
    outer iteration; ``damp`` must lie in ``(0, 1]`` and ``eps`` in
    ``[1e-8, 0.5)`` keeps a positive residual floor. Returns a
    :class:`VarCompResult`. Simulation benchmarks recover ``A`` and ``A+C`` with
    small bias relative to their across-dataset SD. ``se`` is only a within-fit
    Monte-Carlo diagnostic. Use family resampling for a sampling interval when
    clusters are independent and representative. ``seed`` must be a non-boolean
    integer in ``[0, 2**32 - 1]`` or ``None``.

    Common and person-specific one-sided, two-sided, and pinned rectangles are
    accepted. Their geometry cannot establish whether the observation model is
    scientifically coherent, so callers must ensure the supplied bounds represent
    the observation process intended for fitting. Standard NaN and interval-order
    validation still applies."""
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
    damp, eps = _validate_update_controls(damp, eps)
    _validate_population_sampling(sampling, "fit_variance_components")
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

    The HE ``se`` reported by :func:`fit_heritability` and
    :func:`fit_variance_components` is a
    *within-dataset* Monte-Carlo diagnostic and can substantially understate
    sampling variability. This helper resamples families with replacement
    ``n_boot`` times and reports the refit SD and percentile interval. Its sampling
    interpretation assumes independent, non-overlapping, representative family
    clusters and a compatible ascertainment/model; it is not bias-corrected,
    studentized, or automatically calibrated at boundaries.

    ``estimator`` is a callable ``families -> value`` returning the quantity of
    interest as a float or array, e.g.::

        bootstrap_fit(
            fams,
            lambda f: fit_heritability(
                f, seed=1, sampling="population").h2)
        bootstrap_fit(fams, lambda f: fit_variance_components(f, ("A", "C"),
                                          seed=1, sampling="population"
                                      ).components["C"])

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
