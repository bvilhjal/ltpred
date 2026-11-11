"""Stochastic moment fits of liability variance components.

`fit_heritability` fits additive ``h2``; `fit_variance_components` fits A/C/M
proportions in ``Sigma = sum_c theta_c K_c + (1 - sum_c theta_c) I``. Persistent
truncated-normal chains augment liabilities, then a damped Haseman--Elston
regression updates the proportions. Retained fixed-point traces provide the
estimate and a batch-means Monte-Carlo diagnostic, not a sampling SE.

Families must be independent, non-overlapping and have common case/control
thresholds. Population sampling or known positive family-level IPW is required.
`bootstrap_fit` resamples whole families for sampling uncertainty. See
``docs/inference.md`` and Algorithm H in ``docs/algorithm.md``.
"""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike

from ._mathfun import norm_cdf
from ._validation import validate_bounds
from .covariance import get_relatedness, _is_full_sib, _is_mates
from .gibbs import (gibbs_params, _gibbs_advance_chain,
                    _init_chain, _FIXED_TOL, _seed_rng)
from .family import _member_pid_key
from .estimate import (_assert_nonempty_families, _check_unique_roles,
                       _group_by_structure, _batch_means, _scalar_member_bounds)

__all__ = ["FitResult", "fit_heritability", "VarCompResult",
           "fit_variance_components", "BootstrapResult", "bootstrap_fit"]


def _validate_population_sampling(sampling, context, *, weights=None):
    """Resolve the sampling contract, and keep it consistent with ``weights``.

    ``"population"`` is the unascertained contract. ``"ipw"`` says the families
    were selected on observed status with a KNOWN, strictly positive inclusion
    probability per family, and ``weights`` are the reciprocals of those
    probabilities: the per-family moment contributions are then re-mixed to
    population proportions. Returns the resolved mode.
    """
    if sampling is None:
        if weights is not None:
            raise ValueError(
                f"{context}: weights are only meaningful with sampling='ipw'; "
                "pass sampling='ipw' to declare an inverse-probability-weighted "
                "design")
        warnings.warn(
            f"{context} assumes independent, non-overlapping, unascertained "
            "population-sampled families. Case/control or family-history "
            "ascertainment can drive moment estimates to the boundary unless "
            "valid family-level inverse-probability weights are supplied. Pass "
            "sampling='population' only after verifying that contract, or "
            "sampling='ipw' with weights for a known selection probability.",
            RuntimeWarning, stacklevel=3)
        return "population"
    if sampling == "population":
        if weights is not None:
            raise ValueError(
                f"{context}: sampling='population' means no selection to undo, "
                "so weights are not accepted; use sampling='ipw' for a weighted "
                "design")
        return "population"
    if sampling == "ipw":
        if weights is None:
            raise ValueError(
                f"{context}: sampling='ipw' requires weights (one per family, "
                "the reciprocal of that family's inclusion probability)")
        return "ipw"
    raise ValueError(
        f"{context} supports sampling='population' or sampling='ipw'; "
        "ascertained designs whose selection probability is zero for some "
        "stratum (e.g. families ascertained through an affected proband) cannot "
        "be reweighted at all and require an estimator that models the sampling "
        "process")


def _validate_weights(weights, n_families, context):
    """Return per-family IPW weights as a positive finite float array."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != n_families:
        raise ValueError(
            f"{context}: weights must be one-dimensional with one entry per "
            f"family; got shape {w.shape} for {n_families} families")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights must be finite")
    if np.any(w <= 0.0):
        raise ValueError(
            f"{context}: weights must be strictly positive -- reciprocal "
            "inclusion probabilities cannot assign zero contribution to a "
            "sampled family")
    return w


def _assert_nonoverlapping_pids(families, context):
    """Reject a person (by ``pid``) who appears in more than one family.

    The moment fitters treat families as iid clusters. A register extraction
    that places the same parent in many probands' families is the intended
    *prediction* design and an invalid *fitting* design. Members without a
    ``pid`` cannot be checked, so a pid-less input is unchanged. Two copies of
    the same ``fam_id`` and members (bootstrap resampling with replacement)
    are one cluster, not two overlapping pedigrees.
    """
    seen = {}
    for family in families:
        # A repeated cluster is the same fam_id *and* the same people; a reused
        # label over different pedigrees that share a person still overlaps.
        signature = frozenset(k for k in map(_member_pid_key, family.members)
                              if k is not None)
        local = set()
        for member in family.members:
            key = _member_pid_key(member)
            if key is None:
                continue
            if key in local:
                raise ValueError(
                    f"{context}: pid {member.pid!r} appears more than once in "
                    f"family {family.fam_id!r}")
            local.add(key)
            previous = seen.get(key)
            if previous is not None and previous != (family.fam_id, signature):
                previous = previous[0]
                raise ValueError(
                    f"{context}: pid {member.pid!r} appears in families "
                    f"{previous!r} and {family.fam_id!r}; the moment fitters "
                    "require non-overlapping families. Use estimate_liability "
                    "for per-proband scores on overlapping register pedigrees.")
            seen[key] = (family.fam_id, signature)


#: Thresholds for the case-rate check. Sensitivity: in the dose-response of
#: ``benchmarks/bench_ascertainment.py`` (RESULTS §29) a case share 1.51x the
#: assumed prevalence already inflates h2 by +0.48, and every ascertainment
#: scheme fires at z >= +67 on the N = 10,000 grid. Specificity matters more,
#: since a false positive refuses a legitimate analysis: the check runs once
#: per role, and `bootstrap_fit` resamples centre on the cohort's rate rather
#: than on K. A legitimate 1,500-family cohort produced a resample at z = 4.4,
#: hence a bar of 6 (normal-reference null tail ~1e-9 under population sampling).
#: Under IPW the same trigger is only a descriptive heuristic: Kish effective
#: n does not supply a valid binomial variance when weights depend on status.
#: A weighted discrepancy warns rather than refusing a legitimate realization.
_CASE_RATE_RATIO_TOL = 1.15
_CASE_RATE_Z_TOL = 6.0


def _member_bounds(families, n_pheno):
    """Every member's bounds as ``(M, n_pheno)`` arrays, plus roles and family index.

    Rows are stacked family-major with members **sorted by role** within each
    family, so consumers with the same per-family ordering (`fit_pairwise_multi`'s
    pair aggregation) can index their rows directly. Scalars broadcast across
    traits; the per-member fallback raises the same shape error
    ``np.broadcast_to`` gives for a wrong-length bound."""
    members = [(f, m) for f, family in enumerate(families)
               for m in sorted(family.members, key=lambda m: m.role)]
    shape = (len(members), n_pheno)
    try:
        lo = np.asarray([m.lower for _, m in members], dtype=float)
        hi = np.asarray([m.upper for _, m in members], dtype=float)
        lo = np.broadcast_to(lo.reshape(len(members), -1), shape)
        hi = np.broadcast_to(hi.reshape(len(members), -1), shape)
    except ValueError:
        lo = np.array([np.broadcast_to(np.asarray(m.lower, dtype=float), (n_pheno,))
                       for _, m in members]).reshape(shape)
        hi = np.array([np.broadcast_to(np.asarray(m.upper, dtype=float), (n_pheno,))
                       for _, m in members]).reshape(shape)
    return (lo, hi, [m.role for _, m in members],
            np.fromiter((f for f, _ in members), dtype=np.intp, count=len(members)))


def _assert_population_case_rate(families, n_pheno, *, context, weights=None,
                                 member_bounds=None):
    """Screen marginal case-rate/prevalence discrepancies.

    For population sampling, ``|z| >= 6`` and a case-rate ratio outside
    ``[1/1.15, 1.15]`` raises, skipping role/trait cells with n < 30. Under IPW
    the same trigger with Kish effective n is an uncalibrated heuristic and
    only warns: outcome-dependent weights invalidate its binomial variance.
    Sampled weights cannot certify joint positivity or falsify the design;
    even a positive-probability stratum may be absent from a finite sample.
    """
    lo_all, hi_all, roles, fam_index = (member_bounds if member_bounds is not None
                                        else _member_bounds(families, n_pheno))
    w = (np.ones(len(roles)) if weights is None
         else np.asarray(weights, dtype=float)[fam_index])
    # after _assert_common_thresholds: finite lower => case, finite upper =>
    # control, and the finite end IS the threshold. An UNINFORMATIVE member --
    # (-inf, inf), which the prediction path uses routinely to unbind a
    # proband -- is neither, and counting it as a control deflates the role's
    # rate and manufactures a failure on legitimate data.
    is_case = np.isfinite(lo_all)
    informative = is_case | np.isfinite(hi_all)
    thr_all = np.where(is_case, lo_all, hi_all)
    per_role = {}
    for role in set(roles):
        rows = np.fromiter((r == role for r in roles), dtype=bool, count=len(roles))
        for p in range(n_pheno):
            use = rows & informative[:, p]
            if use.any():
                wr = w[use]
                per_role[(role, p)] = (wr.sum(), (wr * wr).sum(),
                                       (wr * is_case[use, p]).sum(),
                                       thr_all[use, p][0])

    worst = None
    for (role, pheno), (sw, sw2, k, thr) in sorted(per_role.items()):
        n = sw * sw / sw2 if sw2 > 0 else 0.0        # Kish effective sample size
        if n < 30:                       # too little information for this screen
            continue
        expected = float(norm_cdf(-thr))     # = 1 - Phi(thr), exact by symmetry
        if not 0.0 < expected < 1.0:
            continue
        observed = k / sw
        se = math.sqrt(expected * (1.0 - expected) / n)
        # A binomial z only for population sampling; under IPW this scale is
        # a warning trigger, not an estimated sampling SD.
        screen_score = (observed - expected) / se if se > 0 else 0.0
        ratio = observed / expected
        if abs(screen_score) >= _CASE_RATE_Z_TOL and not (
                1.0 / _CASE_RATE_RATIO_TOL <= ratio <= _CASE_RATE_RATIO_TOL):
            if worst is None or abs(screen_score) > abs(worst[3]):
                worst = (role, pheno, ratio, screen_score, observed, expected, int(round(n)))

    if worst is None:
        return
    role, pheno, ratio, z, observed, expected, n = worst
    trait = "" if n_pheno == 1 else f" (trait {pheno})"
    if weights is not None:
        warnings.warn(
            f"{context}: descriptive IPW case-rate discrepancy. Role {role!r}{trait} "
            f"has weighted affected rate {observed:.4f} against population "
            f"prevalence {expected:.4f} (ratio {ratio:.2f}, Kish effective n {n}). "
            "This warning is uncalibrated: outcome-dependent weights invalidate "
            "the binomial variance used by the trigger. Check the prevalence "
            "and sampling design; this discrepancy does not establish invalid "
            "weights or a positivity failure. A positive-probability stratum "
            "can be absent from a finite sample. Fitting continues.",
            RuntimeWarning, stacklevel=3)
        return
    raise ValueError(
        f"{context}: the supplied families are not consistent with "
        f"sampling='population'. Role {role!r}{trait} is affected in "
        f"{observed:.4f} of {n} families, but the threshold supplied for it "
        f"asserts a population prevalence of {expected:.4f} -- a factor of "
        f"{ratio:.2f} ({z:+.1f} SD). The pooled Haseman-Elston fixed point "
        "assumes every member is a draw from that same population, so this "
        "mismatch biases it hard and in a direction that looks like real "
        "heritability: in the repository benchmark a 1.51x enrichment inflates "
        "h2 by +0.48 and 2x pins it at the boundary, and on ascertained data "
        "with true h2 = 0 the fitter returns h2 = 1.0. Either the cohort is "
        "ascertained (case/control, family-history or proband-affected "
        "selection), in which case use valid IPW for known, strictly positive "
        "family-level selection probabilities or a selection model otherwise, "
        "or the prevalence "
        "behind the thresholds is wrong for this sample. Fixing the thresholds "
        "is a real fix. See benchmarks/RESULTS.md, ascertainment section.")


def _assert_common_thresholds(families, n_pheno, *, context, member_bounds=None):
    """Require one common one-sided case/control threshold per trait.

    Personalised bounds, pins and two-sided intervals are rejected by these
    moment fitters, not by liability scoring. The restriction is this
    implementation's validated scope, not a failure of total expectation.
    """
    lo_all, hi_all, _roles, _fam_index = (member_bounds if member_bounds is not None
                                          else _member_bounds(families, n_pheno))
    if not lo_all.size:
        return
    # Structural bounds validation runs first so a NaN or reversed interval gets
    # its own precise error rather than being miscounted as a pin/interval below.
    validate_bounds(lo_all, hi_all, context=context)

    pin = hi_all - lo_all < 1e-8
    control = ~pin & np.isneginf(lo_all) & np.isfinite(hi_all)
    case = ~pin & np.isfinite(lo_all) & np.isposinf(hi_all)
    pinned = int(pin.sum())
    interval = int((~pin & np.isfinite(lo_all) & np.isfinite(hi_all)).sum())
    thr_all = np.where(control, hi_all, lo_all)
    thresholds = [thr_all[control[:, p] | case[:, p], p] for p in range(n_pheno)]
    # rtol=1e-6 tolerates the ~1e-7 gap between the float32 and float64
    # representations of one common threshold; genuine age-/CIP-specific
    # thresholds differ by orders of magnitude more.
    varying = any(v.size and not np.allclose(v, v[0], rtol=1e-6, atol=1e-9)
                  for v in thresholds)
    if not (pinned or interval or varying):
        return
    seen = []
    if pinned:
        seen.append(f"{pinned} onset-pinned bound(s)")
    if interval:
        seen.append(f"{interval} two-sided interval bound(s)")
    if varying:
        seen.append("thresholds that differ between individuals")
    raise ValueError(
        f"{context}: found {', '.join(seen)}. The pooled Haseman-Elston fixed "
        "point assumes a single case/control threshold per trait, so "
        "personalised (age-/CIP-specific) LT-FH++ bounds fall outside that "
        "estimating contract. Exploratory failures motivated this restriction, "
        "but do not establish a universal bias direction or magnitude. "
        "Fit from common-threshold bounds (e.g. prevalence_thresholds) instead; "
        "to model onset-age structure explicitly use "
        "research.advanced_fitting.fit_genetic_correlation_decay, whose "
        "likelihood M-step is built for it. Personalised bounds remain correct "
        "for estimate_liability, which conditions on h2 rather than fitting it.")


def _validate_moment_families(families, *, weights, context):
    """Validate the shared single-trait moment-fitting observation contract."""
    _check_unique_roles(families, check_pids=False)
    _assert_nonempty_families(families)
    _assert_nonoverlapping_pids(families, context)
    member_bounds = _member_bounds(families, 1)
    _assert_common_thresholds(families, 1, context=context,
                              member_bounds=member_bounds)
    _assert_population_case_rate(families, 1, context=context,
                                 weights=weights, member_bounds=member_bounds)


def _validate_eps(eps):
    """Validate the numerical variance floor shared by all fitters."""
    if isinstance(eps, (bool, np.bool_)):
        raise TypeError("eps must be a real number, not bool")
    try:
        eps = float(eps)
    except (TypeError, ValueError):
        raise TypeError("eps must be a real number") from None
    if not np.isfinite(eps) or not 1e-8 <= eps < 0.5:
        raise ValueError("eps must lie in [1e-8, 0.5)")
    return eps


def _validate_optimizer_controls(maxiter, tol):
    """Validate the SLSQP controls shared by the pairwise fitters."""
    if isinstance(maxiter, (bool, np.bool_)):
        raise TypeError("maxiter must be a positive integer")
    try:
        maxiter = operator.index(maxiter)
    except TypeError:
        raise TypeError("maxiter must be a positive integer") from None
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if isinstance(tol, (bool, np.bool_)):
        raise TypeError("tol must be a finite positive real number")
    tol = float(tol)
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and in (0, 1)")
    return maxiter, tol


def _normalized_weights(weights):
    """IPW weights rescaled to mean one (max-first, so the rescale itself
    cannot overflow); ``None`` stays ``None`` for callers whose unweighted
    path differs."""
    if weights is None:
        return None
    w = weights / weights.max()
    w = w / w.mean()
    if np.any(w == 0):
        raise ValueError("IPW weights span too wide a numerical range to retain positive contributions")
    return w


def _validate_update_controls(damp, eps):
    """Validate and normalise the stochastic fixed-point controls."""
    if isinstance(damp, (bool, np.bool_)):
        raise TypeError("damp must be a real number, not bool")
    try:
        damp = float(damp)
    except (TypeError, ValueError):
        raise TypeError("damp and eps must be real numbers") from None
    if not np.isfinite(damp) or not 0.0 < damp <= 1.0:
        raise ValueError("damp must lie in (0, 1]")
    return damp, _validate_eps(eps)


def _validate_components(components):
    """The variance components as a list: non-empty, known, and distinct."""
    try:
        comps = list(components)
    except TypeError:
        raise TypeError("components must be a non-empty sequence") from None
    if not comps:
        raise ValueError("components must contain at least one component")
    for c in comps:
        if c not in _COMPONENT_OFFDIAG:
            avail = ", ".join(_COMPONENT_OFFDIAG)
            raise ValueError(f"unknown component {c!r}; choose from {avail} "
                             "(dominance 'D' is not supported)")
    if len(set(comps)) != len(comps):
        raise ValueError(f"duplicate components in {components!r}")
    return comps


def _validate_iteration_controls(n_iter, burn_in, inner_sweeps, *,
                                 min_retained=1):
    """Return strict integer controls for the stochastic fixed point.

    ``min_retained`` is the least ``n_iter - burn_in`` accepted; the public
    fitters pass 4, the two batches `ltpred.estimate._batch_means` needs."""
    values = {}
    for name, value, minimum in (("n_iter", n_iter, 1),
                                 ("burn_in", burn_in, 0),
                                 ("inner_sweeps", inner_sweeps, 1)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer, not bool")
        try:
            value = operator.index(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer") from None
        if value < minimum:
            qualifier = "positive" if minimum else "non-negative"
            raise ValueError(f"{name} must be a {qualifier} integer")
        values[name] = value
    if values["burn_in"] >= values["n_iter"]:
        raise ValueError(
            f"burn_in ({values['burn_in']}) must be non-negative and "
            f"< n_iter ({values['n_iter']})")
    if values["n_iter"] - values["burn_in"] < min_retained:
        raise ValueError(
            f"n_iter - burn_in ({values['n_iter']} - {values['burn_in']}) must "
            f"be >= {min_retained}: the batch-means SE of the retained trace "
            "needs at least two batches")
    return values["n_iter"], values["burn_in"], values["inner_sweeps"]


#: Retained fixed-point iterates the batch-means SE needs (two batches of two).
_MIN_RETAINED = 4


def _validate_h2_init(h2_init):
    """Return a finite real initial heritability, rejecting Boolean aliases."""
    if isinstance(h2_init, (bool, np.bool_)):
        raise TypeError("h2_init must be a real number, not bool")
    try:
        h2_init = float(h2_init)
    except (TypeError, ValueError):
        raise TypeError("h2_init must be a real number") from None
    if not np.isfinite(h2_init) or not 0.0 <= h2_init <= 1.0:
        raise ValueError("h2_init must be in [0, 1]")
    return h2_init


@dataclass
class FitResult:
    """Heritability estimate and its fixed-point traces.

    ``h2_se`` is batch-means Monte-Carlo error in this fit, not a sampling SE
    or a confidence interval. ``samples`` is post-burn-in and ``trace`` full;
    neither is a posterior chain. Use `bootstrap_fit` for sampling uncertainty.
    """
    h2: float
    h2_se: float
    samples: np.ndarray
    trace: np.ndarray
    n_iter: int
    burn_in: int


def fit_heritability(families: Sequence, *, h2_init: float = 0.5,
                     n_iter: int = 1500, burn_in: int = 500,
                     inner_sweeps: int = 5, damp: float = 0.2,
                     seed: int | None = None, eps: float = 1e-4,
                     sampling: str | None = None,
                     weights: ArrayLike | None = None) -> FitResult:
    """Fit liability-scale heritability from common-threshold family data.

    The stochastic fixed point alternates truncated-MVN augmentation with a
    damped Haseman--Elston update. ``n_iter`` and ``burn_in`` count outer
    iterations, ``inner_sweeps`` sets Gibbs work per iteration, and ``damp``
    controls the update. ``h2_init`` lies in [0, 1]; at least four retained
    iterates are required. Related pairs are necessary for identification.

    Families must be independent and non-overlapping. Use
    ``sampling="population"`` for an unascertained sample, or
    ``sampling="ipw"`` with one reciprocal inclusion probability per family
    for a known, strictly positive selection design. Omitting ``sampling``
    warns and assumes population sampling. Personalised bounds, onset pins and
    two-sided intervals are outside this fitter's scope. The marginal case-rate
    screen catches gross inconsistency but does not certify the design.

    Returns a `FitResult`. ``h2_se`` measures Monte-Carlo noise in this fit,
    not sampling uncertainty; use `bootstrap_fit` for the latter.
    """
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps, min_retained=_MIN_RETAINED)
    h2_init = _validate_h2_init(h2_init)
    damp, eps = _validate_update_controls(damp, eps)
    weights = _validate_weights(weights, len(families), "fit_heritability")
    _validate_population_sampling(sampling, "fit_heritability", weights=weights)
    _validate_moment_families(families, weights=weights, context="fit_heritability")
    trace, samples, est, se = _fit_component_engine(
        families, ("A",), np.array([h2_init]), n_iter=n_iter,
        burn_in=burn_in, inner_sweeps=inner_sweeps, damp=damp, seed=seed,
        eps=eps, weights=weights, context="heritability fit bounds",
        identification_error=(
            "no related pairs in the families — cannot fit h2 "
            "(need relatives, not lone probands)."))
    trace = trace[:, 0].copy()
    samples = samples[:, 0].copy()
    return FitResult(h2=float(est[0]), h2_se=float(se[0]), samples=samples,
                     trace=trace, n_iter=n_iter, burn_in=burn_in)


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
# would be a mis-specified component -- `_component_matrix` guards against it.
#
# Dominance ("D") is likewise not offered. It requires an explicit dominance
# relationship kernel and contrasts linearly independent of the additive and
# sibship kernels; MZ/DZ observations can contribute in a richer design but cannot
# identify A, C, and D by themselves.


@dataclass
class VarCompResult:
    """Result of `fit_variance_components`.

    ``components`` maps each fitted component (``"A"`` additive, ``"C"`` sibship
    common environment, ``"M"`` couple/spousal environment) to its estimated
    **proportion** of the liability variance; ``residual`` is the remaining ``e2``.
    So ``A`` is the (narrow-sense) heritability. ``se`` is a within-dataset
    Monte-Carlo diagnostic (same caveat as `FitResult`), not
    across-dataset sampling uncertainty. Use an appropriately designed
    family-cluster bootstrap for sampling uncertainty. ``traces`` are the
    post-burn-in proportion traces per component, and ``n_iter`` and
    ``burn_in`` the controls the fit ran with."""
    components: dict
    residual: float
    se: dict
    traces: dict
    n_iter: int
    burn_in: int


def _component_matrix(roles, comp):
    """Relationship matrix ``K`` for one variance component (diagonal 1).

    Off-diagonals over the sorted ``roles``: ``A`` is the additive relationship
    (``get_relatedness(a, b, 1.0)``), ``C`` is 1 for full sibs, ``M`` is 1 for a
    mate pair (`_COMPONENT_OFFDIAG`).

    A valid variance component has a positive-semidefinite ``K`` (the additive
    relationship is PSD for any consistent pedigree; an equivalence-class partition
    is one sufficient construction for a shared-environment kernel). A non-PSD ``K`` — e.g. a
    vertical parent-offspring "environment" whose sharing chains across generations —
    is not a proper component, so it is rejected here."""
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


def _prepare_component_group(families, idx, comps, weights=None, *,
                             context="variance-component fit bounds"):
    """Precompute one role-set group for Algorithm H.

    Returns kernels, the nonzero-kernel pairs with their design rows, role-sorted
    bounds, IPW weights and the initial chain state (marginal truncated medians)."""
    roles = sorted(m.role for m in families[idx[0]].members)
    k = len(roles)
    F = len(idx)
    # A component kernel may legitimately be singular: an exact shared-class
    # block is rank one. Keep it exact so fitting and prediction use the same
    # kernel; the assembled covariance is strictly PD once the residual is positive.
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
            lowers[slot, c], uppers[slot, c] = _scalar_member_bounds(m, families[f].fam_id)
    validate_bounds(lowers, uppers, context=context)
    fixed = np.ascontiguousarray((uppers - lowers) < _FIXED_TOL)
    ones = np.ones(k)                # unit marginal SD: bounds are pre-standardised
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_chain(lowers[slot], uppers[slot], ones)
    w = (np.ones(F) if weights is None
         else np.asarray(weights, dtype=float)[list(idx)])
    return dict(roles=roles, k=k, F=F, K=K, pairs=pairs, eye=np.eye(k),
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                w=np.ascontiguousarray(w),
                x=np.ascontiguousarray(x))


def _assert_observed_identification(groups, comps, n_pheno=1, *, context):
    """Check covariance contrasts before augmenting missing phenotypes.

    A sampled, unobserved liability cannot supply information about its own
    covariance. For each trait pair require full rank among jointly observed
    coordinates; cross-trait fits also need the within-person residual column.
    Bounds in these groups are phenotype-major.
    """
    for p in range(n_pheno):
        for q in range(p, n_pheno):
            rows = set()
            for g in groups:
                k = g["k"]
                observed = np.isfinite(g["lowers"]) | np.isfinite(g["uppers"])
                kernels = g.get("K", {"A": g.get("A")})
                for i in range(k):
                    for j in range(i + 1 if p == q else 0, k):
                        if not np.any(observed[:, p*k+i] & observed[:, q*k+j]):
                            continue
                        row = tuple(float(kernels[c][i, j]) for c in comps)
                        if p != q:
                            row += (float(i == j),)
                        rows.add(row)
            dimension = len(comps) + int(p != q)
            design = np.asarray(list(rows), dtype=float).reshape(-1, dimension)
            if len(design) < dimension or np.linalg.matrix_rank(design) < dimension:
                raise ValueError(
                    f"{context}: covariance components not identified from observed "
                    f"pairs for traits ({p}, {q}); need informative related pairs "
                    "and independent relationship contrasts (rank-deficient design)")


def _fit_component_engine(families, comps, initial, *, n_iter, burn_in,
                          inner_sweeps, damp, seed, eps, weights, context,
                          identification_error=None):
    """Run Algorithm H steps H2-H7 (module docstring) for components ``comps``.

    Returns ``(trace, post-burn-in samples, estimate, batch-means SE)``; the
    engine behind both `fit_heritability` and `fit_variance_components`."""
    C = len(comps)
    groups = [
        _prepare_component_group(families, idx, comps, weights, context=context)
        for _key, idx in _group_by_structure(families)
    ]

    # X'X is fixed across sweeps. With weights == 1 this is the ordinary
    # unweighted pair design; IPW changes only each family's contribution.
    # Wholly unobserved families (every member (-inf, inf)) carry no data: their
    # latent draws are prior draws, so they would only pull each update back to
    # the current values and slow convergence in proportion to their number.
    for group in groups:
        keep = (np.isfinite(group["lowers"]) | np.isfinite(group["uppers"])).any(axis=1)
        if not keep.all():
            for key in ("lowers", "uppers", "fixed", "w", "x"):
                group[key] = np.ascontiguousarray(group[key][keep])
            group["F"] = int(keep.sum())
    groups = [g for g in groups if g["F"] > 0]
    XtX = np.zeros((C, C))
    for group in groups:
        family_weight = float(group["w"].sum())
        for _i, _j, row in group["pairs"]:
            XtX += family_weight * np.outer(row, row)
    if np.linalg.matrix_rank(XtX, tol=1e-8) < C:
        if identification_error is not None:
            raise ValueError(identification_error)
        raise ValueError(
            "variance components not identified from these families — the "
            "relationship design is rank-deficient (e.g. fitting 'C' with no "
            "full-sib pairs, or no related pairs at all).")
    _assert_observed_identification(groups, comps, context=context)
    # The scalar solve is exact division. The tiny ridge stabilises only an
    # identified multicomponent design.
    XtX_reg = XtX if C == 1 else XtX + 1e-10 * np.eye(C)

    # Unconditional: seed=None reseeds from entropy (see gibbs._seed_rng), so
    # an unseeded fit never replays the generator a previous seeded call left
    # in this thread's persistent-chain stream.
    _seed_rng(seed)

    values = np.asarray(initial, dtype=float).copy()
    trace = np.empty((n_iter, C))
    for it in range(n_iter):
        # Updates stay below 1 - eps. With h2_init=1 the first iteration has
        # zero residual (Sigma = K_A).
        residual = max(1.0 - values.sum(), 0.0)
        Xty = np.zeros(C)
        for group in groups:
            sigma = residual * group["eye"] + sum(
                values[ci] * group["K"][component]
                for ci, component in enumerate(comps))
            precision, conditional_sd = gibbs_params(sigma)
            _gibbs_advance_chain(
                precision, conditional_sd, group["lowers"], group["uppers"],
                group["fixed"], group["x"], inner_sweeps)
            x = group["x"]
            w = group["w"]
            for i, j, row in group["pairs"]:
                Xty += row * float(w @ (x[:, i] * x[:, j]))
        estimate = (Xty / XtX_reg[0, 0] if C == 1
                    else np.linalg.solve(XtX_reg, Xty))
        estimate = np.clip(estimate, eps, 1.0 - eps)
        if estimate.sum() > 1.0 - eps:
            estimate *= (1.0 - eps) / estimate.sum()
        values = (1.0 - damp) * values + damp * estimate
        trace[it] = values

    samples = trace[burn_in:].copy()
    fitted, se = _batch_means(samples)
    return trace, samples, fitted, se


def fit_variance_components(families: Sequence, components: Sequence[str] = ("A", "C"),
                            *, n_iter: int = 1500, burn_in: int = 500,
                            inner_sweeps: int = 5, damp: float = 0.2,
                            seed: int | None = None, eps: float = 1e-4,
                            sampling: str | None = None,
                            weights: ArrayLike | None = None) -> VarCompResult:
    """Fit A/C/M liability-variance proportions by stochastic moment regression.

    ``components`` selects additive genetic (A), full-sibship environment (C)
    and/or couple environment (M). Identifying relationship contrasts must be
    observed; dominance is unsupported. The algorithm, controls and sampling
    contract match `fit_heritability`. Component proportions sum to at most
    one and ``residual`` is the remainder.

    Returns a `VarCompResult`. Its ``se`` values are within-fit Monte-Carlo
    diagnostics, not sampling SEs; use family resampling for uncertainty.
    """
    comps = _validate_components(components)
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps, min_retained=_MIN_RETAINED)
    damp, eps = _validate_update_controls(damp, eps)
    weights = _validate_weights(weights, len(families),
                                "fit_variance_components")
    _validate_population_sampling(sampling, "fit_variance_components",
                                  weights=weights)
    _validate_moment_families(families, weights=weights,
                              context="fit_variance_components")
    C = len(comps)
    _trace, samples, est, se = _fit_component_engine(
        families, comps, np.full(C, 0.5 / C), n_iter=n_iter,
        burn_in=burn_in, inner_sweeps=inner_sweeps, damp=damp, seed=seed,
        eps=eps, weights=weights, context="variance-component fit bounds")
    components_out = {c: float(est[ci]) for ci, c in enumerate(comps)}
    traces = {c: np.ascontiguousarray(samples[:, ci]) for ci, c in enumerate(comps)}
    return VarCompResult(components=components_out,
                         residual=float(1.0 - est.sum()),
                         se={c: float(se[ci]) for ci, c in enumerate(comps)},
                         traces=traces, n_iter=n_iter, burn_in=burn_in)


@dataclass
class BootstrapResult:
    """Result of `bootstrap_fit`.

    ``estimate`` is the point estimate from the full data; ``se`` the bootstrap
    standard error (SD, ``ddof=1``, of the resampled estimates); ``ci_low`` / ``ci_high`` the
    percentile confidence interval at ``ci_level``; ``samples`` the ``(n_boot, …)``
    array of per-resample estimates. Shapes follow whatever the estimator returns
    (scalar → shape ``()``; vector/matrix → that shape)."""
    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_level: float
    n_boot: int
    samples: np.ndarray


def bootstrap_fit(families: Sequence, estimator: Callable, *,
                  weights: ArrayLike | None = None, n_boot: int = 100,
                  seed: int | None = None,
                  ci_level: float = 0.95) -> BootstrapResult:
    """Estimate percentile uncertainty by resampling whole families.

    ``estimator`` receives the family list and returns a scalar or array of
    stable shape. With ``weights``, it instead receives ``(families, weights)``
    and both are resampled by the same indices. Fix the estimator's own seed
    so bootstrap spread reflects family resampling rather than sampler noise.
    The resampling ``seed`` is passed to ``numpy.random.default_rng``; unlike
    sampler seeds, it is not restricted to uint32.
    Returns the full-data estimate, refit SD, percentile interval and samples.
    Independent, non-overlapping clusters and a valid sampling design remain
    necessary; boundary intervals are not automatically calibrated.
    """
    families = list(families)
    n = len(families)
    if n < 2:
        raise ValueError("need at least 2 families to bootstrap")
    weights = _validate_weights(weights, n, "bootstrap_fit")
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
    point_value = (estimator(families) if weights is None
                   else estimator(families, weights))
    point = np.asarray(point_value, dtype=float)
    samples = np.empty((n_boot,) + point.shape, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sampled_families = [families[i] for i in idx]
        sample_value = (estimator(sampled_families) if weights is None
                        else estimator(sampled_families, weights[idx]))
        sample = np.asarray(sample_value, dtype=float)
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
