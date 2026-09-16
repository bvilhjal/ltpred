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
cross-products reconstruct population moments directly for independent,
unascertained population-sampled families. Known, strictly positive family-level
selection probabilities can instead be handled by inverse-probability weighting;
unknown probabilities, zero-probability strata, and overlapping pedigrees require
an estimator that models the sampling process.

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
from .gibbs import (gibbs_params, gibbs_advance,
                    _init_chain, _FIXED_TOL, _seed_rng)
from .family import _is_missing_id
from .estimate import (_assert_nonempty_families, _check_unique_roles,
                       _group_by_structure, batch_means)

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
            f"{context}: weights must be strictly positive -- a zero weight is a "
            "family that could not have been sampled, which is a positivity "
            "failure rather than something to down-weight")
    return w


def _pid_key(pid):
    """Hashable identity for a member ``pid``, or ``None`` if it is missing.

    Missing values cannot witness overlap. Numpy scalars are unwrapped so
    ``np.int64(1)`` and ``1`` compare equal. Unhashable objects fall back to
    ``str`` rather than crashing the check.
    """
    if _is_missing_id(pid):
        return None
    if isinstance(pid, np.generic):
        pid = pid.item()
        if _is_missing_id(pid):
            return None
    if isinstance(pid, bytes):
        pid = pid.decode("utf-8", "replace")
    if isinstance(pid, str):
        pid = pid.strip()
        if _is_missing_id(pid):
            return None
        return pid
    try:
        hash(pid)
    except TypeError:
        return str(pid)
    return pid


def _assert_nonoverlapping_pids(families, context):
    """Reject a person (by ``pid``) who appears in more than one family.

    The moment fitters treat families as iid clusters. A register extraction
    that places the same parent in many probands' families is the intended
    *prediction* design and an invalid *fitting* design. Members without a
    ``pid`` cannot be checked, so a pid-less input is unchanged. Two copies of
    the same ``fam_id`` (bootstrap resampling with replacement) are one
    cluster, not two overlapping pedigrees.
    """
    seen = {}
    for family in families:
        local = set()
        for member in family.members:
            key = _pid_key(member.pid)
            if key is None:
                continue
            if key in local:
                raise ValueError(
                    f"{context}: pid {member.pid!r} appears more than once in "
                    f"family {family.fam_id!r}")
            local.add(key)
            previous = seen.get(key)
            if previous is not None and previous != family.fam_id:
                raise ValueError(
                    f"{context}: pid {member.pid!r} appears in families "
                    f"{previous!r} and {family.fam_id!r}; the moment fitters "
                    "require non-overlapping families. Use estimate_liability "
                    "for per-proband scores on overlapping register pedigrees.")
            seen[key] = family.fam_id


#: Thresholds for the case-rate check, calibrated from BOTH sides.
#:
#: Sensitivity, from the dose-response in ``benchmarks/bench_ascertainment.py``
#: (nuclear families, true h2 = 0.5, K = 0.05, N = 4000): a realised case share
#: 1.27x the assumed prevalence inflates h2 by +0.230, 1.49x by +0.493, and by
#: 2x the estimate is pinned at the clamp (arm H of that benchmark; the curve is
#: steep through this region, so read the shape rather than any single cell).
#:
#: Specificity matters more, because a false positive here refuses a legitimate
#: analysis. Two things push the null z above what a single clean test would
#: give: the check runs once per role (so several correlated tests per fit), and
#: :func:`bootstrap_fit` re-runs the whole estimator on resamples that are
#: centred on the *cohort's* rate rather than on K, so their z carries the
#: cohort's own sampling error as a systematic offset. A legitimate 1500-family
#: cohort (role rate 0.110 against K = 0.100, p = 0.20) produced a resample at
#: z = 4.4 -- which is why the bar is 6.0 and not 4.0.
#:
#: At z >= 6 the per-test null probability is ~1e-9, so chance firing is
#: negligible even across a 25-resample bootstrap, while every ascertainment
#: scheme in the benchmark fires at z >= +67. The cost is power at small N.
#: Detectable enrichment is the ratio at which |z| reaches the bar,
#: ``1 + 6*sqrt((1-K)/(K*n))``, so it depends on BOTH n and the prevalence: at
#: K = 0.05 it is ~1.67x at n = 1,500 and ~1.26x at n = 10,000; at K = 0.10,
#: ~1.46x and ~1.18x. (Quoting one figure without its K mixes the two.) MILD
#: enrichment on a small cohort therefore passes, and remains the caller's
#: responsibility. This is a guard against the catastrophic case, not a
#: certificate of population sampling.
_CASE_RATE_RATIO_TOL = 1.15
_CASE_RATE_Z_TOL = 6.0


def _assert_population_case_rate(families, n_pheno, *, context, weights=None):
    """Check the observed case rate against the one the thresholds assert.

    ``sampling="population"`` was an honour system: it checked a string, not the
    data, so an ascertained cohort passed straight through to a fixed point that
    runs to the boundary. The data can be checked directly, because the supplied
    bounds already encode the assumed prevalence -- a common threshold ``t``
    means ``K = 1 - Phi(t)`` -- and under population sampling each role's case
    indicator is one ``Bernoulli(K)`` per family, independent across families.
    So the count for role ``r`` is ``Binomial(n_families, K)`` and a plain
    binomial z-test applies, with no clustering correction needed.

    This runs after :func:`_assert_common_thresholds`, which guarantees exactly
    the input this assumes: one threshold per trait, every member a one-sided
    case or control, no pins or two-sided intervals.

    A failure does not necessarily mean the sample was ascertained -- an honestly
    population-sampled cohort analysed with a mis-specified ``K`` fails the same
    way, and is wrong for the same reason. Either way the model's own
    precondition is violated, so the message names the observed and asserted
    rates rather than guessing the cause.

    With ``weights`` (``sampling="ipw"``) the same **marginal-calibration** check
    is applied to the weighted case rate. Correct inverse-probability weights must
    re-mix each role's marginal rate back to ``K``, so failure falsifies the
    supplied weighting design. Passing is necessary but not sufficient: matching
    marginals cannot prove that every joint family-status stratum had positive
    inclusion probability, nor that the weights reconstruct their joint
    distribution. The binomial SE uses Kish's effective sample size
    ``(sum w)^2 / sum w^2``, so heavy weights widen the tolerance instead of
    manufacturing significance.
    """
    per_role = {}
    for fam_index, family in enumerate(families):
        w = 1.0 if weights is None else float(weights[fam_index])
        for member in family.members:
            lo = np.broadcast_to(np.asarray(member.lower, dtype=float), (n_pheno,))
            hi = np.broadcast_to(np.asarray(member.upper, dtype=float), (n_pheno,))
            for p in range(n_pheno):
                # after _assert_common_thresholds: finite lower => case,
                # finite upper => control, and the finite end IS the threshold.
                # An UNINFORMATIVE member -- (-inf, inf), which the prediction
                # path uses routinely to unbind a proband -- is neither, and
                # counting it as a control deflates the role's rate and
                # manufactures a failure on legitimate data.
                lo_p, hi_p = lo[p], hi[p]
                is_case = bool(np.isfinite(lo_p))
                if not is_case and not np.isfinite(hi_p):
                    continue
                thr = lo_p if is_case else hi_p
                sw, sw2, k, t = per_role.get((member.role, p), (0.0, 0.0, 0.0, thr))
                per_role[(member.role, p)] = (sw + w, sw2 + w * w,
                                              k + w * int(is_case), t)

    worst = None
    for (role, pheno), (sw, sw2, k, thr) in sorted(per_role.items()):
        n = sw * sw / sw2 if sw2 > 0 else 0.0        # Kish effective sample size
        if n < 30:                       # binomial normal approx not trustworthy
            continue
        expected = float(norm_cdf(-thr))     # = 1 - Phi(thr), exact by symmetry
        if not 0.0 < expected < 1.0:
            continue
        observed = k / sw
        se = math.sqrt(expected * (1.0 - expected) / n)
        z = (observed - expected) / se if se > 0 else 0.0
        ratio = observed / expected
        if abs(z) >= _CASE_RATE_Z_TOL and not (
                1.0 / _CASE_RATE_RATIO_TOL <= ratio <= _CASE_RATE_RATIO_TOL):
            if worst is None or abs(z) > abs(worst[3]):
                worst = (role, pheno, ratio, z, observed, expected, int(round(n)))

    if worst is None:
        return
    role, pheno, ratio, z, observed, expected, n = worst
    trait = "" if n_pheno == 1 else f" (trait {pheno})"
    if weights is not None:
        raise ValueError(
            f"{context}: the weighted marginal case rates are not consistent "
            f"with the supplied thresholds. Role {role!r}{trait} has a weighted "
            "affected "
            f"rate of {observed:.4f} (effective n {n}) against an asserted "
            f"population prevalence of {expected:.4f} -- a factor of {ratio:.2f} "
            f"({z:+.1f} SD). This falsifies the supplied weighting design: the "
            "weights may be wrong, the asserted prevalence may be wrong, or a "
            "joint stratum may be missing (a positivity failure). Passing this "
            "marginal check would not prove joint positivity or weight validity. "
            "A design with a zero-probability stratum needs an estimator that "
            "models selection, not a reweighting of the observed families.")
    raise ValueError(
        f"{context}: the supplied families are not consistent with "
        f"sampling='population'. Role {role!r}{trait} is affected in "
        f"{observed:.4f} of {n} families, but the threshold supplied for it "
        f"asserts a population prevalence of {expected:.4f} -- a factor of "
        f"{ratio:.2f} ({z:+.1f} SD). The pooled Haseman-Elston fixed point "
        "assumes every member is a draw from that same population, so this "
        "mismatch biases it hard and in a direction that looks like real "
        "heritability: in the repository benchmark a 1.27x enrichment inflates "
        "h2 by +0.23 and 2x pins it at the boundary, and on ascertained data "
        "with true h2 = 0 the fitter returns h2 = 1.0. Either the cohort is "
        "ascertained (case/control, family-history or proband-affected "
        "selection), in which case use valid IPW for known, strictly positive "
        "family-level selection probabilities or a selection model otherwise, "
        "or the prevalence "
        "behind the thresholds is wrong for this sample. Fixing the thresholds "
        "is a real fix. See benchmarks/RESULTS.md, ascertainment section.")


def _assert_common_thresholds(families, n_pheno, *, context):
    """Reject person-specific liability bounds in the pooled-moment fitters.

    The Haseman-Elston fixed point pools cross-products across families and reads
    them as estimates of ``h2 * A_ij``, which holds only when every augmented
    liability is a draw from the *same* ``N(0, 1)`` population -- i.e. when one
    threshold per trait separates cases from controls. **Personalised LT-FH++
    bounds fall outside that estimating contract**: an age-/CIP-specific
    threshold per person (and an onset pin for cases) gives each augmented draw
    its own conditional mean, so the pooled cross-products no longer have the
    supported common-threshold interpretation. Exploratory failures motivated
    this guard, but they do not establish a universal direction or magnitude of
    bias for every personalised observation model. This implementation therefore
    refuses the unsupported geometry rather than returning an uncertified fit.

    The prediction estimators (:func:`~ltpred.estimate.estimate_liability` and
    friends) are unaffected -- they *condition* on a supplied ``h2`` rather than
    fitting it, and personalised bounds are exactly what they are designed for.
    """
    los, his = [], []
    for family in families:
        for member in family.members:
            los.append(np.broadcast_to(np.asarray(member.lower, dtype=float), (n_pheno,)))
            his.append(np.broadcast_to(np.asarray(member.upper, dtype=float), (n_pheno,)))
    if not los:
        return
    lo_all = np.asarray(los)
    hi_all = np.asarray(his)
    # Structural bounds validation runs first so a NaN or reversed interval gets
    # its own precise error rather than being miscounted as a pin/interval below.
    validate_bounds(lo_all, hi_all, context=context)

    pinned = interval = 0
    thresholds = [[] for _ in range(n_pheno)]
    for row_lo, row_hi in zip(lo_all, hi_all):
        for p in range(n_pheno):
            lo, hi = row_lo[p], row_hi[p]
            if hi - lo < 1e-8:
                pinned += 1
            elif np.isneginf(lo) and np.isfinite(hi):
                thresholds[p].append(float(hi))
            elif np.isfinite(lo) and np.isposinf(hi):
                thresholds[p].append(float(lo))
            elif np.isfinite(lo) and np.isfinite(hi):
                interval += 1
    # rtol=1e-6 tolerates the ~1e-7 gap between the float32 and float64
    # representations of one common threshold; genuine age-/CIP-specific
    # thresholds differ by orders of magnitude more.
    varying = any(v and not np.allclose(v, v[0], rtol=1e-6, atol=1e-9)
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


def _validate_iteration_controls(n_iter, burn_in, inner_sweeps):
    """Return strict integer controls for the stochastic fixed point."""
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
    return values["n_iter"], values["burn_in"], values["inner_sweeps"]


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


def fit_heritability(families: Sequence, *, h2_init: float = 0.5,
                     n_iter: int = 1500, burn_in: int = 500,
                     inner_sweeps: int = 5, damp: float = 0.2,
                     seed: int | None = None, eps: float = 1e-4,
                     sampling: str | None = None,
                     weights: ArrayLike | None = None) -> FitResult:
    """Estimate liability-scale ``h2`` from family case/control (+age) statuses.

    **Sampling contract:** this moment fitter assumes independent,
    non-overlapping families under one of two designs. ``sampling="population"``
    is the unascertained case; ``sampling="ipw"`` with per-family ``weights``
    covers selection on observed status with a known, strictly positive
    inclusion probability (below). Omitting ``sampling`` emits a compatibility
    warning; anything else raises. There is no ascertainment *likelihood* here,
    so a design that reweighting cannot reach -- because family-level selection
    probabilities are unknown or misspecified, or any joint stratum has
    probability zero -- still produces severe boundary bias.

    That acknowledgement is now **checked against the data**, not merely taken on
    trust: the supplied thresholds assert a prevalence, and each role's case rate
    is compared against it (:func:`_assert_population_case_rate`). A gross
    mismatch raises, because the failure it guards is severe and silent -- on
    ascertained families with a true ``h2`` of 0, this fitter returns
    ``h2 = 1.0``. The check is deliberately conservative, so it catches the
    catastrophic designs rather than certifying population sampling; mild
    enrichment on a small cohort still passes and remains your responsibility.
    When members carry ``pid``, a person who appears in more than one family
    (distinct ``fam_id``) is rejected; prediction on overlapping register
    pedigrees is :func:`~ltpred.estimate.estimate_liability`.

    **Selected samples:** ``sampling="ipw"`` with per-family ``weights`` handles
    selection on observed status when the inclusion probability is known and
    strictly positive for every stratum -- the standard case/control cohort and
    biobank case-enrichment designs. Pass ``weights = 1 / P(family sampled)``.
    The per-family moment contributions are then re-mixed to population
    proportions, which is the right shape of remedy here: the augmentation for a
    *given* family with *given* statuses is already correct, and it is the
    **mix** of families that selection breaks. In the repository benchmark
    (``benchmarks/RESULTS.md`` section 29) this takes a 50/50 case/control
    cohort from an unweighted fit pinned at ``h2 = 1`` back close to the true
    0.5.

    Two limits, both real. **Positivity:** a design that samples no families from
    some joint status stratum has inclusion probability zero there, and no
    weighting can reconstruct what was never observed. The marginal case-rate
    check can falsify many such designs, but passing it does not prove joint
    positivity or weight validity. **Efficiency:** weights reach 19x at K = 0.05
    with a 50/50 cohort, so the effective sample size is far below the nominal
    one; the benchmark reports the inflated across-replicate SD alongside the
    bias.

    The case-rate check has a stated blind spot: the enrichment it can detect is
    ``1 + 6*sqrt((1-K)/(K*n))``, which at K = 0.05 is ~1.67x at n = 1,500 and
    ~1.26x at n = 10,000. Milder enrichment passes, and the benchmark's
    dose-response shows that is not harmless.

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
    must be non-negative and smaller than ``n_iter``.

    **Requires a common case/control threshold per trait.** The pooled
    Haseman-Elston fixed point assumes every augmented liability is drawn from
    the same ``N(0, 1)`` population, so personalised (age-/CIP-specific) or
    onset-pinned LT-FH++ bounds — even perfectly coherent ones — fall outside
    that estimating contract. Those inputs are **rejected**, not silently fitted
    (see :func:`_assert_common_thresholds`). Fit from common-threshold bounds; the
    prediction path (:func:`~ltpred.estimate.estimate_liability`) is unaffected,
    since it conditions on ``h2`` rather than fitting it. Standard NaN and
    interval-order validation still applies."""
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps)
    h2_init = _validate_h2_init(h2_init)
    damp, eps = _validate_update_controls(damp, eps)
    weights = _validate_weights(weights, len(families), "fit_heritability")
    _validate_population_sampling(sampling, "fit_heritability", weights=weights)
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    _assert_nonoverlapping_pids(families, "fit_heritability")
    _assert_common_thresholds(families, 1, context="fit_heritability")
    _assert_population_case_rate(families, 1, context="fit_heritability",
                                 weights=weights)
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


def _prepare_component_group(families, idx, comps, weights, *, context):
    """Precompute one shared-structure group for the component engine."""
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


def _prepare_group_vc(families, idx, comps, weights=None):
    """Compatibility wrapper around the shared component-group preparation."""
    return _prepare_component_group(
        families, idx, comps, weights, context="variance-component fit bounds")


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
    """Run the common collapsed-Gibbs/multiple-HE fixed-point engine."""
    C = len(comps)
    groups = [
        _prepare_component_group(families, idx, comps, weights, context=context)
        for _key, idx in _group_by_structure(families)
    ]

    # X'X is fixed across sweeps. With weights == 1 this is the ordinary
    # unweighted pair design; IPW changes only each family's contribution.
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
    # The scalar solve is exact division, preserving fit_heritability's original
    # update. The tiny ridge stabilises only an identified multicomponent design.
    XtX_reg = XtX if C == 1 else XtX + 1e-10 * np.eye(C)

    if seed is not None:
        _seed_rng(seed)

    values = np.asarray(initial, dtype=float).copy()
    trace = np.empty((n_iter, C))
    for it in range(n_iter):
        # Updates stay below 1 - eps. A user-supplied h2_init=1 retains the
        # single-component fitter's historical first-iteration behaviour.
        residual = max(1.0 - values.sum(), 0.0)
        Xty = np.zeros(C)
        for group in groups:
            sigma = residual * group["eye"] + sum(
                values[ci] * group["K"][component]
                for ci, component in enumerate(comps))
            precision, conditional_sd = gibbs_params(sigma)
            gibbs_advance(
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
    fitted, se = batch_means(samples)
    return trace, samples, fitted, se


def fit_variance_components(families: Sequence, components: Sequence[str] = ("A", "C"),
                            *, n_iter: int = 1500, burn_in: int = 500,
                            inner_sweeps: int = 5, damp: float = 0.2,
                            seed: int | None = None, eps: float = 1e-4,
                            sampling: str | None = None,
                            weights: ArrayLike | None = None) -> VarCompResult:
    """Fit liability-scale variance components by a multiple Haseman-Elston regression.

    **Sampling contract:** like :func:`fit_heritability`, this supports
    independent, non-overlapping families sampled either directly from the
    population or with known, strictly positive family-level selection
    probabilities. Pass ``sampling="population"`` for the former, or
    ``sampling="ipw"`` with reciprocal-probability ``weights`` for the latter (see
    :func:`fit_heritability` for both, including the marginal-check, positivity,
    pid-overlap, and efficiency limits). Without one of those, this fitter does not correct
    case/control or family-history ascertainment; under it the components
    saturate, exhausting the residual variance rather than sitting at the
    elementwise clamp.

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

    **Requires a common case/control threshold per trait**, exactly as
    :func:`fit_heritability` does and for the same reason: personalised or
    onset-pinned LT-FH++ bounds fall outside the supported pooled-moment
    estimating contract, so they are **rejected** rather than silently fitted.
    Standard NaN and interval-order validation still applies."""
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
    n_iter, burn_in, inner_sweeps = _validate_iteration_controls(
        n_iter, burn_in, inner_sweeps)
    damp, eps = _validate_update_controls(damp, eps)
    weights = _validate_weights(weights, len(families),
                                "fit_variance_components")
    _validate_population_sampling(sampling, "fit_variance_components",
                                  weights=weights)
    _check_unique_roles(families)
    _assert_nonempty_families(families)
    _assert_nonoverlapping_pids(families, "fit_variance_components")
    _assert_common_thresholds(families, 1, context="fit_variance_components")
    _assert_population_case_rate(families, 1,
                                 context="fit_variance_components",
                                 weights=weights)
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


def bootstrap_fit(families: Sequence, estimator: Callable, *,
                  weights: ArrayLike | None = None, n_boot: int = 100,
                  seed: int | None = None,
                  ci_level: float = 0.95) -> BootstrapResult:
    """Approximate percentile uncertainty by **resampling family clusters**.

    The HE ``se`` reported by :func:`fit_heritability` and
    :func:`fit_variance_components` is a
    *within-dataset* Monte-Carlo diagnostic and can substantially understate
    sampling variability. This helper resamples families with replacement
    ``n_boot`` times and reports the refit SD and percentile interval. Its sampling
    interpretation assumes independent, non-overlapping family clusters and
    either representative sampling or valid aligned IPW; it is not
    bias-corrected, studentized, or automatically calibrated at boundaries.

    ``estimator`` is a callable ``families -> value`` returning the quantity of
    interest as a float or array, e.g.::

        bootstrap_fit(
            fams,
            lambda f: fit_heritability(
                f, seed=1, sampling="population").h2)
        bootstrap_fit(fams, lambda f: fit_variance_components(f, ("A", "C"),
                                          seed=1, sampling="population"
                                      ).components["C"])

    For an IPW analysis, pass the original per-family ``weights`` and accept a
    second argument in the estimator. The helper resamples the two arrays with
    the same indices, preventing weights from becoming detached from their
    families::

        bootstrap_fit(
            fams,
            lambda f, w: fit_heritability(
                f, sampling="ipw", weights=w, seed=1).h2,
            weights=weights)

    Fix the estimator's internal ``seed`` so each refit is deterministic given its
    resample — then the bootstrap spread reflects family sampling, not the sampler's
    own Monte-Carlo noise. Returns a :class:`BootstrapResult`. Cost is ``n_boot + 1``
    fits, so this is deliberately expensive; lower ``n_boot`` for a quick check.

    ``estimator`` must return a stable scalar/array shape. ``n_boot`` is an integer
    at least 2; substantially more than the default 100 may be needed for stable
    interval endpoints. ``seed`` follows NumPy ``default_rng`` semantics and seeds
    only the resampling; ``ci_level`` sets the percentile interval. When
    ``weights`` is supplied, ``estimator`` is called as
    ``estimator(families, weights)`` for both the point fit and every refit."""
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
