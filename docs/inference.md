# Inference

Beyond estimating each proband's liability, ltpred can **fit the model** from the
family data: the liability-scale heritability and the shared-environment variance
components. That is **use III** of the [vignette](vignette.md) (architecture),
under a declared sampling contract — not a side effect of scoring families for
prediction or a GWAS. It is optional: skip straight to
[estimation](estimation.md) if you already have an `h²`.
Unsupported experimental inferential machinery lives in the checkout-only
`research/` package; see [Unsupported research prototypes](#unsupported-research-prototypes).

!!! danger "Supported sampling contract"

    `fit_heritability`, `fit_variance_components` and `fit_pairwise` assume independent,
    non-overlapping families. When members carry `pid`, a person who appears
    in more than one family (distinct `fam_id`) is rejected; use
    `estimate_liability` for per-proband scores on overlapping register
    pedigrees. For **unascertained** samples pass
    `sampling="population"`; for a sample **selected on observed status with a
    known, strictly positive inclusion probability** pass `sampling="ipw"` with
    per-family `weights` (see [Ascertained samples](#ascertained-samples)).
    Nothing else is supported: `bootstrap_fit` does not repair selection bias,
    and a design that samples *no* families from some stratum cannot be
    reweighted at all.

    `sampling="population"` is **screened for marginal inconsistency** with your
    data, not taken entirely on trust. Your thresholds assert a prevalence, so
    each role's case rate is compared with it and a gross mismatch raises. This
    is worth knowing because
    the failure it guards is severe and silent: on ascertained families with a
    true `h²` of **0**, the unguarded HE fitter returns **`h² = 1.0`**
    ([RESULTS.md §29](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).

    A failure is equally consistent with an honestly sampled cohort analysed
    with the **wrong prevalence** — a real and fixable cause — so the error
    reports the observed and asserted rates rather than assuming ascertainment.

    **It is a guard, not a certificate.** The enrichment it can detect is
    `max(1 + 6·√((1−K)/(K·n)), 1.15)×` — the z-score gate is combined with a
    15% rate-ratio floor. At K = 0.05 that is ~1.67× at N = 1,500 and
    ~1.26× at N = 10,000, with the 1.15× floor binding for N ≳ 30,000. Milder enrichment passes silently, and the
    HE dose-response below shows a 1.51× enrichment already inflates `h²` by
    +0.48. Passing this check is not evidence that your sample is
    population-sampled.

The meaning of a reported `se` depends on the method. For the
Haseman–Elston/data-augmentation fits, it is a *within-dataset* Monte-Carlo
diagnostic, not across-dataset sampling uncertainty.
The opt-in [pairwise fitter](#deterministic-pairwise-fitting) instead reports
conditional, asymptotic family-cluster sampling SEs for interior estimates.
[`bootstrap_fit`](#family-cluster-uncertainty-bootstrap_fit) provides an
approximate family-cluster sampling interval when families are independent.

## Fitting heritability from the family data

If the sampling contract above holds, you can **estimate `h²` from the family
data** when the bounds also encode a coherent observation model. Otherwise use
an external estimate or a fitter that models the sampling design.
`fit_heritability` alternates a Gibbs augmentation of the latent liabilities with a damped
Haseman–Elston update of `h²`, converging to the value consistent with the observed
familial resemblance (see
[algorithm.md](algorithm.md#fitting-the-covariance-heritability)). This is a
**stochastic-approximation fixed point**, not posterior sampling of `h²`: the trace
is not a posterior draw and its mean is not a posterior mean.

The pooled Haseman–Elston fixed point requires **one case/control threshold
per trait**. Personalised (age-/CIP-specific) or onset-pinned LT-FH++ bounds —
including the Quickstart's `age_thresholds` output, even when they are
internally coherent — are **rejected**. This is an identification contract,
motivated by exploratory failures outside the common-threshold design, not a
current quantitative benchmark for personalised fitting. Fit from
common-threshold bounds (`prevalence_thresholds`), or bring an external
liability-scale `h²`. Personalised bounds remain the intended input for
`estimate_liability`, which conditions on `h²` rather than fitting it. For
onset-age-structured genetic correlation see
`research.advanced_fitting.fit_genetic_correlation_decay`.

```python
from ltpred import fit_heritability
fit = fit_heritability(
    fitting_families,
    sampling="population",  # explicit acknowledgement, not a correction
)
fit.h2, fit.h2_se  # fitted liability-scale h² (+ MC diagnostic)
```

It needs relatives (lone probands carry no information and raise). `fit.h2_se` is a
*within-dataset* Monte-Carlo **diagnostic** of the fixed point, not an inferential
standard error; it substantially understated across-dataset variability in the
repository benchmarks. For an approximate iid-family cluster interval, use
`bootstrap_fit` (below). Feed the point estimate back in as
`h2=fit.h2` (or, better, run the [sensitivity analysis](#sensitivity-to-the-assumed-heritability)
around it).

### Family-cluster uncertainty: `bootstrap_fit`

`bootstrap_fit` resamples families with replacement and refits, giving a
family-cluster bootstrap SE and percentile CI:

```python
from ltpred import bootstrap_fit
bs = bootstrap_fit(
    families,
    lambda f: fit_heritability(
        f, seed=1, sampling="population"
    ).h2,
    n_boot=100,
)
bs.estimate, bs.se, (bs.ci_low, bs.ci_high)   # point, bootstrap SE, percentile CI
```

It wraps any of the fitters: pass a `lambda` that returns the quantity of
interest, such as the `"C"` component from
`fit_variance_components(..., sampling="population")`, or a fit from
`research.advanced_fitting`. Fix the estimator's `seed` so the spread reflects
family sampling, not sampler noise. For IPW, pass `weights=weights` to
`bootstrap_fit` and use a two-argument callable, `lambda f, w:
fit_heritability(f, sampling="ipw", weights=w, seed=1).h2`; the helper resamples
families and weights with the same indices. It costs `n_boot`+1 fits.
The interval assumes iid, non-overlapping family clusters and either
representative sampling or valid aligned IPW. It is not automatically calibrated
for zero-probability selection strata, boundary parameters, or overlapping
pedigrees. The `n_boot=100` call above is a computational example, not a
recommended final precision: percentile endpoints can be visibly unstable with
so few resamples. Increase `n_boot` until the SE and interval endpoints are stable
for your analysis, and report the number of successful refits.

## Deterministic pairwise fitting

`fit_pairwise` estimates the same A/C/M covariance parameters using products of
binary relative-pair probabilities. It aggregates common-threshold observations
by relationship pattern and optimizes a small deterministic objective, without
sampling latent liabilities. This is a separate composite-likelihood estimator;
`fit_heritability` and `fit_variance_components` retain their existing algorithms.

```python
from ltpred import fit_pairwise

pw = fit_pairwise(
    fitting_families, components=("A", "C"), sampling="population",
)
pw.components, pw.residual
pw.se, pw.inference_status
```

The initial implementation requires a common single-trait case/control
threshold. It rejects personalized thresholds, onset pins and two-sided
intervals; uninformative members contribute no pairs. Components must have
independent identifying contrasts among the **observed** pairs. Fractions are
nonnegative and sum to at most `1 - eps` (`eps=1e-6` by default). Failed,
infeasible or nonstationary optimizer results raise rather than being returned.

Independent, non-overlapping family clusters and the sampling contract above
remain essential. For a known, strictly positive selection design, use
`fit_pairwise(fitting_families, sampling="ipw", weights=weights)`; weights are
normalized to mean one without changing their relative contributions. Passing
the weighted marginal screen does not establish joint positivity or correct
weights. The fitter does not infer selection probabilities or fit an
ascertainment likelihood.

For an interior fit, `pw.covariance` follows `pw.component_order`, and `pw.se`
contains its square-root diagonal. The sandwich calculation combines the
observed likelihood sensitivity with **family-level** score variability, so
shared members within a family are not treated as independent pairs. It is
conditional on supplied thresholds and weights and excludes uncertainty from
estimating them. These are asymptotic SEs, not finite-sample calibration claims.
At a component or residual constraint boundary, `pw.at_boundary` is true and
the covariance and SEs are `NaN`; `pw.inference_status` also distinguishes
insufficient clusters or information. Ordinary normal intervals are then
inappropriate, and the composite log likelihood does not justify ordinary
chi-square likelihood-ratio tests.

For cluster resampling, an existing `bootstrap_fit` callback can return
`fit_pairwise(f, sampling="population").components["A"]`. For IPW, accept
both `f` and resampled `w` and pass `weights=weights` to `bootstrap_fit`.
No internal random seed is needed. Bootstrap or profile inference at a
boundary still needs separate calibration.

## Ascertained samples

Selection on phenotype also matters for pairwise fitting. The quantified
failure below concerns the HE/data-augmentation fitter. From the
dose-response in `benchmarks/bench_ascertainment.py` (nuclear families,
true `h²` = 0.5, K = 0.05, N = 2,000):

| realised case share ÷ assumed K | 0.95× | 1.51× | 1.96× | ≥ 2× |
|---|---:|---:|---:|---:|
| fitted `h²` | 0.420 | 0.984 | 0.998 | 1.000 |

A 7.6% case rate against an assumed 5.0% already inflates `h²` by +0.48. This is
**bias, not noise**: it does not shrink with N (constant +0.500 from N = 1,000 to
10,000) and it is not an unconverged run (the same value is reached from
`h2_init` 0.05 and 0.95).

### When you can correct it: `sampling="ipw"`

If families were selected on observed status with a **known** inclusion
probability `π` that is **positive for every stratum**, pass the reciprocals as
weights:

```python
import numpy as np
from ltpred import fit_heritability

# 50/50 case/control cohort drawn from a population with K = 0.05:
# every case kept, controls retained with probability K(1-q)/(q(1-K)).
keep_p = 0.05 * 0.5 / (0.5 * 0.95)
weights = np.where(proband_is_case, 1.0, 1.0 / keep_p)

fit = fit_heritability(families, sampling="ipw", weights=weights)
```

Why this is the right shape of correction: the liability augmentation for a
*given* family with *given* statuses is already the correct conditional
distribution. What selection breaks is the **mix** of families, and weighting
re-mixes them to population proportions. Benchmarked
([RESULTS §29](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)),
this takes both a 50/50 case/control cohort and a 20%-enriched cohort from an
unweighted fit pinned at `h² = 1` back close to the true 0.5; the fitted means
and their replicate scatter are in that section.

Two limits, and both matter:

- **Positivity.** A design that samples no families from some stratum has
  `π = 0` there, and no weight reconstructs what was never observed.
  Ascertainment through an affected proband is the standard example. The
  benchmark labels such rows *undefined from the known design* before calling
  the weighted fitter, because no finite valid weights exist. The fitter's
  role-wise weighted case-rate screen can falsify some bad weights, but passing
  it does **not** establish joint-pattern positivity or correct weights.
- **Efficiency.** At K = 0.05 a 50/50 cohort needs weights up to 19×, so the
  effective sample size is far below the nominal one and the across-replicate SD
  grows accordingly. IPW buys accuracy with precision.

Weights must be supplied by you from the sampling design; ltpred cannot infer
them, and estimating `π` from a sampling frame adds its own error. Selection
on family history is not categorically excluded: it is IPW-correctable only when
the inclusion probability of every complete observed family pattern is known
and strictly positive. Deterministic family-history selection usually violates
that condition.

!!! warning "What a scale correction cannot do"

    A Lee et al. (2011)-style observed→liability factor does not help here, and
    the reason is worth stating: it is a multiplicative function of `(K, P)`
    alone, while at fixed `K` a true `h²` of 0.5 and of 0.0 **both** produce
    1.000. No invertible constant maps both back. That correction also targets
    an *observed-scale* estimate, whereas these fitters estimate on the
    liability scale directly, so there is no observed-scale quantity to
    transform. [`observed_to_liability_h2`](api.md) remains the right tool for
    its own job: converting an observed-scale `h²` that some other method
    produced.

    Benchmark arm J is deliberately narrower: it standardises each role by its
    sample case rate, pools multi-relative HE moments, and applies the proband's
    sample fraction in the Lee factor. It is a diagnostic of that improvised
    bridge, not an implementation or evaluation of conventional unrelated-
    sample LDSC or GREML.

## Sensitivity to the assumed heritability

Because no single `h²` is uniquely correct, check how much the score actually
depends on it: re-estimate over a plausible `h²` grid and correlate the scores
across settings:

```python
import numpy as np
from ltpred import estimate_liability
grid = [0.3, 0.4, 0.5, 0.6, 0.7]
scores = [estimate_liability(families, h2=h).est["genetic"] for h in grid]
min_corr = np.corrcoef(scores)[np.triu_indices(len(grid), 1)].min()
```

In the calibration benchmark (§12 of `benchmarks/RESULTS.md`) the ranking is
barely touched by a wrong `h²`: corr(est, true g) stays in 0.429–0.431 across
assumed `h²` 0.2–0.8 while the calibration slope sweeps from 2.24 to 0.68 —
changing the assumed `h²` acts mostly like a near-linear
rescaling of the liability score. This is an empirical sensitivity result, not
rank invariance or a guarantee for other structures. A low cross-setting
correlation is the signal to obtain a better external `h²`, or—only when the
population-sampling contract holds—to fit it with `fit_heritability`. Prevalence/CIP
sensitivity changes the truncation bounds rather than the covariance, so probe it
by rebuilding the families under each prevalence and comparing. (The *scale* is more
sensitive than the ranking — see the calibration benchmark in
[benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).)

## Variance components (A / C / M)

To separate additive heritability from a shared **common-environment** component
`C` (e.g. a full-sib effect that inflates familial resemblance beyond genetics),
`fit_variance_components` fits both at once by a multiple Haseman–Elston
regression:

```python
from ltpred import fit_variance_components
vc = fit_variance_components(
    families, ("A", "C"), sampling="population"
)
vc.components["A"], vc.components["C"], vc.residual   # proportions of liability variance
```

`C` is identified only from **full-sib pairs**, so the families must contain them
(otherwise the fit raises). The same `h2_se` caveat applies — use `bootstrap_fit` for a CI.

Dominance is intentionally not offered. It requires a dominance relationship
kernel with contrasts linearly independent of `A` and `C`. MZ/DZ twin observations
can contribute useful contrasts within a richer design, but MZ/DZ pairs alone
cannot identify `A`, `C`, and `D` simultaneously. The current role bank does not
provide a dominance kernel. With `("A",)` alone the result matches
`fit_heritability`.

A second shared-environment component, `"M"` (couple / spousal environment),
loads on the genetically-unrelated **mate pairs** — the parents `(m, f)` and the
grandparent couples. On a 3-generation population-sampled pedigree,
`fit_variance_components(..., sampling="population")` fits all three at once
(each component needs its identifying pairs; the fit raises on a rank-deficient
design). `M` is a
**descriptive spousal-resemblance component**: it absorbs shared adult environment
and some manifestations of assortative mating, but it is *not* a generative model
of assortative mating (which would also alter the genetic covariance among
offspring and across generations). Because mates have `A = 0`, omitting a real
`M` biases `A` **much less than omitting `C`** — so fit `M` to
quantify spousal resemblance
for its own sake (a parametric-bootstrap component test lives in
`research.advanced_fitting`) rather than to de-bias `h²`. The shipped `C` and `M` kernels are
equivalence-class partitions, which guarantees that they are positive
semi-definite (PSD); equivalence classes are sufficient, not necessary. Any future
kernel must be symmetric and PSD. Every fitted kernel has diagonal one, so each
component also consumes that fraction of marginal liability variance and the
individual residual is `1 - sum(components)`. The naive vertical parent-offspring
indicator considered here is non-PSD and is rejected. Component kernels may be
PSD; the assembled covariance used by Gibbs must be strictly positive-definite.

## Unsupported research prototypes

Multi-trait genetic-correlation and onset-age-decay fitting, common-factor and
genetic-nurture models, MCEM, and parametric-bootstrap tests live in
[`research/advanced_fitting.py`](https://github.com/bvilhjal/ltpred/blob/main/research/advanced_fitting.py).
They are importable from a source checkout but are not installed and are not part
of the supported API. See the
[`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
for the current inventory and the [research extensions](research.md) page for
the models and their identifiability caveats.
