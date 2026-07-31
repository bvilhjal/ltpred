# Inference

Beyond estimating each proband's liability, ltpred can **fit the model** from the
family data: the liability-scale heritability and the shared-environment variance
components. These are optional
— skip straight to [estimation](estimation.md) if you already have an `h²`.
Unsupported experimental inferential machinery lives in the checkout-only
`research/` package; see [Unsupported research prototypes](#unsupported-research-prototypes).

!!! danger "Supported sampling contract"

    `fit_heritability` and `fit_variance_components` support only independent,
    non-overlapping, unascertained population-sampled families. They have no
    ascertainment likelihood or sampling weights. Case/control enrichment or
    selection on family history can drive estimates to the boundary, and
    `bootstrap_fit` does not repair that bias. Pass `sampling="population"` only
    to acknowledge a design that actually satisfies this contract.

The meaning of a reported `se` depends on the method. For the
Haseman–Elston/data-augmentation fits, it is a *within-dataset* Monte-Carlo
diagnostic, not across-dataset sampling uncertainty.
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

The fitter accepts common or person-specific one-sided, two-sided, and pinned
rectangles: their geometry alone cannot reveal how they were constructed. That
flexibility makes provenance the caller's responsibility. The onset-pinned bounds
from the quickstart are accepted, but using them makes that onset/threshold
construction part of the fitted observation model. If that is not the model you
intend, use an external estimate or construct separate fitting bounds.

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
family sampling, not sampler noise. It costs `n_boot`+1 fits.
The interval assumes iid, non-overlapping family clusters and a sampling design
compatible with the fitted model. It is not automatically calibrated under
case/control or family-history ascertainment, boundary parameters, or overlapping
pedigrees. The `n_boot=100` call above is a computational example, not a
recommended final precision: percentile endpoints can be visibly unstable with
so few resamples. Increase `n_boot` until the SE and interval endpoints are stable
for your analysis, and report the number of successful refits.

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

In the documented benchmark this worst-case cross-setting correlation was high
(≈0.97 across `h² 0.2–0.8` for the tested pedigree): changing the assumed `h²`
acted mostly like a near-linear
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
for the current inventory and [algorithm.md](algorithm.md) for model background.
