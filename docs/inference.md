# Fitting model parameters

Fit liability-scale heritability, genetic/residual correlations and
shared-environment covariance from family data. If those parameters are already
available, go to [Scoring](estimation.md). Unsupported prototypes stay in
[research extensions](research.md).

!!! danger "Supported sampling contract"

    All four fitters assume **independent, non-overlapping families** and common
    case/control thresholds. When `pid` is supplied, a person appearing in more
    than one family raises. Use `estimate_liability` for overlapping register
    pedigrees.

    Use `sampling="population"` for an unascertained sample. For selection on
    observed status, use `sampling="ipw"` with one reciprocal **known, strictly
    positive** inclusion probability per family. Omitting `sampling` warns and
    assumes population sampling. No weight can reconstruct a stratum with zero
    inclusion probability, and `bootstrap_fit` does not repair selection bias.

    Under population sampling, each role's case rate is screened for gross
    inconsistency with the threshold's prevalence. A mismatch can mean
    ascertainment **or the wrong prevalence**. The screen is not a certificate:
    at K = 0.05 its detectable
    enrichment is about 1.67× at N = 1,500 and 1.26× at N = 10,000, with a
    1.15× floor binding at large N. The benchmark shows 1.51× enrichment
    already inflates h² by +0.48.

    Under IPW, a large weighted mismatch only warns. Its trigger uses Kish
    effective sample size as an uncalibrated heuristic, not a binomial test:
    outcome-dependent weights change the sampling variance. Sampled weights
    alone cannot establish that a design is invalid or a stratum has zero
    inclusion probability.

The meaning of `se` differs by method. Haseman–Elston (HE) fits report a
within-dataset Monte-Carlo diagnostic, not sampling uncertainty. Pairwise fits
report conditional, asymptotic family-cluster sampling SEs for identifiable
interior estimates. `bootstrap_fit` provides approximate family-cluster
uncertainty when its design assumptions hold.

## Choose a fitter

**Table 1. Fitters for independent families with common binary thresholds.**

| fitter | use when | computation and uncertainty |
|---|---|---|
| `fit_heritability` | additive h² only | stochastic moment fixed point; family bootstrap |
| `fit_variance_components` | separate A, C and/or M with identifying relationships | stochastic moment fixed point; family bootstrap |
| `fit_pairwise` | deterministic single-trait A/C/M | composite likelihood; interior cluster-sandwich SEs |
| `fit_pairwise_multi` | joint trait h², genetic/residual correlations, optional C/M | multivariate composite likelihood; interior cluster/delta-method SEs |

Personalised CIP/onset bounds are for scoring, not these fits. Identification
requires jointly observed relationship contrasts. The stochastic fitters are
one to two orders of magnitude slower at their defaults; shortening their Gibbs
schedules requires separate Monte Carlo checks. See the
[benchmark ledger](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for measured recovery and cost.

## Fitting heritability from the family data

`fit_heritability` alternates Gibbs augmentation of latent liabilities with a
damped Haseman–Elston update. This is a **stochastic-approximation fixed
point**, not posterior sampling: the trace is not a posterior chain and its
mean is not a posterior mean.

```python
from ltpred import fit_heritability
fit = fit_heritability(fitting_families, sampling="population")
fit.h2, fit.h2_se       # estimate and within-fit MC diagnostic
```

Relatives are required. `fit.h2_se` is not an inferential SE and understated
across-dataset variability in benchmarks; use `bootstrap_fit` for that. The
fitter requires one common threshold per trait and rejects personalised bounds,
onset pins and two-sided intervals. That is the validated implementation scope,
not evidence that heterogeneous thresholds invalidate the moment identity.
`estimate_liability` still accepts personalised bounds because it conditions on
`h2` rather than fitting it.

### Family-cluster uncertainty: `bootstrap_fit`

```python
from ltpred import bootstrap_fit
bs = bootstrap_fit(
    families,
    lambda f: fit_heritability(f, seed=1, sampling="population").h2,
    n_boot=100,
)
bs.estimate, bs.se, (bs.ci_low, bs.ci_high)
```

The callback can return any scalar or array from a fitter. Fix its internal
seed so spread reflects family resampling rather than sampler noise. For IPW,
pass `weights=weights` and use a two-argument callback such as
`lambda f, w: fit_heritability(f, sampling="ipw", weights=w, seed=1).h2`;
families and weights are resampled by the same indices.

Cost is `n_boot + 1` fits. The interval assumes independent clusters and a
valid sampling design; zero-probability strata, overlapping pedigrees and
boundary parameters need separate treatment. Increase `n_boot` until the SE
and percentile endpoints are stable and report the value used.

## Deterministic pairwise fitting

`fit_pairwise` estimates the same A/C/M fractions from binary relative-pair
probabilities without sampling latent liabilities:

```python
from ltpred import fit_pairwise
pw = fit_pairwise(fitting_families, components=("A", "C"), sampling="population")
pw.components, pw.residual
pw.se, pw.inference_status
```

Components need independent identifying contrasts among **observed** pairs.
Fractions are nonnegative and sum to at most `1 - eps` (`1e-6` by default).
Failed, infeasible or nonstationary optimization raises. IPW weights are
normalised to mean one; the weighted marginal screen does not prove joint
positivity or weight correctness.

For an interior fit, `pw.covariance` follows `pw.component_order` and `pw.se`
is its diagonal SD. The sandwich uses family-level score variation, so pairs
sharing a member are not treated as independent. It is conditional on supplied
thresholds and weights and is asymptotic, not a finite-sample calibration
claim. At a component/residual boundary (within `max(1e-7, 10*tol)`),
`at_boundary` is true and covariance/SEs are `NaN`; `inference_status` also
distinguishes insufficient clusters or information. Ordinary normal intervals
and likelihood-ratio tests are then inappropriate. Bootstrap/profile inference
at a boundary needs its own calibration.

## Joint heritability and cross-trait correlations

```python
from ltpred import fit_pairwise_multi
joint = fit_pairwise_multi(
    two_trait_families, components=("A", "C", "M"), sampling="population")
joint.h2, joint.rg, joint.re
joint.se["h2"], joint.se["rg"], joint.se["re"]
joint.components["C"], joint.correlations["C"]
joint.inference_status
```

Each member needs one bound per trait. Thresholds may differ between traits
but must be common across people within each trait. Missing phenotypes use
`(-inf, inf)`; see the [input recipe](data-preparation.md#preparing-multiple-traits-for-covariance-fitting).

With phenotype-major coordinates, the fitted model is

\[
\Sigma = G\otimes A + S\otimes C + T\otimes M + E\otimes I,
\qquad \operatorname{diag}(G+S+T+E)=1. \tag{1}
\]

`components=("A",)` omits S and T. Every component is PSD and residual E has
eigenvalues at least `eps`. C means full-sibship environment; M describes
spousal resemblance, not the inheritance consequences of assortative mating.

`h2 = diag(G)`, `rg` standardises G and **`re` standardises residual E**.
Shared-environment correlations are separate in `correlations["C"]` and
`correlations["M"]`. `rp` sums all components, so `rp - genetic_cov` is total
non-genetic covariance and differs from `env_cov` when C/M is present. These
are model decompositions, not causal effects.

Only jointly observed coordinates contribute. Missingness must preserve pair
distributions; imputing a completely missing relative creates no identifying
contrast. Family-cluster sandwich SEs and the correlation delta method are
conditional on thresholds and weights. `parameter_order` indexes free
covariance entries and `parameter_covariance` their joint uncertainty. A
boundary in any component withholds all normal SEs; the tolerance is
`max(1e-7, 10*sqrt(tol))`. Correlations at or below that variance are `NaN`.

`benchmarks/bench_pairwise_multi.py` covers both correlation signs and nulls,
A+C+M, MCAR, IPW and omitted shared environment. Its evidence is limited to
the tested designs. Multi-trait scoring is A+E, so do not pass a C/M fit's `rp`
as `full_corrmat`: that would incorrectly treat shared covariance as residual E.

## Ascertained samples

The HE dose-response (`bench_ascertainment.py`, nuclear families, true h² =
0.5, K = 0.05, N = 2,000, two replicates) shows why selection matters:

| realised case share ÷ assumed K | 0.95× | 1.51× | 1.96× | ≥ 2× |
|---|---:|---:|---:|---:|
| fitted `h²` | 0.420 | 0.984 | 0.998 | 1.000 |

This is bias, not just noise: the proband-ascertained arm remains at +0.500
bias from N = 1,000 to 10,000 and across initial values. See
[RESULTS §29](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#29-ascertainment-what-the-fitters-do-on-selected-samples-bench_ascertainmentpy).

### When you can correct it: `sampling="ipw"`

For a known design with positive inclusion probability `π` in every stratum,
pass `weights = 1 / π`:

```python
import numpy as np
from ltpred import fit_heritability

# 50/50 case/control cohort from population prevalence K = 0.05:
keep_p = 0.05 * 0.5 / (0.5 * 0.95)
weights = np.where(proband_is_case, 1.0, 1.0 / keep_p)
fit = fit_heritability(families, sampling="ipw", weights=weights)
```

Augmentation within a given family is already conditional on its statuses;
selection breaks the **mix** of families, which the weights restore. In the
benchmark this returns case-enriched fits from the boundary to near the true
0.5.

Two limits remain:

- **Positivity.** No finite weight reconstructs a stratum with zero inclusion probability.
  Ascertainment through an affected proband is the standard failure. Weighted
  marginal case rates provide a diagnostic warning; they cannot establish
  invalid weights or joint positivity.
- **Efficiency.** At K = 0.05 a 50/50 cohort needs weights up to 19×, greatly
  reducing effective sample size. IPW buys accuracy with precision.

ltpred cannot infer weights from the data. Family-history selection is
correctable only when every complete observed family pattern has a known,
strictly positive inclusion probability.

!!! warning "A scale correction is not a selection model"

    A Lee observed→liability factor depends only on `(K, P)` and cannot repair
    these fits: at fixed K, true h² = 0 and 0.5 can both return 1.000, so no
    invertible constant maps both back. The fitters already estimate on the
    liability scale. Use `observed_to_liability_h2` only to convert an
    observed-scale estimate from another method.

## Sensitivity to the assumed heritability

```python
import numpy as np
from ltpred import estimate_liability
grid = [0.3, 0.4, 0.5, 0.6, 0.7]
scores = [estimate_liability(families, h2=h).est["genetic"] for h in grid]
min_corr = np.corrcoef(scores)[np.triu_indices(len(grid), 1)].min()
```

In the calibration benchmark, assumed `h²` 0.2–0.8 barely changed ranking
(corr with true g 0.429–0.431) while the calibration slope moved from 2.24 to
0.68: it acted mostly as a near-linear rescaling in that design, not as proof
of rank invariance. Probe prevalence/CIP sensitivity by rebuilding the bounds
and comparing scores. Low cross-setting correlation is a reason to seek a
better external estimate or, only under the sampling contract, fit one.

## Variance components (A / C / M)

```python
from ltpred import fit_variance_components
vc = fit_variance_components(families, ("A", "C"), sampling="population")
vc.components["A"], vc.components["C"], vc.residual
```

C is identified only by full-sib pairs. `vc.se` has the same Monte-Carlo-only
meaning as `h2_se`; use `bootstrap_fit` for sampling uncertainty. With
`("A",)` the result matches `fit_heritability`.

M loads on genetically unrelated mate pairs (parents and grandparent couples).
It describes spousal resemblance from shared adult environment and/or
assortative mating but is not a generative assortative-mating model. Because
mates have A = 0, omitting M usually biases A much less than omitting C.
Every identifying pair set must be present. Dominance is unsupported because
the role bank provides no independent dominance kernel. C and M kernels are
PSD equivalence-class partitions with unit diagonal; residual variance is
`1 - sum(components)`, floored at `eps`.

## Unsupported research prototypes

HE multi-trait genetic-correlation and onset-age-decay fitting, common-factor
and genetic-nurture models, MCEM and parametric-bootstrap tests live in
[`research/advanced_fitting.py`](https://github.com/bvilhjal/ltpred/blob/main/research/advanced_fitting.py).
They are available only from a source checkout and are not part of the
supported distribution. See [research extensions](research.md).
