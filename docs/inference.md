# Inference

Beyond estimating each proband's liability, ltpred can **fit the model** from the
family data: the liability-scale heritability, shared-environment variance
components, the genetic correlation between traits, a common-factor model over
those correlations, and parametric-bootstrap significance tests. These are optional
— skip straight to [estimation](estimation.md) if you already have an `h²`.

The meaning of a reported `se` depends on the method. For the default
Haseman–Elston/data-augmentation fits, it is a *within-dataset* Monte-Carlo
diagnostic, not across-dataset sampling uncertainty. The MCEM path instead
reports an approximate model-based SE, with the assumptions and caveats described
below. [`bootstrap_fit`](#family-cluster-uncertainty-bootstrap_fit) provides an
approximate family-cluster sampling interval when families are independent.

## Fitting heritability from the family data

If you don't have an external `h²`, you can **estimate it from family data** when
the bounds encode a coherent fitting observation model. `fit_heritability`
alternates a Gibbs augmentation of the latent liabilities with a damped
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
fit = fit_heritability(fitting_families)  # bounds from the fitting observation model
fit.h2, fit.h2_se                    # fitted liability-scale heritability (+ MC diagnostic)
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
bs = bootstrap_fit(families, lambda f: fit_heritability(f, seed=1).h2, n_boot=100)
bs.estimate, bs.se, (bs.ci_low, bs.ci_high)   # point, bootstrap SE, percentile CI
```

It wraps any of the fitters (pass a `lambda` that returns the quantity of interest,
e.g. `fit_variance_components(f, ("A","C"), seed=1).components["C"]` or
`fit_genetic_correlation(f, seed=1).rg[0,1]`); fix the estimator's `seed` so the
spread reflects family sampling, not sampler noise. It costs `n_boot`+1 fits.
The interval assumes iid, non-overlapping family clusters and a sampling design
compatible with the fitted model. It is not automatically calibrated under
case/control or family-history ascertainment, boundary parameters, or overlapping
pedigrees. The `n_boot=100` call above is a computational example, not a
recommended final precision: percentile endpoints can be visibly unstable with
so few resamples. Increase `n_boot` until the SE and interval endpoints are stable
for your analysis, and report the number of successful refits.

## Sensitivity to the assumed heritability

Because no single `h²` is uniquely correct, check how much the score actually
depends on it. `liability_sensitivity` sweeps a grid and reports the cross-setting
correlation of the estimates:

```python
from ltpred import liability_sensitivity
sens = liability_sensitivity(families, [0.3, 0.4, 0.5, 0.6, 0.7], method="pa")
sens.min_corr          # worst-case correlation of the score across the grid
sens.mean, sens.sd     # how the scale shifts with h²
```

In the documented benchmark, `min_corr` was high (≈0.97 across `h² 0.2–0.8` for
the tested pedigree): changing the assumed `h²` acted mostly like a near-linear
rescaling of the liability score. This is an empirical sensitivity result, not
rank invariance or a guarantee for other structures. A low `min_corr` is the
signal to pin `h²` down (with `fit_heritability`). Prevalence/CIP
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
vc = fit_variance_components(families, ("A", "C"))
vc.components["A"], vc.components["C"], vc.residual   # proportions of liability variance
```

`C` is identified only from **full-sib pairs**, so the families must contain them
(otherwise the fit raises). The same `h2_se` caveat applies — use `bootstrap_fit` for a CI.

### Onset-age decay

If the genetic correlation should **decay with the difference in age at onset**
between relatives (early- vs late-onset heterogeneity), use
`fit_genetic_correlation_decay` instead. Each member additionally carries `aod`
(onset age for cases, censoring age for controls), per trait:

```python
from ltpred import fit_genetic_correlation_decay
gd = fit_genetic_correlation_decay(families, phen_names=["adhd", "depression"])
gd.rg             # genetic correlation at equal onset age
gd.lambda_cross   # cross-trait decay rate lam (0 = the scalar model)
gd.lambda_within  # per-trait decay rates
gd.converged      # did the EM stabilise? (False -> raise n_em)
```

Useful options: `shared_lambda=True` ties all decay rates to one scalar
(fewer parameters, guaranteed PSD, less ridge -- the recommended default);
`shared_env=True` adds a shared-family environmental component `C` (a proper
onset-age ACE decomposition, so household environment isn't mistaken for
genetics); `n_starts` re-runs the EM from perturbed inits against the
multi-modal likelihood.

The fit is a Monte-Carlo **EM** (likelihood, not the moment regression), because
ascertainment truncation is itself age-dependent and confounds a moment estimator
-- see [the algorithm notes](algorithm.md#onset-age-structured-genetic-correlation).
It is **data-hungry**: the amplitude and the decay rate trade off along a
likelihood ridge, so it needs on the order of thousands of families and several
dozen EM iterations to recover `r_g` and `lam` (see
`benchmarks/bench_aod_decay.py`); with few families or little onset-age spread,
prefer the scalar `fit_genetic_correlation`.
Dominance is intentionally not offered. It requires a dominance relationship
kernel with contrasts linearly independent of `A` and `C`. MZ/DZ twin observations
can contribute useful contrasts within a richer design, but MZ/DZ pairs alone
cannot identify `A`, `C`, and `D` simultaneously. The current role bank does not
provide a dominance kernel. With `("A",)` alone the result matches
`fit_heritability`.

A second shared-environment component, `"M"` (couple / spousal environment), loads
on the genetically-unrelated **mate pairs** — the parents `(m, f)` and the
grandparent couples — so `fit_variance_components(families, ("A", "C", "M"))` fits
all three at once given a 3-generation pedigree (each component needs its
identifying pairs; the fit raises on a rank-deficient design). `M` is a
**descriptive spousal-resemblance component**: it absorbs shared adult environment
and some manifestations of assortative mating, but it is *not* a generative model
of assortative mating (which would also alter the genetic covariance among
offspring and across generations). Because mates have `A = 0`, omitting a real
`M` biases `A` **much less than omitting `C`** — so fit `M` to
quantify, or `test_variance_component(families, "M")` to test, spousal resemblance
for its own sake rather than to de-bias `h²`. The shipped `C` and `M` kernels are
equivalence-class partitions, which guarantees that they are positive
semi-definite (PSD); equivalence classes are sufficient, not necessary. Any future
kernel must be symmetric and PSD. Every fitted kernel has diagonal one, so each
component also consumes that fraction of marginal liability variance and the
individual residual is `1 - sum(components)`. The naive vertical parent-offspring
indicator considered here is non-PSD and is rejected.

Pass `method="mcem"` for the package's **approximate Monte-Carlo EM-style
likelihood fit** instead of the moment regression (`method="reml"` is a deprecated
alias; this is not restricted ML):

```python
vc = fit_variance_components(families, ("A", "C"), method="mcem")
vc.components["A"], vc.se["A"]  # estimate + approximate OPG/BHHH information SE
vc.loglik, vc.aic                # Monte-Carlo GHK log-likelihood + AIC diagnostic
```

The implementation runs a finite `n_iter` trajectory with a fixed damping
coefficient and averages the post-burn-in iterates; it does not stop on an EM
likelihood-convergence criterion. Consequently the component estimates, the
OPG/BHHH information SE, and the GHK log-likelihood/AIC are all Monte-Carlo and
finite-iteration approximations. Use them as diagnostics, check stability across
seeds and iteration settings, and do not replace a family-cluster bootstrap with
the reported `se` unless its repeated-sampling calibration has been established
for your design. AIC can compare nested fits descriptively, but it is particularly
fragile near a variance-component boundary; use `test_variance_component` when
its conditional parametric-bootstrap assumptions are appropriate.

## Genetic correlation between traits

For **several traits**, `fit_genetic_correlation` estimates the genetic
correlation `r_g` between them (and each trait's `h²`). Each member must carry one
interval per trait (length-`n_pheno` `lower`/`upper`, as for the
[multi-trait estimator](estimation.md#multiple-correlated-traits)):

```python
from ltpred import fit_genetic_correlation
gc = fit_genetic_correlation(families, phen_names=["adhd", "depression"])
gc.rg          # (P, P) genetic-correlation matrix (the headline)
gc.re          # (P, P) environmental correlation (phenotypic corr not from shared genes)
gc.h2, gc.rp   # per-trait heritabilities; phenotypic (full-liability) correlations
```

The phenotypic correlation splits into genetic and environmental parts —
`gc.rp` **equals** `gc.genetic_cov + gc.env_cov` — so you get both `r_g` and
`r_e`. Both covariances are positive semi-definite and every reported correlation
is derived from that one pair, so the returned object is a coherent model you can
simulate from or hand to `fit_genetic_factor` directly. It needs related pairs (the
genetic correlation is carried by the cross-relative, cross-trait resemblance). It
was approximately unbiased near the null and mildly attenuated at large `|r_g|`
in the repository benchmarks; use
`bootstrap_fit` for a CI.

## A genetic common-factor model

With several traits, ask whether one genetic factor explains the `r_g` among them —
a **common-factor model** `r_g ≈ Λ Λ' + Ψ` (Genomic-SEM-lite):

```python
from ltpred import fit_genetic_correlation, fit_genetic_factor

gc3 = fit_genetic_correlation(
    three_trait_families,
    phen_names=["adhd", "depression", "anxiety"],
)
fa = fit_genetic_factor(gc3)             # or pass a valid (P, P) matrix
fa.loadings                              # (P, 1) each trait's correlation with the factor
fa.communality                           # per-trait genetic variance the factor explains
fa.srmr, fa.prop_explained               # off-diagonal misfit; fraction of r_g captured
```

`fit_genetic_factor` fits the loadings by MINRES (minimising the **off-diagonal**
residuals, so the factor explains the cross-trait correlations, not each trait's own
variance). `srmr` is a descriptive in-sample misfit measure; values around
0.05–0.08 are sometimes used as informal heuristics, not as a calibrated test of
factor number. Communalities are constrained to `[0, 1]`; a value at 1 is a
Heywood boundary rather than evidence of a perfect measurement. A single factor
needs `P ≥ 3` traits. Although the usual parameter count gives zero formal degrees
of freedom at `P = 3`, sign and communality constraints can still leave non-zero
residual misfit, so the fit is not universally exact. This is a descriptive
decomposition of a point-estimate `r_g`; bootstrap the
`fit_genetic_correlation → fit_genetic_factor` pipeline for uncertainty.

For `n_factors > 1`, factor signs, order and rotation are not identified. Align
bootstrap loading matrices to the reference solution (for example by permutation,
sign matching and Procrustes rotation) before elementwise intervals, or bootstrap
rotation-invariant summaries instead.

## Is a component / correlation significant?

The frequentist analog of the twin-SEM likelihood-ratio test ("is `C` in the
model?", "is the genetic path non-zero?") is a **parametric bootstrap**:

```python
from ltpred import test_variance_component, test_genetic_correlation
tc = test_variance_component(families, "C", n_boot=200)   # H0: c² = 0
tg = test_genetic_correlation(two_trait_families, n_boot=200)  # H0: r_g = 0
tc.p_value, tc.estimate, tc.null      # p-value, observed statistic, null distribution
```

Each fits the full model, then simulates `n_boot` datasets under the null — for
`C`, an `A`-only model; for `r_g[i,j]`, a pair-specific null with only that genetic
correlation set to zero. Each trait's `h²`, the environmental covariance, and
nuisance genetic correlations are preserved as far as positive-semidefinite
coherence permits. Simulation uses the same pedigrees and fixed thresholds, then
refits and locates the observed statistic in that plug-in null. The resulting
p-value is an approximate conditional parametric-bootstrap calibration: it
depends on fitted nuisance parameters, a correctly specified null and observation
model, independent family clusters, and enough bootstrap replicates. It must not
reuse an outcome-derived onset threshold after simulating a different status. The
current helpers therefore require **case/control-style bounds fixed independently
of status**; pinned age-of-onset bounds raise. A common prevalence threshold is
accepted automatically. For individualized thresholds fixed from baseline
covariates, pass `thresholds_are_status_independent=True` only when that statement
is genuinely true; never use it for a threshold derived from age of onset. It
costs about `n_boot` refits. With the plus-one Monte-Carlo p-value used here, `B`
null replicates give a minimum attainable p-value of `1 / (B + 1)`; for example,
`n_boot=200` cannot resolve a p-value below about `0.005`. Increase `n_boot` for
smaller target significance levels and report the simulation count.
