# Model and algorithms

This is the model reference. [Getting started](quickstart.md) chooses the analysis;
[Scoring](estimation.md) and [Fitting](inference.md) document the calls. Full
algorithm derivations and historical simulation tables live in the
[typeset methods report](#methods-report).

## The liability-threshold model

For a non-inbred individual, let full liability be the sum of additive genetic
liability and independent residual liability:

$$
\ell_i = a_i + e_i,\qquad
\operatorname{Var}(a_i)=h^2,\quad
\operatorname{Var}(e_i)=1-h^2,\quad
\operatorname{Var}(\ell_i)=1. \tag{1}
$$

The model is jointly Gaussian across relatives. A case exceeds the threshold
corresponding to population prevalence $K$:

$$
T=\Phi^{-1}(1-K). \tag{2}
$$

Here `h2` is liability-scale heritability. It is supplied to scoring; it is not
estimated as a side effect of constructing a score.

## The estimand

Let $D_F$ denote the observed family records encoded as liability bounds (or a
censoring mixture), $A$ the additive relationship matrix and $K(\cdot)$ the
cumulative-incidence model. LTpred targets

$$
\mu_i=\mathbb E[a_i\mid D_F,A,h^2,K(\cdot)]. \tag{3}
$$

The returned estimate approximates this model-defined posterior mean. It is
neither the realised generating value $a_i$, a SNP polygenic score, nor a disease
probability. Including the proband's diagnosis gives a quantitative phenotype
for a GWAS. For family-history prediction, leave that observation uninformative
and use only relatives' records available at the prediction landmark.

`LiabilityResult.var` is posterior genetic-liability variance;
`LiabilityResult.se` is Monte Carlo error in the estimated mean. PA has zero
Monte Carlo error but can have approximation error. More Gibbs draws can reduce
Monte Carlo error; they do not remove posterior uncertainty.

## Family covariance

`A` is twice kinship: parent–offspring and full-sibling entries are 0.5;
grandparent and half-sibling entries are 0.25. Role labels encode conventional
relationships; the pedigree route constructs `A` from parent links, including
inbreeding. Optional `C` and `M` kernels represent full-sibship and couple
shared environment. Before standardisation, write

$$
V=h^2A+c^2C+m^2M+(1-h^2-c^2-m^2)I. \tag{4}
$$

The component weights are nonnegative and sum to at most one. The kinship API
standardises each full liability using $s_j=\sqrt{V_{jj}}$, so its covariance
is $V_{jk}/(s_js_k)$ and equation (2) retains its prevalence meaning. The genetic
target is scaled by the same $s_i$. In the additive-only inbred case,

$$
\operatorname{Var}(a_i/s_i)=
\frac{h^2 A_{ii}}{1+h^2(A_{ii}-1)}. \tag{5}
$$

For non-inbred pedigrees without shared components, this reduces to equation (1).
`kinship_from_pedigree` returns `A`, not the standardised full-liability
covariance; `construct_covmat_from_kinship` performs that conversion.

### Adding environmental covariance to improve prediction

Shared environment changes which part of familial resemblance is attributed to
genetics. It can improve prediction of full liability while reducing the genetic
score. Supply `c2`/`m2` to single-trait role scoring; on the kinship route also
supply the aligned `c_kernel`/`m_kernel`. Additive relatedness alone cannot
identify whether two equally related people share a sibship or household.

The [scoring reference](estimation.md#shared-environment-components-c-and-m)
owns the kernel conventions and runnable examples. Multi-trait scoring supports
A+E only; fitting C/M components does not extend that scorer.

### Pedigree selection and observation selection

The register driver extracts relatives within `max_degree` and retains the
ancestors required for exact kinship. Those extra ancestors are structural:
their diagnoses remain uninformative unless explicitly requested. In prediction
mode, each relative's follow-up is censored at their own attained age at the
proband's calendar landmark. This is not the proband's age applied to everyone.
See [data preparation](data-preparation.md#beyond-the-role-grammar-arbitrary-pedigrees)
for calendar alignment, unresolved parents and diagnostic counts.

## Connection to selection index and BLUP

For continuously observed full liabilities $\ell_F$, Gaussian conditioning gives

$$
\mathbb E[a_i\mid\ell_F]
=\operatorname{Cov}(a_i,\ell_F)\operatorname{Var}(\ell_F)^{-1}\ell_F. \tag{6}
$$

For interval observations, take the conditional expectation again:

$$
\mathbb E[a_i\mid D_F]
=\operatorname{Cov}(a_i,\ell_F)\operatorname{Var}(\ell_F)^{-1}
  \mathbb E[\ell_F\mid D_F]. \tag{7}
$$

The selection-index weights are unchanged; the phenotypes become truncated-normal
means. The resulting score is nonlinear in binary and censored observations.
Gibbs samples the truncated distribution; PA approximates its moments. Equations
(6)–(7) use a nonsingular observed covariance; redundant exact observations need
support-aware conditioning, which the PA implementation checks explicitly.

## Thresholds and observation models

Method names primarily describe the observations, not different genetic models.
Table 1 separates that choice from the inference engine.

**Table 1. Observation models under the liability-threshold framework.**

| Model | Observations |
|---|---|
| LT-FH | One population prevalence threshold; proband and/or relatives' binary statuses |
| LT-FH++ | Person-specific age/sex/cohort CIP bounds, with relatives |
| ADuLT | The same personalised bounds for the proband alone |
| Base PA-FGRS | Lifetime case intervals plus an age-censored-control mixture, evaluated with PA |

For LT-FH++/ADuLT, controls lie below their follow-up-age threshold. Cases are
pinned at their onset threshold by default (`case_mode="pin"`); the alternative
`"interval"` puts them above it. A missing observation is `(-inf, inf)`, not a
control. `thresholds_from_cip` accepts empirical CIPs; `age_thresholds` is a
logistic demonstration. [Data preparation](data-preparation.md#getting-lowerupper-from-status-and-age)
owns the recipes, and [CIP estimation](cip-estimation.md) distinguishes net from
competing-risk incidence.

### Base PA-FGRS: lifetime cases and censored controls

In the published base encoding, cases use $[\Phi^{-1}(1-K_{pop}),\infty)$.
Age-specific incidence enters the weights of the censored-control mixture,
using per-person `K_i` and `K_pop`. The convenience helpers `pa_thresholds` and
`thresholds_from_cip(case_mode="interval")` instead use onset-age case thresholds;
with the mixture, they define an age-dependent PA-FGRS-style variant. That variant
is neither base PA-FGRS nor the published PA-FGRS_ADT specification.
The mixture is available only with PA. It is also distinct from a
register-standardised family genetic risk score.

## Inference engine 1: Gibbs sampler

Gibbs cycles through conditional univariate normal distributions truncated to
each person's bounds. After burn-in, it averages retained draws and estimates
Monte Carlo error by batch means. The sampler covers single- and multi-trait
scoring, but not the PA-FGRS mixture. Seeds and retained-draw controls determine
reproducibility and precision; family streams remain independent when scoring
identical records or using chunked inputs.

The [scoring reference](estimation.md#choosing-gibbs-vs-pearsonaitken) owns sampler
controls and convergence diagnostics. Sampler convergence does not check the
prevalence, observation model or sampling design.

## Inference engine 2: Pearson–Aitken

PA processes truncated coordinates sequentially, updating the joint mean and
covariance with each coordinate's truncated moments. This is a deterministic
two-moment approximation; the result can depend on processing order. Exact pins
are conditioned jointly and uninformative coordinates can be marginalised before
the sequential updates. Repeated deterministic bound rows can share a result.

PA is the single-trait default. The restricted [nuclear-family quadrature](estimation.md#nuclear-family-quadrature)
route provides deterministic numerical integration for additive, non-inbred
nuclear families without shared-environment kernels or a mixture. Check its
refinement diagnostics; its finite quadrature rule is not exact arithmetic.
[Numerical checks](validation.md) demonstrate agreement and its limits.

## Fitting the covariance (heritability)

Scoring conditions on covariance parameters. Fitting estimates them from
independent, non-overlapping families with common case/control thresholds and
identifying observed relationships. Supported sampling is population sampling or
inverse-probability weighting with known, strictly positive family inclusion
probabilities. A marginal case-rate guard can reject gross inconsistencies;
passing it does not certify the sampling design.

`fit_heritability` and `fit_variance_components` use a stochastic moment fixed
point, not posterior sampling of the parameters. `fit_pairwise` and
`fit_pairwise_multi` maximise a composite likelihood from observed binary pair
patterns. Their family-cluster sandwich SEs are conditional on thresholds and
weights and are withheld at covariance boundaries. The [fitting reference](inference.md)
owns model choice, uncertainty, ascertainment and replicated evidence links.

## Multiple traits

Multi-trait scoring uses Gibbs with vector `h2`, `genetic_corrmat` and
`full_corrmat`. The last is full-liability correlation, not residual correlation.
For covariance fitting, `fit_pairwise_multi` estimates trait heritabilities,
genetic and residual correlations and optional C/M covariance. Its `re` describes
residual E only. With C/M fitted, total non-genetic covariance also includes those
components; do not fold them into an A+E scorer. See the
[joint-fitting contract](inference.md#joint-heritability-and-cross-trait-correlations).
Unsupported alternatives remain in [research extensions](research.md).

## Methods report

The [methods PDF](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
is the portable, detailed derivation and historical evidence snapshot, labelled
v0.7.4 (25 September 2026). It is not an additional onboarding guide or a timing
report for today's checkout. [Report sources and rebuild instructions](https://github.com/bvilhjal/ltpred/blob/main/report/README.md)
live with the PDF; [the benchmark ledger](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
owns the dated measurements and links to retained data and provenance.

Method citations are collected in
[CITATION.cff](https://github.com/bvilhjal/ltpred/blob/main/CITATION.cff) and the
report. [Assumptions](assumptions.md) provides the real-data checklist.
