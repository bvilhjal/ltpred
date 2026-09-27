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

## Algorithms in brief

Each algorithm is stated in full in the docstring of the module named with it;
Algorithms G, P and M carry the step names of the [methods report](#methods-report).
Steps are given as implemented.

**Algorithm G (Gibbs; `ltpred.gibbs`, `ltpred.estimate`).** With $Q=\Sigma^{-1}$,
coordinate $j$ is drawn from $N(\sum_{i\ne j}P_{ij}x_i,\ \mathrm{sd}_j^2)$ truncated
to its bounds, where $P_{ij}=-Q_{ij}/Q_{jj}$ and $\mathrm{sd}_j^2=1/Q_{jj}$.
**G1.** Group families by role set. **G2.** Integrate out coordinates that are
unbounded in every family of the kernel call; the chain runs on the rest, $y$,
and the target contributes $\mathbb E[g\mid y]$ with posterior variance
$\operatorname{Var}(\mathbb E[g\mid y])+\operatorname{Var}(g\mid y)$. **G3.**
Start each coordinate at its marginal truncated median; hold pins. **G4.** Sweep
by inverse CDF (a log-scale tail construction beyond $|z|\approx38.5$). **G5.**
After burn-in, stream sums, squares and batch means ($b=\max(\lfloor\sqrt{n}\rfloor,2)$);
store no draws. **G6.** Repeat rounds of fresh chains for unconverged families,
pooling sums, until every batch-means SE is at most `tol` or `max_rounds` is
reached (then warn). Family seeds are fixed by the family's global index, so a
result does not depend on thread scheduling. Because G2's collapse set is
computed from the families in a call, chunked Gibbs can differ from the
unchunked array call by Monte Carlo error; chunked PA agrees to rounding.

**Algorithm P (Pearson–Aitken; `ltpred.pearson_aitken`).** Selecting coordinate
$i$ from $N(m_i,v_i)$ to moments $(m^\star,v^\star)$ updates the others by
$m_j\leftarrow m_j+\Sigma_{ji}(m^\star-m_i)/v_i$ and
$\Sigma_{jk}\leftarrow\Sigma_{jk}+\Sigma_{ji}\Sigma_{ik}(v^\star-v_i)/v_i^2$.
**P1.** Put the target first (the object API sorts the other roles by name; array
and kinship callers keep their own order). **P2.** Drop absent rows and condition
on all pins jointly, $m_X=\Sigma_{XP}\Sigma_{PP}^{-1}p$,
$V_X=\Sigma_{XX}-\Sigma_{XP}\Sigma_{PP}^{-1}\Sigma_{PX}$, sharing $V_X$ between
families with the same observation mask; incompatible exact observations raise.
**P3.** Fold the remaining intervals last to first with their truncated-normal
moments. **P4.** Apply the target's own interval. **P5.** Return its mean and
variance. The result is exact for pins plus at most one interval (the target's own
included) and otherwise a sequential two-moment approximation that can depend on
fold order.

**Algorithm M (censored-control mixture; PA only).** A control observed below
its current-age threshold is a lifetime control or a not-yet-onset case. In P3
its moments are those of the two-component mixture split at the lifetime
threshold $T=\Phi^{-1}(1-K_{pop})$, with weight
$\pi=\Phi_{\rm below}/\{\Phi_{\rm below}+(1-\Phi_{\rm below})(K_{pop}-K_i)/K_{pop}\}$.
The factor $(K_{pop}-K_i)/K_{pop}$ assumes onset timing independent of liability
among eventual cases. A call containing a mixture row skips P2: pins, absent rows
and intervals are folded sequentially, without P2's compatibility checks.

**Algorithm Q (nuclear-family quadrature; `ltpred.quadrature`).** For an
additive, non-inbred nuclear family with unrelated parents and $h^2<1$, the
parental genetic values $z=(a_m,a_f)\sim N(0,h^2I)$ make all observed liabilities
conditionally independent. **Q1.** Use analytic shortcuts where they apply. **Q2.**
Condition $z$ on the pins exactly; with at most one interval left, return the
exact answer. **Q3.** Project onto the span of the interval rows' loadings (one or
two dimensions). **Q4.** Find the posterior mode by Newton's method with an
Armijo line search. **Q5.** Integrate on a Gauss–Hermite grid centred at the
mode: $\mathbb E[\mu(z)]$ and $\mathbb E[V(z)]+\operatorname{Var}[\mu(z)]$. **Q6.**
Double the nodes from 16 up to `quadrature_max_nodes` and accept when both of the
last two changes are within `quadrature_atol`; otherwise raise. The reported
error is a resolution change, not a certified bound.

**Algorithm K (kinship; `ltpred.covariance`, `ltpred._selected_kinship`).** Over
a parents-first order, $A_{ii}=1+\tfrac12A_{s_id_i}$ and
$A_{ij}=\tfrac12(A_{s_ij}+A_{d_ij})$ for $i$ later than $j$; an unknown parent
contributes zero. The dense route fills one row per person. The selected route
evaluates only requested pairs and their dependencies with an explicit stack and a
bounded cache; eviction changes the work, never the value. The entries are exact
binary fractions, so both routes agree bit for bit at realistic pedigree depths.

**Algorithm R (register driver; `ltpred.pipeline`).** **R1.** Validate the
inputs and each CIP once. **R2.** Build the parent graph and check it for cycles;
report unresolved parents. **R3.** For each proband, extract relatives within
`max_degree` plus the ancestors needed for exact kinship (closure-only).
**R4.** Form the observation set: under `use="prediction"` censor each relative
at their own age at the proband's `index_time`, and leave the proband's and
closure-only rows uninformative. **R5.** Convert records to pinned-onset bounds.
**R6.** When the positive-definiteness bound allows, keep only the target and
informative rows (exact marginalisation) and compute their kinship by the dense
or selected route of Algorithm K; otherwise use the full matrix. **R7.** Score
by PA (a family-free, non-inbred, unpinned proband uses the exact scalar ADuLT
moments instead). **R8.** Report
the per-proband observation counts and call-level diagnostics.

**Algorithm H (moment fit; `ltpred.fit`).** For
$\Sigma=\sum_c\theta_cK_c+(1-\sum\theta)I$: **H1.** Check non-overlapping IDs,
common thresholds and the case-rate screen. **H2.** Form the within-family pair
design and check its rank. **H3.** Initialise $\theta$ and the chains. **H4.**
Advance each family's latent-liability chain by `inner_sweeps` Gibbs sweeps.
**H5.** Update $\theta$ by the (weighted) Haseman–Elston regression of
$\ell_i\ell_j$ on the kernel entries. **H6.** Clip to $[\epsilon,1-\epsilon]$ and
damp. **H7.** After `n_iter` iterations, report the mean of the post-burn-in
trace; `h2_se` is its batch-means Monte Carlo SE, not a sampling SE.

**Algorithm L (pairwise likelihood; `ltpred.pairwise`, `ltpred.pairwise_multi`).**
**L1.** Screen as in H1. **L2.** Tabulate jointly observed within-family pairs
into weighted $2\times2$ tables keyed by their kernel row. **L3.** Evaluate the
composite likelihood from bivariate-normal cell probabilities. **L4.** Maximise
under the non-negativity and residual constraints and certify stationarity.
**L5.** Report the family-cluster sandwich $H^{-1}JH^{-1}$, withheld at a
boundary. The multi-trait fit parametrises $\Sigma=G\otimes A+S\otimes C+T\otimes M+E\otimes I$
with PSD components and unit total variance per trait, and derives $h^2$, $r_g$,
$r_e$ and their SEs by the delta method.

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
