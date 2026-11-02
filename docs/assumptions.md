# Assumptions, checklist & pitfalls

What the model does and does not represent, a pre-flight checklist for a production
run, and the common mistakes.

## Modelling assumptions

By default, the high-level predictor uses **additive genetic sharing only**: every
off-diagonal entry is `shared_DNA × h²` (the optional `c2`/`m2` shared-environment
components below add their own off-diagonals). It assumes jointly Gaussian liabilities,
correct relationships and diagnoses, and correctly specified prevalence/CIPs and
liability-scale `h²`. Under its standard model, individual-specific residuals are
independent of relatives' additive genetic values and of each other.

Diagnosis quality and family-history reporting are therefore part of the
observation model, not clerical details. An Alzheimer-disease proxy-GWAS analysis
documented bias from survival and non-random participation in parental-history
surveys ([Wu et al. 2024](https://doi.org/10.1038/s41588-024-01963-9)), while a
broader psychiatric-genetics Perspective argues that shallow EHR or self-reported
phenotypes can carry heritable confounding
([Cai et al. 2026](https://doi.org/10.1038/s41588-025-02465-y)). Liability
modelling alone cannot correct a biased diagnosis or reporting process.

The age-dependent formulation assumes a non-decreasing cumulative-incidence curve
(hence a non-increasing liability threshold) and, in the current high-level model,
the same liability covariance and genetic architecture across age at diagnosis,
sex and cohort. In LT-FH++/ADuLT those variables alter thresholds; in base PA-FGRS
they alter a censored control's mixture weight through `K_i`. They do not alter
`h²` or genetic correlations. Follow-up/censoring must be represented by a
defensible observation model; independent censoring is required for ordinary
Kaplan–Meier risk estimates, and with competing death the thresholds need a
deliberate choice between net and crude incidence
([the estimand choice](cip-estimation.md#the-estimand-choice-the-most-important-decision-on-this-page)).
The PA-FGRS censored-control mixture inherits both requirements: its
weight `(K_pop − K_i)/K_pop` reads a future case's not-yet-onset probability off
the stratum's population CIP, which assumes censoring is non-informative given
the stratum and onset timing among future cases independent of liability (see
[algorithm.md](algorithm.md#base-pa-fgrs-lifetime-cases-and-censored-controls)).
When that independence fails (liability-dependent onset, ρ = 0.6), the mixture
still leaves ranking intact and shifts calibration the way it is supposed to;
it is the pinned case encoding that then over-conditions (RESULTS.md §16).

Gibbs targets the truncated-Gaussian conditional moments by Monte Carlo. PA is a
deterministic sequential-moment approximation when several interval observations
are folded in (exact for one truncation or point conditioning), so unusual family
structures should be checked against Gibbs.

Shared household/cultural transmission, dominance/epistasis, indirect genetic
effects and generative assortative mating are not included. Assortative mating is
not an environmental component: it changes genetic covariances among mates and
descendants. A directional maternal effect is likewise not generally represented
by a symmetric shared-environment matrix. When omitted processes contribute to
familial aggregation, read the output as the **additive-model projection of the
family history**, not a pure causal genetic value.

PA can operate on a symmetric positive-semidefinite covariance when its
non-deterministic interval coordinates are linearly independent after pin
conditioning. Dependent intervals on singular support are rejected, including
compatible ones: sequential moment projection cannot enforce that support.
Singular target or unobserved rows alone are permitted. In contrast,
`gibbs_params` and `rtmvnorm_gibbs` require a **strictly positive-definite**
covariance so their conditional variances exist. Gibbs scoring supplements
coordinate sweeps with joint Gaussian moves for strongly coupled blocks.
Its batch-means SE still measures estimated sampling precision, not proof
of stationarity; check difficult problems with independent seeds, longer
burn-in/chains and a deterministic reference where available.
The high-level estimators nudge
a numerically singular assembled covariance to strict positive-definiteness and
warn when they do so. Which entry points accept `c2`/`m2` is listed under
[Scoring](estimation.md#shared-environment-components-c-and-m); multi-trait
scoring is A+E only.

All common-threshold fitters require identifying contrasts among **jointly
observed** phenotypes. Missing relatives cannot identify a component merely
because the model can impute their liabilities. The joint fitter's `re` is
the within-person residual E correlation; C/M correlations are separate.
Omitting shared environment can distort the genetic/residual decomposition.
These model components do not identify causal genetic or environmental effects.

The family-data fitters assume **independent, non-overlapping families**, and
one of two sampling contracts. When members carry `pid`, a person who appears
in more than one family is rejected rather than silently pooled. `sampling="population"` for an unascertained
sample — *screened for gross marginal inconsistency* with your data, since
the thresholds assert a prevalence the observed case rates should match.
`sampling="ipw"` with per-family
`weights` for a sample selected on observed status with a **known, strictly
positive** inclusion probability (case/control cohorts, biobank case
enrichment); see [Inference](inference.md#ascertained-samples).

Overlapping pedigrees invalidate the iid-family moments, and bootstrap does not
correct selection bias. Selection on family history can be reweighted only when
the inclusion probability of every complete observed family pattern is known
and strictly positive. A design that samples *no* families from some stratum
cannot be corrected — ascertainment through an affected proband is the standard
example, where inclusion probability is zero for unaffected probands.

## Real-data checklist

First pick a **use** ([Getting started](quickstart.md#choose-the-analysis), Table 1): **I** family-history
risk prediction (own status out; optional PGS afterwards), **II** a
quantitative GWAS phenotype (own status in), or **III** architecture /
relationships / aetiology from `h²`, `r_g`, residual `r_e` and/or the CIP. Use III can skip
pedigree scoring; where uses I and II part company is item 8 below.

Then, before running a production analysis:

1. Validate the **diagnosis and family-history source**: case definition,
   reporting accuracy, missingness, follow-up, and participation/selection process.
2. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
3. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period where
   incidence differs. With independent right censoring and no competing events,
   `1 − Kaplan–Meier` estimates risk; with competing death or diagnoses,
   Aalen–Johansen gives the crude curve and Kaplan–Meier with those events
   censored the net curve; decide which the thresholds need
   ([the estimand choice](cip-estimation.md#the-estimand-choice-the-most-important-decision-on-this-page)).
4. Make each CIP age grid cover the analysed onset/follow-up ages and supply
   `k_pop` explicitly unless its final value is a defensible lifetime prevalence;
   the helper holds endpoint values constant outside the grid.
5. Convert `h²` to the **liability scale** (`observed_to_liability_h2`),
   passing the actual study case fraction as `prop_cases` — the ascertainment
   correction matters for anything but a representative sample.
6. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
7. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator raises on duplicates).
8. Decide whether to **condition on the proband's own status**. Include `o` for
   use II (a diagnosis-derived GWAS phenotype); for use I (prospective
   prediction/classification of that diagnosis) keep `o` and its `pid` with
   uninformative `(-inf, inf)` bounds (omitting `o` is allowed only when no member
   carries a `pid`).
9. Decide **case encoding** — onset-pinned (`thresholds_from_cip`,
   `case_mode="pin"`), lifetime-interval (base PA-FGRS), or age-specific
   interval (`case_mode="interval"`; with the mixture, an age-dependent
   PA-FGRS-style variant) — and record it. Pin only when you believe onset is
   the CIP inverse of liability; under partial or noisy onset dependence the
   pin over-conditions and the lifetime interval is the safer encoding
   (RESULTS.md §16). The age-specific interval `[T(onset), ∞)` keeps most of
   the pin's ranking gain (§3) but over-disperses calibration (slope ≈ 0.83, §16).
10. Choose **Gibbs vs Pearson–Aitken**; for unusual no-mixture pedigrees cross-check
    PA against Gibbs. The PA-FGRS censoring mixture has no Gibbs implementation.
11. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch). Prefer
     non-overlapping target families; if related targets remain, use and validate an
     association method that handles their relatedness, shared family-history
     phenotype, case-control imbalance, and tail behaviour
     ([Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)).
12. For built-in family-data **fitting or bootstrap inference**, require
    independent, non-overlapping families (a `pid` that appears in more than
    one family is rejected), and match the contract to the design:
    `sampling="population"` for an unascertained sample, or `sampling="ipw"`
    with `weights = 1 / P(family sampled)` when selection was on observed status
    with a known positive probability for every stratum. Check that your
    observed case rates plausibly match the prevalence behind your thresholds.
    The fitter raises only on gross mismatch (about 1.67× at K = 0.05 with
    1,500 families), and a match does not prove joint-pattern positivity. A
    mismatch may equally mean the prevalence is wrong rather than the sample
    selected. Neither contract covers proband-ascertained families; those need
    an estimator that models the selection.
13. For **joint covariance fitting**, supply one common case/control threshold
    per trait, model defensible shared-environment components, and inspect
    `inference_status`. Family-cluster SEs condition on the thresholds and
    weights; any covariance boundary withholds normal SEs. Missingness must
    preserve the modelled pair distributions, and zero-variance correlations
    are undefined. See the [joint inference contract](inference.md#joint-heritability-and-cross-trait-correlations).

## Pitfalls

- **`h²` must be liability-scale.** Convert observed-scale estimates first.
- **Ages mean different things by status.** For `thresholds_from_cip`,
  `age_thresholds` and `pa_thresholds`, `age` is the **age of onset** for cases
  and the **age at last follow-up** for controls.
- **Number repeated relatives** (`s1`, `s2`) — an unnumbered duplicate role
  collides.
- **The proband's optional observed-status role is `o`**, not `g`. `g` (what you
  estimate) is added for you.
- **Prevalence and CIPs should match the population** the thresholds refer to;
  stratify by sex/birth-year if your incidence differs across strata (one CIP
  per stratum when building the bounds with `thresholds_from_cip`; the
  per-person `K_i` matters only for the PA mixture, `use_mixture=True`).
- **`use_mixture=True` needs `K_i`/`K_pop`** on the members (from `pa_thresholds`
  or `thresholds_from_cip`) — otherwise it raises. In mixture mode, age enters
  through `K_i`; the implementation splits at the lifetime threshold and uses
  the finite control bound only as a censoring flag, so an age-specific control
  bound from these helpers is not applied twice. Their age-specific **case**
  bounds (pins or intervals) nevertheless remain a variant of base PA-FGRS.
- **Missing or uninformative family observations** leave less conditioning
  information. If `o` is included, the estimate then relies mostly on the
  proband's own status; if `o` is also unbound, it shrinks toward the prior mean.
  Low prevalence alone does not imply little information: an affected person with
  a rare disease can be highly informative.
- **Install `[fast]`** (Numba) for large runs; the pure-Python fallback is
  numerically identical but much slower.
