# Assumptions, checklist & pitfalls

What the model does and does not represent, a pre-flight checklist for a production
run, and the common mistakes.

## Modelling assumptions

The high-level predictor uses **additive genetic sharing only**: every
off-diagonal entry is `shared_DNA × h²`. It assumes jointly Gaussian liabilities,
correct relationships and diagnoses, and correctly specified prevalence/CIPs and
liability-scale `h²`. Under its standard model, individual-specific residuals are
independent of relatives' additive genetic values and of each other.

Diagnosis quality and family-history reporting are therefore part of the
observation model, not clerical details. An Alzheimer-disease proxy-GWAS analysis
documented bias from survival and non-random participation in parental-history
surveys ([Wu et al. 2024](https://doi.org/10.1038/s41588-024-01963-9)), while a
broader psychiatric-genetics Perspective argues that shallow EHR or self-reported
phenotypes can carry heritable confounding
([Cai et al. 2026](https://doi.org/10.1038/s41588-025-02465-y)). These sources do
not imply that every register phenotype is biased. They show or argue, respectively,
that liability modelling alone cannot correct a biased diagnosis or reporting
process.

The age-dependent formulation assumes a non-decreasing cumulative-incidence curve
(hence a non-increasing liability threshold) and, in the current high-level model,
the same liability covariance and genetic architecture across age at diagnosis,
sex and cohort. In LT-FH++/ADuLT those variables alter thresholds; in base PA-FGRS
they alter a censored control's mixture weight through `K_i`. They do not alter
`h²` or genetic correlations. Follow-up/censoring must be represented by a
defensible observation model; independent censoring is required for ordinary
Kaplan–Meier risk estimates, and competing events require a cumulative-incidence
estimator. The PA-FGRS censored-control mixture inherits both requirements: its
weight `(K_pop − K_i)/K_pop` reads a future case's not-yet-onset probability off
the stratum's population CIP, which assumes censoring is non-informative given
the stratum and onset timing among future cases independent of liability (see
[algorithm.md](algorithm.md#base-pa-fgrs-lifetime-cases-and-censored-controls)).
A liability-dependent onset arm (ρ = 0.6) leaves ranking intact and still
shifts calibration the way the mixture is supposed to; it is the pinned
case encoding that then over-conditions (RESULTS.md §16).

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

PA can operate on a symmetric positive-semidefinite covariance, but
`gibbs_params` and `rtmvnorm_gibbs` require a **strictly positive-definite**
covariance so their conditional variances exist. The high-level estimators nudge
a numerically singular assembled covariance to strict positive-definiteness and
warn when they do so. The fitters ship descriptive sibship `C` and mate/couple
`M` kernels, and the single-trait high-level predictor accepts them as `c2`/`m2`
on both the role/object and array paths. The high-level multi-trait dispatcher
rejects nonzero `c2`/`m2` rather than silently dropping them, pending a defined
cross-trait component covariance; see
[algorithm.md](algorithm.md#adding-environmental-covariance-to-improve-prediction)
and [Inference](inference.md#variance-components-a-c-m).

The family-data fitters assume **independent, non-overlapping families**, and
one of two sampling contracts. `sampling="population"` for an unascertained
sample — now *verified* against your data, since the thresholds assert a
prevalence the observed case rates must match. `sampling="ipw"` with per-family
`weights` for a sample selected on observed status with a **known, strictly
positive** inclusion probability (case/control cohorts, biobank case
enrichment); see [Inference](inference.md#ascertained-samples).

Everything else invalidates the iid-family moments, and bootstrap does not
correct it: overlapping pedigrees, selection on family history, and any design
that samples *no* families from some stratum — ascertainment through an
affected proband being the standard example, where inclusion probability is
zero for unaffected probands and no weighting can reconstruct them. The
severity is worth internalising: on ascertained families with a **true `h²` of
0**, the unguarded fitter returns **`h² = 1.0`**, and a case rate only 1.17×
the assumed prevalence already inflates `h²` by +0.12
([RESULTS.md §29](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).

## Real-data checklist

Before running a production analysis:

1. Validate the **diagnosis and family-history source**: case definition,
   reporting accuracy, missingness, follow-up, and participation/selection process.
2. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
3. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period where
   incidence differs. With independent right censoring and no competing events,
   `1 − Kaplan–Meier` estimates risk; with competing death or diagnoses, use a
   cause-specific cumulative-incidence estimator such as Aalen–Johansen.
4. Make each CIP age grid cover the analysed onset/follow-up ages and supply
   `k_pop` explicitly unless its final value is a defensible lifetime prevalence;
   the helper holds endpoint values constant outside the grid.
5. Convert `h²` to the **liability scale** (`observed_to_liability_h2`),
   passing the actual study case fraction as `prop_cases` — the ascertainment
   correction matters for anything but a representative sample.
6. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
7. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator now raises on duplicates).
8. Decide whether to **condition on the proband's own status**. Include `o` when
   intentionally constructing a diagnosis-derived GWAS phenotype; omit it or make
   it uninformative for prospective prediction/classification of that diagnosis.
9. Decide **case encoding** — onset-pinned (`age_thresholds`), lifetime-interval
   (base PA-FGRS), or age-specific interval (`pa_thresholds`, an age-dependent
   PA-FGRS-style variant) — and record it. Pin only when you believe onset is
   the CIP inverse of liability; under partial or noisy onset dependence the
   pin over-conditions, and most of the onset increment survives the interval
   `[T(onset), ∞)` (RESULTS.md §3 and §16).
10. Choose **Gibbs vs Pearson–Aitken**; for unusual no-mixture pedigrees cross-check
    PA against Gibbs. The PA-FGRS censoring mixture has no Gibbs implementation.
11. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch). Prefer
     non-overlapping target families; if related targets remain, use and validate an
     association method that handles their relatedness, shared family-history
     phenotype, case-control imbalance, and tail behaviour
     ([Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)).
12. For built-in family-data **fitting or bootstrap inference**, require
    independent, non-overlapping families, and match the contract to the design:
    `sampling="population"` for an unascertained sample, or `sampling="ipw"`
    with `weights = 1 / P(family sampled)` when selection was on observed status
    with a known positive probability for every stratum. Check that your
    observed case rates actually match the prevalence behind your thresholds —
    the fitter now checks this for you and raises, but a mismatch may equally
    mean the prevalence is wrong rather than the sample selected. Neither
    contract covers proband-ascertained families; those need an estimator that
    models the selection.

## Pitfalls

- **`h²` must be liability-scale.** Convert observed-scale estimates first.
- **Ages mean different things by status.** For `age_thresholds`/`pa_thresholds`,
  `age` is the **age of onset** for cases and the **age at last follow-up** for
  controls.
- **Number repeated relatives** (`s1`, `s2`) — an unnumbered duplicate role
  collides.
- **The proband's optional observed-status role is `o`**, not `g`. `g` (what you
  estimate) is added for you. If `o` is omitted, it is inserted unbounded; use that
  path to avoid outcome leakage in disease prediction/classification.
- **Prevalence and CIPs should match the population** the thresholds refer to;
  stratify by sex/birth-year if your incidence differs across strata (pass the
  per-person `K_i`).
- **CIP interpolation is flat beyond its age grid.** It does not extrapolate an
  incidence trend; cover the analysed ages and do not equate the last observed CIP
  with lifetime prevalence unless the curve reaches that horizon.
- **`use_mixture=True` needs `K_i`/`K_pop`** on the members (from `pa_thresholds`
  or `thresholds_from_cip`) — otherwise it now raises rather than silently doing
  nothing. In mixture mode, age enters through `K_i`; the implementation splits at
  the lifetime threshold and uses the finite control bound only as a censoring flag,
  so an age-specific control bound from these helpers is not applied twice. Their
  age-specific **case** intervals nevertheless remain a variant of base PA-FGRS.
- **Missing or uninformative family observations** leave less conditioning
  information. If `o` is included, the estimate then relies mostly on the
  proband's own status; if `o` is also unbound, it shrinks toward the prior mean.
  Low prevalence alone does not imply little information: an affected person with
  a rare disease can be highly informative.
- **Install `[fast]`** (Numba) for large runs; the pure-Python fallback is
  numerically identical but much slower.
