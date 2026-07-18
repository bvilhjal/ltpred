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
estimator.

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

The low-level covariance entry points (`rtmvnorm_gibbs`, `pa_algorithm`,
`pa_estimate_batched`) accept a symmetric positive-semidefinite covariance, so a
valid shared-environment kernel can be added deliberately. The fitters ship
descriptive sibship `C` and mate/couple `M` kernels, but the high-level predictor
does not yet accept fitted `C` or `M`; see
[algorithm.md](algorithm.md#adding-environmental-covariance-to-improve-prediction)
and [Inference](inference.md#variance-components-a-c-m).

The family-data fitters target population variance components only when families
are independent sampling clusters and ascertainment is absent or correctly
represented by the fitted observation model. Overlapping pedigrees, case/control
sampling, or selection on family history invalidate the ordinary iid-family
moments, information estimates and bootstrap unless the design is handled
explicitly. In those settings, treat fitted values as design-dependent model
projections and validate them under the actual sampling scheme.

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
5. Convert `h²` to the **liability scale** (`convert_observed_to_liability_scale`),
   passing the actual study case fraction; its `sample_prev=0.5` default represents
   a balanced case/control design only.
6. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
7. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator now raises on duplicates).
8. Decide whether to **condition on the proband's own status**. Include `o` when
   intentionally constructing a diagnosis-derived GWAS phenotype; omit it or make
   it uninformative for prospective prediction/classification of that diagnosis.
9. Decide **case encoding** — onset-pinned (`age_thresholds`), lifetime-interval
   (base PA-FGRS), or age-specific interval (`pa_thresholds`, an age-dependent
   PA-FGRS-style variant) — and record it.
10. Choose **Gibbs vs Pearson–Aitken**; for unusual no-mixture pedigrees cross-check
    PA against Gibbs. The PA-FGRS censoring mixture has no Gibbs implementation.
11. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch). Prefer
     non-overlapping target families; if related targets remain, use and validate an
     association method that handles their relatedness, shared family-history
     phenotype, case-control imbalance, and tail behaviour
     ([Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)).
12. For family-data **fitting or bootstrap inference**, verify that sampled family
     clusters do not overlap and either avoid ascertainment or model it explicitly.

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
