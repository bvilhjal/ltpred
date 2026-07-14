# Assumptions, checklist & pitfalls

What the model does and does not represent, a pre-flight checklist for a production
run, and the common mistakes.

## Modelling assumptions

The family covariance models **additive genetic sharing only**: every off-diagonal
entry is `shared_DNA × h²`. Not modelled are shared environment, household/cultural
transmission, assortative mating (parents are assumed genetically unrelated —
`m`–`f` covariance is 0), dominance/epistasis, and indirect genetic effects. When
these contribute to familial aggregation — common for psychiatric, reproductive,
metabolic and social traits — read the output as the **additive-genetic-model
projection of the family history**, not a pure causal genetic value, and expect
some over- or under-statement of "genetic" liability. The estimate is also
conditional on the assumed `h²`, prevalence and CIPs; treat those as inputs whose
uncertainty propagates (see the checklist).

The covariance is modular, though: you can **add environmental covariance**
(shared environment `c²`, maternal effects, assortative mating) to the
between-relative covariance to separate genetic from shared-environmental
resemblance and improve prediction. `construct_covmat` ships only the
additive-genetic table, but the covariance-level entry points (`rtmvnorm_gibbs`,
`pa_algorithm`, `pa_estimate_batched`) accept an arbitrary covariance — see
[algorithm.md](algorithm.md#adding-environmental-covariance-to-improve-prediction).
Several of those components (sibship `C`, couple `M`) can also be **fit and tested**
from the data — see [Inference](inference.md#variance-components-a-c-m).

## Real-data checklist

Before running a production analysis:

1. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
2. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period
   where incidence differs; use a censoring-aware / competing-risk estimator
   (Kaplan–Meier, Aalen–Johansen) if death/emigration/competing diagnoses matter.
3. Convert `h²` to the **liability scale** (`convert_observed_to_liability_scale`).
4. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
5. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator now raises on duplicates).
6. Decide **case encoding** — pinned (`age_thresholds`) vs interval
   (`pa_thresholds`) — and record it.
7. Choose **Gibbs vs Pearson–Aitken**; for unusual pedigrees cross-check PA
   against Gibbs.
8. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch) and
   handle related probands (LMM / pruning) before the GWAS.

## Pitfalls

- **`h²` must be liability-scale.** Convert observed-scale estimates first.
- **Ages mean different things by status.** For `age_thresholds`/`pa_thresholds`,
  `age` is the **age of onset** for cases and the **age at last follow-up** for
  controls.
- **Number repeated relatives** (`s1`, `s2`) — an unnumbered duplicate role
  collides.
- **The proband is role `o`**, not `g`. `g` (what you estimate) is added for you.
- **Prevalence and CIPs should match the population** the thresholds refer to;
  stratify by sex/birth-year if your incidence differs across strata (pass the
  per-person `K_i`).
- **`use_mixture=True` needs `K_i`/`K_pop`** on the members (from `pa_thresholds`
  or `thresholds_from_cip`) — otherwise it now raises rather than silently doing
  nothing. And do **not** combine an already age-adjusted control bound with the
  mixture (it would double-correct); the mixture does the age adjustment itself.
- **Very low prevalence + tiny families** carry little information; the estimate
  approaches the population mean and the gain over case/control shrinks.
- **Install `[fast]`** (Numba) for large runs; the pure-Python fallback is
  numerically identical but much slower.
