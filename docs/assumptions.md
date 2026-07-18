# Assumptions, checklist & pitfalls

What the model does and does not represent, a pre-flight checklist for a production
run, and the common mistakes.

## Modelling assumptions

The high-level predictor uses **additive genetic sharing only**: every
off-diagonal entry is `shared_DNA × h²`. It assumes jointly Gaussian liabilities,
correct relationships and diagnoses, and correctly specified prevalence/CIPs and
liability-scale `h²`. Under its standard model, individual-specific residuals are
independent of relatives' additive genetic values and of each other.

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

1. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
2. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period where
   incidence differs. With independent right censoring and no competing events,
   `1 − Kaplan–Meier` estimates risk; with competing death or diagnoses, use a
   cause-specific cumulative-incidence estimator such as Aalen–Johansen.
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
9. For family-data **fitting or bootstrap inference**, verify that sampled family
   clusters do not overlap and either avoid ascertainment or model it explicitly.

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
  nothing. In mixture mode, age enters through `K_i`; the implementation splits at
  the lifetime threshold and uses the finite control bound only as a censoring flag,
  so an age-specific bound from these helpers is safe and is not applied twice.
- **Missing or uninformative family observations** make the estimate rely mostly
  on the proband's own status. Low prevalence alone does not imply little
  information: an affected person with a rare disease can be highly informative.
- **Install `[fast]`** (Numba) for large runs; the pure-Python fallback is
  numerically identical but much slower.
