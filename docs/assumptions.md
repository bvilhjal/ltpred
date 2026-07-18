# Assumptions, checklist & pitfalls

What the model does and does not represent, a pre-flight checklist for a production
run, and the common mistakes.

## Modelling assumptions

The high-level predictor uses **additive genetic sharing only**: every
off-diagonal entry is `shared_DNA × h²`. It assumes jointly Gaussian liabilities,
correct relationships and diagnoses, and correctly specified prevalence/CIPs and
liability-scale `h²`. Under its standard model, individual-specific residuals are
independent of relatives' additive genetic values and of each other.

The age-dependent formulation assumes a non-decreasing cumulative-incidence curve
(hence a non-increasing liability threshold) and, in the current high-level model,
the same liability covariance and genetic architecture across age at diagnosis,
sex and cohort. Those variables alter thresholds, not `h²` or genetic
correlations. Follow-up/censoring must be represented by a defensible observation
model; independent censoring is required for ordinary Kaplan–Meier risk estimates,
and competing events require a cumulative-incidence estimator.

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

1. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
2. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period where
   incidence differs. With independent right censoring and no competing events,
   `1 − Kaplan–Meier` estimates risk; with competing death or diagnoses, use a
   cause-specific cumulative-incidence estimator such as Aalen–Johansen.
3. Make each CIP age grid cover the analysed onset/follow-up ages and supply
   `k_pop` explicitly unless its final value is a defensible lifetime prevalence;
   the helper holds endpoint values constant outside the grid.
4. Convert `h²` to the **liability scale** (`convert_observed_to_liability_scale`),
   passing the actual study case fraction; its `sample_prev=0.5` default represents
   a balanced case/control design only.
5. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
6. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator now raises on duplicates).
7. Decide **case encoding** — pinned (`age_thresholds`) vs interval
   (`pa_thresholds`) — and record it.
8. Choose **Gibbs vs Pearson–Aitken**; for unusual pedigrees cross-check PA
   against Gibbs.
9. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch) and
   handle related probands (LMM / pruning) before the GWAS.
10. For family-data **fitting or bootstrap inference**, verify that sampled family
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
- **CIP interpolation is flat beyond its age grid.** It does not extrapolate an
  incidence trend; cover the analysed ages and do not equate the last observed CIP
  with lifetime prevalence unless the curve reaches that horizon.
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
