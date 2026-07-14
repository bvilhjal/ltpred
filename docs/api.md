# API reference

A quick map of every public name, then the full signatures and docstrings
(generated from the source by [mkdocstrings](https://mkdocstrings.github.io/) when
this site is built).

## Quick reference

| function | purpose |
|---|---|
| `estimate_liability` | end-to-end estimator (PA-FGRS by default, Gibbs for multi-trait; `method=` to force) |
| `liability_sensitivity` | sweep the assumed `h²` and report how stable the score is |
| `estimate_liability_pa` | deterministic PA-FGRS estimator |
| `estimate_liability_pa_arrays` / `_gibbs_arrays` | array API — skip `Family` objects for biobank scale |
| `fit_heritability` | **fit** liability-scale `h²` from family data (data-augmentation fixed point) |
| `fit_variance_components` | **fit** additive `A` + shared-environment `C` (sibship) / `M` (couple) as proportions |
| `fit_genetic_correlation` | **fit** the genetic correlation `r_g` between traits (cross-trait HE regression) |
| `fit_genetic_factor` | **fit** a common-factor model `r_g ≈ ΛΛ' + Ψ` (Genomic-SEM-lite) |
| `bootstrap_fit` | family-resampling bootstrap SE / CI for any of the fitters |
| `test_variance_component` / `test_genetic_correlation` | parametric-bootstrap significance tests |
| `set_num_threads` | set the Numba-parallel thread count |
| `pa_algorithm` / `pa_estimate_batched` | Pearson–Aitken selection updates |
| `tnorm_moments` / `tnorm_mixture_conditional` | truncated-normal moments (+ censoring mixture) |
| `construct_covmat` / `_single` / `_multi` | family covariance from relatedness |
| `get_relatedness` | shared-DNA × `h²` for a pair of roles |
| `kinship_from_pedigree` | additive relationship matrix `A` from a pedigree (`id`, `father`, `mother`) |
| `construct_covmat_from_kinship` | liability covariance from a kinship/`A` matrix |
| `estimate_liability_from_kinship` | estimate a target's liability from a pedigree `A` + per-member bounds |
| `rtmvnorm_gibbs` | truncated-MVN Gibbs sampler |
| `prevalence_thresholds` / `age_thresholds` / `pa_thresholds` | status (+age) → liability bounds |
| `thresholds_from_cip` | bounds from an empirical (population) CIP curve — for real data |
| `convert_age_to_cir` / `convert_age_to_thresh` / … | age ↔ incidence ↔ threshold |
| `convert_observed_to_liability_scale` | observed → liability-scale `h²` (Lee et al.) |
| `simulate_under_LTM_single` | simulate families for testing/benchmarking |
| `families_from_columns` | build family inputs from flat columns |

## Estimation — `ltpred.estimate`

::: ltpred.estimate

## Thresholds — `ltpred.thresholds`

::: ltpred.thresholds

## Covariance — `ltpred.covariance`

::: ltpred.covariance

## Model fitting — `ltpred.fit`

::: ltpred.fit

## Families — `ltpred.family`

::: ltpred.family

## Gibbs sampler — `ltpred.gibbs`

::: ltpred.gibbs

## Pearson–Aitken — `ltpred.pearson_aitken`

::: ltpred.pearson_aitken

## Simulation — `ltpred.simulate`

::: ltpred.simulate
