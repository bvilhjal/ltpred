# API reference

A quick map of the primary workflow, followed by advanced and numerical APIs and
then signatures/docstrings generated from the source. The top-level
`ltpred.__all__` controls wildcard imports and is intentionally limited to the
ordinary single-trait path. A wider set of names is available as explicit
top-level compatibility imports; the remaining numerical functions are available
from their owning modules shown below. Do not assume every generated module member
is also a top-level export.

## Primary workflow

| function | purpose |
|---|---|
| `estimate_liability` | end-to-end estimator (PA inference by default for one trait, Gibbs for multi-trait) |
| `fit_heritability` | **fit** liability-scale `h²` from independent, non-overlapping families; `sampling="population"` (unascertained, screened for gross marginal case-rate inconsistency but not certified) or `sampling="ipw"` with `weights = 1 / P(family sampled)` for a known, strictly positive selected design |
| `prevalence_thresholds` / `age_thresholds` | classic or personalised pinned bounds; family rows determine LT-FH++ vs ADuLT |
| `pa_thresholds` | age-specific interval-case and PA-FGRS censored-control-mixture inputs; an age-dependent variant, not base PA-FGRS, exact PA-FGRS_ADT, or a requirement merely to use the PA engine |
| `thresholds_from_cip` | bounds from an empirical (population) CIP curve — `case_mode="interval"` is likewise an age-dependent PA-FGRS-style variant rather than exact PA-FGRS_ADT; endpoint values are held constant outside its age grid |
| `families_from_columns` | build family inputs from flat columns |
| `kinship_from_pedigree` / `estimate_liability_from_kinship` | arbitrary-pedigree input and PA-default estimation; pass `use_mixture=True` with `K_i`/`K_pop` for the PA-FGRS censored-control mixture |
| `simulate_under_LTM_single` | simulate families for testing/benchmarking |
| `observed_to_liability_h2` / `liability_to_observed_h2` | observed ↔ liability-scale `h²` (Lee et al.) |
| `set_num_threads` | set the Numba-parallel thread count |

## Advanced fitting and scale APIs

| function | purpose |
|---|---|
| `estimate_liability_pa_arrays` / `estimate_liability_gibbs_arrays` | array API — skip `Family` objects for biobank scale |
| `fit_variance_components` | **fit** additive `A` + shared-environment `C` (sibship) / `M` (couple) as proportions under the same population-sampling contract |
| `bootstrap_fit` | iid-family cluster bootstrap SD / percentile interval; it does not correct ascertainment bias |

The heavier inferential machinery — the multi-trait genetic-correlation,
onset-age-decay, common-factor and genetic-nurture fits, the MCEM
variance-component fit, and the parametric-bootstrap significance tests — lives
in the unsupported checkout-only
[`research/` package](https://github.com/bvilhjal/ltpred/tree/main/research);
they are not installed with the distribution.

For the Gibbs estimator and model-fit samplers, `seed` is either `None` or an
integer in `[0, 2**32 - 1]`; booleans are rejected. Derived sampler streams wrap
deterministically within that 32-bit range. Resampling seeds on bootstrap helpers
follow NumPy's `default_rng` contract.

## Low-level numerical APIs

| function | purpose |
|---|---|
| `pa_algorithm` / `pa_estimate_batched` | Pearson–Aitken selection updates |
| `construct_covmat_single` / `construct_covmat_multi` | family covariance from relatedness |
| `get_relatedness` | shared-DNA × `h²` for a pair of roles |
| `construct_covmat_from_kinship` | liability covariance from a kinship/`A` matrix |
| `rtmvnorm_gibbs` | truncated-MVN Gibbs sampler; covariance must be strictly positive-definite |
| `convert_age_to_cir` / `convert_age_to_thresh` / … | age ↔ incidence ↔ threshold |

PA accepts a symmetric positive-semidefinite covariance with strictly positive
marginal variances. Gibbs requires strict positive-definiteness because it forms
precision-based conditional variances.
The sex-limited and genetic-nurture covariance constructors live in the
checkout-only `research/covariance_extensions.py`.

## Estimation — `ltpred.estimate`

::: ltpred.estimate

## Liability-scale transformations — `ltpred.liability_scale`

::: ltpred.liability_scale

## Tetrachoric correlation — `ltpred.tetrachoric`

::: ltpred.tetrachoric

## Register pipeline — `research.pipeline`

The end-to-end register pipeline (`PopulationScores`, `estimate_liabilities`)
moved to the unsupported `research/` package at the repository root
(`research/pipeline.py`); it is importable from a checkout but is not part of
the installed `ltpred` distribution, so it has no generated reference here.

## CIP estimation — `ltpred.cip`

::: ltpred.cip

## Pedigree discovery — `ltpred.pedigree`

::: ltpred.pedigree

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

## Runtime configuration — `ltpred.set_num_threads`

::: ltpred._numba.set_num_threads
