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
| `kinship_from_pedigree` / `estimate_liability_from_kinship` | arbitrary-pedigree input and PA-default estimation; caller-supplied `c_kernel`/`m_kernel` support environmental components, and `use_mixture=True` with `K_i`/`K_pop` enables the PA-FGRS censored-control mixture |
| `estimate_liabilities` | supported population-trio driver: pedigree discovery, pinned-onset empirical-CIP bounds, explicit GWAS versus prospective-prediction observation sets, and calendar-time censoring |
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
| `construct_covmat_from_kinship` | liability covariance from a kinship/`A` matrix, optionally with caller-supplied `C`/`M` kernels |
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

## Register pipeline — `ltpred.pipeline`

`PopulationScores` and `estimate_liabilities` are installed public APIs and are
also available as explicit lazy imports from `ltpred`. They are deliberately
absent from the curated wildcard-import list. Prediction requires per-person
`birth_time` and per-proband `index_time` on a common numeric calendar scale
whose unit matches `age`; this is what gives relatives from different birth
years their correct attained-age censoring landmarks. Structural ancestors in
the pedigree's exact-kinship closure are not diagnosis observations unless
`condition_closure=True` is explicitly requested. Under `use="prediction"` the
estimand is `E[g | relatives' records at the landmark]` — it is *not*
additionally conditioned on the proband being disease-free at the landmark; the
two agree on ranking within an age but differ in level across ages, because
surviving to an older age disease-free is evidence of lower liability. The
driver's payoff and throughput evidence (RESULTS §§20–21, regenerated
2026-09-03 under the provenance wrapper): the API is supported, with the
payoff quantified there -- degree-3 corr(est, true g) 0.574 ± 0.021 vs
degree-1 0.539 ± 0.015 (paired contrast +0.035 ± 0.012), prospective
familywise-censored AUC 0.680 ± 0.039, and ~380 probands/s throughput.
The relatives'-events contrast is unresolved at R = 5 on both metrics and
is not quoted as a payoff.

The old `research.pipeline` implementation is retained only as historical
checkout scaffolding. Its common `index_age` shortcut assigns a proband's
attained age to every relative and is not a valid general familywise
calendar-time censor across generations.

Two boundary diagnostics are surfaced on `PopulationScores` rather than
silently absorbed: `frac_records_with_unresolved_parents` (the fraction of records with a
non-null parent reference matching no id — a zero resolved share raises, a
share above half warns) and, under `use="prediction"`, `proband_state`
(`disease_free_and_followed` / `prevalent_case` / `exited_before_index` at the
landmark, with a warning on prevalent cases).

::: ltpred.pipeline

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
