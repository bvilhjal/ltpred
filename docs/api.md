# API reference

A quick map of the primary workflow, followed by signatures and docstrings
generated from the source for every public module (advanced array, pairwise
and low-level numerical APIs included). The top-level
`ltpred.__all__` controls wildcard imports and is intentionally limited to the
ordinary single-trait path. A wider set of names is available as explicit
top-level compatibility imports; the remaining numerical functions are available
from their owning modules shown below. Do not assume every generated module member
is also a top-level export.

## Primary workflow

**Table 1. Primary workflow functions.**

| function | purpose |
|---|---|
| `estimate_liability` | end-to-end estimator (PA by default for one trait, Gibbs for multi-trait; opt-in `method="quadrature"` for additive nuclear families) |
| `fit_heritability` | **fit** liability-scale `h²` from independent, non-overlapping families with a common case/control threshold; `sampling="population"` (unascertained, screened for gross marginal case-rate inconsistency but not certified) or `sampling="ipw"` with `weights = 1 / P(family sampled)` for a known, strictly positive selected design |
| `prevalence_thresholds` / `age_thresholds` | classic or personalised pinned bounds; family rows determine LT-FH++ vs ADuLT |
| `pa_thresholds` | age-specific interval-case and PA-FGRS censored-control-mixture inputs; an age-dependent variant, not base PA-FGRS, exact PA-FGRS_ADT, or a requirement merely to use the PA engine |
| `thresholds_from_cip` | bounds from an empirical (population) CIP curve — `case_mode="interval"` is LTFHPlus's default case interval and, with the mixture, a PA-FGRS-style variant (not PA-FGRS_ADT); endpoint values are held constant outside its age grid |
| `families_from_columns` | build family inputs from flat columns |
| `kinship_from_pedigree` / `estimate_liability_from_kinship` | arbitrary-pedigree input and PA-default estimation; caller-supplied `c_kernel`/`m_kernel` support environmental components, and `use_mixture=True` with `K_i`/`K_pop` enables the PA-FGRS censored-control mixture |
| `estimate_liabilities` | supported population-trio driver: pedigree discovery, pinned-onset empirical-CIP bounds, explicit GWAS versus prospective-prediction observation sets, and calendar-time censoring |
| `simulate_under_LTM_single` | simulate role-grammar families for one trait, for testing/benchmarking |
| `simulate_pedigree` / `pedigree_birth_times` | simulate a population trio table (`ids`, `father`, `mother`) and give it generation-coherent calendar birth times |
| `simulate_register_liabilities` | simulate **one** consistent liability field over a pedigree and return the register columns plus the true genetic liability, so a score can be checked against truth |
| `simulate_followup_records` | simulate registry follow-up (entry age, exit age, event code) from a known incidence curve, with optional competing mortality and delayed entry — the input to `kaplan_meier_cip` / `aalen_johansen_cip` |
| `simulate_under_LTM_multi` | simulate families with two traits under A/C/M/E components, for `fit_pairwise_multi` |
| `observed_to_liability_h2` / `liability_to_observed_h2` | observed ↔ liability-scale `h²` (Lee et al.) |
| `set_num_threads` | set the Numba-parallel thread count |


The heavier inferential machinery — the HE multi-trait genetic-correlation,
onset-age-decay, common-factor and genetic-nurture fits, the MCEM
variance-component fit, and the parametric-bootstrap significance tests — lives
in the unsupported checkout-only
[`research/` package](https://github.com/bvilhjal/ltpred/tree/main/research);
they are not installed with the distribution. See
[research extensions](research.md) for the models.

For the Gibbs estimator and model-fit samplers, `seed` is either `None` or an
integer in `[0, 2**32 - 1]`; booleans are rejected. Derived sampler streams wrap
deterministically within that 32-bit range. Resampling seeds on bootstrap helpers
follow NumPy's `default_rng` contract.


PA accepts a symmetric positive-semidefinite covariance with strictly positive
marginal variances. Gibbs requires strict positive-definiteness because it forms
precision-based conditional variances.
The sex-limited and genetic-nurture covariance constructors live in the
checkout-only `research/covariance_extensions.py`.


The older array APIs keep their tuple contracts: PA `(est, var)`, Gibbs
`(est, se)` or `(est, se, var)` with `return_var=True`, and kinship
`(est, se, var)`. Posterior variance and Monte Carlo SE are distinct quantities.

## Estimation — `ltpred.estimate`

::: ltpred.estimate

## Nuclear-family quadrature — `ltpred.quadrature`

This explicit array API returns `QuadratureResult`, not `LiabilityResult`.
Its `error` is a successive-refinement diagnostic, not a certified error bound
or Monte-Carlo SE; nonconvergence raises. The high-level
`estimate_liability(..., method="quadrature")` route instead includes
`quadrature_error` and `quadrature_nodes` dictionaries in `LiabilityResult`;
its zero `se` means no Monte-Carlo noise. See the
[worked example and model scope](estimation.md#nuclear-family-quadrature).

::: ltpred.quadrature
## Chunked array drivers — `ltpred.chunked`

::: ltpred.chunked

## Liability-scale transformations — `ltpred.liability_scale`

::: ltpred.liability_scale

## Tetrachoric correlation — `ltpred.tetrachoric`

`ltpred.tetrachoric` is the function; import these names from `ltpred` or with `from ltpred.tetrachoric import ...`.

::: ltpred.tetrachoric

## Register pipeline — `ltpred.pipeline`

Prediction requires per-person `birth_time` and per-proband `index_time` on a
common numeric calendar scale whose unit matches `age`; this is what gives
relatives from different birth years their correct attained-age censoring
landmarks. Structural ancestors in the pedigree's exact-kinship closure are not
diagnosis observations unless `condition_closure=True` is explicitly requested.
Under `use="prediction"` the estimand is
`E[g | relatives' records at the landmark]` — it is *not* additionally
conditioned on the proband being disease-free at the landmark; the two rank
probands of one age and family structure almost identically but differ in level
across ages, because surviving to an older age disease-free is evidence of lower
liability. Accuracy and throughput evidence:
[RESULTS §21](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#21-end-to-end-register-pipeline-bench_register_pipelinepy).

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

## Pairwise model fitting — `ltpred.pairwise`

`PairwiseFitResult` reports a component covariance matrix and sampling SEs only
for identifiable interior fits; boundary or insufficient-information results
mark those fields `NaN` and provide `inference_status`. Thresholds and weights
are treated as fixed. See [pairwise fitting](inference.md#deterministic-pairwise-fitting)
for the sampling contract and distinction from the existing stochastic fitters.

::: ltpred.pairwise

## Families — `ltpred.family`

::: ltpred.family

## Gibbs sampler — `ltpred.gibbs`

::: ltpred.gibbs
    options:
      members: [rtmvnorm_gibbs, gibbs_params]

## Pearson–Aitken — `ltpred.pearson_aitken`

::: ltpred.pearson_aitken

## Simulation — `ltpred.simulate`

::: ltpred.simulate

## Runtime configuration — `ltpred.set_num_threads`

::: ltpred.set_num_threads

## Joint pairwise fitting — `ltpred.pairwise_multi`

::: ltpred.pairwise_multi
