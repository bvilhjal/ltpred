# Changelog

All notable changes to ltpred are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the major
version is 0 the public API may still change between minor releases.

## Unreleased

## 0.2.0 — 2026-07-29

### Added

- `construct_covmat_sex_limited` puts sex into the covariance rather than only
  the thresholds: sex-specific heritabilities and a cross-sex genetic
  correlation, `Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) *
  rg_cross^[sex_i != sex_j]`. Equal heritabilities with `rg_cross = 1`
  reproduce `construct_covmat_single` exactly, and the result is positive
  semi-definite for any `|rg_cross| <= 1`. Because the BLUP weights are
  threshold-free, this is what makes sex a ranking lever rather than only a
  calibration correction. The parameters are inputs, not fitted;
  see the sex-limitation section of `docs/algorithm.md` for the identification
  requirements.

### Fixed

- The high-level multi-trait dispatcher now rejects nonzero `c2`/`m2` instead
  of silently dropping them; defining their cross-trait covariance remains
  future work.
- The covariance fitters again accept valid person-specific one-sided,
  two-sided, and pinned rectangles. Geometry-only filtering no longer blocks
  those observations or the `mcem`/`ml`/`reml` likelihood routes; callers remain
  responsible for supplying a coherent fitting observation model.
- The MkDocs site has a real root page, the sdist includes the benchmark helper
  imported by its shipped tests, and documentation fence validation now matches
  delimiter type and opening length.

## 0.1.0 — 2026-07-22

First tagged release. A from-scratch Python implementation of the
liability-threshold family of genetic-liability estimators, ported from the R
package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus).

### Models and inference

- **LT-FH** (family history), **LT-FH++** (age-, birth-year- and sex-dependent
  prevalence) and **ADuLT** (the personalised construction without family
  history), from the same threshold/covariance machinery.
- Two inference engines: a Numba-JIT truncated-multivariate-normal **Gibbs**
  sampler with batch-means convergence, and a deterministic **Pearson–Aitken**
  sequential-moment approximation (the single-trait default). Across matched
  no-mixture bounds their posterior-mean estimates correlated ≥ 0.997, with PA
  315–510× faster in the controlled threaded benchmark.
- The PA-only **PA-FGRS** age-censored-control mixture.
- Single- and multi-trait `estimate_liability`, plus an array API
  (`estimate_liability_pa_arrays` / `_gibbs_arrays`) that skips `Family` objects
  for biobank-scale runs, with an optional float32 bounds path.

### Covariance, thresholds and data preparation

- Role-based covariance over the LTFHPlus relative grammar, and arbitrary
  pedigrees via `kinship_from_pedigree` / `estimate_liability_from_kinship`.
- Sibship (`C`) and couple (`M`) shared-environment components, accepted as
  `c2`/`m2` by the single-trait role/object and array estimators. The tagged
  high-level multi-trait dispatcher exposed these arguments but did not apply
  them; this is fixed under Unreleased above.
- Threshold builders (`prevalence_thresholds`, `age_thresholds`, `pa_thresholds`,
  `thresholds_from_cip`), CIP estimation from follow-up records
  (Kaplan–Meier and Aalen–Johansen, `ltpred.cip`), pedigree discovery from trio
  records (`ltpred.pedigree`), and an end-to-end register pipeline
  (`ltpred.pipeline`).

### Model fitting

- `fit_heritability`, `fit_variance_components` (A + C + M, by multiple
  Haseman–Elston or a Monte-Carlo EM-style likelihood fit),
  `fit_genetic_correlation`, `fit_genetic_correlation_decay` (onset-age-structured
  `r_g`), and `fit_genetic_factor` (a Genomic-SEM-style common-factor model).
- `bootstrap_fit` for family-cluster intervals, `liability_sensitivity` for `h²`
  sensitivity, and parametric-bootstrap significance tests.
- Liability-scale transformations (`ltpred.liability_scale`) and
  tetrachoric-correlation diagnostics (`ltpred.tetrachoric`).

### Notes for users

**The tagged 0.1.0 fitters enforce a common case/control threshold per trait.**
`fit_heritability`, `fit_variance_components`, and
`fit_genetic_correlation` reject person-specific, two-sided, or pinned bounds;
the guard also precedes the variance-component likelihood aliases. The
Unreleased changes above remove that geometry-only restriction while retaining
ordinary bounds validation and documenting the caller's observation-model
responsibility.

For the Haseman–Elston and genetic-correlation fits, reported `se` values are
**within-dataset Monte-Carlo diagnostics**, not sampling standard errors. The
MCEM variance-component fit instead reports an approximate OPG/BHHH
observed-information SE, still subject to Monte-Carlo, finite-iteration, and
model-correctness assumptions. Use `bootstrap_fit` when its independent-cluster
assumptions hold. The PA-FGRS censoring mixture and the inference machinery
(bootstrap intervals, MCEM SEs, parametric-bootstrap tests) have not been
calibration-benchmarked; see `docs/assumptions.md` and `benchmarks/RESULTS.md`
for what is and is not validated. The `fit_heritability` benchmark grid covers
prevalence 0.1 with parents + 2 sibs at 3,000 families; rarer disease or sparser
families are outside it and show noticeably more spread.

`h2` must lie in `(0, 1]`: a zero heritability makes the genetic liability
identically zero, leaving a covariance no positive-definite correction can
repair. This is now rejected at the API boundary rather than failing opaquely
further down.
