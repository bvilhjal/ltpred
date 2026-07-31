# Changelog

All notable changes to ltpred are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the major
version is 0 the public API may still change between minor releases.

## 0.3.0 — 2026-07-31

### Changed (breaking)

Leanness refactor: the supported public API is now the lean estimation core.
The advanced fitting and covariance machinery moved to an **unsupported
`research/` package** at the repository root — importable as
`research.<module>` from a checkout, not installed with the distribution and
not covered by the compatibility policy. A capability (re)joins `ltpred`
proper only when wired into the core estimation path and benchmarked.

**Moved to `research/`** (same names unless noted):

- `ltpred.pipeline` (`PopulationScores`, `estimate_liabilities`) →
  `research/pipeline.py`.
- `fit_genetic_correlation`, `fit_genetic_correlation_decay`,
  `fit_genetic_factor`, `fit_nurture`, `test_variance_component`,
  `test_genetic_correlation` and their result types →
  `research/advanced_fitting.py`. The MCEM variance-component route is now
  `fit_variance_components_mcem` there (was
  `fit_variance_components(..., method="mcem")`, aliases `"ml"`/`"reml"`);
  its `MCEMVarCompResult` keeps the `loglik`/`aic` fields.
- `construct_covmat_sex_limited` and `construct_covmat_nurture` →
  `research/covariance_extensions.py`.

**Removed outright** (no replacement unless noted):

- `liability_sensitivity` / `SensitivityResult` — re-estimate over an `h2`
  grid and correlate the scores directly.
- `convert_observed_to_liability_scale` — an exact duplicate of
  `liability_scale.observed_to_liability_h2`; use that.
- `observed_to_liability_gencov`, `liability_to_observed_gencov` and
  `observed_to_liability_rg` from `ltpred.liability_scale`.
- `tnorm_moments` and `tnorm_mixture_conditional` from
  `ltpred.pearson_aitken` (internal helpers, no longer exported).
- `truncated_normal_cdf` from `ltpred.thresholds`; `convert_cir_to_age` is
  now private (`_convert_cir_to_age`).
- `extract_pedigrees` — a trivial generator; iterate `extract_pedigree`.
- The `construct_covmat` dispatcher — call `construct_covmat_single` /
  `construct_covmat_multi` directly.
- `estimate_liability_single` / `estimate_liability_pa` /
  `estimate_liability_multi` are now private (`_estimate_liability_*`); use
  `estimate_liability` or the array APIs.

**Signature changes:**

- `fit_heritability` and `fit_variance_components` add a transitional
  `sampling=` contract. Pass `sampling="population"` to affirm independent,
  non-overlapping population-sampled families; omission warns, and no
  ascertained-sample mode is implemented.
- `fit_variance_components` no longer takes `method=` — the Haseman–Elston
  moment regression is the only core route (MCEM lives in `research/`).
- `liability_r2_from_z` now subtracts the unit expected null contribution,
  using `(z² - 1) / N` by default. Individual estimates may therefore be
  negative; aggregate before interpretation. Pass `subtract_null=False` only
  to reproduce the former raw `z² / N` second moment.
- `VarCompResult` drops the `loglik`/`aic` fields.
- `Covmat` drops the write-only `h2` dataclass field.
- `LiabilityResult` drops the `column` attribute.
- `estimate_liability`'s `out=` accepts only the exact strings `"genetic"` /
  `"full"` (singly or as a tuple).
- `aalen_johansen_cip` drops `n_boot`/`seed` — the closed-form Aalen (1978)
  variance is the only standard error.
- `gibbs_params` / `rtmvnorm_gibbs` now reject singular as well as indefinite
  covariance matrices, including when precomputed Gibbs parameters are passed.
- `tetrachoric_matrix` warns when independently fitted pairwise correlations
  do not form a positive-semidefinite matrix; pass `check_psd=False` only after
  making an explicit downstream handling choice.
- `convert_age_to_thresh` and `convert_liability_to_aoo` drop `dist=` (and
  the normal-branch min/max parameters) — they always use the logistic
  mapping.

### Fixed

- Aalen-Johansen pointwise uncertainty now uses the finite-risk-set,
  grouped-tie `cmprsk::cuminc` recurrence rather than a large-risk-set
  approximation. Event-coded zero follow-up and censoring as `cause=0` are
  rejected; `n_entered` counts records with positive follow-up.
- Moment fitting preserves the exact theoretical `A`, `C`, and `M` kernels
  instead of nudging singular-but-valid component matrices. Public update
  controls are validated so the assembled covariance retains a numerical
  positive residual floor.
- The genetic-nurture covariance derivation now includes the changed proband
  genetic row, and the PGS/family-history correlation identity states
  conditional independence as an assumption rather than inferring it from
  non-overlapping cohorts.
- Benchmark reports now distinguish historical outputs from current-tree
  validation, scope uncertainty claims to the evidence retained, and provide
  a provenance runner with hashed source state, console logs, and canonical
  artifacts.

### Added

- The genetic-nurture additions below landed in `research/`
  (`construct_covmat_nurture` in `research/covariance_extensions.py`,
  `fit_nurture` in `research/advanced_fitting.py`), not in the core package.
  `construct_covmat_nurture` separates a proband's **direct** genetic effect
  from parental **indirect** (genetic-nurture) effects, building the covariance
  from the path model rather than from kinship. Parent-offspring covariance
  becomes `h2/2 + n*h2` and sib-sib `h2/2 + 2*n*h2 + 2*n^2*h2` -- inflated by
  different amounts, which is what identifies `n` -- while `Cov(A_o, l_parent)`
  stays at `h2/2`, `Cov(A_o, l_o)` becomes `h2*(1+n)`, and
  `Cov(A_o, l_sibling)` becomes `h2/2 + n*h2`. That selective, directional
  pattern is what a single symmetric kinship-scaled matrix cannot express.
  `nurture = 0` reproduces `construct_covmat_single` exactly. Nuclear roles
  only; the closed form was verified against a 4,000,000-family Monte-Carlo
  simulation of the path model. Note that on sibling covariance alone nurture is
  indistinguishable from a sibship `C`; parent-offspring covariance is what
  separates them.
- `fit_nurture` makes the indirect coefficient a **fitted** quantity rather than
  a supplied one. The parent-offspring and sib-sib moments are two equations in
  two unknowns, so the ratio isolates the indirect path and the estimates are
  closed-form and exact: `1 + 2n = cov_sib / cov_parent_offspring`,
  `h2 = 2*cov_parent_offspring^2 / cov_sib`. It also reports what a
  nurture-blind additive model would claim from each relative type alone, whose
  disagreement is the diagnostic for an indirect path and is zero exactly when
  `n` is zero. A moment estimator: no standard errors, and inputs must be
  liability-scale (use `ltpred.tetrachoric` on binary data).

## 0.2.0 — 2026-07-29

### Added

- `construct_covmat_sex_limited` puts sex into the covariance rather than only
  the thresholds: sex-specific heritabilities and a cross-sex genetic
  correlation, `Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) *
  rg_cross^[sex_i != sex_j]`. Equal heritabilities with `rg_cross = 1`
  reproduce `construct_covmat_single` exactly, and the result is positive
  semi-definite for any `|rg_cross| <= 1`. This changes the relative weights
  carried by same- and opposite-sex relatives. Personalised thresholds already
  affect calibration and can also change ordering through the truncated means;
  the covariance model adds sex-specific genetic weighting. The parameters are
  inputs, not fitted;
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
