# Roadmap

Where ltpred stands and where it is going. See [guide.md](guide.md) for usage,
[algorithm.md](algorithm.md) for the model and estimators, and
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md) for the method comparison.

## Where things stand

ltpred is a from-scratch Python port of LT-FH++. It provides two interchangeable
estimators of the posterior-mean genetic liability:

- **Gibbs (LT-FH++)** — a truncated-multivariate-normal sampler (Rcpp port) with
  batch-means convergence.
- **Pearson–Aitken (PA-FGRS)** — a deterministic selection-formula sweep with the
  age-censored-control mixture. It agrees with Gibbs to correlation 0.9997 and
  runs 100–360× faster.

Both support single- and multi-trait analyses and the classic LT-FH,
age-of-onset ADuLT, and PA-FGRS threshold encodings.

**Performance and scale.** The core is Numba-JIT'd and `prange`-parallel, with
families grouped by structure (canonical form). Streaming batch-means keeps
standard-error memory at `O(F)`; a precision-matrix `gibbs_params` path is
~7.7× faster; an array API that skips Python objects is ~105× faster at biobank
scale; and a float32 bounds option halves memory. (The exploration also showed
why int8-quantising the covariance, ldpred3-style, is the wrong lever here.)

**Variance-component fitting.** `fit_heritability` is a data-augmentation Gibbs
(Haseman–Elston update), validated unbiased across h² 0.2–0.8.
`fit_variance_components` fits additive `A` and common-environment `C` together
by a **multiple Haseman–Elston regression** on the same well-mixing collapsed
data-augmentation — validated unbiased for `A` and `A+C` across family
structures, with negligible false-positive `C`. (This replaced an earlier
experimental Bayesian animal-model Gibbs, which mixed poorly and showed
structure-dependent bias. Dominance `D` is intentionally not offered — it needs
MZ/DZ twin contrasts to estimate honestly.) `fit_genetic_correlation` estimates
the **genetic correlation `r_g`** between traits by the cross-trait analogue of
the same regression — validated ~unbiased near the null with mild attenuation at
large `|r_g|`.

**Benchmarks** (`benchmarks/`, `RESULTS.md`) cover accuracy, runtime scaling,
age-of-onset, and GWAS power (LT-FH++ and PA both ~1.52× effective-N over
case/control at λ_GC ≈ 1), plus `fit_heritability` quality (unbiased, but
`h2_se` understates the true SD ~20–30×, so use `bootstrap_fit`), `A+C` recovery,
and `r_g` recovery. Real-LD runs go through an opt-in HAPNEST path.

**Docs.** README, a user guide, and an algorithm/model doc (with the
BLUP / selection-index framing, the Pak–Sham liability-threshold-risk
connection, and the environmental-covariance extension), plus `CITATION.cff`
(15 references).

The test suite is 150 tests passing.

## Near-term — variance-component thread ✅ complete

1. ~~**Fix the animal-model Gibbs mixing.**~~ **Done — resolved by replacing the
   sampler.** The Bayesian animal-model Gibbs proved fragile: PX-DA / ASIS /
   blocked-`(l,u)` / joint-`u` variants all left ESS in the single digits and
   structure-dependent bias (A-only itself hit 0.69 on a 4-sib pedigree). The fix
   was to abandon it for a **multiple Haseman–Elston regression** — the validated
   `fit_heritability` data-augmentation generalised to several relationship
   matrices at once. It is unbiased and precise for `A` and `A+C` across
   structures with negligible false-positive `C` (`bench_variance_components.py`).
   Dominance `D` was dropped: from sib-only data the non-negativity constraint
   biases it upward (a spurious `D` on additive-only data), so it needs twin
   contrasts. The "experimental" label is lifted.

2. ~~**Multi-trait genetic correlations `r_g`.**~~ **Done.**
   `fit_genetic_correlation` estimates `r_g` between traits by a **cross-trait
   Haseman–Elston regression** — the multivariate analogue of the fit above:
   regress same-trait cross-relative products on `A` for each `h2_p`, cross-trait
   cross-relative products on `A` for the genetic covariance `G[p,q]`, and
   within-individual cross-trait products for the phenotypic correlation;
   `r_g = G/sqrt(h2_p h2_q)`. Validated ~unbiased near the null (no spurious `r_g`
   when traits are genetically independent but phenotypically correlated) with
   mild attenuation at large `|r_g|` (`bench_genetic_correlation.py`). An
   inverse-Wishart Gibbs on `G ⊗ A` remains an option only if posterior *draws* of
   `G` are wanted; the moment estimator covers the point estimate.

3. ~~**Bootstrap SE for `fit_heritability`.**~~ **Done.** `bootstrap_fit` resamples
   families with replacement and refits any estimator, returning a bootstrap SE and
   percentile CI. On one 3 000-family dataset it recovers a SE of 0.047 vs the
   reported `h2_se` of 0.002 (23×), matching the true across-dataset SD. Works for
   all three fitters (pass a `lambda` returning the quantity of interest).

## SEM-inspired inference

Bringing twin/family structural-equation-modelling strengths (model comparison,
likelihood-based inference) to the pedigree/registry setting.

- ~~**Significance tests for components / correlations.**~~ **Done.**
  `test_variance_component` (is `C` needed?) and `test_genetic_correlation` (is
  `r_g ≠ 0`?) — the frequentist analog of the SEM likelihood-ratio test, done as a
  **parametric bootstrap**: fit the null, simulate under it on the same pedigrees
  and thresholds, refit, locate the observed statistic. Because the null is
  simulated and refit the same way, the moment estimator's boundary bias cancels —
  validated calibrated under H0 (VC test uniform p, FPR ≈ nominal; `r_g` test
  controls Type-I error, slightly conservative) with good power. Case/control
  bounds only.

- ~~**ML / REML backend (Monte-Carlo EM).**~~ **Done.**
  `fit_variance_components(..., method="reml")` runs a Monte-Carlo EM
  maximum-likelihood fit: the E-step is the truncated-MVN liability draw already
  used; the M-step maximises the Gaussian likelihood of the imputed liabilities
  (`min log|Σ| + tr(Σ⁻¹ S)`) instead of the HE regression. Validated **unbiased and
  ~30 % more efficient** than HE, with a **model-based SE** from the observed
  information (outer product of per-family observed-data scores, via Fisher's
  identity) that approximates the true across-dataset SD (se/SD ≈ 0.9–1.4) — where
  the HE `se` understates it ~15-20×. Also returns a Monte-Carlo (GHK) observed-
  data **log-likelihood** and **AIC** for nested-model comparison (AIC strongly
  prefers `A+C` on real `A+C` data; near the boundary it under-penalises, so the
  parametric-bootstrap test above remains the calibrated decision tool).

- **Latent factor model on the multi-trait genetic covariance** (Genomic-SEM-lite):
  fit `G ≈ ΛΛ' + Ψ` to the estimated genetic covariance — does one genetic factor
  explain the `r_g` among traits?

## Medium-term — rigor and real data

4. ~~**Pedigree/kinship-matrix input.**~~ **Done.** `kinship_from_pedigree(id,
   father, mother)` builds the additive relationship matrix `A` from an arbitrary
   pedigree (recursive tabular method, handles inbreeding);
   `construct_covmat_from_kinship` turns `A` into the liability covariance and
   `estimate_liability_from_kinship` runs the sampler on it. This generalises past
   the fixed role grammar — it reproduces the role-based covariance and estimates
   entry-for-entry where they overlap, and additionally handles half-sibs of any
   degree, cousins and inbred pedigrees. (The role grammar itself is the compact
   special case; a role-less array interface — `A` + per-member bounds — replaces
   the originally-envisaged `families_from_pedigree` object builder.)

5. **Expand benchmark diagnostics.** Slope/intercept and tail calibration (not
   just correlation), PA fold-in-ordering sensitivity, large/rare/densely-
   affected pedigrees, and mixture validation.

6. **Censoring-aware CIPs.** Helpers and guidance for Kaplan–Meier /
   Aalen–Johansen incidence with competing risks (death, emigration), for
   registry data.

7. ~~**Sensitivity utility.**~~ **Done.** `liability_sensitivity(families,
   h2_values)` re-estimates over an h² grid and reports the cross-setting
   correlation of the scores (`min_corr` = worst-case rank stability) plus how the
   scale shifts. In practice `min_corr ≈ 0.97` across h² 0.2–0.8 — the assumed h²
   mostly rescales the liability without changing the ranking, so a linear GWAS on
   it is nearly invariant. Prevalence/CIP sensitivity (which moves the bounds, not
   the covariance) is done by rebuilding families per prevalence and comparing.

## Longer-term — scale and ecosystem

8. **Sparse covariance for a single giant pedigree** (thousands of relatives) —
   a sparse relationship matrix with sparse solves.

9. **Chunked/streaming driver** so biobank runs never hold all families in
   memory at once.

10. **End-to-end GWAS example** (HAPNEST genotypes → liability → LMM GWAS) plus
    covariate/residualisation helpers.

11. **API-docs site** (MkDocs/Sphinx) and PyPI packaging.

12. **Multi-trait PA approximation**, if it can be made accurate, for scalable
    multi-trait analysis.
