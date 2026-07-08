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
MZ/DZ twin contrasts to estimate honestly.)

**Benchmarks** (`benchmarks/`, `RESULTS.md`) cover accuracy, runtime scaling,
age-of-onset, and GWAS power (LT-FH++ and PA both ~1.52× effective-N over
case/control at λ_GC ≈ 1), plus `fit_heritability` quality (unbiased, but
`h2_se` understates the true SD ~20–30×, so bootstrap) and `A+C` recovery. Real-LD
runs go through an opt-in HAPNEST path.

**Docs.** README, a user guide, and an algorithm/model doc (with the
BLUP / selection-index framing, the Pak–Sham liability-threshold-risk
connection, and the environmental-covariance extension), plus `CITATION.cff`
(15 references).

The test suite is 134 tests passing.

## Near-term — finish the variance-component thread

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

2. **Multi-trait genetic correlations `r_g`.** Estimate the genetic covariance
   between traits. The natural fit is the multivariate analogue of the HE
   regression above (regress cross-trait liability products on the relationship
   matrix), or a multi-trait Gibbs with `a ~ N(0, G ⊗ A)` and an inverse-Wishart
   update on `G` (the bipred `iw_df` pattern) if posterior draws of `G` are
   wanted. Note the single-trait animal-model Gibbs is no longer a dependency.

3. **Bootstrap SE for `fit_heritability`.** Family resampling for honest
   confidence intervals, since `h2_se` understates uncertainty.

## Medium-term — rigor and real data

4. **Pedigree/kinship-matrix input.** `families_from_pedigree(id, mother,
   father)` and `construct_covmat_from_kinship`, generalising past the fixed
   role grammar (the R LTFGRS graph feature) and unlocking arbitrary pedigrees.

5. **Expand benchmark diagnostics.** Slope/intercept and tail calibration (not
   just correlation), PA fold-in-ordering sensitivity, large/rare/densely-
   affected pedigrees, and mixture validation.

6. **Censoring-aware CIPs.** Helpers and guidance for Kaplan–Meier /
   Aalen–Johansen incidence with competing risks (death, emigration), for
   registry data.

7. **Sensitivity utility.** An h²/prevalence/CIP sensitivity analysis in one
   call, to make the "run a sensitivity analysis" advice concrete.

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
