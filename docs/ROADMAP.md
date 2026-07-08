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
`fit_variance_components` is a Bayesian animal-model Gibbs (A/C/D components)
with HE initialisation and Rao-Blackwellisation; it is committed but marked
**experimental** — the formulation is correct, but the single-site threshold
sampler still mixes slowly.

**Benchmarks** (`benchmarks/`, `RESULTS.md`) cover accuracy, runtime scaling,
age-of-onset, and GWAS power (LT-FH++ and PA both ~1.52× effective-N over
case/control at λ_GC ≈ 1), plus `fit_heritability` quality (unbiased, but
`h2_se` understates the true SD ~20–30×, so bootstrap). Real-LD runs go through
an opt-in HAPNEST path.

**Docs.** README, a user guide, and an algorithm/model doc (with the
BLUP / selection-index framing, the Pak–Sham liability-threshold-risk
connection, and the environmental-covariance extension), plus `CITATION.cff`
(15 references).

The test suite is 132 tests passing.

## Near-term — finish the variance-component thread

1. **Fix the animal-model Gibbs mixing.** Implement parameter-expanded data
   augmentation (PX-DA) / blocked `(l, u)` updates per Sorensen & Gianola.
   Validate unbiasedness across h² and family structures, and that A+C and A+D
   recover the real components with no false positives. Then lift the
   "experimental" label. This unblocks item 2.

2. **Multi-trait genetic correlations `r_g`.** Extend the sampler to
   `a ~ N(0, G ⊗ A)` with an inverse-Wishart update on `G` (the bipred `iw_df`
   pattern) — genetic correlations via a Gibbs sampler in the animal-breeding
   style.

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
