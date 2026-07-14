# Roadmap

Where ltpred stands and where it is going. See [guide.md](guide.md) for usage,
[algorithm.md](algorithm.md) for the model and estimators, and
[benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for model and inference-engine comparisons.

## Where things stand

ltpred is a from-scratch Python port of LT-FH++. It provides two interchangeable
inference engines for posterior-mean genetic liability:

- **Gibbs** — a truncated-multivariate-normal sampler (Rcpp port) with
  batch-means convergence.
- **Pearson–Aitken (PA)** — a deterministic selection-formula sweep. Its optional
  PA-FGRS extension adds the age-censored-control mixture. PA agrees with Gibbs
  to correlation ≥0.997 across the benchmark grid and ran 323–569× faster in the
  controlled 10-thread timing benchmark.

Both support classic LT-FH; personalised pinned bounds used as LT-FH++ with
relatives or ADuLT without them; and the PA-FGRS interval/mixture encoding.
Multi-trait estimation is Gibbs-only (PA is single-trait).

**Performance and scale.** The core is Numba-JIT'd and `prange`-parallel, with
families grouped by structure (canonical form). Streaming batch-means keeps
standard-error memory at `O(F)`; the array API skips Python objects and, in the
current warmed timing run, adds another 7–31× over the PA object path while
processing 1.6–9.4 million already-aligned families/s. A float32 bounds option
halves memory. (The exploration also showed
why int8-quantising the covariance, ldpred3-style, is the wrong lever here.)

**Variance-component fitting.** `fit_heritability` is a data-augmentation
fixed point (Gibbs augmentation + damped Haseman–Elston update, not posterior
sampling of h²). Across h² 0.2–0.8, current simulations show small bias relative
to the across-dataset SD rather than exact unbiasedness.
`fit_variance_components` fits additive `A` and a **bank of relationship-specific
shared-environment components** — `C` (sibship, from the full-sib excess) and `M`
(couple, from the `A = 0` mate pairs) — together by a **multiple Haseman–Elston
regression** on the same well-mixing collapsed data-augmentation, validated
approximately unbiased for `A`, `A+C` and `A+M` across family structures. At a
zero component, the constrained estimates show a small positive boundary floor;
formal false-positive control comes from the parametric-bootstrap component test.
Environment components are validated to be equivalence-class
(PSD) partitions, so a non-PSD vertical parent-offspring "environment" is rejected.
(This replaced an earlier experimental Bayesian animal-model Gibbs, which mixed
poorly and showed structure-dependent bias. Dominance `D` is intentionally not
offered — it needs MZ/DZ twin contrasts to estimate honestly.) `fit_genetic_correlation` estimates
the **genetic correlation `r_g`** between traits by the cross-trait analogue of
the same regression — validated ~unbiased near the null with mild attenuation at
large `|r_g|`. On top of that `r_g` matrix, `fit_genetic_factor` fits a
**common-factor model `r_g ≈ ΛΛ' + Ψ`** (Genomic-SEM-lite, by MINRES): does one
latent genetic factor explain the correlations among the traits? — with `srmr` as
an in-sample misfit diagnostic, not a calibrated factor-number test.

**Benchmarks** (`benchmarks/`, `RESULTS.md`) cover accuracy, runtime scaling,
age-of-onset, replicated classic LT-FH GWAS power (Gibbs and PA both
1.47 ± 0.04× causal-SNP NCP ratio over case/control at λ_GC ≈ 1), and an integrated
personalized LT-FH++ GWAS with
age-, sex-, and cohort-dependent CIP, plus `fit_heritability` quality (small bias
relative to sampling SD, while `h2_se` understates that SD ~20–30×, so use
`bootstrap_fit`), `A+C` recovery,
and `r_g` recovery, plus **calibration** of the score (generally near the
posterior-mean target under the tested correct models, with a rare extended-family
outlier; ranking robust but scale sensitive to a wrong `h²`), **cohort
confounding / `λ_GC`** (personalised thresholds keep genomic control valid), and
**PA robustness / fold-order** (PA tracks Gibbs to corr ≥ 0.998 on stressful
pedigrees). Real-LD runs go through an opt-in HAPNEST path.

**Docs.** README, a user guide, and an algorithm/model doc (with the
BLUP / selection-index framing, the Pak–Sham liability-threshold-risk
connection, and the environmental-covariance extension), plus `CITATION.cff`
(15 references).

The full test suite passes (`pytest`), with CI running it and the `ruff` gate on
Python 3.9 and 3.12.

## Near-term — variance-component thread ✅ complete

1. ~~**Fix the animal-model Gibbs mixing.**~~ **Done — resolved by replacing the
   sampler.** The Bayesian animal-model Gibbs proved fragile: PX-DA / ASIS /
   blocked-`(l,u)` / joint-`u` variants all left ESS in the single digits and
   structure-dependent bias (A-only itself hit 0.69 on a 4-sib pedigree). The fix
   was to abandon it for a **multiple Haseman–Elston regression** — the validated
   `fit_heritability` data-augmentation generalised to several relationship
   matrices at once. It is unbiased and precise for `A` and `A+C` across
   structures. At true `C=0`, the constrained point estimate has a small positive
   boundary floor; it is not itself a false-positive rate
   (`bench_variance_components.py`).
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

- ~~**ML backend (Monte-Carlo EM).**~~ **Done.**
  `fit_variance_components(..., method="mcem")` (aliases `"ml"`/`"reml"`) runs a
  Monte-Carlo EM **maximum-likelihood** fit — ML on the imputed liabilities, *not*
  restricted ML: the E-step is the truncated-MVN liability draw already used; the
  M-step maximises the Gaussian likelihood of the imputed liabilities
  (`min log|Σ| + tr(Σ⁻¹ S)`) instead of the HE regression. In the benchmarked
  configurations it was **unbiased and ~30 % more efficient** than HE, with an
  **approximate model-based SE** (an OPG/BHHH observed-information estimate, subject
  to Monte-Carlo error) that approximated the true across-dataset SD (se/SD ≈
  0.9–1.4) — where the HE `se` understates it ~15-20×. Also returns a Monte-Carlo
  (GHK) observed-data **log-likelihood** and **AIC** (both Monte-Carlo estimates)
  for nested-model comparison (AIC strongly prefers `A+C` on real `A+C` data; near
  the boundary it under-penalises, so the parametric-bootstrap test above remains
  the calibrated decision tool).

- ~~**Latent factor model on the multi-trait genetic covariance** (Genomic-SEM-lite).~~
  **Done.** `fit_genetic_factor` fits `r_g ≈ ΛΛ' + Ψ` — a common-factor model — to
  the genetic correlation matrix from `fit_genetic_correlation`, by **MINRES**
  (minimising the off-diagonal residuals, so the factor(s) explain the cross-trait
  correlations, not each trait's own variance). `srmr` / `prop_explained` read off
  the fit; a single factor needs `P ≥ 3` traits (and `P ≥ 4` to *test* it), and
  `n_factors` must leave `df = ½((P−m)²−(P+m)) ≥ 0`. In the planted simulation it
  recovers loadings end-to-end, and `srmr` rises when a one-factor model is fit to
  two-factor data (`bench_genetic_factor.py`). That demonstrates a diagnostic,
  not validated model selection. It is a descriptive decomposition of a
  point-estimate `r_g` (bootstrap the pipeline for uncertainty); an optional DWLS
  weighting hook is there for when honest per-`r_g` weights are supplied.

- **Relationship-specific environmental components.** *Partly done.* The single `C`
  is now a **bank**: `_COMPONENT_OFFDIAG` ships `C` (full-sib / sibship) and `M`
  (couple / spousal, identified from the `A = 0` mate pairs), fitted jointly with
  `A` by `fit_variance_components(fams, ("A", "C", "M"))` and testable with
  `test_variance_component(fams, "M")`. Each environment component must be a valid
  **equivalence-class partition** (PSD `K_c`); the fitter now rejects one that is
  not — which rules out a naive **vertical** parent-offspring "environment" (its
  sharing chains across generations, so `K_c` is indefinite; that is the directional
  maternal-effect case below, not a symmetric variance component). Remaining bank
  ideas that *are* valid partitions: a maternal-lineage rearing environment
  (full-sibs + maternal half-sibs) and cousin environments — each needs the
  matching relative types present. The binding constraint stays identifiability —
  #components ≤ #distinct relationship contrasts — so extra components pay off on
  **extended registry pedigrees**, not nuclear families (see algorithm.md,
  *Relationship-specific environments and identifiability*). Note the caveats there:
  symmetric shared-environment ≠ directional maternal effect, and an environment
  `∝ A` is confounded with `h2`.

## Medium-term — rigor and real data

4. ~~**Pedigree/kinship-matrix input.**~~ **Done.** `kinship_from_pedigree(id,
   father, mother)` builds the additive relationship matrix `A` from an arbitrary
   pedigree (recursive tabular method, handles inbreeding);
   `construct_covmat_from_kinship` turns `A` into the liability covariance and
   `estimate_liability_from_kinship` runs PA by default or Gibbs on request. This generalises past
   the fixed role grammar — it reproduces the role-based covariance and estimates
   entry-for-entry where they overlap, and additionally handles half-sibs of any
   degree, cousins and inbred pedigrees. (The role grammar itself is the compact
   special case; a role-less array interface — `A` + per-member bounds — replaces
   the originally-envisaged `families_from_pedigree` object builder.)

5. **Expand benchmark diagnostics.** *Mostly done.* Four benchmarks landed:
   `bench_calibration.py` adds slope/intercept and **decile (tail) calibration** to
   the correlation-only accuracy story (most correctly specified cells are near
   slope 1, with a rare extended-family outlier; a wrong `h²` tilts the scale but
   barely changes ranking — the complement to `liability_sensitivity`);
   `bench_confounding.py` shows
   cohort-blind (single-K) thresholds **inflate `λ_GC`** under a secular prevalence
   trend while the cohort-aware family ablation stays near 1 on average;
   `bench_pa_robustness.py`
   confirms **PA tracks Gibbs to corr ≥ 0.998** on large/rare/densely-affected
   pedigrees, with median fold-order spread below 0.12% and p95 below 3.4% of the
   between-proband SD; `bench_ltfhpp_personalization.py` is the integrated genotype-GWAS
   benchmark with age-, sex-, and cohort-specific CIP, onset/follow-up, competing
   mortality, ascertainment, and stratified null variants. Its matched ADuLT arm
   uses identical proband bounds without relatives, directly measuring the
   LT-FH++ family-history increment. Its prespecified
   sex-isolation panel shows a clear stratum-calibration benefit but no resolved
   adjusted-power increment, rather than conflating those two claims. **Mixture validation**
   remains deferred: the implementation now correctly splits at the lifetime
   threshold (the age-specific upper bound only flags censoring), but a dedicated
   generative benchmark against the PA-FGRS censoring model is still needed.

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

10. **End-to-end real-LD GWAS example** (HAPNEST genotypes → liability → LMM GWAS).
    The independent-SNP integrated LT-FH++ benchmark and covariate residualisation
    are now present; real-LD orchestration remains opt-in work.

11. **PyPI packaging.** The MkDocs API/user-guide site and strict CI build are in place.

12. **Multi-trait PA approximation**, if it can be made accurate, for scalable
    multi-trait analysis.
