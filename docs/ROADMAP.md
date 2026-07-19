# Roadmap

Where ltpred stands and where it is going. See [guide.md](guide.md) for usage,
[algorithm.md](algorithm.md) for the model and estimators, and
[benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for model and inference-engine comparisons.

## Where things stand

ltpred is a from-scratch Python port of LT-FH++. For ordinary liability bounds it
provides two alternative inference engines for posterior-mean genetic liability:

- **Gibbs** — a truncated-multivariate-normal sampler (Rcpp port) with
  batch-means convergence.
- **Pearson–Aitken (PA)** — a deterministic selection-formula sweep. Its optional
  PA-FGRS extension adds the age-censored-control mixture. On the **no-mixture**
  benchmark grid, PA and Gibbs posterior-mean estimates had correlation ≥0.997,
  and PA ran 315–510× faster in the controlled 10-thread timing benchmark. The
  PA-only mixture was not part of that comparison.

Both support classic LT-FH and personalised pinned bounds used as LT-FH++ with
relatives or ADuLT without them. The base PA-FGRS lifetime-case/censored-control-
mixture encoding is PA-only; the convenience helpers also expose an age-specific
interval-case variant that is not an exact implementation of PA-FGRS_ADT.
Multi-trait estimation is Gibbs-only (PA is single-trait).

**Performance and scale.** The core is Numba-JIT'd and `prange`-parallel, with
families grouped by structure (canonical form). Streaming batch-means keeps
standard-error memory at `O(F)`; the array API skips Python objects and, in the
current warmed timing run, adds another 6–31× over the PA object path while
processing 1.6–8.6 million already-aligned families/s. A float32 bounds option
halves memory. (The exploration also showed
why int8-quantising the covariance, ldpred3-style, is the wrong lever here.)

**Variance-component fitting.** `fit_heritability` is a data-augmentation
fixed point (Gibbs augmentation + damped Haseman–Elston update, not posterior
sampling of h²). Across h² 0.2–0.8, current simulations show small bias relative
to the across-dataset SD rather than exact unbiasedness.
`fit_variance_components` fits additive `A` and a **bank of relationship-specific
shared-environment components** — `C` (sibship, from the full-sib excess) and `M`
(couple, from the `A = 0` mate pairs) — together by a **multiple Haseman–Elston
regression** on the same well-mixing collapsed data-augmentation. Repository
benchmarks found small bias relative to sampling variability for `A`, `A+C` and
`A+M` across the tested family structures. At a
zero component, the constrained estimates show a small positive boundary floor;
formal false-positive control comes from the parametric-bootstrap component test.
The shipped environment components are equivalence-class partitions, a sufficient
construction for PSD kernels; any future kernel must likewise be symmetric and
PSD. A non-PSD vertical parent-offspring "environment" is rejected. Each fitted
kernel has diagonal one and therefore reduces the residual variance by its fitted
proportion.
(This replaced an earlier experimental Bayesian animal-model Gibbs, which mixed
poorly and showed structure-dependent bias. Dominance `D` is intentionally not
offered: it needs an explicit dominance kernel and independent relationship
contrasts. MZ/DZ observations can contribute within a richer design, but MZ/DZ
pairs alone cannot identify `A`, `C`, and `D` simultaneously.) `fit_genetic_correlation` estimates
the **genetic correlation `r_g`** between traits by the cross-trait analogue of
the same regression — approximately unbiased near the null with mild attenuation
at large `|r_g|` in the repository benchmarks. On top of that `r_g` matrix,
`fit_genetic_factor` fits a
**common-factor model `r_g ≈ ΛΛ' + Ψ`** (Genomic-SEM-lite, by MINRES): does one
latent genetic factor explain the correlations among the traits? — with `srmr` as
an in-sample misfit diagnostic, not a calibrated factor-number test.

**Pedigree discovery.** `ltpred.pedigree` goes from population trio records
(ids, father, mother) to per-proband pedigrees: BFS ego-extraction within
`max_degree` relationship-degrees (full-sib edges give the standard degree
scale) plus a full ancestral closure, making extracted kinship exactly equal
to the full-population values restricted to the members (benchmarked at
0.0 max abs diff; extraction ~0.1 ms/proband). This is the Pedersen et al.
(2025) graph-extraction niche, with exact tabular kinship in place of the
paper's path-counting approximation.

**Benchmarks** (`benchmarks/`, `RESULTS.md`) cover accuracy, runtime scaling,
age-of-onset, replicated classic LT-FH GWAS power (Gibbs and PA both
1.47 ± 0.04× causal-SNP NCP ratio over case/control at λ_GC ≈ 1), and an integrated
personalized LT-FH++ GWAS with
age-, sex-, and cohort-dependent CIP, plus `fit_heritability` quality (small bias
relative to sampling SD, while `h2_se` substantially understates that SD, so use
`bootstrap_fit`), `A+C` recovery,
and `r_g` recovery, plus **calibration** of the score (generally near the
posterior-mean target under the tested correct models, with a rare extended-family
outlier; score correlation changed little while scale moved under the tested wrong-`h²`
settings), **cohort
confounding / `λ_GC`** (personalised thresholds removed the tested
threshold-misspecification inflation after ordinary covariate adjustment), and
  **PA robustness / fold-order** (under no-mixture bounds, PA and Gibbs
  posterior-mean estimates had
correlation ≥ 0.998 on stressful pedigrees). Real-LD runs go through an opt-in
HAPNEST path.

**Docs.** README, a user guide, and an algorithm/model doc (with the
BLUP / selection-index framing, the Pak–Sham liability-threshold-risk
connection, and the environmental-covariance extension), plus `CITATION.cff`.

The full test suite passes (`pytest`), with CI running it and the `ruff` gate on
Python 3.9 and 3.12.

## Completed implementation history — variance components

1. ~~**Fix the animal-model Gibbs mixing.**~~ **Done — resolved by replacing the
   sampler.** The Bayesian animal-model Gibbs proved fragile: PX-DA / ASIS /
   blocked-`(l,u)` / joint-`u` variants all left ESS in the single digits and
   structure-dependent bias (A-only itself hit 0.69 on a 4-sib pedigree). The fix
   was to abandon it for a **multiple Haseman–Elston regression** — the validated
   `fit_heritability` data-augmentation generalised to several relationship
   matrices at once. In the repository benchmarks its bias was small relative to
   across-dataset variability for `A` and `A+C` across the tested structures. At
   true `C=0`, the constrained point estimate has a small positive
   boundary floor; it is not itself a false-positive rate
   (`bench_variance_components.py`).
   Dominance `D` was dropped: from sib-only data the non-negativity constraint
   biases it upward (a spurious `D` on additive-only data). It needs an explicit
   dominance kernel in a richer relationship design; MZ/DZ pairs alone cannot
   separate `A`, `C`, and `D`. The "experimental" label is lifted.

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
   all three fitters (pass a `lambda` returning the quantity of interest). Its
   sampling interpretation assumes iid, non-overlapping family clusters and a
   design compatible with the fitted model.

## Completed and partial implementation history — SEM-inspired inference

Bringing twin/family structural-equation-modelling strengths (model comparison,
likelihood-based inference) to the pedigree/registry setting.

- ~~**Significance tests for components / correlations.**~~ **Done.**
  `test_variance_component` (is `C` needed?) and `test_genetic_correlation` (is
  `r_g ≠ 0`?) — the frequentist analog of the SEM likelihood-ratio test, done as a
  **parametric bootstrap**: fit the null, simulate under it on the same pedigrees
  and fixed thresholds, refit, locate the observed statistic. This is a
  conditional plug-in calibration, not a guarantee of uniform null p-values: it
  depends on the fitted nuisance model, independent family clusters and the
  sampling design. Common case/control thresholds work automatically;
  individualized thresholds require an explicit assertion that they were fixed
  from baseline covariates, never age of onset.

- ~~**Approximate Monte-Carlo EM backend.**~~ **Done.**
  `fit_variance_components(..., method="mcem")` (aliases `"ml"`/`"reml"`) runs a
  finite-iteration, damped Monte-Carlo EM-style fit, *not* restricted ML: the
  E-step is the truncated-MVN liability draw already used; the
  M-step maximises the Gaussian likelihood of the imputed liabilities
  (`min log|Σ| + tr(Σ⁻¹ S)`) instead of the HE regression. The current implementation
  uses fixed damping and averages post-burn-in iterates rather than checking an
  observed-likelihood convergence criterion. Its OPG/BHHH information SE, GHK
  observed-data log-likelihood, and AIC are therefore approximate diagnostics with
  Monte-Carlo and finite-iteration error. Their sampling calibration must be
  validated for the target design; the conditional parametric-bootstrap test or
  family bootstrap remains the relevant uncertainty route under its assumptions.

- ~~**Latent factor model on the multi-trait genetic covariance** (Genomic-SEM-lite).~~
  **Done.** `fit_genetic_factor` fits `r_g ≈ ΛΛ' + Ψ` — a common-factor model — to
  the genetic correlation matrix from `fit_genetic_correlation`, by **MINRES**
  (minimising the off-diagonal residuals, so the factor(s) explain the cross-trait
  correlations, not each trait's own variance). `srmr` / `prop_explained` read off
  the fit; a single factor needs `P ≥ 3` traits, and `n_factors` must leave
  nominal `df = ½((P−m)²−(P+m)) ≥ 0`. At `P=3, m=1`, sign and communality
  constraints can still leave residual misfit despite nominal `df=0`. In the planted simulation it
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
  symmetric PSD kernel. The shipped kernels are **equivalence-class partitions**,
  which guarantees PSD but is not the only valid construction; the fitter rejects
  a kernel that is not PSD. This rules out the naive **vertical** parent-offspring
  "environment" considered here (its
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

4. ~~**Pedigree/kinship-matrix input.**~~ **Done.** `kinship_from_pedigree(ids,
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
   confirms **correlation ≥ 0.998 between PA and Gibbs posterior-mean estimates**
   on large/rare/densely-affected no-mixture pedigrees, with median fold-order spread below
   0.12% and p95 below 3.4% of the
   between-proband SD; `bench_ltfhpp_personalization.py` is the integrated genotype-GWAS
   benchmark with age-, sex-, and cohort-specific CIP, onset/follow-up, competing
   mortality, ascertainment, and stratified null variants. Its matched ADuLT arm
   uses identical proband bounds without relatives, directly measuring the
   LT-FH++ family-history increment. Its prespecified
   sex-isolation panel shows a clear stratum-calibration benefit but no resolved
   adjusted-power increment, rather than conflating those two claims. **Mixture validation**
   is now done (`bench_pafgrs_mixture.py`, RESULTS.md section 16): generative
   validation under both the threshold-crossing and the stochastic-onset
   observation models. The mixture is implemented correctly and never costs
   correlation, but its censoring correction is small at the tested settings;
   the case encoding (pinned vs lifetime vs interval) dominates calibration
   there.

6. ~~**Censoring-aware CIPs.**~~ **Done.** `ltpred/cip.py` estimates the
   cumulative-incidence curve from registry-style follow-up records (entry age,
   exit age, event code): `kaplan_meier_cip` (left-truncated product-limit,
   Greenwood SEs) for the no-competing-risks case, and `aalen_johansen_cip`
   (delayed entry, right censoring, death/emigration as competing events --
   the LT-FH++ construction, Pedersen et al. 2022) with the closed-form Aalen
   (1978) variance and an optional person-level bootstrap cross-check. Output
   feeds `thresholds_from_cip` directly. Validated in
   `bench_cip_estimation.py`: exact recovery of a known curve (max err ~0.002),
   the competing-risks overestimation of KM quantified (~0.022 at 42% death
   share), delayed entry handled, and end-to-end the estimated curve costs
   ~0.005 of calibration slope vs the oracle CIP.

7. ~~**Sensitivity utility.**~~ **Done.** `liability_sensitivity(families,
   h2_values)` re-estimates over an h² grid and reports the cross-setting
   Pearson correlation of the scores (`min_corr` = lowest cross-setting score
   correlation) plus how the scale shifts. In the tested pedigree,
   `min_corr ≈ 0.97` across h² 0.2–0.8, consistent with a near-linear rescaling in
   that setting; this is not general rank invariance. Prevalence/CIP sensitivity
   (which moves the bounds, not the covariance) is done by rebuilding families per
   prevalence and comparing.

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
