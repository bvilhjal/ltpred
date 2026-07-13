# ltpred benchmark results

Summary of the ltpred benchmark suite, comparing the two fitting methods — the
**Gibbs sampler** (LT-FH++) and the deterministic **Pearson–Aitken** estimator
(PA-FGRS) — on simulated data where the true genetic liability is known.

- **Generated:** 2026-07-07 (regenerated after the efficiency pass — precision-
  matrix Gibbs params, canonical grouping, streamed batch means, faster PA
  kernels), numpy 2.2.6 / scipy 1.15 / numba 0.66, 10 cores. Accuracy / GWAS /
  age-of-onset numbers are unchanged (the optimisations are equivalent); Gibbs
  timings dropped ~15–20%.
- **Reproduce:** `OMP_NUM_THREADS=10 python benchmarks/<script>.py` (see
  [`README.md`](README.md) for what each measures).
- **Caveat:** these are *stochastic* benchmarks — each number is one Monte-Carlo
  draw, so re-running shifts values by sampling noise. The conclusions are stable.
  They validate PA-FGRS against Gibbs for the **simulated structures included
  here** (small/realistic pedigrees, additive-genetic model); they do not prove
  exact equivalence for arbitrary pedigrees, extreme prevalences/heritabilities,
  densely affected families, or the censoring mixture. The reported metric is a
  correlation — it can hide scale/tail/calibration shifts; extending the diagnostics
  (mean error, slope/intercept, tail calibration, PA fold-in ordering, large/rare
  pedigrees) is future work.

## Headline findings

- **PA-FGRS closely matches the Gibbs LT-FH++ posterior mean.** It is a
  deterministic moment approximation (exact for a single truncation); across all 27
  accuracy cells the two estimates correlate **≥ 0.997** (usually ≥ 0.999), and in
  the genotype GWAS they give the **same** effective sample size (1.52×).
- **PA-FGRS is 100–350× faster** — a deterministic sweep with no MCMC —
  processing **~170 000 families/second** vs ~700/s for the Gibbs sampler, at
  identical accuracy. (Via the object API; the array API removes the remaining
  Python overhead — see the efficiency note below.)
- **Both recover 1.2–2.8× the effective sample size of a raw case/control
  label.** The gain grows with heritability, with *lower* prevalence, and with
  more informative relatives (siblings, extended pedigrees).
- **The power gain carries into a genotype GWAS with no inflation:** LT-FH++ and
  PA-FGRS both reach **1.52× effective N** over case/control at λ_GC ≈ 1.0; the
  oracle (true genetic liability) ceiling is 9.6×.
- **Age-of-onset (the liability→onset map) adds a further ~1–10%**, growing with
  prevalence; PA and Gibbs exploit it identically.

---

## 1. Accuracy — Gibbs vs PA-FGRS (`bench_accuracy.py`)

corr(estimate, true g) at **h²=0.5**, 1500 families, and the effective-N gain
over the raw case/control label. Gibbs and PA are indistinguishable; PA is
~250–350× faster per cell.

| Family | Prev | corr Gibbs | corr PA | eff-N gain | Gibbs≈PA |
|---|---:|---:|---:|---:|---:|
| parents | 0.05 | 0.394 | 0.394 | 1.30× | 0.9997 |
| parents+2 sibs | 0.05 | 0.454 | 0.454 | 1.88× | 0.9997 |
| extended (7 rel) | 0.05 | 0.374 | 0.374 | 2.16× | 0.9997 |
| parents+2 sibs | 0.01 | 0.296 | 0.297 | 2.83× | 0.9990 |
| parents+2 sibs | 0.20 | 0.601 | 0.601 | 1.54× | 0.9999 |

Accuracy rises with heritability and prevalence; the *relative* gain over
case/control is largest at low prevalence and with siblings in the pedigree
(rarer cases carry more family-history information). Minimum Gibbs–PA agreement
across the whole grid was 0.997.

## 2. Runtime scaling (`bench_scaling.py`)

Wall time, h²=0.5, K=0.05, Numba on 10 cores.

| #families (trios) | Gibbs | PA-FGRS | speed-up |
|---:|---:|---:|---:|
| 1 000 | 1.5 s | 0.006 s | 244× |
| 2 000 | 3.0 s | 0.012 s | 252× |
| 8 000 | 11.4 s | 0.046 s | 247× |

Both scale linearly in the number of families; PA sustains ~170 000 families/s
against ~700/s for Gibbs. Growing the family from 2 to 10 relatives raises the
Gibbs cost 2.5 s → 8.4 s (n=2000) while PA stays under 0.03 s — the speed-up
holds (230–350×) across family sizes. These use the object API; the **array API**
(`estimate_liability_pa_arrays`) removes the per-family Python overhead entirely —
measured ~10 M families/s (≈100× over the object PA path) on trios after warm-up.

## 3. Age-of-onset information (`bench_age_onset.py`)

Pinning cases at their onset threshold (ADuLT/LT-FH++) vs plain case/control,
both fit with PA (Gibbs shown to agree). corr(estimate, true g), 3000 families,
8-relative pedigree.

| h² | Prev | case/control | age-of-onset (PA) | Gibbs | gain |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.428 | 0.433 | 0.433 | 1.03× |
| 0.5 | 0.30 | 0.643 | 0.673 | 0.673 | 1.10× |
| 0.8 | 0.30 | 0.751 | 0.784 | 0.783 | 1.09× |

Onset information helps most when cases are common (more relatives contribute an
onset age); at low prevalence the gain is small because case relatives are rare.
PA and Gibbs use the onset map identically (curves overlap).

## 4. GWAS power (`bench_gwas_power.py`)

Linear-regression GWAS on 10 000 probands × 5 000 SNPs (30 causal), h²=0.5,
K=0.05, trios. Phenotypes: case/control, LT-FH++ (Gibbs), PA-FGRS, and the oracle
true genetic liability.

| Phenotype | mean χ² at causal | eff-N vs c/c | power (p<5e-8) | λ_GC |
|---|---:|---:|---:|---:|
| case/control | 35.4 | 1.00× | 36.7% | 0.99 |
| **LT-FH++ (Gibbs)** | 53.8 | **1.52×** | 43.3% | 1.01 |
| **PA-FGRS** | 53.8 | **1.52×** | 43.3% | 1.02 |
| oracle (true g) | 341.5 | 9.64× | 76.7% | 1.04 |

Both family-based estimators lift the mean association χ² at causal SNPs by ~52%
— a real effective-sample-size gain — and raise detection power, while staying
calibrated at null SNPs (λ_GC ≈ 1). LT-FH++ and PA-FGRS are interchangeable in
power; the oracle marks the ceiling if the genetic liability were known exactly.
For real-LD genotypes, rerun with `--plink` on a HAPNEST fileset
([`hapnest/README.md`](hapnest/README.md)).

## 5. Variance-component inference (`bench_fit_heritability.py`)

Quality of `fit_heritability` — the data-augmentation Gibbs that *fits*
liability-scale h² from family statuses — measured by fitting many independent
simulated cohorts (25 replicates), so the spread of the fits is the true sampling
distribution. Prevalence 0.10, `parents+2 sibs` unless noted.

**Bias & precision** at 3000 families:

| true h² | fitted (mean) | bias | SD (across datasets) | reported `h2_se` | SD / se |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.187 | −0.013 | 0.046 | 0.0022 | 21× |
| 0.4 | 0.386 | −0.014 | 0.057 | 0.0019 | 31× |
| 0.6 | 0.596 | −0.004 | 0.061 | 0.0025 | 24× |
| 0.8 | 0.786 | −0.014 | 0.061 | 0.0025 | 24× |

- **Approximately unbiased** across 0.2–0.8 (|bias| ≤ 0.014, within the
  replicate-averaging noise).
- **Precision improves with data**: SD falls ~`1/√N` (0.069 at 1 000 families →
  0.029 at 4 000–8 000), and with more informative relatives — 0.098 (parents
  only) → 0.058 (parents + 2 sibs) → 0.046 (extended, 8 relatives) at 3 000
  families.
- **The reported `h2_se` under-states the true uncertainty by ~20–30×.** It is the
  *within-dataset* Monte-Carlo error of one fit, **not** the sampling SD across
  datasets. Do not use it as a confidence interval — use **`bootstrap_fit`**
  (family resampling) for a real CI. On one 3 000-family dataset it gives a
  bootstrap SE of 0.047 against a reported `h2_se` of 0.002 (a 23× gap), matching
  the across-dataset SD above — the same helper works for `fit_variance_components`
  and `fit_genetic_correlation`. (This is the single most important caveat of these
  estimators.)

## 6. Multi-component variance components (`bench_variance_components.py`)

`fit_variance_components` generalises `fit_heritability` to a **multiple**
Haseman–Elston regression, fitting additive `A` and common-environment `C`
together. Measured over 20 replicate cohorts, prevalence 0.10, a full-sib-rich
structure (`m, f, s1…s4`) so `C` — identified from the full-sib excess — is
powered.

**A+C recovery** at 3000 families:

| true (a², c²) | A fitted (bias) | A SD | C fitted (bias) | C SD |
|:---:|---:|---:|---:|---:|
| (0.4, 0.2) | 0.408 (+0.008) | 0.052 | 0.193 (−0.007) | 0.031 |
| (0.5, 0.1) | 0.484 (−0.016) | 0.048 | 0.101 (+0.001) | 0.032 |
| (0.3, 0.3) | 0.299 (−0.001) | 0.057 | 0.294 (−0.006) | 0.045 |
| (0.6, 0.0) | 0.579 (−0.021) | 0.047 | 0.011 (+0.011) | 0.007 |

- **Unbiased for both components** across the grid (|bias| ≤ 0.021), and the two
  do not trade off — `A` is pinned by the parent-offspring / grandparent
  relatednesses, `C` by the full-sib excess.
- **Negligible false positive.** Fitting `A, C` on purely additive data
  (true c² = 0) gives C = 0.013 ± 0.011 — it does not manufacture a
  common-environment component.
- **Precision improves ~`1/√N`**: C SD 0.070 → 0.051 → 0.036 → 0.026 from
  1 000 to 8 000 families. Same `h2_se` caveat as `fit_heritability` — bootstrap
  families for a CI.
- **Dominance is not offered.** From sib-only pedigrees `D` is identified only by
  the small full-sib excess beyond additive, so the non-negativity constraint
  biases it upward (a spurious `D` on additive-only data); honest estimation needs
  MZ-vs-DZ twin contrasts. (This replaced an earlier experimental Bayesian
  animal-model Gibbs that mixed poorly and was structure-dependent-biased.)

## 7. Genetic correlation (`bench_genetic_correlation.py`)

`fit_genetic_correlation` estimates the genetic correlation `r_g` between traits
by a cross-trait Haseman–Elston regression. Two traits, h² = (0.5, 0.4),
phenotypic correlation `r_p = 0.2`, prevalence 0.10, `parents+2 sibs`, 20
replicate cohorts.

**Bias & precision vs true r_g** at 3000 families:

| true r_g | fitted (mean) | bias | SD (across datasets) |
|---:|---:|---:|---:|
| 0.0 *(null)* | +0.016 | +0.016 | 0.075 |
| 0.3 | +0.289 | −0.011 | 0.058 |
| 0.6 | +0.572 | −0.028 | 0.065 |

- **No false positive at the null**: with genetically independent but
  *phenotypically* correlated traits (`r_p = 0.2`, `r_g = 0`), the fit returns
  +0.016 — it does not read the phenotypic correlation as a genetic one.
- **Approximately unbiased**, with a **mild attenuation at large `|r_g|`**
  (−0.03 at 0.6) from the bounded ratio estimator `G / √(h²_p h²_q)`.
- **Precision improves ~`1/√N`**: SD of `r_g` 0.132 → 0.098 → 0.072 → 0.038 from
  1 000 to 8 000 families. Same `se` caveat — bootstrap families for a CI.

## 8. Does modelling shared environment help? (`bench_shared_env.py`)

Families simulated under the true `A+C+E` model (so the proband's *true genetic
liability* `g` is known), then the genetic-liability score estimated under models
that ignore vs. fit the shared-environment component `C`. Metric: corr(estimate,
true `g`) — i.e. how well the score predicts the genetic value. h²=0.5, prevalence
0.10, proband + parents + a sib-ship, 4 replicates × 3 000 families.

**Accuracy vs true c²** (3 sibs):

| true c² | ignore C (fitted h²) | fit `A+C` | oracle `A+C` | gain | fitted h² (additive) |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.5195 | 0.5195 | 0.5196 | +0.000 | 0.48 ✓ |
| 0.1 | 0.5063 | 0.5064 | 0.5066 | +0.000 | 0.56 |
| 0.2 | 0.4866 | 0.4886 | 0.4888 | +0.002 | 0.66 |
| 0.3 | 0.4759 | 0.4813 | 0.4820 | +0.005 | **0.75** |

**Gain vs sib-ship size** (c²=0.3): 2 sibs +0.003, 4 sibs +0.006, 6 sibs +0.007.

- **Modelling `C` barely changes the score's prediction accuracy** — the gain is
  ≤ ~0.007 corr (≈1 % relative) even with strong shared environment (c²=0.3) and a
  large sib-ship. It grows with c² and sib-ship size but stays small, and the
  **fitted `A+C` captures essentially all of it** (it sits right at the oracle).
  The threshold-model BLUP is forgiving of the h²/c² split for *point prediction*.
- **The real cost of ignoring `C` is a badly inflated heritability.** Additive
  `fit_heritability` reports ĥ²=0.75 at c²=0.3 (true 0.5) — `C`'s sib resemblance
  leaks into ĥ² — whereas `fit_variance_components` recovers ≈(0.46, 0.28). So the
  practical value of fitting `C` is getting **h² and its interpretation right**
  (and hence calibration), not sharpening the per-person score.

## 9. Couple / spousal environment `M` (`bench_couple_env.py`)

`fit_variance_components` fits a **bank** of shared-environment components. Besides
sibship `C` it ships `M`, a couple/spousal environment that loads on the
genetically-unrelated mate pairs — the proband's parents `(m, f)` and the
grandparent couples. Families simulated from `a² A + s² K + e² I` (thresholded, so
ground truth is known), true `a²=0.4`, prevalence 0.10, structure
`o+s1+s2+m+f+mgm+mgf+pgm+pgf`, 25 replicate cohorts × 3 000 families.

**(a) `A+M` recovery** (mean fitted, bias in parentheses, across-cohort SD):

| true `m²` | `Â` (bias) | SD | `M̂` (bias) | SD |
|---:|---:|---:|---:|---:|
| 0.0 | 0.394 (−0.006) | 0.031 | 0.017 (+0.017) | 0.014 |
| 0.1 | 0.395 (−0.005) | 0.027 | 0.096 (−0.004) | 0.029 |
| 0.2 | 0.400 (+0.000) | 0.032 | 0.205 (+0.005) | 0.042 |
| 0.3 | 0.387 (−0.013) | 0.029 | 0.293 (−0.007) | 0.038 |

**(b) Bias in the additive-only `Â` from ignoring shared environment** — the *same*
variance `s²` placed once as `C` (sibship) and once as `M` (couple), then fit the
misspecified `("A",)` model:

| true `s²` | ignore `C` → `Â` (bias) | ignore `M` → `Â` (bias) |
|---:|---:|---:|
| 0.1 | 0.447 (**+0.047**) | 0.407 (+0.007) |
| 0.2 | 0.492 (**+0.092**) | 0.420 (+0.020) |
| 0.3 | 0.547 (**+0.147**) | 0.430 (+0.030) |

- **`A+M` is recovered unbiased** across the sweep (biases ≤ 0.013, within one
  across-cohort SD), with **no spurious `M`** at the null (`M̂=0.017`, a small
  boundary floor, not a manufactured component).
- **`C` and `M` bias heritability oppositely if omitted.** Ignoring a real sibship
  `C` inflates `Â` substantially and ~linearly in `s²` (up to +0.15 at `s²=0.3`,
  a 37 % over-estimate) because sibs share both genes and `C`; ignoring a real
  couple `M` barely moves `Â` (≤ +0.03, ~5× smaller) because mates are genetically
  unrelated (`A=0`) and carry ~no weight in the additive regression. So `M` is
  worth fitting to **quantify / test spousal resemblance** (shared environment or
  assortative mating, which parent data cannot separate), not to de-bias `h²`.

## 10. LT-FH / LT-FH++ vs case/control — ascertainment, heritability, birth cohort (`bench_fh_prediction.py`)

The GWAS *phenotype* is `E[g | own status + family history (+ age of onset + birth
cohort)]`, scored against the plain **case/control** label, using **population** CIP
thresholds so it stays valid under case ascertainment (Pedersen 2022/2023).

**Age-, cohort-, and mortality-consistent generative model.** A liability `ℓ` is
fixed; the threshold `T(age; birth_year) = Φ⁻¹(1 − CIP(age; by))` falls with age and
shifts with birth cohort — `CIP(age; by) = K(by)/(1+exp((60−age)/8))`, cohort
prevalence `K(by) = K·R^((by−1965)/30)`. `ℓ` has an onset age under its *own*
cohort's CIP. **Death is a competing risk**: each relative has an age at death
(other-cause) and is observed only up to `c = min(death, age now)`; **observed case**
iff onset ≤ `c` (pinned at `T(a*) ≈ ℓ`), else **censored control** at `(−∞, T(c))` —
a disease-free death is a control censored at the death age, not a phantom
centenarian. Ages are generationally consistent (proband 40–70 born ≈1950–1980,
parents ≈29–31 y older, grandparents ≈56–58 y), so with mortality grandparents are
observed to death (~80, ~98 % deceased, correct ≈1908 cohort); observed prevalence
tracks the CIP (~2 % at K=0.05, not the lifetime K). 3-generation pedigree, 4 000
families × 3 reps, PA backend.

Two threshold policies are compared: **cohort-aware** — the LT-FH++/ADuLT personalised
threshold `Tᵢ = Φ⁻¹(1 − K(ageᵢ; birth_yearᵢ))`, anchoring each person to *their own*
cohort's prevalence `K(by)` — versus **single-K** — the classical-LTM / original-LT-FH
baseline that uses **one** lifetime prevalence `K` (the 1965 reference) for *everyone*,
blind to birth cohort (and sex). They coincide when there is no secular trend (`R=1`).

**(a) vs ascertainment** (proband case fraction `P`; h²=0.5, K=0.05, no trend):

| P | cc | count | LT-FH | LT-FH++ | LT-FH/cc | LT-FH++/LT-FH |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 (pop) | 0.228 | 0.266 | 0.344 | 0.346 | **2.27×** | 1.016× |
| 0.10 | 0.471 | 0.318 | 0.527 | 0.532 | 1.25× | 1.019× |
| 0.25 | 0.627 | 0.357 | 0.661 | 0.669 | 1.11× | 1.024× |
| 0.50 | 0.699 | 0.376 | 0.730 | 0.744 | 1.09× | 1.040× |

**(b) vs heritability** (K=0.05, 50 % ascertained):

| h² | cc | LT-FH | LT-FH++ | LT-FH/cc | LT-FH++/LT-FH |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.490 | 0.522 | 0.535 | 1.13× | 1.050× |
| 0.4 | 0.650 | 0.679 | 0.696 | 1.09× | 1.053× |
| 0.6 | 0.740 | 0.769 | 0.786 | 1.08× | 1.045× |
| 0.8 | 0.807 | 0.828 | 0.846 | 1.05× | 1.043× |

**(c) vs secular prevalence trend `R`** (× per 30 y; h²=0.5, K=0.05, 50 % ascertained)
— cohort-aware vs single-`K` thresholds on the **pedigree**; here the harm is mostly
**bias** (the living proband spans a narrow cohort — see (d) for the ranking gain):

| R | corr (cohort-aware) | corr (single-K) | single-K liability bias |
|---:|---:|---:|---:|
| 1.0 | 0.742 | 0.742 | +0.000 |
| 1.5 | 0.740 | 0.740 | −0.040 |
| 2.0 | 0.741 | 0.740 | −0.058 |
| 3.0 | 0.741 | 0.739 | **−0.073** |

**(d) cohort *ranking* gain vs cohort span of cases** (own age of onset, no family,
R=3) — isolates the re-ranking the pedigree dilutes:

| cohort half-span | corr (cohort-aware) | corr (single-K) | Δ |
|---|---:|---:|---:|
| ±10 y (1955–1975) | 0.426 | 0.422 | +0.004 |
| ±25 y (1940–1990) | 0.456 | 0.414 | +0.042 |
| ±40 y (1925–2005) | 0.506 | 0.414 | +0.092 |
| ±55 y (1910–2020) | **0.524** | 0.418 | **+0.106** |

- **Family history (LT-FH over case/control) is largest for rare observed disease.**
  In a population sample the observed prevalence is ~2 %, case/control is weak
  (corr 0.23), and family history is worth **2.27× effective N**; under 50 %
  ascertainment the balanced label is strong (0.70) and the gain shrinks to 1.09×.
- **Age of onset (LT-FH++ over LT-FH) grows with ascertainment** (1.6 % → 4.0 %) and,
  separately, with prevalence (reproduce with `--K`: it reached ~10 % at K=0.20) —
  the Pedersen 2022 direction (~4 % → ~18 % under ascertainment), at smaller magnitude
  (Gaussian not survival model, one pedigree, no sex thresholds, corr-based eff-N).
- **Heritability scales everything**; the family-history gain shrinks a little as h²
  rises because a high-h² case/control label is already informative.
- **Birth cohort helps both calibration *and* ranking.** Ignoring a secular
  prevalence trend (i) shifts the liability estimate systematically — up to
  **−0.073 at R=3**, a bias that correlates with birth year and can confound — and
  (ii) **loses ranking/power** whenever cases span a range of birth cohorts, because
  two cases with the *same age of onset* but different cohorts have *different* true
  liabilities (the one born in a low-prevalence era is more extreme), and a single-`K`
  analysis collapses them to one value. The ranking gain grows with the cohort span
  of the **cases** (panel (d)): negligible at ±10 y (0.426 vs 0.422 — like the narrow
  living-proband pedigree in (c)), rising to corr **0.524 (cohort-aware) vs 0.418
  (single-K)** at ±55 y — a **~1.57× effective-N** gain. So on a real pedigree the
  cohort correction shows up mostly as the (c) bias (the high-weight proband spans a
  narrow cohort; the wide-cohort grandparents are low-relatedness controls), but the
  underlying power gain (d) is large when cases themselves span many cohorts. This is
  *why* LT-FH++ personalises thresholds by birth year. (GWAS λ_GC not tested.)
- **Death as a competing risk** makes the ages realistic (relatives observed to death,
  grandparents ≈98 % deceased at ≈80) with the correct birth cohorts, and the results
  above are essentially unchanged from the no-mortality version — deceased relatives
  observed over their full life carry the same lifetime signal.

## 11. Genetic factor model (`bench_genetic_factor.py`)

`fit_genetic_factor` fits a common-factor model `r_g ≈ ΛΛ' + Ψ` to the genetic
correlation matrix from `fit_genetic_correlation` — the whole pipeline run
end-to-end on family case/control data (simulate → fit `r_g` → fit the factor
model). P = 5 traits, h² = (0.5, 0.45, 0.4, 0.35, 0.3), `parents+2 sibs`,
prevalence 0.10, 15 replicate cohorts of 3 000 families.

**(a) Single-factor loading recovery** (truth: one factor, loadings
Λ = 0.8, 0.7, 0.6, 0.5, 0.4):

| trait | true loading | fitted (mean ± SD) |
|---:|---:|---:|
| 0 | 0.80 | 0.779 ± 0.082 |
| 1 | 0.70 | 0.711 ± 0.088 |
| 2 | 0.60 | 0.594 ± 0.068 |
| 3 | 0.50 | 0.524 ± 0.057 |
| 4 | 0.40 | 0.411 ± 0.049 |

The one-factor fit gives `srmr = 0.046 ± 0.014` and `prop_explained = 0.984` — one
latent genetic factor reproduces the `r_g` matrix, with loadings recovered close to
truth (the mild attenuation on the largest loading is inherited from the `r_g`
estimator, §7).

**(b) Does `srmr` flag too few factors?** (truth: *two* independent genetic
factors, blocks {0,1,2} and {3,4}):

| model fit | `srmr` |
|---|---:|
| 1 factor (well-specified, from (a)) | 0.046 |
| 1 factor on two-factor data | **0.155** |
| 2 factors on two-factor data | 0.017 |

- **A mis-specified one-factor model triples the off-diagonal misfit** — `srmr`
  0.046 → 0.155 — while adding the second factor drops it back to 0.017. So `srmr`
  distinguishes "one general genetic axis" from "several genetic dimensions" even
  through the noisy pedigree → `r_g` → factor pipeline.
- It is a **descriptive decomposition of a point estimate**: the loadings carry no
  inference of their own, so bootstrap the whole pipeline over families for
  uncertainty (the within-dataset `se` understates it, as everywhere in §5–7).

## Bottom line

PA-FGRS is a drop-in, deterministic replacement for the LT-FH++ Gibbs sampler:
same accuracy and GWAS power to three decimals, two-to-three orders of magnitude
faster. Use `method="pearson-aitken"` for large biobank-scale runs and
`method="gibbs"` when you want posterior draws or a sampling-based check.

`fit_heritability` recovers liability-scale h² approximately without bias (SD
~0.05 at a few thousand informative families); `fit_variance_components` adds a
bank of shared-environment components — sibship `C` and couple `M` (both unbiased,
negligible false positives) — and `fit_genetic_correlation` recovers the genetic
correlation `r_g` between traits (unbiased near the null). On top of `r_g`,
`fit_genetic_factor` fits a common-factor model — one genetic factor cleanly
reproduces a one-factor `r_g` (srmr ≈ 0.05), and `srmr` rises sharply when the truth
has more factors. Report their uncertainty by bootstrapping families, not from the
reported `se`.
