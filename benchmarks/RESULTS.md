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

## Bottom line

PA-FGRS is a drop-in, deterministic replacement for the LT-FH++ Gibbs sampler:
same accuracy and GWAS power to three decimals, two-to-three orders of magnitude
faster. Use `method="pearson-aitken"` for large biobank-scale runs and
`method="gibbs"` when you want posterior draws or a sampling-based check.

`fit_heritability` recovers liability-scale h² approximately without bias (SD
~0.05 at a few thousand informative families); `fit_variance_components` adds a
common-environment `C` (unbiased, negligible false positives) and
`fit_genetic_correlation` recovers the genetic correlation `r_g` between traits
(unbiased near the null). Report their uncertainty by bootstrapping families, not
from the reported `se`.
