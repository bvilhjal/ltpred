# ltpred benchmark results

Current results for the 15 local benchmark scripts in this directory. Bounds
distinguish non-personalised, personalised pinned and interval-case encodings;
family-history inclusion distinguishes LT-FH++ (with relatives) from ADuLT
(index person only). Gibbs and Pearson–Aitken (PA) are alternative inference
engines for those bounds. PA-FGRS is a separate PA-specific specification; its
censoring mixture is not included in the PA–Gibbs comparisons below.

- **Generated:** 2026-07-14.
- **Environment:** Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, Numba 0.61.0,
  10 logical cores.
- **Reproduce:** set `NUMBA_NUM_THREADS=10` (and optionally
  `OMP_NUM_THREADS=10` for linked numerical libraries), then run the scripts
  listed in [`README.md`](README.md).
- **Coverage:** every default local benchmark was rerun. The HAPNEST real-LD path
  was not run because it requires an external multi-GB dataset; it remains an
  explicit opt-in workflow in [`hapnest/README.md`](hapnest/README.md).

Unless stated otherwise, `±` denotes the standard error of a mean across
independent simulated cohorts. Tables labelled SD instead report the empirical
across-cohort standard deviation. Some diagnostic grids remain single-seed
illustrations; those are identified rather than dressed up as certainty.

Metric names matter here. **NCP ratio** means a ratio of causal-SNP chi-square
noncentrality components. **Eff-N proxy** means a squared-correlation ratio to
case/control. The latter is useful for prediction comparisons but is not an
observed GWAS noncentrality ratio, so the two magnitudes are not interchangeable.

## Headline findings

- **PA is the right default for the tested single-trait, no-mixture work.** Across the 27-cell accuracy
  grid, corr(PA, Gibbs) is 0.9972–0.9999. The stressful-pedigree benchmark remains
  at least 0.9984. PA and Gibbs also give indistinguishable downstream GWAS
  results. In the isolated 10-thread timing run, the PA object path is
  **315–510× faster** than grouped Gibbs across the tested sizes and pedigrees.
- **Classic LT-FH improves genotype-GWAS signal without average null inflation.**
  Across three genotype/effect/cohort replicates, the same classic LT-FH model
  inferred by either PA or Gibbs delivers a causal-SNP NCP ratio of
  **1.47 ± 0.04×** over case/control.
- **Full personalised LT-FH++ adds family-history value beyond matched ADuLT.**
  With identical age/sex/cohort proband bounds, ADuLT reaches **1.049 ± 0.004×**
  adjusted causal-SNP NCP ratio over case/control and full LT-FH++ reaches
  **1.194 ± 0.006×**. The paired LT-FH++ minus ADuLT increment is
  **+0.1454 ± 0.0154** NCP-ratio units (95% CI half-width). Full LT-FH++ has
  adjusted calibration slope **0.995 ± 0.015** and PA/Gibbs agreement 0.99990.
- **Sex-specific CIP improves stratum calibration, not proven adjusted power.**
  In a prespecified sex-only scenario, correct sex curves close the female-minus-
  male mean-score-error gap by **0.05091 ± 0.00096** (paired 95% CI), while the
  adjusted NCP-ratio increment is **0.0040 ± 0.0046** and remains unresolved.
- **Cohort personalisation has two distinct benefits.** In a narrow living-proband
  pedigree it mainly corrects a mean-score shift; in the family-free ADuLT panel,
  cases spanning ±55 birth years reach corr 0.512 ± 0.007 with cohort-aware
  thresholds versus 0.397 ± 0.017 when cohort is ignored (about 1.66× in
  the squared-correlation eff-N proxy).
- **Variance-component point estimates need sampling uncertainty.** The fitter's
  reported Monte Carlo SE is much smaller than empirical across-cohort SD. Use
  family bootstrap intervals for inference. A constrained component estimate at
  zero is a boundary estimate, not a false-positive rate.

## 1. PA versus Gibbs accuracy (`bench_accuracy.py`)

Single simulated cohorts, 1,500 families per cell. At h²=0.5 and K=0.05:

| Family structure | corr Gibbs | corr PA | PA squared-correlation eff-N proxy / case-control | corr(PA, Gibbs) |
|---|---:|---:|---:|---:|
| parents | 0.394 | 0.394 | 1.29× | 0.9997 |
| parents + 2 siblings | 0.440 | 0.440 | 1.78× | 0.9997 |
| extended | 0.412 | 0.412 | 1.81× | 0.9997 |

Across all 27 cells, agreement is 0.9972–0.9999. Absolute accuracy increases
with prevalence, heritability, and informative relatives. Relative gain over a
case/control label is often largest for rarer disease; the largest current grid
value is 2.67× (parents + siblings, h²=0.2, K=0.05).

## 2. Controlled runtime scaling (`bench_scaling.py`)

Five warmed timings per point, reported as medians; Python 3.13.5, Numba 0.61.0,
10 Numba threads, h²=0.5, K=0.05, and 25,000 Gibbs draws. No other benchmark ran
concurrently.

### Scaling with number of families (parents + one sibling)

| families | Gibbs families/s | PA object families/s | PA array families/s | object speed-up |
|---:|---:|---:|---:|---:|
| 500 | 654 | 224,027 | 2.11 M | 343× |
| 1,000 | 606 | 211,930 | 3.18 M | 349× |
| 2,000 | 626 | 261,006 | 1.55 M | 417× |
| 4,000 | 677 | 240,743 | 2.74 M | 355× |
| 8,000 | 724 | 280,243 | 8.63 M | 387× |

The object path includes `Family`/`Member` bounds assembly and grouping. The
array path receives already aligned, repeatedly reused arrays; its 1.6–8.6
million families/s is therefore a hot-kernel measurement, not end-to-end input
preparation. Its very short calls also make cache and scheduler effects visible,
so use the CSV IQRs rather than interpreting the non-monotone point rates.

### Family size at 2,000 families

| relatives | structure | Gibbs time | PA object time | PA array time | object speed-up |
|---:|---|---:|---:|---:|---:|
| 2 | parents | 2.03 s | 0.0064 s | 0.00028 s | 315× |
| 3 | + sibling | 2.67 s | 0.0073 s | 0.00032 s | 367× |
| 5 | + two grandparents | 3.66 s | 0.0101 s | 0.00054 s | 363× |
| 7 | extended | 5.05 s | 0.0102 s | 0.00058 s | 495× |
| 10 | extended + aunts | 6.73 s | 0.0132 s | 0.00090 s | 510× |

The small PA times are not strictly monotone; five repeats quantify timing
variation but do not abolish operating-system noise. The defensible claim on
this machine is the observed **315–510×** object-path speed-up, not a universal
hardware-independent constant.

This is the only benchmark used for performance claims. It warms all paths,
records five timings per point with median/IQR, uses `perf_counter`, records the
Numba thread count and simulation configuration, and measures PA's object and
array APIs separately.

## 3. Age-of-onset information (`bench_age_onset.py`)

Three independent cohorts per cell, 3,000 families, eight
relatives. Gibbs is a first-replicate cross-check only (`gibbs_reps=1`).

| h² | K | classic LT-FH corr (PA) | FH + onset corr (PA) | first-rep FH + onset Gibbs | squared-correlation eff-N proxy: onset / classic |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.4328 ± 0.0042 | 0.4349 ± 0.0040 | 0.4345 | 1.0097 ± 0.0029× |
| 0.5 | 0.30 | 0.6365 ± 0.0065 | 0.6651 ± 0.0042 | 0.6666 | 1.0924 ± 0.0092× |
| 0.8 | 0.30 | 0.7381 ± 0.0049 | 0.7723 ± 0.0037 | 0.7787 | 1.0949 ± 0.0061× |

Both main columns use PA and condition on the same family; Gibbs is only the
first-replicate agreement check. This is an LT-FH++ age-component ablation over
classic LT-FH, not ADuLT and not raw case/control. The mean squared-correlation
eff-N proxy over the eight-cell grid is 1.038×. Onset information matters
most when enough relatives are observed as cases; at low prevalence its
increment is small. Minimum first-replicate PA/Gibbs agreement is 0.998997.

## 4. Replicated classic-LT-FH genotype GWAS (`bench_gwas_power.py`)

Three independent genotype/effect/cohort replicates; each has 10,000 probands,
5,000 independent SNPs, 30 causal SNPs, h²=0.5, K=0.05, and parents plus one
sibling. The NCP ratio uses `(mean causal chi² - 1)`, not raw mean chi².

| Phenotype | mean causal chi² | causal-SNP NCP ratio / c-c | power at 5e-8 | lambda GC |
|---|---:|---:|---:|---:|
| case/control | 39.63 ± 2.19 | 1.00× | 41.1 ± 2.2% | 1.011 ± 0.009 |
| classic LT-FH (Gibbs) | 57.53 ± 1.87 | 1.468 ± 0.040× | 48.9 ± 4.0% | 1.012 ± 0.019 |
| classic LT-FH (PA) | 57.60 ± 1.91 | 1.469 ± 0.039× | 48.9 ± 4.0% | 1.006 ± 0.021 |
| oracle true g | 337.37 ± 2.07 | 8.77 ± 0.57× | 75.6 ± 1.1% | 1.047 ± 0.015 |

The former 1.53× headline was one seed; 1.47 ± 0.04× is the replicated NCP
ratio. Both rows are the same classic LT-FH model with different inference
engines. In real-LD mode, variants with r² >= 0.1 to any causal SNP are excluded
from lambda/QQ calibration by default because causal proxies are associated, not
null.

## 5. Heritability fitting (`bench_fit_heritability.py`)

Twenty-five independent cohorts per recovery cell, K=0.10, 3,000 families with
parents + two siblings. Fitter RNG seeds are distinct across cohorts.

| true h² | fitted mean | bias | empirical SD | reported MC SE | SD / MC SE |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.201 | +0.001 | 0.040 | 0.0019 | 21× |
| 0.4 | 0.386 | -0.014 | 0.057 | 0.0020 | 28× |
| 0.6 | 0.596 | -0.004 | 0.061 | 0.0024 | 25× |
| 0.8 | 0.786 | -0.014 | 0.062 | 0.0025 | 25× |

At 3,000 families, empirical SD is 0.105 with parents only, 0.054 with parents +
two siblings, and 0.046 with the extended structure. The 4,000/8,000-family SDs
are 0.039/0.041; their chi-square SD intervals overlap substantially, so the tiny
uptick is replicate noise, not evidence against 1/sqrt(N) scaling.

The returned `h2_se` is within-run Monte Carlo error. It is not a sampling
standard error. Use `bootstrap_fit` for family-resampling intervals.

## 6. A+C variance components (`bench_variance_components.py`)

Twenty-five independent 3,000-family cohorts per recovery cell, with distinct
fitter seeds. Values are fitted means (empirical across-cohort SD):

| true A | true C | fitted A (SD) | fitted C (SD) |
|---:|---:|---:|---:|
| 0.4 | 0.2 | 0.398 (0.055) | 0.197 (0.034) |
| 0.5 | 0.1 | 0.486 (0.051) | 0.099 (0.032) |
| 0.3 | 0.3 | 0.295 (0.064) | 0.295 (0.042) |
| 0.6 | 0.0 | 0.583 (0.046) | 0.010 (0.007) |

At true A=0.4 and C=0.2, increasing N from 1,000 to 8,000 families reduces
empirical SD from 0.053 to 0.029 for A and from 0.066 to 0.025 for C. The small
positive biases at N=1,000 shrink toward zero with N.

The null panel reports a constrained boundary mean and SD. Formal component
testing belongs to `test_variance_component`; a point estimate near zero is not a
test rejection rate.

## 7. Genetic correlation (`bench_genetic_correlation.py`)

Twenty-five independent 3,000-family cohorts per recovery cell; the simulation
includes phenotypic correlation 0.2 even when genetic correlation is zero.

| true r_g | fitted mean | bias | empirical SD |
|---:|---:|---:|---:|
| 0.0 | 0.0068 | +0.0068 | 0.0688 |
| 0.3 | 0.2755 | -0.0245 | 0.0581 |
| 0.6 | 0.5778 | -0.0222 | 0.0653 |

At true r_g=0.5, the across-cohort SD is 0.124, 0.099, 0.064, and 0.037
for N=1,000, 2,000, 4,000, and 8,000 families (15 cohorts per point). Bias at
8,000 is +0.0009. The null cell shows that non-genetic phenotypic correlation is
not spuriously recovered as genetic correlation on average.

## 8. Shared environment and prediction (`bench_shared_env.py`)

Four independent 3,000-family cohorts per cell, h²=0.5, parents + three full
sibs. The paired gain is fitted A+C minus fitted additive-only prediction;
uncertainty shown here is a t-based 95% CI half-width.

| true c² | ignore C, fitted h² | fit A+C | paired gain ± 95% CI |
|---:|---:|---:|---:|
| 0.0 | 0.53356 | 0.53351 | -0.00005 ± 0.00034 |
| 0.1 | 0.50160 | 0.50291 | +0.00131 ± 0.00019 |
| 0.2 | 0.48912 | 0.49319 | +0.00407 ± 0.00199 |
| 0.3 | 0.47151 | 0.47805 | +0.00654 ± 0.00204 |

At c²=0.3, the paired gains with 2, 4, and 6 full siblings are respectively
+0.00480 ± 0.00472, +0.00668 ± 0.00127, and +0.00601 ± 0.00229 (95% CI
half-widths). More relatives help identify C, but four replicates are too few to
claim monotone gain with sibship size.

The benchmark now retains per-family Gibbs MCSEs and warns when estimates miss
the requested tolerance. That instrumentation was added immediately after this
30-minute canonical run, whose pre-instrumentation code discarded the MCSEs;
therefore convergence flags cannot honestly be reconstructed for these rows.

The predictive increment is intentionally evaluated as a paired difference on
the same cohorts. Its main practical value is smaller than the parameter-
interpretation benefit: ignoring C causes sib resemblance to leak into fitted h².

## 9. Couple/spousal environment M (`bench_couple_env.py`)

Twenty-five independent cohorts per cell, 3,000 extended families.

| true m² | fitted A (true 0.4) | A SD | fitted M | M SD |
|---:|---:|---:|---:|---:|
| 0.0 | 0.398 | 0.034 | 0.018 | 0.012 |
| 0.1 | 0.400 | 0.031 | 0.106 | 0.038 |
| 0.2 | 0.396 | 0.027 | 0.197 | 0.042 |
| 0.3 | 0.393 | 0.036 | 0.301 | 0.029 |

Omission comparison, using the same shared variance as sibship C or couple M:

| shared variance | bias in A if C omitted (SD) | bias in A if M omitted (SD) |
|---:|---:|---:|
| 0.1 | +0.044 (0.034) | +0.004 (0.033) |
| 0.2 | +0.108 (0.034) | +0.012 (0.052) |
| 0.3 | +0.163 (0.034) | +0.038 (0.033) |

Omitting M biases A much less than omitting C, but not identically zero. The
M=0 result is a constrained boundary floor, not evidence of a false-positive
rate. This simulation generates shared adult environment; it does not validate a
generative assortative-mating interpretation.

## 10. Registry family-history and ADuLT prediction (`bench_fh_prediction.py`)

Age-, cohort-, and mortality-consistent three-generation pedigrees, 4,000
families and three independent cohorts per main cell. Relatives are censored at
death or current age. Values are scored against known true genetic liability.

### Ascertainment

| observed proband case fraction | case/control corr | classic LT-FH corr (PA) | FH + age/cohort corr (PA) | classic LT-FH squared-correlation eff-N proxy / c-c | age/cohort proxy / classic |
|---:|---:|---:|---:|---:|---:|
| 0.019 (population) | 0.233 | 0.347 | 0.348 | 2.212 ± 0.015× | 1.009 ± 0.011× |
| 0.10 | 0.464 | 0.524 | 0.530 | 1.276 ± 0.006× | 1.021 ± 0.002× |
| 0.25 | 0.623 | 0.658 | 0.668 | 1.118 ± 0.006× | 1.029 ± 0.001× |
| 0.50 | 0.698 | 0.728 | 0.744 | 1.086 ± 0.009× | 1.045 ± 0.002× |

Both family-history scores use PA. The first is classic LT-FH; the second changes
the bounds to add age and cohort information, not the inference engine. Thus the
population-sampling **2.212 ± 0.015×** is
`(corr(classic LT-FH) / corr(case/control))²`, a squared-correlation eff-N
proxy, not the causal-SNP NCP ratio reported in section 4. Family history is most
valuable relative to case/control in population sampling; the incremental
age/cohort benefit grows under ascertainment.

### Cohort effects

At a 3× lifetime-prevalence trend per 30 birth years, cohort-aware and single-K
pedigree scores have similar narrow-cohort ranking (0.7445 versus 0.7427), but
their mean scores differ by -0.07246 ± 0.00031. Truth-referenced mean errors are
+0.0031 ± 0.0059 for cohort-aware and -0.0693 ± 0.0062 for single-K.

The replicated own-onset panel has **no family-history inputs** and is therefore
ADuLT. It isolates cohort-aware versus cohort-blind ranking across broader birth
cohorts:

| cohort half-span | ADuLT cohort-aware corr | ADuLT cohort-blind corr |
|---:|---:|---:|
| ±10 years | 0.387 ± 0.021 | 0.378 ± 0.025 |
| ±25 years | 0.427 ± 0.015 | 0.374 ± 0.023 |
| ±40 years | 0.477 ± 0.015 | 0.383 ± 0.018 |
| ±55 years | 0.512 ± 0.007 | 0.397 ± 0.017 |

## 11. Genetic factor diagnostic (`bench_genetic_factor.py`)

Fifteen independent cohorts, 3,000 families, five traits. Under planted
one-factor truth, loadings 0.8/0.7/0.6/0.5/0.4 are recovered as
0.822/0.704/0.606/0.487/0.404, with SD 0.056–0.102.

| truth and fitted model | SRMR mean | SRMR SD |
|---|---:|---:|
| true 1F, fit 1F | 0.047 | 0.015 |
| true 2F, fit 1F | 0.146 | 0.030 |
| true 2F, fit 2F | 0.017 | 0.011 |

This demonstrates that SRMR diagnoses this planted misspecification. It is not a
calibrated factor-number test; extra factors improve in-sample fit by construction.
Bootstrap the whole correlation-to-factor pipeline for uncertainty.

## 12. Score calibration (`bench_calibration.py`)

These are single-seed, 3,000-family diagnostic cells. Correctly specified PA:

| structure | K | slope | corr | top-decile realised/predicted |
|---|---:|---:|---:|---:|
| parents + siblings | 0.01 | 1.006 | 0.265 | 1.033 |
| parents + siblings | 0.05 | 1.014 | 0.421 | 1.010 |
| parents + siblings | 0.20 | 0.996 | 0.589 | 0.999 |
| extended | 0.01 | 0.916 | 0.215 | 0.896 |
| extended | 0.05 | 1.017 | 0.428 | 1.009 |
| extended | 0.20 | 1.015 | 0.608 | 0.987 |

Most cells are near slope 1; rare disease with the extended pedigree is an
outlier in this single seed, so the benchmark does not justify a universal
calibration claim.

Misspecified h², with true h²=0.5 and K=0.05:

| assumed h² | slope | corr | top realised/predicted | calibration RMSE |
|---:|---:|---:|---:|---:|
| 0.2 | 2.249 | 0.419 | 2.272 | 0.160 |
| 0.4 | 1.224 | 0.421 | 1.224 | 0.059 |
| 0.5 | 1.014 | 0.421 | 1.010 | 0.026 |
| 0.6 | 0.870 | 0.421 | 0.866 | 0.046 |
| 0.8 | 0.684 | 0.420 | 0.681 | 0.124 |

Ranking barely changes, while scale changes sharply. A realised/predicted ratio
above 1 means the score under-predicted the realised top decile; it does not mean
the score overstated it.

## 13. Cohort confounding (`bench_confounding.py`)

Three independent cohorts per trend. Lambda GC below is for SNPs correlated with
birth cohort; truly independent null SNPs stay near 1 for every method.

| prevalence trend R per 30 y | FH + cohort-specific K | single-K FH | case/control |
|---:|---:|---:|---:|
| 1 | 0.960 ± 0.011 | 0.960 ± 0.011 | 0.899 ± 0.034 |
| 2 | 0.965 ± 0.054 | 4.539 ± 0.252 | 2.008 ± 0.257 |
| 3 | 0.922 ± 0.057 | 10.274 ± 0.077 | 4.616 ± 0.220 |
| 4 | 1.008 ± 0.041 | 16.461 ± 0.397 | 7.666 ± 0.094 |

This benchmark isolates the cohort component in a family model; it is not the
full age/sex/cohort LT-FH++ design. At R=1 there is no trend-driven inflation;
the methods need not have identical finite-sample lambda values. Under strong
trends, cohort-aware thresholds remove the induced stratified-SNP inflation on
average.

## 14. PA robustness (`bench_pa_robustness.py`)

Single-seed stress cells:

| regime | corr(PA, Gibbs) | corr(PA, true g) | corr(Gibbs, true g) |
|---|---:|---:|---:|
| baseline | 0.99969 | 0.4380 | 0.4377 |
| large pedigree | 0.99973 | 0.4490 | 0.4501 |
| rare, K=0.005 | 0.99843 | 0.2300 | 0.2308 |
| densely affected | 0.99909 | 0.4657 | 0.4665 |

Fold-order spread as a percentage of the between-proband score SD:

| pedigree | median | p95 |
|---|---:|---:|
| trio | 0.084% | 1.96% |
| parents + 2 siblings | 0.114% | 2.24% |
| extended | 0.094% | 3.36% |
| large | 0.069% | 2.82% |

The typical order effect is tiny. The p95 values are small but not monotone in
pedigree size. This script does not compare fold-order spread with Gibbs Monte
Carlo noise and does not directly measure rank changes, so it makes neither claim.

## 15. Integrated personalised LT-FH++ genotype GWAS (`bench_ltfhpp_personalization.py`)

### Integrated panel

Ten paired replicates, 4,000 ascertained probands each, 1,200 SNPs (30 causal,
300 sex/cohort-stratified null), age-, sex-, and cohort-dependent CIP, coherent
onset/follow-up, and sex-dependent competing mortality. Accuracy, slope, and
GWAS values below are adjusted for proband sex and birth year; `±` is replicate
SE. The final column shows the stratified-null lambda before and after the same
standard covariate adjustment.

| Phenotype | adjusted corr | adjusted slope | adjusted causal-SNP NCP ratio / c-c | stratified-null lambda raw -> adjusted |
|---|---:|---:|---:|---:|
| case/control | 0.586 ± 0.008 | 1.123 ± 0.021 | 1.000× | 1.140 ± 0.051 -> 1.019 ± 0.024 |
| ADuLT (same full personalised proband CIP, no FH) | 0.600 ± 0.008 | 1.004 ± 0.018 | 1.049 ± 0.004× | 1.024 ± 0.047 -> 1.014 ± 0.031 |
| LT-FH single-K | 0.629 ± 0.008 | 1.183 ± 0.020 | 1.149 ± 0.007× | 1.131 ± 0.046 -> 0.990 ± 0.041 |
| FH + age CIP (ablation) | 0.639 ± 0.008 | 1.009 ± 0.016 | 1.189 ± 0.006× | 0.998 ± 0.033 -> 1.018 ± 0.041 |
| FH + age + sex CIP (ablation) | 0.639 ± 0.008 | 1.008 ± 0.016 | 1.188 ± 0.006× | 0.986 ± 0.032 -> 1.011 ± 0.036 |
| FH + age + cohort CIP (ablation) | 0.640 ± 0.008 | 0.995 ± 0.015 | 1.194 ± 0.006× | 1.014 ± 0.034 -> 1.020 ± 0.034 |
| **LT-FH++ (full age + sex + cohort CIP)** | **0.640 ± 0.008** | **0.995 ± 0.015** | **1.194 ± 0.006×** | **0.996 ± 0.036 -> 1.010 ± 0.039** |

The matched ADuLT row uses exactly the full personalised proband bounds but no
relative columns. Adding relatives to reach full LT-FH++ improves adjusted
correlation by **+0.04073 ± 0.00358** and causal-SNP NCP ratio by
**+0.1454 ± 0.0154** (paired 95% CI half-widths). Here and below these are
causal-SNP NCP-ratio units. ADuLT itself gains +0.0490 ± 0.0096 NCP-ratio
units over case/control.

Other paired t-based 95% intervals isolate the components. Single-K gains
+0.1489 ± 0.0147 NCP-ratio units over case/control; age adds
+0.0399 ± 0.0076 beyond single-K; cohort adds +0.0056 ± 0.0022 beyond age.
Full LT-FH++ adds **+0.0455 ± 0.0073** beyond classic LT-FH.
Sex adds -0.0004 ± 0.0010 beyond age, and adding sex to age+cohort adds
-0.00004 ± 0.00076. Thus this cancellation-prone integrated scenario establishes
age and cohort gains, but no conditional sex-power gain.

The stratified-null panel is a **covariate-adjustment sanity check**, not evidence
that personalization substitutes for standard GWAS adjustment. After proper FWL
adjustment, lambda is near one for every phenotype.

### Prespecified sex-CIP isolation

Five paired replicates, 3,000 ascertained probands, 600 SNPs (30 causal), age-
dependent CIP, female:male lifetime-risk ratio 2, equal onset midpoints, and no
cohort or sex-dependent-mortality effect. This isolates sex-specific thresholds
without retrofitting the integrated parameters after seeing its result.

| Phenotype | adjusted corr | adjusted causal-SNP NCP ratio / c-c | female mean error | male mean error |
|---|---:|---:|---:|---:|
| FH + age CIP (ablation) | 0.6298 ± 0.0078 | 1.292 ± 0.016× | +0.0301 ± 0.0086 | -0.0142 ± 0.0084 |
| FH + age + sex CIP (ablation) | 0.6307 ± 0.0080 | 1.296 ± 0.016× | -0.0068 ± 0.0087 | -0.0002 ± 0.0084 |

The adjusted ranking and NCP-ratio increments are small and unresolved:
Δcorr = +0.00097 ± 0.00110 and ΔNCP ratio = +0.0040 ± 0.0046
(paired 95% CIs).
The calibration benefit is decisive because the paired errors are highly
correlated: adding the correct sex curve shifts female error by
-0.03689 ± 0.00041 and male error by +0.01401 ± 0.00064, closing the
female-minus-male error gap by **0.05091 ± 0.00096**.

On the first two 300-family, no-mixture main-panel cross-checks, PA/Gibbs agreement is
0.999901. Gibbs reaches the requested MCSE tolerance for every score (maximum
MCSE 0.0090 at tolerance 0.03); PA-vs-Gibbs normalized RMSE is 0.0174 score SD,
the Gibbs-on-PA slope is 0.990, and the mean difference is -0.0019.

## 16. PA-FGRS censoring-mixture validation (`bench_pafgrs_mixture.py`)

The PA-FGRS age-censored-control mixture previously had unit tests but no
generative benchmark. This one simulates families (proband + parents + sib)
under the liability-threshold model with a logistic CIP (h2 = 0.5, lifetime
prevalence 0.10, mid-point 60), censors honestly (a case is observed only if
its onset precedes the current age), and scores the estimate against the true
genetic liability over 5 replicates of 20,000 families. Two censoring regimes
(mid-life, heavy; old, light) and two observation models: the LT-FH++
threshold-crossing convention, and a stochastic-onset model in which onset age
is drawn from the CIP independent of liability (the mixture's native model).
Slope = regress(true on estimate); 1.0 is a calibrated posterior mean.

Threshold-crossing observation model:

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base (lifetime case) + no-mixture | 0.3281 | 1.1992 | 0.4693 | 1.0293 |
| base + mixture | 0.3277 | 1.1738 | 0.4691 | 1.0209 |
| interval case + mixture | 0.3334 | 0.8444 | 0.4766 | 0.8263 |
| pinned case + no-mixture | 0.3337 | 0.9977 | 0.4770 | 0.9885 |
| pinned case + mixture | 0.3337 | 0.9766 | 0.4769 | 0.9811 |

(Gibbs cross-check on base + no-mixture: corr(PA, Gibbs) 0.9992 / 0.9998,
matching slopes 1.2192 / 1.0217 -- the MID slope > 1 is the case encoding's
information loss, not a PA artifact.)

Stochastic-onset observation model:

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base + no-mixture | 0.2757 | 1.0184 | 0.4528 | 0.9957 |
| base + mixture | 0.2758 | 0.9974 | 0.4528 | 0.9878 |

### Verdict: implemented correctly; small censoring effect; case encoding dominates

- **The mixture is implemented correctly**: it moves calibration toward 1 for
  the under-conditioned lifetime-case encoding and is near-exact under its
  native stochastic-onset model; it never costs correlation (<= 0.0005).
- **Its effect at these settings is small** (calibration slope shifts of
  0.01-0.03, correlation unchanged). Under the threshold-crossing model the
  plain age truncation is already exact for censored controls, so the mixture
  has no work to do; under the stochastic-onset model the naive encoding is
  already nearly calibrated at these settings.
- **Case encoding dominates calibration**: pinned (LT-FH++-exact) cases give
  slope 0.98-1.00 in every cell; the lifetime case interval loses onset-age
  information (slope up to 1.20 under heavy censoring); the age-specific
  interval over-disperses (slope 0.83, consistent with the unit guard's
  documented 0.84). Guidance: pin cases at their onset threshold where the
  crossing model is believed; use the lifetime interval only when onset ages
  are unreliable.

## What changed in this rerun

- Added independent-replicate SEs to GWAS power, age-onset, family-history,
  confounding, and cohort-span panels.
- Corrected genotype-GWAS NCP ratios to use chi-square noncentrality rather than
  raw mean chi-square.
- Warmed and replicated runtime measurements; added a directly measured PA array
  path and recorded configuration/thread metadata.
- Used distinct deterministic inference seeds across fitted cohorts.
- Added uncertainty intervals for empirical SD curves and small prediction gains.
- Rebuilt the integrated LT-FH++ benchmark around ten paired main replicates,
  consistent adjusted metrics, a matched family-free ADuLT arm, a prespecified
  sex-CIP isolation panel, richer Gibbs diagnostics, and a self-describing CSV
  with safe output prefixes.
- Recast constrained C/M null estimates as boundary behavior rather than false-
  positive rates.
- Added LD-proxy filtering to the real-LD calibration path.
- Removed unsupported claims about fold-order monotonicity, Gibbs-noise dominance,
  universal score calibration, and zero omission bias for M.

## Remaining limitations

- Accuracy, calibration, and PA stress grids still contain single-seed diagnostic
  cells. Treat small differences there as descriptive.
- Three- or four-replicate panels give useful SEs but still estimate tail
  uncertainty coarsely. The integrated main panel now uses ten replicates and
  its sex isolation uses five; tail calibration remains noisier than paired
  score contrasts.
- Component boundary means do not establish Type-I error or interval coverage.
  Use the package's parametric-bootstrap tests and family bootstrap intervals.
- The HAPNEST path was not executed here. The PA-FGRS censoring-mixture
  benchmark now exists (section 16); its censoring correction is small at
  the tested settings, and case encoding dominates calibration there.
- The lightweight GWAS helper uses the large-sample `n * r²` score statistic.
  A finite-sample regression test would use residual degrees of freedom and the
  `r² / (1-r²)` correction; the matched NCP ratios are robust to this
  small approximation, but genome-wide discovery counts are only illustrative.
- Most benchmark scripts still write canonical artifacts unconditionally. The
  integrated-personalization script now accepts `--output-prefix` and writes a
  self-describing CSV, but a common runner and run manifest remain outstanding.
