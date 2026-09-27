# ltpred benchmark results

The evidence ledger for the scripts in this directory. §§1–30 and §32 are
simulation and cross-package results, each backed by a committed CSV beside its
script except §17 and §18 (stdout only); §31 and §33 summarise JSON capsules
under [`results/`](results/). How to run and archive a benchmark is in the
[README](README.md); `scripts/check_evidence.py` pins the release-defining
numbers here to their artifacts. Nothing here is a timing or validation of the
current source tree: each value applies to its recorded design and provenance,
and should be refreshed after numerical or benchmark-source changes.

Bounds define the observation encoding (non-personalised, personalised pinned,
interval case). Family-history inclusion distinguishes LT-FH++ (with
relatives) from ADuLT (index person only). Gibbs and Pearson–Aitken (PA) are
alternative inference engines for the same bounds. PA-FGRS is a separate
PA-specific specification; its censoring mixture is not in the PA–Gibbs
comparisons.

**Table 0.1. Provenance by section.** Stack A is free-threaded Python 3.14.6,
NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0 on an Apple M2 Pro (10 logical cores,
arm64); stack B is Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3.

| date | version | sections | stack, threads | note |
|---|---|---|---|---|
| 2026-07-14 to 08-15 | — | first campaign (07-14); focused runs 07-31, 08-01, 08-14/15 | 08-14/15 reruns on B | prose values the 2026-08-20 rerun reproduced within sampling error were left as recorded |
| 2026-08-20 | v0.4.0 | all 26 scripts; every CSV regenerated in place | A, Numba 8 (`bench_scaling`, `bench_ltfhplus_compare` 4) | §30 and §28 artifacts later refreshed at v0.4.1 |
| 2026-09-03 | `bad41ad` | §20, §21 | A, Numba 4, BLAS 1 | provenance wrapper, clean tree |
| 2026-09-10 | `b6065ff` | §32 | CPython 3.10.20, NumPy 1.26.4, SciPy 1.15.3; 6 workers, BLAS 1 | provenance wrapper |
| 2026-09-14 | v0.6.2 | §15, §16, affected cells of §3 and §10; §19 | A | only arms combining family history with pinned onsets moved (the v0.6.0 joint conditioning of pins: scale shifts, ranking does not); every column PA does not compute is bit-identical |
| 2026-09-14 | v0.6.2 | suite verification rerun | A | `bench_tetrachoric`, `bench_liability_scale`, `bench_calibration`, `bench_pa_robustness` byte-identical; `bench_accuracy`, `bench_pedigree_inference` identical except wall-clock columns; `bench_register_pipeline` ~1e-3 per row, replicate means unchanged; timing artifacts not refreshed (one-minute load 13–28) |
| 2026-09-21 | — | §21 accuracy | BLAS 1 | Cholesky `h2 A` draw, 2026-09-03 design |
| 2026-09-23 | v0.7.1 `5b42b13` | §21 throughput | BLAS 1 | one-minute load 2.1 |
| 2026-09-09, 09-16, 09-23 | v0.6.1, dev, v0.7.1 | §31, §33, §31a | per capsule README | JSON capsules |

Runs before `run_manifest.jsonl` was reinstated on 2026-09-03 have no
surviving per-run source record.

Unless stated otherwise `±` is the standard error of a mean across independent
simulated cohorts; tables labelled SD give the empirical across-cohort standard
deviation, tables labelled 95% CI a half-width. Single-seed diagnostic grids
are identified as such.

Metric names matter. **NCP ratio** is a ratio of causal-SNP chi-square
noncentrality components (`mean chi² - 1`); **eff-N proxy** is a
squared-correlation ratio to case/control. The latter suits prediction
comparisons but is not an observed GWAS noncentrality ratio, so the two
magnitudes are not interchangeable. Every causal-SNP NCP and detection-power
result used independent SNPs, non-overlapping simulated families and the
lightweight marginal score statistic in `_common.py`: evidence for that
simulation estimand, not for a real-LD, related-sample mixed-model GWAS.

Core fitter benchmarks (§§5–9) use unascertained, population-sampled families
and characterise the fitters only under that supported contract; §29 measures
what they return on ascertained samples and underlies the marginal case-rate
screen applied under `sampling="population"`.

## Headline findings

**Table H.1.** Headline findings; each is qualified further in its section.

| finding | evidence | § |
|---|---|---|
| **PA is the right default for the tested single-trait, no-mixture work.** | Mean corr(PA, Gibbs) 0.9995–1.0000 over the 27-cell grid (five seeds per cell); the stressful-pedigree benchmark remains at least 0.9991 (worst single-seed 0.99916); indistinguishable downstream causal-NCP results. In the 4-thread timing run the PA object path is **392–518× faster** than grouped Gibbs. The ratio is not thread-count-free (Gibbs is parallel, the PA object path largely serial): quote it with its thread count or not at all. | 1, 2, 14 |
| **ltpred Gibbs matches public LTFHPlus on the same families.** | Against LTFHPlus 2.2.0, corr = 0.9999, RMSE = 0.0041 ± 0.0001 (200 nuclear families, three cohorts, the R package's own Gibbs settings). | 30 |
| **Public PA is LTFGRS, not LTFHPlus** (LTFHPlus 2.2.0 is Gibbs-only). | ltpred PA and LTFGRS 1.0.1 `method="PA"` agree at corr = 1.0000 (RMSE 0.000087 ± 0.000008). Same-algorithm folds on this machine: **6.786 ± 0.079×** (LTFHPlus Gibbs / ltpred Gibbs), **1418 ± 30×** (LTFGRS PA / ltpred PA). Mixing algorithms, LTFHPlus / ltpred PA is **8086 ± 250×**, which is not "LT-FH++, but faster". | 30 |
| **Classic LT-FH improves simulated independent-SNP causal NCP without average null inflation.** | Causal-SNP NCP ratio **1.47 ± 0.04×** over case/control by PA or Gibbs, three replicates; not a real-LD result. | 4 |
| **Full personalised LT-FH++ adds independent-SNP NCP beyond matched ADuLT.** | Same age/sex/cohort proband bounds: ADuLT **1.049 ± 0.004×**, full LT-FH++ **1.194 ± 0.006×**; paired increment **+0.1454 ± 0.0155** NCP-ratio units (95% CI half-width). LT-FH++ adjusted calibration slope **1.004 ± 0.015**, PA/Gibbs agreement 0.99996. | 15 |
| **Cohort-blind family thresholds inflate stratified-null λ_GC; cohort-aware thresholds do not.** | At a 4× lifetime-prevalence trend per 30 birth years: single-K **16.460 ± 0.399**, cohort-specific K **1.007 ± 0.040**; independent null SNPs near 1 for every method. Personalised CIP is a confounding-control device, not only a power tweak. | 13 |
| **Sex-specific CIP improves stratum calibration, not proven adjusted independent-SNP NCP.** | Prespecified sex-only scenario: correct sex curves close the female-minus-male mean-score-error gap by **0.05092 ± 0.00096** (paired 95% CI); the adjusted NCP-ratio increment **0.0040 ± 0.0046** is unresolved. | 15 |
| **A correctly specified posterior mean is self-calibrating; a wrong h² is not.** | Slope 1.006 ± 0.008 at K=0.05; assumed h² 0.2 to 0.8 sweeps it from 2.240 ± 0.015 to 0.678 ± 0.006 while ranking stays in 0.429–0.431. Use the score as a ranker or GWAS phenotype under a roughly right h²; its scale is E[g \| family] only at the h² it was computed at. | 12 |
| **A PGS and the family-history score are complementary.** | 50/50 train/test split, test R² against held-out g: **0.236 ± 0.009** (PGS), **0.170 ± 0.003** (classic LT-FH), **0.338 ± 0.009** (two-fold cross-fitted OLS on both). | 28 |
| **The PA-FGRS mixture has a detectable but practically negligible ranking effect; case encoding dominates calibration.** | Four of ten paired Δcorr 95% CIs exclude zero; the largest shift is −0.00034 (95% CI ± 0.00011). Under threshold crossing pinned cases stay near slope 1; when onset only *tends* to track liability (ρ = 0.6) pinning over-conditions (0.93 under heavy censoring); the lifetime interval under-conditions (up to 1.20); an age-specific case interval over-disperses (≈ 0.83). | 16 |
| **Phenotype ascertainment pins the heritability fitter at the clamp.** | At true h² = 0 every phenotype-selected design returns h² = 1.0. IPW recovers a 50/50 case/control cohort from 1.000 to **0.468** (truth 0.5; weights up to 19) and a 20%-enriched cohort to **0.521** (3,000-family grid). Designs with zero inclusion probability in some stratum cannot be reweighted. | 29 |
| **Variance-component point estimates need sampling uncertainty.** | The reported Monte Carlo SE is much smaller than the empirical across-cohort SD: use family bootstrap intervals. A constrained estimate at zero is a boundary estimate, not a false-positive rate. | 5, 6, 17 |

## 1. PA versus Gibbs accuracy (`bench_accuracy.py`)

**Table 1.1.** h² = 0.5, K = 0.05; five independent cohorts (seeds 1–5) of
1,500 families per cell; across-seed mean ± SE (sd/√5).

| Family structure | corr Gibbs | corr PA | PA squared-correlation eff-N proxy / case-control | corr(PA, Gibbs) |
|---|---:|---:|---:|---:|
| parents | 0.382 ± 0.009 | 0.382 ± 0.009 | 1.36 ± 0.03× | 0.9999 |
| parents + 2 siblings | 0.437 ± 0.009 | 0.437 ± 0.009 | 1.66 ± 0.07× | 0.9999 |
| extended | 0.420 ± 0.014 | 0.420 ± 0.014 | 1.71 ± 0.09× | 0.9999 |

Across all 27 cells mean PA–Gibbs agreement is 0.9995–1.0000 with across-seed
SE ≤ 0.0001 (bare means shown). Absolute accuracy increases with prevalence,
heritability and informative relatives. Relative gain over a case/control label
is often largest for rarer disease; the largest grid mean is 2.53 ± 0.52×
(extended, h²=0.2, K=0.01). Low-prevalence cells are noisy (parents +
siblings, h²=0.2, K=0.05 gives 2.11 ± 0.15×), so their ordering is descriptive.

## 2. Controlled runtime scaling (`bench_scaling.py`)

Five warmed `perf_counter` timings per point, reported as medians (IQRs in the
CSV); stack A, **4 Numba threads**, h²=0.5, K=0.05, 25,000 Gibbs draws, with
the collapsed-genetic Gibbs path (untruncated `g` integrated out of the
sweep). No other benchmark ran concurrently, but the 1-minute load average
moved from 2.56 to 5.86 during the run. This is the only benchmark used for
performance claims; four threads is the project's baseline operating point.

**Table 2.1.** Scaling with number of families (parents + one sibling).

| families | Gibbs families/s | PA object families/s | PA array families/s | object speed-up |
|---:|---:|---:|---:|---:|
| 500 | 504 | 207,980 | 2.63 M | 412× |
| 1,000 | 512 | 215,713 | 3.74 M | 421× |
| 2,000 | 503 | 218,190 | 5.05 M | 434× |
| 4,000 | 506 | 220,534 | 5.53 M | 436× |
| 8,000 | 512 | 219,743 | 6.29 M | 429× |

The object path includes `Family`/`Member` bounds assembly and grouping. The
array path receives already aligned, repeatedly reused arrays; its 2.63–6.29
million families/s is a hot-kernel measurement, not end-to-end input
preparation, and its very short calls make cache and scheduler effects visible
(use the CSV IQRs rather than the non-monotone point rates).

**Table 2.2.** Family size at 2,000 families.

| relatives | structure | Gibbs time | PA object time | PA array time | object speed-up |
|---:|---|---:|---:|---:|---:|
| 2 | parents | 2.91 s | 0.00741 s | 0.00034 s | 392× |
| 3 | + sibling | 3.92 s | 0.00919 s | 0.00042 s | 427× |
| 5 | + two grandparents | 5.95 s | 0.0128 s | 0.00058 s | 463× |
| 7 | extended | 8.02 s | 0.0166 s | 0.00073 s | 483× |
| 10 | extended + aunts | 11.24 s | 0.0217 s | 0.00099 s | 518× |

The small PA times are not strictly monotone; five repeats quantify timing
variation but do not abolish operating-system noise under background load. The
defensible claim is the observed **392–518×** object-path
speed-up *at four threads on this machine* — not a hardware-independent
constant, and not transferable to another thread count, since more threads
speed Gibbs up far more than the largely serial PA object path.

## 3. Age-of-onset information (`bench_fh_prediction.py` panel (e), formerly `bench_age_onset.py`)

*Rows in `bench_fh_prediction.csv` (panel (e)); refreshed cells per Table 0.1.*

**Table 3.1.** Three independent cohorts per cell, 3,000 families, eight
relatives. The same families are scored as classic LT-FH (lifetime case
interval), interval (`[T(onset), ∞)`) and pin (`T(onset)` as a point); onset
is the CIP inverse of true liability. All columns use PA; Gibbs is a
first-replicate check on the pin only (`gibbs_reps=1`).

| h² | K | classic | pin | interval | pin / classic | interval / classic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.431 ± 0.001 | 0.436 ± 0.001 | 0.436 ± 0.001 | 1.024 ± 0.009× | 1.023 ± 0.007× |
| 0.5 | 0.30 | 0.642 ± 0.006 | 0.669 ± 0.005 | 0.665 ± 0.005 | 1.086 ± 0.006× | 1.074 ± 0.004× |
| 0.8 | 0.30 | 0.747 ± 0.001 | 0.779 ± 0.001 | 0.772 ± 0.000 | 1.086 ± 0.006× | 1.069 ± 0.003× |

An LT-FH++ age-component ablation over classic LT-FH, not ADuLT and not raw
case/control. Mean pin / classic squared-correlation eff-N proxy over the
eight-cell grid is 1.039×; interval / classic is 1.034×. At low prevalence pin
and interval are indistinguishable. At K = 0.30 the pin adds a further
~0.01–0.02× over the interval: most of the onset increment is knowing the case
is at least that extreme, not the pin-equals-liability identity. Minimum
first-replicate PA/Gibbs agreement is 0.99971.

## 4. Replicated classic-LT-FH independent-SNP association benchmark (`bench_gwas_power.py`)

**Table 4.1.** Three independent genotype/effect/cohort replicates of 10,000
probands, 5,000 independent SNPs (30 causal), h²=0.5, K=0.05, parents plus one
sibling. The NCP ratio uses `(mean causal chi² - 1)`, not raw mean chi².

| Phenotype | mean causal chi² | causal-SNP NCP ratio / c-c | power at 5e-8 | lambda GC |
|---|---:|---:|---:|---:|
| case/control | 39.63 ± 2.19 | 1.00× | 41.1 ± 2.2% | 1.011 ± 0.009 |
| classic LT-FH (Gibbs) | 57.53 ± 1.87 | 1.468 ± 0.040× | 48.9 ± 4.0% | 1.012 ± 0.019 |
| classic LT-FH (PA) | 57.60 ± 1.91 | 1.469 ± 0.039× | 48.9 ± 4.0% | 1.007 ± 0.021 |
| oracle true g | 337.37 ± 2.07 | 8.77 ± 0.57× | 75.6 ± 1.1% | 1.047 ± 0.015 |

Both LT-FH rows are the same classic model with different inference engines.
Real-LD mode (not run here) excludes variants with r² >= 0.1 to any causal SNP
from lambda/QQ calibration, since causal proxies are associated, not null.
Neither mode is a related-sample mixed-model analysis.

## 5. Heritability fitting (`bench_fit_heritability.py`)

**Table 5.1.** Twenty-five independent, unascertained population cohorts per
cell, K=0.10, 3,000 families with parents + two siblings, distinct fitter seeds.

| true h² | fitted mean | bias | empirical SD | reported MC SE | SD / MC SE |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.201 | +0.001 | 0.037 | 0.0020 | 19× |
| 0.4 | 0.387 | -0.013 | 0.055 | 0.0022 | 25× |
| 0.6 | 0.577 | -0.023 | 0.063 | 0.0023 | 28× |
| 0.8 | 0.786 | -0.014 | 0.061 | 0.0026 | 24× |

At 3,000 families, empirical SD is 0.107 with parents only, 0.052 with parents +
two siblings, and 0.051 with the extended structure. The 4,000/8,000-family SDs
are 0.038/0.040; their chi-square SD intervals overlap substantially, so the
uptick is replicate noise, not evidence against 1/sqrt(N) scaling. The returned
`h2_se` is within-run Monte Carlo error, not a sampling standard error; use
`bootstrap_fit` for family-resampling intervals.

## 6. A+C variance components (`bench_variance_components.py`)

**Table 6.1.** Fitted mean (empirical across-cohort SD); twenty-five
independent, unascertained cohorts of 3,000 families per cell, distinct fitter
seeds.

| true A | true C | fitted A (SD) | fitted C (SD) |
|---:|---:|---:|---:|
| 0.4 | 0.2 | 0.399 (0.057) | 0.190 (0.034) |
| 0.5 | 0.1 | 0.492 (0.053) | 0.101 (0.032) |
| 0.3 | 0.3 | 0.297 (0.062) | 0.294 (0.038) |
| 0.6 | 0.0 | 0.583 (0.057) | 0.013 (0.012) |

At true A=0.4 and C=0.2, increasing N from 1,000 to 8,000 families reduces
empirical SD from 0.072 to 0.029 for A and from 0.067 to 0.018 for C; the small
positive biases at N=1,000 shrink toward zero with N. The null panel reports a
constrained boundary mean and SD: formal component testing belongs to
`test_variance_component`, and a point estimate near zero is not a test
rejection rate.

## 7. Genetic correlation (`bench_genetic_correlation.py`)

**Table 7.1.** Twenty-five independent 3,000-family cohorts per cell;
phenotypic correlation 0.2 even when genetic correlation is zero.

| true r_g | fitted mean | bias | empirical SD |
|---:|---:|---:|---:|
| 0.0 | -0.0061 | -0.0061 | 0.0798 |
| 0.3 | 0.2894 | -0.0106 | 0.0825 |
| 0.6 | 0.5976 | -0.0024 | 0.0706 |

At true r_g=0.5 the across-cohort SD is 0.148, 0.108, 0.078 and 0.042 for
N=1,000, 2,000, 4,000 and 8,000 families (15 cohorts per point); bias at 8,000
is +0.0068. Non-genetic phenotypic correlation is not recovered as genetic
correlation on average.

## 8. Shared environment and prediction (`bench_shared_env.py`)

**Table 8.1.** Four independent 3,000-family cohorts per cell, h²=0.5, parents +
three full sibs. Accuracy is `corr(estimate, true g)` under each fitted model;
the paired gain is fitted A+C minus fitted additive-only, with a t-based 95% CI
half-width; the last two columns are the fitted parameters behind them.

| true c² | accuracy, ignore C | accuracy, fit A+C | paired gain ± 95% CI | fitted h², ignore C | fitted h², fit A+C |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.53276 | 0.53264 | -0.00012 ± 0.00040 | 0.502 | 0.492 |
| 0.1 | 0.49920 | 0.50017 | +0.00097 ± 0.00063 | 0.550 | 0.468 |
| 0.2 | 0.48888 | 0.49246 | +0.00358 ± 0.00146 | 0.643 | 0.469 |
| 0.3 | 0.47083 | 0.47671 | +0.00588 ± 0.00236 | 0.748 | 0.478 |

At c²=0.3 the paired gains with 2, 4 and 6 full siblings are +0.00265 ±
0.00346, +0.00627 ± 0.00564 and +0.00814 ± 0.00616 (95% CI half-widths): more
relatives help identify C, but four replicates are too few to claim monotone
gain with sibship size. The stored rows carry per-family Gibbs MCSE
instrumentation: per-model MCSE maxima span 0.0048–0.0120, with zero nonfinite
and zero unconverged estimates in every cell. The canonical run took 542 s.
Panel (c) is the §24 C/M wiring check (fitted A/C/M = 0.427/0.142/0.150).

The predictive increment is a paired difference on the same cohorts. Its
practical value is smaller than the parameter-interpretation benefit: ignoring
C lets sib resemblance leak into fitted h².

## 9. Couple/spousal environment M (`bench_couple_env.py`)

**Table 9.1.** Twenty-five independent cohorts per cell, 3,000 extended families.

| true m² | fitted A (true 0.4) | A SD | fitted M | M SD |
|---:|---:|---:|---:|---:|
| 0.0 | 0.394 | 0.032 | 0.014 | 0.011 |
| 0.1 | 0.392 | 0.036 | 0.102 | 0.035 |
| 0.2 | 0.399 | 0.032 | 0.197 | 0.038 |
| 0.3 | 0.387 | 0.029 | 0.286 | 0.022 |

**Table 9.2.** Omission bias, same shared variance as sibship C or couple M.

| shared variance | bias in A if C omitted (SD) | bias in A if M omitted (SD) |
|---:|---:|---:|
| 0.1 | +0.047 (0.031) | -0.005 (0.031) |
| 0.2 | +0.101 (0.034) | +0.025 (0.040) |
| 0.3 | +0.156 (0.037) | +0.034 (0.035) |

Omitting M biases A much less than omitting C, but not identically zero. The
M=0 result is a constrained boundary floor, not a false-positive rate. The
simulation generates shared adult environment; it does not validate a
generative assortative-mating interpretation.

## 10. Registry family-history and ADuLT prediction (`bench_fh_prediction.py`)

Age-, cohort- and mortality-consistent three-generation pedigrees, 4,000
families and three independent cohorts per main cell; relatives censored at
death or current age; scored against known true genetic liability.

**Table 10.1.** Ascertainment. Both family-history scores use PA; the second
changes the bounds (adds age and cohort), not the engine.

| observed proband case fraction | case/control corr | classic LT-FH corr (PA) | FH + age/cohort corr (PA) | classic LT-FH squared-correlation eff-N proxy / c-c | age/cohort proxy / classic |
|---:|---:|---:|---:|---:|---:|
| 0.0215 (population) | 0.232 | 0.346 | 0.350 | 2.233 ± 0.127× | 1.024 ± 0.003× |
| 0.10 | 0.471 | 0.525 | 0.531 | 1.242 ± 0.018× | 1.022 ± 0.002× |
| 0.25 | 0.621 | 0.654 | 0.663 | 1.107 ± 0.008× | 1.030 ± 0.002× |
| 0.50 | 0.694 | 0.722 | 0.738 | 1.081 ± 0.003× | 1.045 ± 0.002× |

The population-sampling **2.233 ± 0.127×** is
`(corr(classic LT-FH) / corr(case/control))²`, a squared-correlation eff-N
proxy, not the causal-SNP NCP ratio of §4. Family history is most valuable
relative to case/control under population sampling; the incremental
age/cohort benefit grows under ascertainment.

Cohort effects: at a 3× lifetime-prevalence trend per 30 birth years,
cohort-aware and single-K pedigree scores have similar narrow-cohort ranking
(0.7419 versus 0.7399), but their mean scores differ by -0.07335 ± 0.00025.
Truth-referenced mean errors are +0.0041 ± 0.0057 (cohort-aware) and
-0.0693 ± 0.0057 (single-K).

**Table 10.2.** Cohort span, own-onset panel with **no family-history inputs**
(therefore ADuLT): cohort-aware versus cohort-blind ranking.

| cohort half-span | ADuLT cohort-aware corr | ADuLT cohort-blind corr |
|---:|---:|---:|
| ±10 years | 0.387 ± 0.021 | 0.378 ± 0.025 |
| ±25 years | 0.427 ± 0.015 | 0.374 ± 0.023 |
| ±40 years | 0.477 ± 0.015 | 0.383 ± 0.018 |
| ±55 years | 0.512 ± 0.007 | 0.397 ± 0.017 |

## 11. Genetic factor diagnostic (`bench_genetic_correlation.py` panel (c), formerly `bench_genetic_factor.py`)

*Rows in `bench_genetic_correlation.csv` (panel (c)).*

Fifteen independent cohorts, 3,000 families, five traits. Under planted
one-factor truth, loadings 0.8/0.7/0.6/0.5/0.4 are recovered as
0.809/0.695/0.596/0.522/0.411, with SD 0.052–0.088.

**Table 11.1.** SRMR by truth and fitted model.

| truth and fitted model | SRMR mean | SRMR SD |
|---|---:|---:|
| true 1F, fit 1F | 0.042 | 0.011 |
| true 2F, fit 1F | 0.145 | 0.033 |
| true 2F, fit 2F | 0.016 | 0.011 |

SRMR diagnoses this planted misspecification. It is not a calibrated
factor-number test (extra factors improve in-sample fit by construction);
bootstrap the whole correlation-to-factor pipeline for uncertainty.

## 12. Score calibration (`bench_calibration.py`)

**Table 12.1.** Correctly specified PA; five independent 3,000-family cohorts
per cell (seeds 1–5), across-seed mean ± SE.

| structure | K | slope | corr | top-decile realised/predicted |
|---|---:|---:|---:|---:|
| parents + siblings | 0.01 | 1.004 ± 0.024 | 0.259 ± 0.003 | 0.982 ± 0.059 |
| parents + siblings | 0.05 | 1.006 ± 0.008 | 0.431 ± 0.004 | 1.003 ± 0.030 |
| parents + siblings | 0.20 | 1.006 ± 0.005 | 0.593 ± 0.003 | 1.001 ± 0.007 |
| extended | 0.01 | 0.989 ± 0.027 | 0.249 ± 0.004 | 0.946 ± 0.018 |
| extended | 0.05 | 1.002 ± 0.019 | 0.429 ± 0.008 | 0.991 ± 0.023 |
| extended | 0.20 | 1.002 ± 0.007 | 0.595 ± 0.006 | 1.007 ± 0.004 |

Every slope mean is within 1.3 SE of 1 and every PA intercept within 0.01 of 0.
The panel is PA-only; Gibbs agreement is in §1 and §14. One departure
replicates: at K=0.01 with the extended pedigree the top-decile
realised/predicted ratio is 0.946 ± 0.018, about 3 SE below 1, so the top of
that score is over-stated by roughly 5%. Calibration is good in every tested
cell, but the rare-disease top decile blocks a universal calibration claim.

**Table 12.2.** Misspecified h², true h²=0.5, K=0.05.

| assumed h² | slope | corr | top realised/predicted | calibration RMSE |
|---|---:|---:|---:|---:|
| 0.2 | 2.240 ± 0.015 | 0.429 ± 0.004 | 2.256 ± 0.065 | 0.165 ± 0.004 |
| 0.4 | 1.216 ± 0.009 | 0.431 ± 0.004 | 1.215 ± 0.036 | 0.066 ± 0.004 |
| 0.5 | 1.006 ± 0.008 | 0.431 ± 0.004 | 1.003 ± 0.030 | 0.040 ± 0.006 |
| 0.6 | 0.864 ± 0.007 | 0.430 ± 0.004 | 0.862 ± 0.025 | 0.058 ± 0.006 |
| 0.8 | 0.678 ± 0.006 | 0.429 ± 0.004 | 0.678 ± 0.020 | 0.135 ± 0.006 |

Ranking barely changes (corr 0.429–0.431, SE ≤ 0.004) while the slope sweeps
from 2.240 ± 0.015 to 0.678 ± 0.006: ranking-robust, scale-fragile. A
realised/predicted ratio above 1 means the score under-predicted the realised
top decile, not that it overstated it.

## 13. Cohort confounding (`bench_confounding.py`)

**Table 13.1.** Lambda GC for SNPs correlated with birth cohort; three
independent cohorts per trend. Truly independent null SNPs stay near 1 for
every method.

| prevalence trend R per 30 y | FH + cohort-specific K | single-K FH | case/control |
|---:|---:|---:|---:|
| 1 | 0.960 ± 0.012 | 0.960 ± 0.012 | 0.899 ± 0.034 |
| 2 | 0.964 ± 0.054 | 4.539 ± 0.253 | 2.008 ± 0.257 |
| 3 | 0.922 ± 0.057 | 10.275 ± 0.078 | 4.616 ± 0.220 |
| 4 | 1.007 ± 0.040 | 16.460 ± 0.399 | 7.666 ± 0.094 |

This isolates the cohort component in a family model; it is not the full
age/sex/cohort LT-FH++ design. At R=1 there is no trend-driven inflation (the
methods need not have identical finite-sample lambda). Under strong trends,
cohort-aware thresholds remove the induced stratified-SNP inflation on average.

## 14. PA robustness (`bench_pa_robustness.py`)

**Table 14.1.** Three independent seeds per cell; across-seed mean ± SE
(per-seed rows in `bench_pa_robustness.csv`).

| regime | corr(PA, Gibbs) | corr(PA, true g) | corr(Gibbs, true g) |
|---|---:|---:|---:|
| baseline | 0.999912 ± 0.000004 | 0.438 ± 0.005 | 0.438 ± 0.005 |
| large pedigree | 0.999879 ± 0.0000003 | 0.465 ± 0.011 | 0.465 ± 0.011 |
| rare, K=0.005 | 0.999504 ± 0.000025 | 0.216 ± 0.007 | 0.216 ± 0.007 |
| densely affected | 0.999189 ± 0.000020 | 0.486 ± 0.002 | 0.487 ± 0.002 |

The worst single-seed PA–Gibbs agreement is 0.99916 (densely affected).

**Table 14.2.** Fold-order spread as a percentage of the between-proband score
SD (across-seed mean ± SE).

| pedigree | median | p95 |
|---|---:|---:|
| trio | 0.081% ± 0.011 | 1.90% ± 0.03 |
| parents + 2 siblings | 0.099% ± 0.010 | 2.25% ± 0.09 |
| extended | 0.083% ± 0.011 | 2.34% ± 0.15 |
| large | 0.069% ± 0.006 | 3.24% ± 0.05 |

The typical order effect is tiny. The p95 values are small and rise with
pedigree size in this grid (the largest pedigree has the highest p95 in all
three seeds); the medians are not monotone. The script does not compare
fold-order spread with Gibbs Monte Carlo noise and does not measure rank
changes, so it makes neither claim.

## 15. Integrated personalised LT-FH++ independent-SNP association benchmark (`bench_ltfhpp_personalization.py`)

**Table 15.1.** Integrated panel. Ten paired replicates, 4,000 ascertained
probands each, 1,200 SNPs (30 causal, 300 sex/cohort-stratified null); age-,
sex- and cohort-dependent CIP, coherent onset/follow-up, sex-dependent
competing mortality. Accuracy, slope and association values are adjusted for
proband sex and birth year; `±` is replicate SE. The last column is the
stratified-null lambda before and after the same standard covariate adjustment.

| Phenotype | adjusted corr | adjusted slope | adjusted causal-SNP NCP ratio / c-c | stratified-null lambda raw -> adjusted |
|---|---:|---:|---:|---:|
| case/control | 0.586 ± 0.008 | 1.123 ± 0.021 | 1.000× | 1.140 ± 0.051 -> 1.019 ± 0.024 |
| ADuLT (same full personalised proband CIP, no FH) | 0.600 ± 0.008 | 1.004 ± 0.018 | 1.049 ± 0.004× | 1.024 ± 0.047 -> 1.014 ± 0.031 |
| LT-FH single-K | 0.629 ± 0.008 | 1.183 ± 0.020 | 1.149 ± 0.007× | 1.131 ± 0.046 -> 0.990 ± 0.041 |
| FH + age CIP (ablation) | 0.639 ± 0.008 | 1.017 ± 0.016 | 1.189 ± 0.006× | 0.997 ± 0.032 -> 1.021 ± 0.042 |
| FH + age + sex CIP (ablation) | 0.639 ± 0.008 | 1.016 ± 0.016 | 1.189 ± 0.006× | 0.983 ± 0.032 -> 1.015 ± 0.038 |
| FH + age + cohort CIP (ablation) | 0.640 ± 0.008 | 1.004 ± 0.016 | 1.194 ± 0.006× | 1.013 ± 0.034 -> 1.026 ± 0.033 |
| **LT-FH++ (full age + sex + cohort CIP)** | **0.640 ± 0.008** | **1.004 ± 0.015** | **1.194 ± 0.006×** | **0.994 ± 0.035 -> 1.009 ± 0.041** |

The matched ADuLT row uses exactly the full personalised proband bounds but no
relative columns. The stratified-null panel is a **covariate-adjustment sanity
check**, not evidence that personalization substitutes for standard GWAS
adjustment: after FWL adjustment lambda is near one for every phenotype.

**Table 15.2.** Paired increments of Table 15.1 (paired t-based 95% CI
half-widths; NCP-ratio units).

| contrast | Δ adjusted corr | Δ causal-SNP NCP ratio |
|---|---:|---:|
| full LT-FH++ − ADuLT (adding relatives) | **+0.04073 ± 0.00358** | **+0.1454 ± 0.0155** |
| ADuLT − case/control | | +0.0490 ± 0.0096 |
| single-K − case/control | | +0.1489 ± 0.0147 |
| + age, beyond single-K | | +0.0399 ± 0.0076 |
| + cohort, beyond age | | +0.0056 ± 0.0022 |
| full LT-FH++ − classic LT-FH | | **+0.0455 ± 0.0073** |
| + sex, beyond age | | -0.0004 ± 0.0010 |
| + sex, beyond age + cohort | | -0.00004 ± 0.00076 |

This cancellation-prone scenario establishes age and cohort gains, but no
conditional sex-specific NCP gain in this independent-SNP design.

**Table 15.3.** Prespecified sex-CIP isolation. Five paired replicates, 3,000
ascertained probands, 600 SNPs (30 causal), age-dependent CIP, female:male
lifetime-risk ratio 2, equal onset midpoints, no cohort or sex-dependent
mortality effect; fixed before seeing the integrated result.

| Phenotype | adjusted corr | adjusted causal-SNP NCP ratio / c-c | female mean error | male mean error |
|---|---:|---:|---:|---:|
| FH + age CIP (ablation) | 0.6298 ± 0.0078 | 1.292 ± 0.016× | +0.0301 ± 0.0086 | -0.0142 ± 0.0084 |
| FH + age + sex CIP (ablation) | 0.6307 ± 0.0080 | 1.296 ± 0.016× | -0.0068 ± 0.0087 | -0.0002 ± 0.0084 |

The adjusted ranking and NCP-ratio increments are small and unresolved:
Δcorr = +0.00097 ± 0.00111 and ΔNCP ratio = +0.0040 ± 0.0046 (paired 95% CIs).
The calibration benefit is decisive because the paired errors are highly
correlated: the correct sex curve shifts female error by -0.03691 ± 0.00040
and male error by +0.01401 ± 0.00064, closing the female-minus-male error gap
by **0.05092 ± 0.00096**.

On the first two 300-family, no-mixture main-panel cross-checks, PA/Gibbs
agreement is 0.999964. Gibbs reaches the requested MCSE tolerance for every
score (maximum MCSE 0.0049 at tolerance 0.03); PA-vs-Gibbs normalized RMSE is
0.0132 score SD, the Gibbs-on-PA slope is 0.990, and the mean difference is
-0.0018.

## 16. PA-FGRS censoring-mixture validation (`bench_pafgrs_mixture.py`)

Families (proband + parents + sib) under the liability-threshold model with a
logistic CIP (h2 = 0.5, lifetime prevalence 0.10, mid-point 60), censored
honestly (a case is observed only if onset precedes current age), scored
against true genetic liability over 5 replicates of 20,000 families. Two
censoring regimes (MID: mid-life, heavy; OLD: old, light) and three observation
models: threshold crossing (the LT-FH++ convention); stochastic onset, drawn
from the CIP independently of liability (the mixture's native model); and a
liability-dependent copula (ρ = 0.6), the assumption-stress arm, since the
mixture treats future cases as a random draw from the above-threshold tail.
Slope = regress(true on estimate); `±` is across-replicate SE. Every
replicate and paired contrast is in `bench_pafgrs_mixture.csv` (one row per
model × regime × rep × arm).

**Table 16.1.** Threshold-crossing observation model.

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base (lifetime case) + no-mixture | 0.3281 ± 0.0047 | 1.1992 ± 0.0151 | 0.4693 ± 0.0030 | 1.0293 ± 0.0069 |
| base + mixture | 0.3277 ± 0.0047 | 1.1738 ± 0.0149 | 0.4691 ± 0.0030 | 1.0209 ± 0.0068 |
| interval case + mixture | 0.3334 ± 0.0047 | 0.8444 ± 0.0097 | 0.4766 ± 0.0033 | 0.8263 ± 0.0050 |
| pinned case + no-mixture | 0.3337 ± 0.0048 | 0.9977 ± 0.0109 | 0.4770 ± 0.0034 | 0.9885 ± 0.0057 |
| pinned case + mixture | 0.3337 ± 0.0048 | 0.9766 ± 0.0109 | 0.4769 ± 0.0034 | 0.9811 ± 0.0057 |

Gibbs cross-check on base + no-mixture, corr(PA, Gibbs) MID / OLD with Gibbs
slopes: crossing 0.9998 / 0.9999 (1.2192 / 1.0217 — the MID slope > 1 is the
case encoding's information loss, not a PA artifact); stochastic 0.9998 /
0.9999 (1.0685 / 0.9845); liability-dependent 0.9998 / 0.9999
(1.1419 / 1.0127).

**Table 16.2.** Stochastic-onset observation model.

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base + no-mixture | 0.2781 ± 0.0018 | 1.0267 ± 0.0069 | 0.4536 ± 0.0037 | 0.9990 ± 0.0082 |
| base + mixture | 0.2782 ± 0.0018 | 1.0057 ± 0.0067 | 0.4537 ± 0.0037 | 0.9911 ± 0.0082 |

**Table 16.3.** Liability-dependent onset (ρ = 0.6).

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base + no-mixture | 0.3089 ± 0.0020 | 1.1321 ± 0.0103 | 0.4638 ± 0.0040 | 1.0162 ± 0.0083 |
| base + mixture | 0.3088 ± 0.0020 | 1.1088 ± 0.0101 | 0.4638 ± 0.0040 | 1.0081 ± 0.0083 |
| pinned + no-mixture | 0.3084 ± 0.0021 | 0.9259 ± 0.0089 | 0.4644 ± 0.0041 | 0.9685 ± 0.0075 |
| pinned + mixture | 0.3083 ± 0.0021 | 0.9008 ± 0.0086 | 0.4642 ± 0.0042 | 0.9559 ± 0.0075 |

**Table 16.4.** Paired mixture-minus-no-mixture contrasts on identical cohorts;
mean ± SE with t-based 95% CI half-width, 5 replicates.

| cell | case encoding | Δcorr | Δslope |
|---|---|---|---|
| crossing MID | lifetime interval | -0.00034 ± 0.00004 (CI ± 0.00011) | -0.02537 ± 0.00032 (CI ± 0.00088) |
| crossing MID | pinned | -0.00004 ± 0.00004 (CI ± 0.00010) | -0.02682 ± 0.00032 (CI ± 0.00090) |
| crossing OLD | lifetime interval | -0.00017 ± 0.00005 (CI ± 0.00013) | -0.00844 ± 0.00013 (CI ± 0.00036) |
| crossing OLD | pinned | -0.00010 ± 0.00005 (CI ± 0.00014) | -0.01251 ± 0.00013 (CI ± 0.00036) |
| stochastic MID | lifetime interval | +0.00012 ± 0.00005 (CI ± 0.00013) | -0.02101 ± 0.00030 (CI ± 0.00083) |
| stochastic OLD | lifetime interval | +0.00008 ± 0.00005 (CI ± 0.00015) | -0.00785 ± 0.00016 (CI ± 0.00046) |
| dependent MID | lifetime interval | -0.00013 ± 0.00003 (CI ± 0.00008) | -0.02336 ± 0.00024 (CI ± 0.00066) |
| dependent MID | pinned | -0.00007 ± 0.00005 (CI ± 0.00013) | -0.02509 ± 0.00029 (CI ± 0.00081) |
| dependent OLD | lifetime interval | -0.00007 ± 0.00004 (CI ± 0.00012) | -0.00812 ± 0.00014 (CI ± 0.00038) |
| dependent OLD | pinned | -0.00020 ± 0.00006 (CI ± 0.00015) | -0.01265 ± 0.00016 (CI ± 0.00045) |

**Verdict.**

- *Correlation shifts are measurable but negligible.* Four of ten Δcorr CIs
  exclude zero; the largest is -0.00034 (95% CI ± 0.00011), about 0.1% of the
  correlation level — including when the independence assumption is false.
- *Calibration shifts are real and directional.* Every Δslope CI excludes zero;
  the mixture lowers the slope by 0.008-0.027, most under heavy censoring.
  Where the no-mixture encoding under-conditions (lifetime interval, MID:
  1.20, 1.03, 1.13 across the three onset models) this moves calibration
  toward 1, including under the native stochastic model (1.027 -> 1.006).
  Where it is already calibrated the mixture tilts slightly past (crossing
  pinned MID 0.998 -> 0.977): under threshold crossing plain age truncation is
  already exact for censored controls.
- *Case encoding dominates calibration, and the right one depends on the onset
  model.* Under threshold crossing pinned cases give slope 0.98-1.00, the
  lifetime interval loses onset-age information (up to 1.20) and the
  age-specific interval over-disperses (0.83). At ρ = 0.6 the pin
  over-conditions (0.93 MID / 0.97 OLD: the *wrong* encoding)
  while the lifetime interval still under-conditions (MID 1.13, between
  crossing 1.20 and stochastic 1.03).
- *Guidance.* Pin cases at their onset threshold only where the crossing model
  is believed; use the lifetime interval when onset ages are unreliable or only
  partly liability-dependent.

## 17. Inference-machinery calibration (`bench_inference_calibration.py`)

*Stdout only: no archived artifact; the values below are the only record.*

**Table 17.1.** R = 25 independent datasets per panel (n_boot = 50, reduced fit
iterations). Rows 1–3: A-only data, 400 families (proband + parents + sib),
prevalence 0.1, true h2 = 0.5, C = 0. Row 4: two traits, h2 = (0.5, 0.4), r_g
= 0, r_p = 0.2, prevalence 0.1, parents + 2 sibs (§7's null cell at the same
R, n_fam, n_boot and fit settings). Intervals are Clopper–Pearson two-sided 95%.

| check | result | 95% CI | detail |
|---|---|---|---|
| Type-I, `test_variance_component("C")` | 0/25 rejections at 0.05 | 0.00–0.14 | p mean 0.428, median 0.392, min 0.059 |
| `bootstrap_fit` coverage of h2 = 0.5 | 24/25 = 96% at nominal 95% | 0.796–0.999 | bootstrap SE mean 0.168 vs across-dataset SD 0.132 (ratio 1.27) |
| MCEM OPG SE | mean SE 0.127 vs across-dataset SD 0.136 (ratio 0.93) | — | point estimate mean A = 0.502 (truth 0.5) |
| Type-I, `test_genetic_correlation` | 0/25 rejections at 0.05 | 0.00–0.14 | p mean 0.577, median 0.569, min 0.059 |

No anti-conservatism: the component test is if anything mildly conservative at
this resolution (a uniform null would give mean 0.5, min ~0.04), and the
phenotypic correlation is not mistaken for a genetic one. The bootstrap SE is
mildly conservative, consistent with the slight over-coverage. The internal
`h2_se` is a within-dataset Monte-Carlo diagnostic that substantially
understates the across-dataset SD (`ltpred/fit.py` `FitResult`; the >5x gap is
locked by `test_bootstrap_fit_scalar_and_calibration`), so the bootstrap
remains the right route. The OPG SE is close to the sampling SD at this design,
slightly narrow. R = 25 bounds the resolution (a true 10% Type-I rate would
show 0/25 with ~7% chance): read "no gross miscalibration", not proof of exact
calibration.

## 18. Model-misspecification stress (`bench_misspecification.py`)

*Stdout only: no archived artifact; the values below are the only record.*

**Table 18.1.** One assumption violated at a time; 8,000 families, 5
replicates, PA, classic case/control bounds, h2 = 0.5, true prevalence 0.10.

| arm | corr | slope |
|---|---|---|
| control | 0.4983 | 1.0002 |
| heavy-tail env (t_5) | 0.4820 | 0.9898 |
| assortative mating (phenotypic rho ~ 0.3) | 0.4915 | 0.9741 |
| sibship shared env (c2 = 0.15, unmodeled) | 0.4934 | 0.9776 |
| wrong prevalence 0.05 (true 0.10) | 0.4982 | 0.8886 |
| wrong prevalence 0.20 (true 0.10) | 0.4980 | 1.1328 |

**Verdict: structurally robust; prevalence is the calibration hazard.** Heavy
tails, generative assortative mating and an unmodeled sibship environment each
move the slope by only 0.01-0.03 (in the direction theory predicts: familial
resemblance over-credited to genes). The ranking cost is small but not uniformly
negligible: assortative 0.007, sibship 0.005, heavy-tail 0.016 (0.4983 +/-
0.0045 vs 0.4820 +/- 0.0024), about 3% of the control correlation and roughly
3 SE — real if minor. A factor-2 prevalence/threshold error moves the slope to
0.89-1.13: the risk the personalised CIPs of LT-FH++ exist to remove, and the
reason threshold provenance matters more than model refinement here.

## 19. CIP estimation from follow-up records (`bench_cip_estimation.py`)

`ltpred.cip` estimates the cumulative-incidence curve that `thresholds_from_cip`
consumes from registry-style follow-up records (entry age, exit age, event
code), with left truncation and competing risks. It matches the LT-FH++
construction (Pedersen et al. 2022: Aalen-Johansen with death and emigration as
competing events, sex × birth-year strata).

**Table 19.1.** One simulated registry cohort (N = 50,000; known logistic CIP,
lifetime prevalence 0.10; Gompertz mortality; administrative censoring; one arm
with a 1995 register start). Artifact: `bench_cip_estimation.csv`.

| arm | estimator and estimand | result |
|---|---|---|
| no mortality | Kaplan-Meier | max abs error 0.0013; all 13 prespecified age-grid truths inside the pointwise ±2SE bands |
| with mortality (42% death share) | Aalen-Johansen, crude cumulative incidence | max abs error 0.0017 |
| with mortality | Kaplan-Meier treating death as censoring, vs crude curve | overestimates by up to 0.0218 (competing-risks bias) |
| with mortality | same, vs its own marginal no-death-world curve | max abs error 0.0019 |
| delayed entry (register starts 1995) | Aalen-Johansen | max abs error 0.0019 |
| end-to-end | estimated curve -> `thresholds_from_cip` -> `estimate_liability` | calibration slope 1.0023 vs 1.0069 with the oracle curve; correlation identical (0.3861) |

Grid containment is single-dataset, not repeated-sample coverage (a
person-level bootstrap would give that), and repeated-run uncertainty of the
end-to-end scores was not retained. Estimand choice is the user's call; for
LT-FH++ thresholds the crude curve is the right one (the dead cannot be
diagnosed).

## 20. Pedigree inference from trio records (`bench_pedigree_inference.py`)

*Regenerated 2026-09-03 (Table 0.1) with the supported observation contract:
ancestors added solely for ancestral closure keep uninformative bounds, so the
payoff arm measures relatives reached within `max_degree` only.*

`ltpred.pedigree` discovers a proband's relatives from population
parent-offspring records (the Pedersen et al. 2025 graph-extraction niche;
full-sibling edges give the standard degree scale: parents/siblings 1,
grandparents/half-sibs/aunts 2, first cousins 3). Exactness and scale use one
simulated 3-generation population (2,683 persons, remarriages and cousin
links); the payoff is replicated over five independent populations (2,529 to
2,719 persons). `±` is across-replicate SE (sd/sqrt(R)); the paired contrast
carries a t-based 95% CI. Per-replicate values are in
`bench_pedigree_inference.csv` (long format `rep, metric, value`; rep 0 marks
single-run parts). Kinship is the exact tabular method (inbreeding-aware), not
the 2025 paper's path-counting approximation.

- **Exactness:** with a full ancestral closure (every recorded ancestor of the
  extracted set included), the extracted sub-pedigree's kinship equals the
  full-population kinship restricted to the members -- max abs diff **0.0**
  over 300 probands (single run). Without the closure, boundary members who are
  siblings are split into unrelated founders (diff 0.5). The degree limit
  truncates only *which relatives* are included, never the kinship among them.
- **Payoff (the LT-FGRS point):** on 300 probands per replicate,
  corr(est, truth) is **0.569 ± 0.013** using all relatives up to third degree
  vs **0.498 ± 0.016** with the named-role subset the grammar encodes (parents,
  full siblings, grandparents); the paired contrast is **+0.0706 ± 0.0060**,
  95% CI [+0.054, +0.087] (ratio of means 1.14; the two scores correlate
  0.8987 ± 0.0062). Which relatives you include matters, consistent with the
  LT-FGRS package (Pedersen et al.).
- **Scale:** 3,000 extractions at degree 3 in 0.19 s (single-run timing,
  ~0.06 ms per proband); neighbourhoods stay small (tens of nodes), so
  per-proband extraction plus a small dense kinship covariance is the right
  architecture.

## 21. End-to-end register pipeline (`bench_register_pipeline.py`)

*Regenerated 2026-09-21 from `simulate_register_liabilities`' Cholesky draw of
`h2 A`, same design as the 2026-09-03 run; claims are replicate means, not
seed-locked constants.*

`ltpred.pipeline.estimate_liabilities` chains trio records -> pedigree
discovery -> per-stratum CIP thresholds -> per-proband scores. Five independent
synthetic registers (3-generation populations of 2,529 to 2,719 with
remarriages; one consistent liability field `G ~ N(0, h2 A)`, `L = G + E`;
crossing model with a logistic CIP, lifetime prevalence 0.10). `±` is
across-replicate SE (sd/sqrt(R)); paired contrasts carry t-based 95% CIs.
Per-replicate values are in `bench_register_pipeline.csv` (long format
`rep, metric, value`; rep 0 marks the single-run throughput part).

**Table 21.1.** Accuracy against true g.

| arm | corr(est, true g) | calibration slope |
|---|---:|---:|
| degree 3 (to first cousins) | **0.567 ± 0.029** | 1.05 ± 0.05 |
| degree 1 (first-degree only) | **0.522 ± 0.031** | 1.05 ± 0.05 |
| CIP estimated from the register | 0.5671 ± 0.0288 | — |
| CIP oracle curve | 0.5668 ± 0.0291 | — |

**Table 21.2.** Prospective prediction: diagnosis in (40, 70] after index age
40; observed future-case rate 0.081 ± 0.006.

| score | corr with future outcome | rank (Mann-Whitney) AUC |
|---|---:|---:|
| (a) honest, familywise-censored | 0.167 ± 0.007 | **0.654 ± 0.021** |
| (b) + relatives' post-index events | 0.211 ± 0.023 | 0.697 ± 0.027 |
| (c) + proband's own future outcome (leak) | 0.692 ± 0.018 | 0.991 ± 0.002 |

For (a), mean predicted future-case risk is 0.071 ± 0.001 and the linear
calibration slope of future on score is 0.19 ± 0.02.

**Table 21.3.** Paired contrasts.

| contrast | difference | 95% CI |
|---|---:|---:|
| degree 3 − degree 1, corr | **+0.0451 ± 0.0092** | [+0.020, +0.071] |
| estimated − oracle CIP, corr | +0.0003 ± 0.0003 | [-0.0006, +0.0012] |
| (b) − (a), corr | **+0.0436 ± 0.0220** | [-0.018, +0.105] |
| (b) − (a), AUC | **+0.0429 ± 0.0248** | [-0.026, +0.112] |
| (c) − (a), corr | **+0.524 ± 0.014** | [+0.486, +0.562] |
| (c) − (a), AUC | +0.337 ± 0.020 | [+0.282, +0.391] |

The second/third-degree contribution is a small real gain (the LT-FGRS effect).
corr ~0.57 is the accuracy at these registers' case rates and age structure
under the driver's pinned-onset LT-FH++ bounds; §20's 0.569 used a 10% rate
with a uniform lifetime threshold. Estimating the CIP from the register costs
at most a few thousandths of a correlation point (consistent with §19). **The
relatives'-events contrast is unresolved at R = 5 on both metrics**: do not
quote a relatives'-events payoff from this panel. The proband's-own-outcome
leakage is unambiguous and large: honest censoring costs real accuracy.

Throughput: 987 probands/s (400 probands in 0.41 s; single run, BLAS pinned to
1; v0.7.1 commit `5b42b13`, 2026-09-23, one-minute load 2.1), against 276
probands/s (1.45 s) at v0.7.0. A proband with many selected relatives now gets
its relationships from a dense matrix over the extracted pedigree instead of
memoized pair recursion; every accuracy, CIP and prospective value above
reproduced exactly.

## 22. Tetrachoric correlations (`bench_tetrachoric.py`)

`ltpred.tetrachoric` estimates the latent liability correlation from 2x2
case/control tables by maximum likelihood (thresholds from the marginals,
bounded scalar optimisation over rho, SciPy's bivariate-normal CDF with
requested absolute and relative tolerances of `1e-10`, which are not a proven
error bound; SEs from the observed information).

**Table 22.1.** Liability-threshold families, h2 = 0.5, prevalence 0.1, 5
replicates of 20,000 families; ± is the SE of the mean across the five
replicates (`np.std(..., ddof=1)/sqrt(REPS)`), not the sample SD. Artifact:
`bench_tetrachoric.csv`.

| pair | expected h2*A | tetrachoric | latent corr |
|---|---|---|---|
| o-m | 0.250 | 0.258 ± 0.009 | 0.251 |
| o-f | 0.250 | 0.240 ± 0.010 | 0.252 |
| o-s1 | 0.250 | 0.247 ± 0.011 | 0.248 |
| m-s1 | 0.250 | 0.267 ± 0.007 | 0.253 |
| o-mgm | 0.125 | 0.115 ± 0.006 | 0.127 |
| o-mau1 | 0.125 | 0.127 ± 0.008 | 0.128 |
| m-f (mates) | 0.000 | 0.016 ± 0.006 | 0.000 |

The pairwise means broadly track h2 * A and the latent Pearson correlations,
but several cells differ from the nominal target by more than one reported SE
(notably mother-sibling and the mate pair); five replicates are too few for a
calibrated equivalence claim. The Falconer estimate h2 ~ 2 x
tetrachoric(first-degree) gives 0.497 ± 0.016 (truth 0.5) from binary relative
pairs alone, agreeing with `fit_heritability` on 4,000-family subsets of those
cohorts (0.515 ± 0.031); the tetrachoric uses all 20,000 families, so this does
not establish greater efficiency at equal sample size. `tetrachoric_matrix`
produces the expected h2*A block for multi-variable status matrices. Use it as
a fast diagnostic of the family model, before any fitting.

## 23. Liability-scale transformations (`bench_liability_scale.py`)

`ltpred.liability_scale` implements the Lee et al. (2011) observed/liability
heritability bridge and probit estimation of residual-scale genetic variance
`q` (the probit model IS the liability-threshold model: per-SNP `q` is the
identity 2 f (1-f) beta²; the z-statistic route is Lee & Wray 2013 with the
master factor, null-adjusted by default as `(z² - 1) / N`). The default `q` is
not a total-liability fraction; aggregate it before applying `q / (1 + q)`.

**Table 23.1.** Polygenic disease on the probit convention (liab = X beta +
eps, Var(X beta) = 0.5, prevalence 0.1). Artifact: `bench_liability_scale.csv`.

| route | estimate | target |
|---|---|---|
| (a) joint probit fit -> sum 2f(1-f)beta² | 0.510 ± 0.017 | 0.500 (exact) |
| (a') marginal probit fits (GWAS practice) | 0.352 ± 0.007 | ~1/(1+V_bg) attenuated |
| (b) null-adjusted probit z², Lee & Wray 2013 factor | 0.320 ± 0.024 | same as (a'), attenuated |
| (c) OLS observed-scale total | 0.119 ± 0.011 | (observed scale) |
| (c) Lee-2011 bridged to liability | 0.334 ± 0.032 | 1/3 (Lee fraction) |

The joint probit fit recovers the probit residual-scale total (0.5); marginal
per-SNP fits attenuate by the polygenic background in their residuals
(~1/(1+V_bg), material at h²=0.5); the null-adjusted z route agrees with that
attenuated total; and the OLS/Lee-2011 route recovers the
*fraction-of-total-liability* form (1/3), the McKelvey-Zavoina R² of Lee et al.
2012 (Genet Epidemiol, eq. 9), implemented as
`probit_liability_r2(..., fraction=True)`. Paper-fixture tests cover the Lee
2011 Discussion factors (observed 0.18/0.54/0.91 -> liability 0.1/0.3/0.5 at
K=0.01, P=0.5) and the Crohn's Table 3 fixture (0.61 -> 0.22).

## 24. Environment components in estimation (`bench_shared_env.py` panel (c), formerly `bench_env_components.py`)

*Rows in `bench_shared_env.csv` (panel (c)). corr(o) reflects
`estimate_liability(out="full")` applying the proband's own bound after the
relative fold.*

The sibship (C) and couple (M) components fitted by `fit_variance_components`
are wired into liability estimation: `construct_covmat_single(..., c2=...,
m2=...)` and the estimator front doors (`estimate_liability`,
`estimate_liability_pa_arrays`, `estimate_liability_gibbs_arrays`) accept them.
The genetic target stays coupled to relatives only through h2 * A; the
components only change how relatives are conditioned.

**Table 24.1.** True h2 = 0.4, sibship c2 = 0.15, couple m2 = 0.15; 5
replicates of 4,000 families.

| arm | corr(g) | slope(g) | corr(o) |
|---|---|---|---|
| additive-only (misspecified) | 0.454 | 0.927 | 0.605 |
| oracle-wired (c2/m2 at truth) | 0.455 | 0.986 | 0.608 |
| fitted-wired (fit then wire) | 0.455 | 0.983 | 0.607 |

Wiring restores calibration: the additive-only model over-credits
environmental clustering to genetics (slope 0.93, over-dispersed) and wiring
recalibrates to 0.99. Full-liability prediction sharpens (corr(E[l_o | family],
truth) 0.605 -> 0.608). The fit->wire loop works end to end: fitted components
(A 0.43, C 0.14, M 0.15 vs truth 0.4/0.15/0.15) give the oracle's calibration.
Ranking is untouched (corr(g) flat).

## 25. Onset-age-structured genetic correlation (`bench_aod_decay.py`)

`fit_genetic_correlation_decay` fits a genetic correlation that **decays with the
onset-age difference** between relatives,
`Cov(g_i^p, g_j^q) = A_ij sqrt(h2_p h2_q) rho_g K(|a_ip - a_jq|; lam)`, by
Monte-Carlo EM (a likelihood M-step; a Haseman-Elston moment step is confounded
by age-dependent ascertainment truncation). The analytic cross-trait M-step
gradient is pinned against finite differences
(`research/tests/test_decay.py::GradientTests`).

**Table 25.1.** Two-trait nuclear families (h2 = 0.5/0.5, r_p = 0.3, prevalence
0.2, onset ages iid U(15, 65), a pessimistic choice that maximises the
amplitude-decay ridge); `n_em = 45`, `n_draw = 100`, 3 replicates per cell.

| panel | truth | fitted r_g | fitted lam_cross |
|---|---|---|---|
| scalar null | r_g = 0.5, lam = 0 | 0.506 +/- 0.049 | 0.001 +/- 0.002 |
| decay | r_g = 0.5, lam = 0.04 | 0.515 +/- 0.178 | 0.041 +/- 0.009 |
| data req (n = 1000) | r_g = 0.5, lam = 0.04 | 0.578 +/- 0.107 | 0.063 +/- 0.029 |
| data req (n = 2500) | r_g = 0.5, lam = 0.04 | 0.526 +/- 0.114 | 0.044 +/- 0.022 |
| r_g null | r_g = 0, lam = 0.04 | 0.009 +/- 0.089 | (unidentified) |

- **The headline r_g is recovered** (0.51-0.53 at `n_fam = 2500`), and the decay
  rate is identified in the right place (`lam_cross ~ 0.041`, `lam_within ~ 0.039`
  vs true 0.04).
- **The scalar null is clean**: `lam = 0` refits give `r_g ~ 0.51` and
  `lam ~ 0.001` -- no manufactured decay. No spurious correlation at the
  `r_g = 0` null (`0.009 +/- 0.089`).
- **The model is data-hungry**: at `n_fam = 1000` both `r_g` and `lam` run high
  (0.58 / 0.063) along the amplitude-decay ridge, converging as `n` grows into
  the thousands. Below that, prefer the scalar `fit_genetic_correlation`.
- **Sampling variability is large**: even at `n_fam = 2500` the across-replicate
  SD of `r_g` is ~0.11-0.18; use `bootstrap_fit` for sampling uncertainty.

### Robustness (`bench_aod_decay.py --robustness`)

The main grid is *circular* (the simulator draws from the fitted covariance),
so it cannot reveal misspecification. Two adversarial arms (`n_fam = 2000`,
`n_em = 35`, 2 reps):

- **Wrong kernel is benign**: truth Gaussian decay, fit OU gives `r_g ~ 0.55`
  (true 0.5).
- **Unmodelled shared family environment biases r_g DOWN**: a cross-relative
  environmental correlation (`c2 = 0.10`) the model cannot represent drops
  `r_g` from `0.50` to `~0.33`. The extra same-trait familial covariance
  inflates `h2`, and since `r_g = G / sqrt(h2_0 h2_1)` the inflated denominator
  attenuates `r_g`. The model has **no shared-family environmental component
  across relatives**, so traits with household effects will be mis-estimated --
  the most important caveat for application.

## 26. Sex in the covariance vs sex in the thresholds (`bench_sex_limitation.py`, now panel (a) of `bench_covariance_extensions.py`)

`construct_covmat_sex_limited` lets sex enter `Sigma` (sex-specific `h2`, a
cross-sex genetic correlation `rg`) rather than only the thresholds. The algebra
guarantees the BLUP weights change, not that the score improves. Families
(proband + both parents + one sibling) are drawn from the **true** sex-limited
covariance; proband and sibling sexes are balanced across the four cells;
`K_female = 0.05`, `K_male = 0.10`; 3 replicates of 2,000 families. Arms are
scored by `corr(estimate, true g)` and slope `regress(true on estimate)`; `±`
is a t-based 95% half-width.

**Table 26.1.** Arms.

| arm | thresholds | covariance |
|---|---|---|
| pooled | one pooled `K` | scalar pooled `h2` |
| sex thresholds | sex-specific `K` | scalar pooled `h2` |
| sex in Sigma (true) | sex-specific `K` | true `(h2_F, h2_M, rg)` |
| sex in Sigma (rg=1) | sex-specific `K` | true `h2` pair, `rg` wrongly 1 |

**Table 26.2.** Sweeping the true `rg` at `h2_F = 0.6`, `h2_M = 0.2`. `eff-N`
is the squared-correlation effective-N proxy for the covariance fix, thresholds
held fixed.

| true rg | sex thresholds | sex in Sigma | gain | eff-N proxy |
|---:|---:|---:|---:|---:|
| 1.0 | 0.3735 | 0.4149 | +0.0414 ± 0.0213 | 1.235x ± 0.126 |
| 0.8 | 0.3604 | 0.4023 | +0.0419 ± 0.0289 | 1.249x ± 0.199 |
| 0.6 | 0.3550 | 0.4013 | +0.0462 ± 0.0237 | 1.278x ± 0.151 |
| 0.4 | 0.3348 | 0.3937 | +0.0589 ± 0.0106 | 1.384x ± 0.064 |
| 0.2 | 0.3306 | 0.3921 | +0.0614 ± 0.0030 | 1.407x ± 0.046 |

**Table 26.3.** Sweeping the heritability gap at `rg = 1`, mean `h2 = 0.4`.

| gap | sex thresholds | sex in Sigma | gain | eff-N proxy |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4001 | 0.4001 | +0.0000 ± 0.0000 | 1.000x ± 0.000 |
| 0.2 | 0.3837 | 0.3948 | +0.0111 ± 0.0026 | 1.059x ± 0.023 |
| 0.4 | 0.3537 | 0.3942 | +0.0405 ± 0.0144 | 1.241x ± 0.070 |
| 0.6 | 0.3395 | 0.4335 | +0.0940 ± 0.0272 | 1.639x ± 0.334 |

At gap 0 with `rg = 1` the sex-limited covariance **is** the scalar one, so the
arms are bit-identical and the gain is exactly zero (the sanity check). The gain
is real but conditional: `1.06x` at a gap of 0.2, `1.24x` at 0.4, `1.64x` at
0.6. **A sex-limited model with no sex difference to find buys nothing**, the
case to expect by default. Mis-specifying `rg` as 1 when it is truly 0.2 costs
about a quarter of the gain.

**Table 26.4.** Mechanism. Proband + mother + sister versus proband + father +
brother: matched on relatedness (two first-degree relatives), differing only in
sex configuration; `h2 = 0.5` for both sexes, so any gain is attributable to
`rg`.

| true rg | composition | sex thresholds | sex in Sigma | gain |
|---:|---|---:|---:|---:|
| 1.0 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 0.6 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 0.2 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 1.0 | cross-sex | 0.4271 | 0.4271 | +0.0000 ± 0.0000 |
| 0.6 | cross-sex | 0.3740 | 0.3871 | +0.0131 ± 0.0096 |
| 0.2 | cross-sex | 0.3035 | 0.3538 | +0.0502 ± 0.0342 |

The effect appears exactly where the model predicts: zero in every same-sex row
regardless of `rg`, zero cross-sex when `rg = 1`, monotone in the remaining two.
The parameter acts on the pairs it is defined to act on; the model is worth
reaching for in pedigrees rich in opposite-sex relatives.

**Calibration is the cleaner signal.** Only the sex-limited covariance is
calibrated (slope `0.945-1.019` in every cell). The threshold-only arm degrades
as the gap grows (`0.9609` at gap 0 to `0.8390` at gap 0.6); the pooled arm sits
between.

**Sex thresholds alone can hurt.** Under genuine sex limitation with a
sex-blind covariance, personalising the *thresholds* by sex was **worse** than
ignoring sex: `0.3395` vs `0.3906` in correlation at gap 0.6, and worse
calibration (`0.8390` vs `0.9469`). Thresholds set `mu` but the estimate is
`w' mu`, and sharpening `mu` under wrong weights `w` need not improve the
product; the pooled arm's two mis-specifications partially cancel. At gap 0 the
pooled/threshold difference is `-0.0051`, within the CI, and the effect appears
only once the gap is nonzero. Sex-specific thresholds and covariance are not
substitutes; this is not an argument for sex-blind thresholds.

Caveats: one pedigree shape, a balanced sex design, PA inference, supplied
rather than fitted `h2_F`, `h2_M`, `rg`; "sex in Sigma (true)" is a ceiling
that a real analysis reaches only as well as its parameter estimates allow.

## 27. Ignoring genetic nurture (`bench_nurture.py`, now panel (b) of `bench_covariance_extensions.py`)

`construct_covmat_nurture` separates a proband's **direct** additive value from
the **indirect** path by which the parents' genotypes shape the rearing
environment. Families (proband + both parents + one full sib) are drawn from the
**true** path model, so the direct value `A_o` is known; `h2 = 0.4` (direct),
`K = 0.05`, 5 replicates of 3,000 families; `±` is a t-based 95% half-width.

**Table 27.1.** Estimators, scored against `A_o`.

| arm | covariance |
|---|---|
| additive, true h2 | ordinary, at the true *direct* `h2` |
| additive, moment h2 | ordinary, at the `h2` a moment fitter reads off parent-offspring pairs |
| A + C matched to sibs | `c2` chosen to reproduce the nurture sib-sib covariance exactly |
| nurture | the true path model -- the ceiling |

**Table 27.2.** Ranking.

| true n | additive (moment h2) | A + C matched | nurture | cost |
|---:|---:|---:|---:|---:|
| 0.0 | 0.3805 | 0.3805 | 0.3805 | +0.0000 ± 0.0000 |
| 0.1 | 0.3952 | 0.3949 | 0.3958 | +0.0006 ± 0.0008 |
| 0.2 | 0.4237 | 0.4221 | 0.4253 | +0.0017 ± 0.0010 |
| 0.3 | 0.4602 | 0.4550 | 0.4645 | +0.0043 ± 0.0025 |

At `n = 0` every arm is identical (the sanity check). Correlation *rises* with
`n` in all arms: a stronger parental contribution makes relatives more
informative about the parental genetic values, which correlate with `A_o`. The
relative cost of ignoring nurture stays small.

**Table 27.3.** Calibration and heritability.

| true n | c2 matched | h2 from parent-offspring | h2 from sibs | A+C slope | nurture slope |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | 0.4000 | 0.4000 | 1.0157 | 1.0157 |
| 0.1 | 0.0880 | 0.4800 | 0.5760 | 1.0693 | 1.0000 |
| 0.2 | 0.1920 | 0.5600 | 0.7840 | 1.1458 | 1.0077 |
| 0.3 | 0.3120 | 0.6400 | 1.0000 | 1.1957 | 1.0005 |

The nurture model is calibrated at every setting (slope `1.000-1.016`); the A+C
model, which reproduces the sibling covariance **exactly**, drifts to `1.196`,
under-predicting the direct effect by about a fifth. The true direct `h2` is
`0.4` throughout, but a nurture-blind moment fitter returns `0.64` from
parent-offspring pairs and `1.000` (the boundary) from sibs at `n = 0.3`. **The
two estimates disagree by up to 0.36, and that disagreement is the
diagnostic**: no single additive `h2` fits both relative types when an indirect
path is present. Nurture here threatens the score's **scale** and, far more,
any heritability estimated from the same families, not its ranking.

Caveats: one pedigree shape (nuclear only, as the constructor requires), PA
inference, supplied rather than fitted `n`; the moment heritabilities are
computed analytically from the true covariance, not by `fit_heritability`, so
they show the estimator's target without finite-sample noise.

## 28. PGS comparison and the joint model (`bench_pgs_comparison.py`)

Five independent replicates of the `bench_gwas_power.py` framework (10,000
probands, 2,000 independent SNPs with 30 causal, h²=0.5, K=0.05, parents plus
one sibling), each split 50/50: the discovery GWAS and PGS weights use only the
5,000 train probands, and everything predictive is scored on the 5,000 held-out
test probands. The two-score combiner is two-fold cross-fitted within the test
half, so no subject is scored by a combiner trained on its own `g`. The PGS is
marginal Z-scored weights (the LDpred-infinitesimal limit with independent
SNPs; the optional `--pgs-backend ldpred3` agreed to within 0.002 correlation
on a shared replicate); the family-history score is classic LT-FH by PA (no
age/sex/cohort structure here; §15 covers the personalised case; PA matches
Gibbs to corr ≥ 0.997, §1). The script
docstring has the full construction. `±` is replicate SE.

**Table 28.1.** GWAS arms on the train cohort (NCP ratios, as in §4).

| Phenotype | mean causal chi² | causal-SNP NCP ratio / c-c | lambda GC |
|---|---:|---:|---:|
| case/control | 22.28 ± 1.11 | 1.000× | 0.999 ± 0.021 |
| LT-FH (PA) | 31.73 ± 1.43 | 1.446 ± 0.023× | 1.018 ± 0.023 |
| oracle true g | 168.38 ± 1.47 | 7.95 ± 0.43× | 1.017 ± 0.018 |

The LT-FH NCP ratio matches §4's 1.468 ± 0.040× (the ratio is sample-size
independent in expectation).

**Table 28.2.** Prediction arms on the test cohort, against held-out true `g`
(squared-correlation R², not comparable with NCP ratios).

| Score | corr with held-out g | R² |
|---|---:|---:|
| case/control label | 0.343 ± 0.007 | 0.118 ± 0.005 |
| PGS | 0.485 ± 0.009 | 0.236 ± 0.009 |
| LT-FH (PA) | 0.413 ± 0.003 | 0.170 ± 0.003 |
| PGS + LT-FH joint (cross-fitted OLS) | — | 0.338 ± 0.009 |

The PGS beats the raw label (paired ΔR² = **+0.1176 ± 0.0199**), LT-FH beats
case/control on correlation (paired Δcorr = +0.0694 ± 0.0104), and the joint
model beats either score alone: the incremental R² of the PGS over LT-FH is
**+0.1677 ± 0.0182** and of LT-FH over the PGS is **+0.1025 ± 0.0075** (paired
95% CI half-widths; the LT-FH-vs-case/control NCP-ratio increment is
+0.4464 ± 0.0638).

The scores are weakly correlated and complementary, as the measurement model in
`docs/algorithm.md` predicts: with `p = h²_SNP / h²_total = 1` by construction,
`Corr(PGS, FH) = a·b·√p = a·b` (`a = Corr(PGS, g)`, `b = Corr(FH, g)` on the
test cohort). Observed corr(PGS, LT-FH) is **0.2009 ± 0.0046** against a theory
value of 0.2003 ± 0.0050; the paired theory-minus-observed difference is
**-0.0006 ± 0.0160** (95% CI). Conditional independence holds by construction
(PGS error is train-cohort noise, LT-FH error is posterior uncertainty, the
cohorts do not overlap), so this verifies the identity, not its failure modes
(ancestry, assortment, selection).

Caveats: independent SNPs only (no LD); the joint regression targets the true
`g`, an oracle evaluation rather than an observed outcome; at 2,000 SNPs the
multiple-testing dilution is mild, so the joint model's *increment* over the
PGS is the portable number, not the absolute R².

## 29. Ascertainment: what the fitters do on selected samples (`bench_ascertainment.py`)

The other fitter benchmarks draw unascertained population families, the only
design `fit_heritability` / `fit_variance_components` /
`fit_genetic_correlation` support; this section measures what happens when that
contract is violated. The user-facing account of the failure and of
`sampling="ipw"` is in [docs/inference.md](../docs/inference.md#ascertained-samples);
this section keeps the numbers.

Each replicate draws a **population** under a known model, applies one selection
rule and fits the selected subset; model, analysed N, thresholds and fitter
settings are held fixed. Five replicates, N = 3,000 analysed families, K = 0.05,
true h² = 0.5, n_iter = 800 (the default; `--full` gives the 10-replicate,
N = 10,000, n_iter = 1500 grid). `random_50` keeps half the population
*independently of phenotype* through the identical accept/reject, redraw and
truncation path: the negative control that isolates selection-on-phenotype from
the harness.

**Table 29.1.** Fitted h² by selection scheme.

| scheme | realised case share | fitted h² (nuclear) | fitted h² (sibship) |
|---|---:|---:|---:|
| population | 0.051 | 0.515 (SD 0.147) | 0.566 (SD 0.056) |
| random_50 *(negative control)* | 0.050 | 0.602 (SD 0.139) | 0.543 (SD 0.049) |
| proband_case | 1.000 | **1.000** | **1.000** |
| case_control | 0.495 | **1.000** | **1.000** |
| enriched_20 | 0.198 | **1.000** | **1.000** |
| family_history | 0.291 | **1.000** | **1.000** |
| fh_proband_control | 0.000 | **1.000** | **1.000** |

Every phenotype-selected scheme is pinned at the clamp in every replicate
(boundary fraction 1.00, across-replicate SD 0.000); the population and
negative-control arms are not. Case shares are the nuclear arm's; the sibship
arm realises the same except `family_history` (0.219).

**The strongest cell is true h² = 0** (5 replicates, nuclear; not in the
default grid: `--h2 0 --tag _h2null --arms h2 mechanism --structures nuclear --n-fam 10000 --n-iter 1500 --burn-in 500`,
`bench_ascertainment_h2null.csv`, reproduced 2026-09-14 to 1.5e-14; the script
now writes four columns the artifact predates): population returns 0.018 and
`random_50` 0.025, while `proband_case`, `case_control`, `enriched_20`,
`family_history` and `fh_proband_control` **all return 1.000**.

**Table 29.2.** A+C (`fit_variance_components`, sibship, true A = 0.4,
C = 0.2). The multi-component fit renormalises onto the simplex instead of
pinning a component at 1-eps, so the diagnostic is the exhausted residual (the
CSV's `residual` column).

| scheme | A | C | residual | saturated |
|---|---:|---:|---:|---:|
| population | 0.428 | 0.203 | 0.369 | 0.00 |
| random_50 | 0.427 | 0.209 | 0.364 | 0.00 |
| proband_case | 0.500 | 0.500 | **0.0001** | 1.00 |
| case_control | 0.498 | 0.502 | **0.0001** | 1.00 |
| enriched_20 | 0.500 | 0.499 | **0.0001** | 1.00 |
| family_history | 0.594 | 0.406 | **0.0001** | 1.00 |
| fh_proband_control | 0.832 | 0.168 | **0.0001** | 1.00 |

**Genetic correlation** (ascertained on trait 1, true r_g = 0.5): population
stays unpinned at 0.381 (SD 0.033) with per-trait h² 0.504/0.518;
`proband_case` and `case_control` return r_g = 1.000 with **both** per-trait h²
at 1.000. Since r_g = G/sqrt(h²₁h²₂), that 1.000 is a ratio of two pinned
denominators, not an estimate (the `h2pin` column).

**Bias, not noise, not non-convergence.** Across N = 1,000 / 3,000 / 10,000 the
population bias is -0.077 / -0.066 / +0.013 (SD 0.254 → 0.033) while
`proband_case` holds at **+0.500 / +0.500 / +0.500**. `population` gives
0.477 / 0.450 / 0.468 at n_iter = 250 / 800 / 2000; `proband_case` gives 0.9999
at all three with a trace-tail slope of 0, and reaches 0.9999 from h2_init =
0.05, 0.5 and 0.95 alike (spread 0.0): it *climbs* to the ceiling. `random_50`
recovers h² within this grid's noise (+0.102 nuclear, +0.043 sibship;
across-replicate SD 0.139 / 0.049).

**Table 29.3.** The clamp censors the magnitude (every ascertained cell reports
0.9999); the Haseman-Elston moment on the selected families' own liabilities
is uncensored (nuclear).

| scheme | HE moment | × truth | centered |
|---|---:|---:|---:|
| population | 0.507 | 1.01 | 0.507 |
| random_50 | 0.517 | 1.03 | 0.517 |
| proband_case | 1.688 | **3.38** | 0.202 |
| case_control | 1.048 | 2.10 | 0.725 |
| enriched_20 | 0.685 | 1.37 | 0.652 |
| family_history | 1.094 | 2.19 | -0.047 |
| fh_proband_control | 0.828 | 1.66 | 0.010 |

**This is a decomposition, not the mechanism.** At true h² = 0 every
ascertained scheme still fits 1.000 while this statistic sits at ~0
(`proband_case` +0.006) or negative (`family_history` -0.107, and -0.562 once
centered). The fitter sees truncated-MVN draws conditional on the selected
status pattern, not these liabilities: under proband ascertainment every
augmented proband is redrawn above threshold, relatives are pulled with it, and
the cross-products stay positive whatever the truth (augmentation feedback).

**Table 29.4.** Dose-response (arm H; nuclear, N = 2,000, 2 replicates,
realised case share against an assumed K = 0.05).

| enrichment | 0.95× | 1.51× | 1.96× | 2.92× | 4.10× |
|---|---:|---:|---:|---:|---:|
| fitted h² | 0.420 | 0.984 | 0.998 | 1.000 | 1.000 |
| bias | -0.080 | **+0.484** | **+0.498** | +0.500 | +0.500 |
| across-rep SD | 0.070 | 0.001 | 0.001 | 0.000 | 0.000 |

A 7.6% case rate against an assumed 5.0% already inflates h² by +0.48, and by
2× the estimate is at the clamp; the enriched cells saturate in both replicates,
so read the shape, not any single cell. The unenriched 0.420 is this arm's
small-N noise floor (SD 0.070), not a fitter bias; the N = 10,000 population arm
recovers 0.513.

**Table 29.5.** Correction by `sampling="ipw"` (per-family
`weights = 1/P(sampled)` in both numerator and denominator of the
Haseman-Elston ratio); same grid, 5 replicates, N = 3,000.

| scheme | max weight | unweighted | **IPW** | bias | SD (IPW) |
|---|---:|---:|---:|---:|---:|
| population | 1.0 | 0.481 | 0.481 | −0.019 | 0.088 |
| random_50 | 1.0 | 0.469 | 0.469 | −0.031 | 0.123 |
| case_control | 19.0 | **1.000** | **0.468** | −0.032 | 0.099 |
| enriched_20 | 4.7 | **1.000** | **0.521** | +0.021 | 0.040 |
| proband_case | — | 1.000 | *undefined* | — | — |
| family_history | — | 1.000 | *undefined* | — | — |
| fh_proband_control | — | 1.000 | *undefined* | — | — |

A 50/50 case/control cohort goes from pinned at 1.000 to 0.468 against a truth
of 0.5. **Positivity:** the three *undefined* rows sample no families from some
complete status-pattern stratum; the benchmark identifies that from the known
rule and does not call the weighted fitter (they are examples of
zero-probability designs, not a claim that all family-history selection is
unweightable). **Efficiency:** weights reach 19× at K = 0.05 and the IPW SD
(0.099) exceeds the population arm's (0.088).

A Lee et al. observed→liability factor cannot substitute: it is a function of
(K, P) alone, while at fixed K true h² of 0.5 and 0.0 **both** produce 1.000,
and it transforms an observed-scale estimate while this fitter is already on the
liability scale. Measured directly (arm J: each family role standardised by its
sample case rate, multi-relative HE moments pooled, the proband's sample
fraction in the Lee factor), observed-scale HE + Lee lands at **0.224** (50/50)
and **0.332** (20% enriched) against 0.5, and is itself +0.23 off under clean
population sampling. It is an in-repository diagnostic, not an evaluation of
conventional unrelated-sample LDSC or GREML.

**Table 29.6.** Population screen (`ltpred.fit._assert_population_case_rate`):
under population sampling each role's case count is Binomial(n_families, K),
so a binomial z-test applies per role.

| property | value |
|---|---|
| sensitivity: z for the five phenotype-selected schemes | +67 to +436 on the N = 10,000 grid; +43 to +276 at N = 4,000 |
| specificity (arm I): 6 unascertained cohorts, N ∈ {500 … 10,000}, K ∈ {0.02 … 0.20} | **0 false positives** |
| bar | z ≥ 6 and a ratio outside [1/1.15, 1.15]; a z ≥ 4 bar fired on a legitimate 1,500-family cohort, because `bootstrap_fit` resamples are centred on the cohort's rate rather than on K |
| detectable enrichment `1 + 6·√((1−K)/(K·n))`, K = 0.05 | ~1.67× at N = 1,500; **~1.48× at this section's N = 3,000**; ~1.26× at N = 10,000 |
| same, K = 0.10 | ~1.46× at N = 1,500; ~1.18× at N = 10,000 |

The check catches the catastrophic designs and does **not** certify population
sampling: mild enrichment on a small cohort still passes, and Table 29.4 shows
that is not harmless. With `sampling="ipw"` the test uses the **weighted**
counts and Kish's effective sample size; passing establishes only compatible
role-wise case marginals, not joint family-pattern positivity or weight
correctness.

## 30. Locked comparison to R LTFHPlus and LTFGRS (`bench_ltfhplus_compare.py`)

Same classic LT-FH families and bounds, three independent cohorts of 200
nuclear pedigrees (parents + one sibling), h² = 0.5, K = 0.05, tol = 0.01,
n_sim = 100,000, burn_in = 1,000 — LTFHPlus 2.2.0's own Gibbs settings.
LTFHPlus 2.2.0 is Gibbs-only; public PA is LTFGRS 1.0.1
(`estimate_liability(..., method="PA", useMixture=FALSE)`). Numba used 4
threads, both R packages 1 `future` worker (sequential plan). The CSV records
each replicate's LTFHPlus seed, R/Python/NumPy/Numba versions and resolved
R/Numba/OMP/OpenBLAS thread settings. Wall-clock is the estimator call after
in-process warmup; ms/family is total / 200 (equal nuclear pedigrees, not a
size sweep). Peak RSS is the isolated child via `wait4` (ldpred3's
`_peak_launcher.py`), a fresh process per arm, interpreter and packages
included. `±` is across-replicate SE. Opt-in: needs R and LTFHPlus (the script
exits 2 without them).

**Table 30.1.** Scores, times and peak memory.

| Estimator | corr vs LTFHPlus | RMSE vs LTFHPlus | total s / 200 fam. | ms / family | peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| LTFHPlus Gibbs | — | — | 10.27 ± 0.06 | 51.33 ± 0.28 | 440.7 ± 0.7 |
| LTFGRS PA | 0.9999 ± 0.0000 | 0.0046 ± 0.0003 | 1.802 ± 0.011 | 9.01 ± 0.05 | 259.8 ± 0.7 |
| ltpred Gibbs | 0.9999 ± 0.0000 | 0.0041 ± 0.0002 | 1.513 ± 0.012 | 7.57 ± 0.06 | 180.1 ± 0.1 |
| ltpred PA | 0.9999 ± 0.0000 | 0.0046 ± 0.0003 | 0.00127 ± 0.00003 | 0.00636 ± 0.00017 | 178.7 ± 0.3 |

The Gibbs scores agree at the scale of the Monte Carlo error (LTFHPlus reports
`genetic_se` ≈ 0.004). The PA scores agree much more tightly: corr(ltpred PA,
LTFGRS PA) = 1.0000, RMSE = 0.000087 ± 0.000008 (max abs ≈ 0.0007) — the same
sequential two-moment update, not two approximations. Peak RSS at this n is
mostly runtime: LTFHPlus retains 10⁵ draws; the PA processes differ mainly in R
versus Python heaps.

**Table 30.2.** Fold times: mean ± SE of the three per-replicate ratios (not
the ratio of mean times).

| Comparison | fold | same algorithm? |
|---|---:|:---|
| LTFHPlus Gibbs / ltpred Gibbs | 6.786 ± 0.079× | yes |
| LTFGRS PA / ltpred PA | 1418 ± 30× | yes |
| LTFHPlus Gibbs / ltpred PA | 8086 ± 250× | no |
| LTFGRS PA / ltpred Gibbs | 1.19 ± 0.01× | no |

Row 1 is Gibbs versus Gibbs across language and parallelisation; row 2 the same
sequential PA update in R versus compiled Python; the 8086× row mixes
algorithms. None is a hardware-independent constant.

**Table 30.3.** Four threads versus one (`bench_ltfhplus_compare_1thread.csv`:
same three cohorts, R workers = 1, Numba threads = 1; the 1-thread column is 12
independently seeded replicates, `±` their scatter; per-replicate load not
recorded).

| Comparison | fold at 4 threads | fold at 1 thread |
|---|---:|---:|
| LTFHPlus Gibbs / ltpred Gibbs | 6.786 ± 0.079× | **1.751 ± 0.012×** |
| LTFGRS PA / ltpred PA | 1418 ± 30× | **1425 ± 22×** |

ltpred's Gibbs is `prange`-parallel over families, so most of the 6.79× is the
4:1 resource asymmetry: matched at one thread the gap is 1.75×. PA is
effectively serial, and its fold is unchanged at one thread (1425×) and four
(1418×). The 4-thread rows are what a user gets at each package's defaults
(`future` is sequential by default); the 1-thread rows isolate the
implementation and are more portable, but still machine-specific. Quote either
with its thread count.

## 31. Time and memory: v0.6.1 versus v0.6.0

The [clean-source rerun](results/2026-09-09-time-memory-v061-rerun/README.md)
of 9 September 2026 compares v0.6.1 (`b516271`) with v0.6.0 (`52dec52`): seven
cases, two versions, separate runtime/RSS and allocation processes, all 28
workers complete, sources stable before and after. Python 3.10.20, NumPy 2.2.6,
SciPy 1.15.3, Numba 0.67.0, one Numba thread, arm64 macOS, worker-start load
2.46–3.29. The capsule README has first-call times, workload definitions and
limits.

**Table 31.1. Runtime and peak memory, v0.6.0 / v0.6.1.** Warm time is the
median of five calls after the first call. RSS covers the complete timing
process, including inputs, imports and JIT work. Call allocations are measured
with `tracemalloc` in a separate warmed worker; they exclude inputs/imports
and do not capture every native/JIT workspace allocation. MiB = 2^20 bytes.

| Workload | Warm median (s) | Warm speedup | Peak process RSS (MiB) | Peak call allocations (MiB) |
|---|---:|---:|---:|---:|
| Parent graph, 200,000 records | 0.432 / 0.214 | 2.02× | 290.5 / 255.2 | 109.5 / 68.3 |
| Parent graph, 1,000,000 records | 2.101 / 1.213 | 1.73× | 930.1 / 690.5 | 536.6 / 330.2 |
| PA, mixed pin masks | 1.120 / 0.497 | 2.25× | 491.9 / 337.3 | 179.2 / 50.2 |
| PA, common pin mask | 0.429 / 0.366 | 1.17× | 495.6 / 348.0 | 237.1 / 92.1 |
| PA, intervals only | 0.393 / 0.370 | 1.06× | 347.9 / 243.6 | 77.8 / 12.8 |
| PA, censoring mixture | 0.692 / 0.643 | 1.08× | 447.5 / 354.7 | 132.8 / 42.0 |
| Register scoring, 300 probands | 0.087 / 0.085 | 1.03× | 159.3 / 156.8 | 3.1 / 3.1 |

Every PA mean and variance, register score and diagnostic, and both graph
adjacency digests matched exactly. The small register case changes by about 3%
with an unchanged allocation peak; this does not establish full-register
scaling or a general end-to-end speedup, and no statistical, Gibbs, quadrature
or fitting comparison was made. Repeated measurements on one machine are not
hardware replications or confidence intervals.

### 31a. v0.7.1 versus v0.7.0

The [v0.7.1 capsule](results/2026-09-23-time-memory-v071/README.md) reran the
same seven cases (`5b42b13` against `ac78ba3`) under `ltpred314`, since the
`.venv` that measured v0.6.1 no longer imports SciPy on this macOS; absolute
times are not comparable with Table 31.1. All outputs matched exactly and every
warm median is within 2% (register scoring 0.084 / 0.082 s). The v0.7.1
speedups sit on paths this driver does not measure; the many-relative register
gain is in §21.

## 32. Pairwise composite-likelihood recovery (`bench_pairwise_recovery.py`)

*Commit `b6065ff`, 2026-09-10 (Table 0.1); 806 s. `--jobs` sets only the worker
count: per-replicate seeds make the output identical at any value (verified 1
against 6).*

11,364 independent, unascertained population cohorts of `o, m, f, s1, s2`
families at a common threshold, 10% prevalence, truths `A = 0.30`,
`C = 0.15`, `M = 0.10`, totalling 43,974,000 simulated families. Replicates
are allocated proportional to `1/N`, so SE(bias) is 0.0009, 0.0007 and 0.0009
for A, C and M at *every* cohort size.

**Table 32.1.** Bias, SE calibration, coverage and boundary pinning by N.

| N | replicates | bias A | bias C | bias M | SE/SD A | coverage A | pinned |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 5,866 | -0.0032 | -0.0005 | -0.0019 | 1.01 | 0.948 | 10.3% |
| 3,000 | 2,933 | -0.0014 | -0.0009 | -0.0024 | 1.00 | 0.952 | 2.9% |
| 6,000 | 1,466 | -0.0005 | -0.0009 | -0.0007 | 1.00 | 0.953 | 0.4% |
| 12,000 | 733 | +0.0002 | +0.0005 | +0.0008 | 0.96 | 0.944 | 0.0% |
| 24,000 | 366 | -0.0009 | +0.0007 | +0.0002 | 0.97 | 0.940 | 0.0% |

- **Consistent, with a small negative `1/N` finite-sample bias.** Weighted
  least squares on `bias(N) = b0 + b1/N` puts `b0` at +0.00005 for A (95% CI
  -0.00113 to +0.00123), +0.00020 for C (-0.00066 to +0.00107) and +0.00024
  for M (-0.00099 to +0.00147), all within 0.0002 of zero; `b1` is -4.67,
  -1.57 and -3.96, and the fit is adequate (chi2/df 0.42 to 1.21, p 0.30 to
  0.74). The largest deviation, A at N = 1,500, is -0.0032 (about 1% of the
  parameter) and decays.
- **The negative sign is predicted.** For one component and one
  parent-offspring pair per family at threshold zero the estimator is
  `2 sin(pi (p_hat - 1/2))`, concave for `A > 0`, so Jensen pulls the mean low;
  that case's exact binomial distribution gives `N x bias -> -0.3667` at
  `A = 0.30` (0.975 to 1.000 of the delta-method constant over N = 50 to
  5,000). The fitted `b1 = -4.67` has the same sign and is about thirteen times
  larger, as the harder design (ten dependent pairs, three components, 10%
  case rate) should make it.
- **Sandwich SEs are calibrated and intervals cover.** Mean reported SE over
  across-replicate SD is 0.96–1.10 across all components and sizes; coverage of
  the nominal 95% normal interval 0.940–0.975. No fit failed or was discarded.
- **Boundary pinning is confined to the component nearest zero.** `M` (truth
  0.10, SD 0.065 at N = 1,500) is pinned at zero in 10.3% of replicates there,
  then 2.9%, 0.4% and nil by N = 12,000; `A` never. SEs are unavailable at the
  boundary by design, so those replicates are counted in the last column and
  excluded from coverage.

Scope: one design under `sampling="population"` with a known threshold; other
prevalences, relationship structures and IPW weighting are not covered, so the
0.6.0 caveat that efficiency and interval coverage need broader validation is
narrowed, not retired.

## 33. Joint pairwise h² / genetic / residual correlation inference

The [combined evidence note](results/2026-09-16-joint-pairwise/README.md) holds
the tables, manifests and every attempted fit of the 16 September 2026
development checkout. Across 1,800 fits (200 independent cohorts in each of
nine scenarios) none failed. In the correctly specified scenarios (signed and
null correlations, multivariate sibship/couple components, 20% MCAR phenotypes,
known positive IPW sampling) absolute heritability bias was below 0.004 and
residual-correlation interior 95% coverage 0.920–0.981; 20/200
shared-environment and 44/200 missing-data fits reached a covariance boundary
and withheld normal SEs. Omitting real shared components inflated heritability
by approximately 0.11 and the generating-model residual correlation by 0.18;
the correctly specified A+C+M fit reduced the residual-correlation bias to
-0.011 (MC SE 0.010). Scope: common per-trait thresholds, prevalence 0.10/0.20,
independent nuclear families; not boundary intervals, personalised bounds,
overlapping pedigrees or informative missingness.

## Historical report changes

The dated record of reruns, corrections and re-derivations of this ledger is in
[`CHANGELOG.md`](../CHANGELOG.md) and `git log -- benchmarks/RESULTS.md`;
Table 0.1 gives the provenance of the values now shown.

## Remaining limitations

- Three- or four-replicate panels give useful SEs but estimate tail uncertainty
  coarsely; the PA stress grid (§14) is 3-seed, so finer seed-level detail
  there is descriptive. The integrated LT-FH++ main panel uses ten replicates
  and its sex isolation five; tail calibration remains noisier than paired
  score contrasts.
- Component-test Type-I error, bootstrap coverage and MCEM SEs (§17) are
  calibrated only at R = 25 resolution ("no gross miscalibration"), and §17 and
  §18 have no archived artifact.
- The LTFHPlus / LTFGRS lock (§30) is classic no-mixture bounds only, 200 equal
  nuclear families, LTFHPlus 2.2.0 and LTFGRS 1.0.1: no PA-FGRS mixture,
  personalised CIP or pedigree-size sweep.
- The HAPNEST real-LD path was not run for any artifact: it needs an external
  multi-GB dataset and stays opt-in ([`hapnest/README.md`](hapnest/README.md)).
- The lightweight GWAS helper uses the large-sample `n * r²` score statistic.
  A finite-sample test would use residual degrees of freedom and the
  `r² / (1-r²)` correction; the matched NCP ratios are robust to this, but
  genome-wide discovery counts are only illustrative.
- Most scripts write canonical artifacts unconditionally (the
  integrated-personalization script accepts `--output-prefix`; custom-path
  outputs must be preserved separately), and artifacts older than the
  2026-09-03 manifest are incompletely attributable.
