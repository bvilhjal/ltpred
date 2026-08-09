# ltpred benchmark results

Historical results from the 28 local benchmark scripts in this directory
(`bench_aod_decay_robustness.py` is reported as a subsection of section 25).
Bounds
distinguish non-personalised, personalised pinned and interval-case encodings;
family-history inclusion distinguishes LT-FH++ (with relatives) from ADuLT
(index person only). Gibbs and Pearson–Aitken (PA) are alternative inference
engines for those bounds. PA-FGRS is a separate PA-specific specification; its
censoring mixture is not included in the PA–Gibbs comparisons below.

- **Snapshot:** assembled from focused runs on 2026-07-31 and 2026-08-01, each
  recorded in `run_manifest.jsonl`; this was not one atomic rerun of all 28
  scripts. The first 14-section campaign was generated on 2026-07-14.
- **Recorded environment for the stored artifacts:** Python 3.14.6
  (free-threading), NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, 10 logical cores
  (Apple M2 Pro, arm64), with the full machine-readable environment, command,
  and Git state recorded per run in `run_manifest.jsonl`.
- **Provenance limit:** these checked-in values are historical artifacts. They
  do not automatically validate later source changes, including a dirty working
  tree. Claims should be refreshed after numerical or benchmark-source changes.
- **Future runs:** use
  `python benchmarks/run_benchmark.py SCRIPT -- [ARGS]`; it appends the exact
  command, environment, exit status, Git commit, tracked-diff hash, content
  hashes for untracked source files, captured stdout/stderr log hashes, and
  hashes of changed top-level `bench_*.{csv,png}` artifacts to
  `run_manifest.jsonl`. Custom output paths are not discovered automatically.
  Set `NUMBA_NUM_THREADS` and, where needed, `OMP_NUM_THREADS` explicitly.
- **External-data exception:** the HAPNEST real-LD path was not run because it
  requires an external multi-GB dataset; it remains opt-in in
  [`hapnest/README.md`](hapnest/README.md).

Unless stated otherwise, `±` denotes the standard error of a mean across
independent simulated cohorts. Tables labelled SD instead report the empirical
across-cohort standard deviation; tables labelled 95% CI report a half-width.
Some diagnostic grids remain single-seed illustrations; those are identified
rather than dressed up as certainty. The saved sex-limitation and nurture
tables used normal `1.96 × SE` half-widths; their source scripts now use
small-sample t half-widths and must be rerun before those interval fields are
treated as current.

All core fitter benchmarks use unascertained, population-sampled simulated
families. They do not validate heritability or variance-component fitting under
case/control or family-history ascertainment.

Metric names matter here. **NCP ratio** means a ratio of causal-SNP chi-square
noncentrality components. **Eff-N proxy** means a squared-correlation ratio to
case/control. The latter is useful for prediction comparisons but is not an
observed GWAS noncentrality ratio, so the two magnitudes are not interchangeable.

## Headline findings

- **PA is the right default for the tested single-trait, no-mixture work.** Across the 27-cell accuracy
  grid (five seeds per cell), mean corr(PA, Gibbs) is 0.9977–0.9999. The
  stressful-pedigree benchmark remains
  at least 0.9984. PA and Gibbs also give indistinguishable downstream GWAS
  results. In the 4-thread timing run, the PA object path is **203–492× faster**
  than grouped Gibbs across the tested sizes and pedigrees. The ratio is not
  thread-count-free: Gibbs is the parallel engine while the PA object path is
  largely serial, so fewer threads inflate the speed-up. Quote it with its
  thread count or not at all.
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

Five independent simulated cohorts (seeds 1–5) per cell, 1,500 families
each. Values are across-seed means ± SE (sd/sqrt(5)); per-cell settings are
unchanged from the former single-seed grid. At h²=0.5 and K=0.05:

| Family structure | corr Gibbs | corr PA | PA squared-correlation eff-N proxy / case-control | corr(PA, Gibbs) |
|---|---:|---:|---:|---:|
| parents | 0.382 ± 0.009 | 0.382 ± 0.009 | 1.36 ± 0.03× | 0.9997 |
| parents + 2 siblings | 0.437 ± 0.009 | 0.437 ± 0.009 | 1.66 ± 0.07× | 0.9997 |
| extended | 0.419 ± 0.014 | 0.420 ± 0.014 | 1.71 ± 0.09× | 0.9997 |

Across all 27 cells, the mean PA–Gibbs agreement is 0.9977–0.9999 with
across-seed SEs of at most 0.0001 (bare means shown). Absolute accuracy
increases with prevalence, heritability, and informative relatives.
Relative gain over a case/control label is often largest for rarer
disease; the largest grid mean is 2.53 ± 0.52× (extended, h²=0.2, K=0.01).
The extreme low-prevalence cells are themselves noisy — the former
single-seed maximum, 2.67× at parents + siblings, h²=0.2, K=0.05,
replicates as 2.11 ± 0.15× — so treat their ordering as descriptive.

## 2. Controlled runtime scaling (`bench_scaling.py`)

Five warmed timings per point, reported as medians; Python 3.14.6, NumPy 2.4.6,
SciPy 1.18.0, Numba 0.66.0, **4 Numba threads**, on an Apple M2 Pro (10 cores,
arm64), h²=0.5, K=0.05, and 25,000 Gibbs draws. No other benchmark ran
concurrently, but the machine was not otherwise idle: system processes held
1-minute load average near 7–11 throughout. `benchmarks/run_manifest.jsonl`
records the machine, resolved thread count and load for every run, so a timing
taken under load is identifiable rather than silently slow.

These supersede an earlier 10-thread reference (315–510×) taken on different
hardware and a different stack, which does not reproduce on this machine at any
thread count. Four threads is the operating point this project now baselines on.

### Scaling with number of families (parents + one sibling)

| families | Gibbs families/s | PA object families/s | PA array families/s | object speed-up |
|---:|---:|---:|---:|---:|
| 500 | 296 | 71,264 | 1.93 M | 241× |
| 1,000 | 295 | 73,863 | 2.65 M | 250× |
| 2,000 | 300 | 73,319 | 3.55 M | 244× |
| 4,000 | 299 | 78,229 | 3.79 M | 261× |
| 8,000 | 305 | 71,878 | 3.76 M | 236× |

The object path includes `Family`/`Member` bounds assembly and grouping. The
array path receives already aligned, repeatedly reused arrays; its 1.2–3.8
million families/s is therefore a hot-kernel measurement, not end-to-end input
preparation. Its very short calls also make cache and scheduler effects visible,
so use the CSV IQRs rather than interpreting the non-monotone point rates.

### Family size at 2,000 families

| relatives | structure | Gibbs time | PA object time | PA array time | object speed-up |
|---:|---|---:|---:|---:|---:|
| 2 | parents | 5.52 s | 0.0272 s | 0.00055 s | 203× |
| 3 | + sibling | 6.78 s | 0.0278 s | 0.00071 s | 244× |
| 5 | + two grandparents | 10.68 s | 0.0296 s | 0.00089 s | 361× |
| 7 | extended | 12.61 s | 0.0348 s | 0.00114 s | 362× |
| 10 | extended + aunts | 17.16 s | 0.0349 s | 0.00169 s | 492× |

The small PA times are not strictly monotone; five repeats quantify timing
variation but do not abolish operating-system noise, and this run carried
background load. The defensible claim is the observed **203–492×** object-path
speed-up *at four threads on this machine* — not a universal hardware-independent
constant, and not transferable to another thread count, since raising the thread
count speeds Gibbs up far more than the largely serial PA object path.

This is the only benchmark used for performance claims. It warms all paths,
records five timings per point with median/IQR, uses `perf_counter`, records the
Numba thread count and simulation configuration, and measures PA's object and
array APIs separately.

## 3. Age-of-onset information (`bench_age_onset.py`)

Three independent cohorts per cell, 3,000 families, eight
relatives. Gibbs is a first-replicate cross-check only (`gibbs_reps=1`).

| h² | K | classic LT-FH corr (PA) | FH + onset corr (PA) | first-rep FH + onset Gibbs | squared-correlation eff-N proxy: onset / classic |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.4275 ± 0.0100 | 0.4297 ± 0.0101 | 0.4232 | 1.0102 ± 0.0016× |
| 0.5 | 0.30 | 0.6342 ± 0.0036 | 0.6604 ± 0.0028 | 0.6646 | 1.0844 ± 0.0081× |
| 0.8 | 0.30 | 0.7434 ± 0.0050 | 0.7754 ± 0.0046 | 0.7834 | 1.0879 ± 0.0018× |

Both main columns use PA and condition on the same family; Gibbs is only the
first-replicate agreement check. This is an LT-FH++ age-component ablation over
classic LT-FH, not ADuLT and not raw case/control. The mean squared-correlation
eff-N proxy over the eight-cell grid is 1.037×. Onset information matters
most when enough relatives are observed as cases; at low prevalence its
increment is small. Minimum first-replicate PA/Gibbs agreement is 0.998964.

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

Twenty-five independent, unascertained population cohorts per recovery cell,
K=0.10, 3,000 families with parents + two siblings. Fitter RNG seeds are
distinct across cohorts.

| true h² | fitted mean | bias | empirical SD | reported MC SE | SD / MC SE |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.201 | +0.001 | 0.037 | 0.0020 | 19× |
| 0.4 | 0.387 | -0.013 | 0.055 | 0.0022 | 25× |
| 0.6 | 0.577 | -0.023 | 0.063 | 0.0023 | 28× |
| 0.8 | 0.786 | -0.014 | 0.061 | 0.0026 | 24× |

At 3,000 families, empirical SD is 0.107 with parents only, 0.052 with parents +
two siblings, and 0.051 with the extended structure. The 4,000/8,000-family SDs
are 0.038/0.040; their chi-square SD intervals overlap substantially, so the tiny
uptick is replicate noise, not evidence against 1/sqrt(N) scaling.

The returned `h2_se` is within-run Monte Carlo error. It is not a sampling
standard error. Use `bootstrap_fit` for family-resampling intervals.

## 6. A+C variance components (`bench_variance_components.py`)

Twenty-five independent, unascertained population cohorts of 3,000 families per
recovery cell, with distinct fitter seeds. Values are fitted means (empirical
across-cohort SD):

| true A | true C | fitted A (SD) | fitted C (SD) |
|---:|---:|---:|---:|
| 0.4 | 0.2 | 0.399 (0.057) | 0.190 (0.034) |
| 0.5 | 0.1 | 0.492 (0.053) | 0.101 (0.032) |
| 0.3 | 0.3 | 0.297 (0.062) | 0.294 (0.038) |
| 0.6 | 0.0 | 0.583 (0.057) | 0.013 (0.012) |

At true A=0.4 and C=0.2, increasing N from 1,000 to 8,000 families reduces
empirical SD from 0.072 to 0.029 for A and from 0.067 to 0.018 for C. The small
positive biases at N=1,000 shrink toward zero with N.

The null panel reports a constrained boundary mean and SD. Formal component
testing belongs to `test_variance_component`; a point estimate near zero is not a
test rejection rate.

## 7. Genetic correlation (`bench_genetic_correlation.py`)

Twenty-five independent 3,000-family cohorts per recovery cell; the simulation
includes phenotypic correlation 0.2 even when genetic correlation is zero.

| true r_g | fitted mean | bias | empirical SD |
|---:|---:|---:|---:|
| 0.0 | -0.0061 | -0.0061 | 0.0798 |
| 0.3 | 0.2894 | -0.0106 | 0.0825 |
| 0.6 | 0.5976 | -0.0024 | 0.0706 |

At true r_g=0.5, the across-cohort SD is 0.148, 0.108, 0.078, and 0.042
for N=1,000, 2,000, 4,000, and 8,000 families (15 cohorts per point). Bias at
8,000 is +0.0068. The null cell shows that non-genetic phenotypic correlation is
not spuriously recovered as genetic correlation on average.

## 8. Shared environment and prediction (`bench_shared_env.py`)

Four independent 3,000-family cohorts per cell, h²=0.5, parents + three full
sibs. The first two columns are prediction accuracy `corr(estimate, true g)`
under the two fitted models; the paired gain is fitted A+C minus fitted
additive-only prediction; uncertainty shown here is a t-based 95% CI half-width.
The last two columns are the fitted parameters behind those predictions.

| true c² | accuracy, ignore C | accuracy, fit A+C | paired gain ± 95% CI | fitted h², ignore C | fitted h², fit A+C |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.53251 | 0.53238 | -0.00013 ± 0.00041 | 0.502 | 0.492 |
| 0.1 | 0.49899 | 0.49996 | +0.00096 ± 0.00061 | 0.550 | 0.468 |
| 0.2 | 0.48869 | 0.49226 | +0.00357 ± 0.00144 | 0.643 | 0.469 |
| 0.3 | 0.47059 | 0.47649 | +0.00590 ± 0.00231 | 0.748 | 0.478 |

At c²=0.3, the paired gains with 2, 4, and 6 full siblings are respectively
+0.00266 ± 0.00342, +0.00636 ± 0.00574, and +0.00829 ± 0.00658 (95% CI
half-widths). More relatives help identify C, but four replicates are too few to
claim monotone gain with sibship size.

The benchmark retains per-family Gibbs MCSEs and warns when estimates miss the
requested tolerance, and the stored rows carry that instrumentation: across the
four fitted models the per-model MCSE maxima span 0.0093–0.0200, with zero
nonfinite and zero unconverged estimates in every cell. The canonical run took
542 s (`run_manifest.jsonl`), not the ~30 minutes an earlier transcription of
this paragraph claimed.

The predictive increment is intentionally evaluated as a paired difference on
the same cohorts. Its main practical value is smaller than the parameter-
interpretation benefit: ignoring C causes sib resemblance to leak into fitted h².

## 9. Couple/spousal environment M (`bench_couple_env.py`)

Twenty-five independent cohorts per cell, 3,000 extended families.

| true m² | fitted A (true 0.4) | A SD | fitted M | M SD |
|---:|---:|---:|---:|---:|
| 0.0 | 0.394 | 0.032 | 0.014 | 0.011 |
| 0.1 | 0.392 | 0.036 | 0.102 | 0.035 |
| 0.2 | 0.399 | 0.032 | 0.197 | 0.038 |
| 0.3 | 0.387 | 0.029 | 0.286 | 0.022 |

Omission comparison, using the same shared variance as sibship C or couple M:

| shared variance | bias in A if C omitted (SD) | bias in A if M omitted (SD) |
|---:|---:|---:|
| 0.1 | +0.047 (0.031) | -0.005 (0.031) |
| 0.2 | +0.101 (0.034) | +0.025 (0.040) |
| 0.3 | +0.156 (0.037) | +0.034 (0.035) |

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
| 0.0215 (population) | 0.232 | 0.346 | 0.350 | 2.233 ± 0.127× | 1.023 ± 0.003× |
| 0.10 | 0.471 | 0.525 | 0.531 | 1.242 ± 0.018× | 1.022 ± 0.002× |
| 0.25 | 0.621 | 0.654 | 0.663 | 1.107 ± 0.008× | 1.030 ± 0.002× |
| 0.50 | 0.694 | 0.722 | 0.738 | 1.081 ± 0.003× | 1.045 ± 0.002× |

Both family-history scores use PA. The first is classic LT-FH; the second changes
the bounds to add age and cohort information, not the inference engine. Thus the
population-sampling **2.233 ± 0.127×** is
`(corr(classic LT-FH) / corr(case/control))²`, a squared-correlation eff-N
proxy, not the causal-SNP NCP ratio reported in section 4. Family history is most
valuable relative to case/control in population sampling; the incremental
age/cohort benefit grows under ascertainment.

### Cohort effects

At a 3× lifetime-prevalence trend per 30 birth years, cohort-aware and single-K
pedigree scores have similar narrow-cohort ranking (0.7419 versus 0.7399), but
their mean scores differ by -0.07335 ± 0.00025. Truth-referenced mean errors are
+0.0041 ± 0.0057 for cohort-aware and -0.0693 ± 0.0057 for single-K.

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
0.809/0.695/0.596/0.522/0.411, with SD 0.052–0.088.

| truth and fitted model | SRMR mean | SRMR SD |
|---|---:|---:|
| true 1F, fit 1F | 0.042 | 0.011 |
| true 2F, fit 1F | 0.145 | 0.033 |
| true 2F, fit 2F | 0.016 | 0.011 |

This demonstrates that SRMR diagnoses this planted misspecification. It is not a
calibrated factor-number test; extra factors improve in-sample fit by construction.
Bootstrap the whole correlation-to-factor pipeline for uncertainty.

## 12. Score calibration (`bench_calibration.py`)

Five independent 3,000-family cohorts per cell (seeds 1–5); values are
across-seed means ± SE. Correctly specified PA:

| structure | K | slope | corr | top-decile realised/predicted |
|---|---:|---:|---:|---:|
| parents + siblings | 0.01 | 1.004 ± 0.024 | 0.259 ± 0.003 | 0.982 ± 0.059 |
| parents + siblings | 0.05 | 1.006 ± 0.008 | 0.431 ± 0.004 | 1.003 ± 0.030 |
| parents + siblings | 0.20 | 1.006 ± 0.005 | 0.593 ± 0.003 | 1.001 ± 0.007 |
| extended | 0.01 | 0.989 ± 0.027 | 0.249 ± 0.004 | 0.946 ± 0.018 |
| extended | 0.05 | 1.002 ± 0.019 | 0.429 ± 0.008 | 0.991 ± 0.023 |
| extended | 0.20 | 1.002 ± 0.007 | 0.595 ± 0.006 | 1.007 ± 0.004 |

Every slope mean is within 1.3 SE of 1 and every PA intercept within 0.01
of 0; Gibbs and PA slope means agree within 0.01 cell for cell. The former
single-seed outlier (extended pedigree, K=0.01, slope 0.916) replicates as
0.989 ± 0.027 — mostly cohort noise. One departure does replicate: at
K=0.01 with the extended pedigree the top-decile realised/predicted ratio
is 0.946 ± 0.018, about 3 SE below 1, so the top of that score is
over-stated by roughly 5%. Calibration is good in every tested cell, but
the rare-disease top decile still blocks a universal calibration claim.

Misspecified h², with true h²=0.5 and K=0.05:

| assumed h² | slope | corr | top realised/predicted | calibration RMSE |
|---|---:|---:|---:|---:|
| 0.2 | 2.240 ± 0.015 | 0.429 ± 0.004 | 2.256 ± 0.065 | 0.165 ± 0.004 |
| 0.4 | 1.216 ± 0.009 | 0.431 ± 0.004 | 1.215 ± 0.036 | 0.066 ± 0.004 |
| 0.5 | 1.006 ± 0.008 | 0.431 ± 0.004 | 1.003 ± 0.030 | 0.040 ± 0.006 |
| 0.6 | 0.864 ± 0.007 | 0.430 ± 0.004 | 0.862 ± 0.025 | 0.058 ± 0.006 |
| 0.8 | 0.678 ± 0.006 | 0.429 ± 0.004 | 0.678 ± 0.020 | 0.135 ± 0.006 |

Ranking barely changes — corr spans 0.429–0.431 with SE ≤ 0.004 — while
the slope sweeps from 2.240 ± 0.015 down to 0.678 ± 0.006: the
ranking-robust, scale-fragile contrast is now replicated, not single-seed.
A realised/predicted ratio above 1 means the score under-predicted the
realised top decile; it does not mean the score overstated it.

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
| large pedigree | 0.99972 | 0.4483 | 0.4491 |
| rare, K=0.005 | 0.99843 | 0.2300 | 0.2308 |
| densely affected | 0.99907 | 0.4893 | 0.4902 |

Fold-order spread as a percentage of the between-proband score SD:

| pedigree | median | p95 |
|---|---:|---:|
| trio | 0.084% | 1.96% |
| parents + 2 siblings | 0.114% | 2.24% |
| extended | 0.090% | 2.39% |
| large | 0.068% | 3.33% |

The typical order effect is tiny. The p95 values are small and rise with
pedigree size in this grid, while the medians are not monotone; with one seed
per cell, either pattern is descriptive rather than established. This script
does not compare fold-order spread with Gibbs Monte Carlo noise and does not
directly measure rank changes, so it makes neither claim.

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

### Verdict: behaves as intended here; small censoring effect; case encoding dominates

- **The mixture behaves as intended in this design**: it moves calibration
  toward 1 for the under-conditioned lifetime-case encoding and is near-exact
  under its native stochastic-onset model. Mean correlations differ by at most
  0.0005, but the report does not retain uncertainty for those paired
  differences, so this does not establish non-inferiority or “no cost”.
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

## 17. Inference-machinery calibration (`bench_inference_calibration.py`)

The inferential layer (as opposed to point-estimate bias) validated over R = 25
independent A-only datasets (400 families, proband + parents + sib, prevalence
0.1, true h2 = 0.5, C = 0; n_boot = 50, reduced fit iterations):

- **Type-I of `test_variance_component("C")`:** 0/25 rejections at 0.05
  (Clopper–Pearson two-sided 95% CI 0.00–0.14); p-values mean 0.428, median 0.392, min
  0.059 -- no anti-conservatism; if anything mildly conservative at this
  resolution (a uniform null would give mean 0.5, min ~0.04).
- **`bootstrap_fit` interval coverage:** 24/25 = 96% coverage of the true
  h2 = 0.5 at nominal 95% (Clopper–Pearson 95% CI 0.796–0.999). The bootstrap SE is
  mildly conservative: mean 0.168 vs across-dataset SD 0.132 (ratio 1.27),
  consistent with the slight over-coverage -- and, per the ROADMAP, the
  internal `h2_se` understates that sampling SD by >20x, so the bootstrap
  remains the right route.
- **MCEM OPG SE:** mean reported SE 0.127 vs across-dataset SD 0.136 (ratio
  0.93); point estimate mean A = 0.502 (truth 0.5). The approximate
  information SE is close to the sampling SD at this design, slightly narrow.

Resolution note: R = 25 bounds what these can resolve (a true 10% Type-I rate
would have ~7% chance of showing 0/25). The read is "no gross
miscalibration", not proof of exact calibration.

## 18. Model-misspecification stress (`bench_misspecification.py`)

Every other benchmark simulates under exactly the assumed model. This one
violates one assumption at a time (8,000 families, 5 replicates, PA estimator,
classic case/control bounds, h2 = 0.5, true prevalence 0.10):

| arm | corr | slope |
|---|---|---|
| control | 0.4983 | 1.0002 |
| heavy-tail env (t_5) | 0.4820 | 0.9898 |
| assortative mating (phenotypic rho ~ 0.3) | 0.4915 | 0.9741 |
| sibship shared env (c2 = 0.15, unmodeled) | 0.4934 | 0.9776 |
| wrong prevalence 0.05 (true 0.10) | 0.4982 | 0.8886 |
| wrong prevalence 0.20 (true 0.10) | 0.4980 | 1.1328 |

### Verdict: structurally robust; prevalence is the calibration hazard

The score is robust to structural misspecification at these strengths: heavy
tails, generative assortative mating, and an unmodeled sibship environment
each move the calibration slope by only 0.01-0.03 (direction as theory says --
familial resemblance over-credited to genes) and cost almost no ranking
(<= 0.007 corr). The dominant calibration risk is a wrong prevalence/
threshold model (slope 0.89-1.13 for a factor-2 error) -- exactly what the
personalised CIPs of LT-FH++ exist to remove, and the reason threshold
provenance matters more than model refinement here.

## 19. CIP estimation from follow-up records (`bench_cip_estimation.py`)

`ltpred.cip` estimates the cumulative-incidence curve that `thresholds_from_cip`
consumes, from registry-style follow-up records (entry age, exit age, event
code), with left truncation and competing risks. Simulated registry cohort
(N = 50,000; known logistic CIP, lifetime prevalence 0.10; Gompertz mortality;
administrative censoring; one arm with a 1995 register start):

- **No mortality:** Kaplan-Meier recovers the true curve to max abs error
  0.0013; all 13 prespecified age-grid truths fall inside the pointwise ±2SE
  bands. This is single-dataset grid containment, not repeated-sample coverage.
- **With mortality (42% death share):** Aalen-Johansen recovers its estimand
  (the crude cumulative incidence in the presence of death) to 0.0017;
  Kaplan-Meier treating death as censoring overestimates it by up to 0.0218
  (the classic competing-risks bias) while remaining consistent for its own,
  different estimand (the marginal no-death-world curve, 0.0019). Estimand
  choice is the user's call; for LT-FH++ thresholds the crude curve is the
  right one (the dead cannot be diagnosed).
- **Delayed entry (register starts 1995):** AJ still accurate to 0.0019.
- **End-to-end:** the estimated curve -> `thresholds_from_cip` ->
  `estimate_liability` on a family cohort gives calibration slope 0.9969 vs
  1.0015 with the oracle curve, with identical correlation (0.3861). The
  scores were nearly identical in this run; repeated-run uncertainty was not
  retained.

This matches the LT-FH++ construction (Pedersen et al. 2022: Aalen-Johansen
with death and emigration as competing events, sex x birth-year strata).
The saved curve point estimates remain informative, but this historical run
predates the corrected finite-risk-set Aalen variance and therefore does not
validate the current SE implementation; rerun it for uncertainty claims. A
person-level bootstrap remains an option.

## 20. Pedigree inference from trio records (`bench_pedigree_inference.py`)

`ltpred.pedigree` discovers a proband's relatives from population
parent-offspring records (the Pedersen et al. 2025 graph-extraction niche,
with full-sibling edges giving the standard relationship-degree scale:
parents/siblings degree 1, grandparents/half-sibs/aunts degree 2, first
cousins degree 3). On a simulated 3-generation population (2,683 persons,
remarriages and cousin links):

- **Exactness:** with a full ancestral closure (every recorded ancestor of the
  extracted set included), the extracted sub-pedigree's kinship equals the
  full-population kinship restricted to the members -- max abs diff **0.0**
  over 300 probands. (Without the closure, boundary members who are actually
  siblings were split into unrelated founders, diff 0.5; the closure is what
  makes the extracted pedigree safe to estimate from. The degree limit
  truncates only *which relatives* are included, never the kinship among
  them.)
- **Payoff (the LT-FGRS point):** estimating genetic liability on 300
  probands, corr(est, truth) is 0.532 using all relatives up to third degree
  vs 0.452 with the fixed named-role subset the grammar encodes (parents, full
  siblings, and grandparents)
  (+17.6%); the two scores correlate 0.88 -- which relatives you include
  matters, consistent with the LT-FGRS package (Pedersen et al.).
- **Scale:** 3,000 extractions at degree 3 in 0.18–0.32 s over the two stored
  runs (~0.1 ms per proband); per-proband neighborhoods stay small (tens of
  nodes), so per-proband extraction plus a small dense kinship covariance is
  the right architecture.

Kinship here is the exact tabular method (inbreeding-aware), not the 2025
paper's path-counting approximation.

## 21. End-to-end register pipeline (`bench_register_pipeline.py`)

`research.pipeline.estimate_liabilities` chains trio records -> pedigree
discovery -> per-stratum CIP thresholds -> per-proband scores. On a synthetic
register (3-generation population of 2,683 with remarriages; one CONSISTENT
liability field `G ~ N(0, h2 A)`, `L = G + E` -- an earlier per-pedigree draw
silently decorrelates probands' g from relatives' statuses; the crossing
model with a logistic CIP, lifetime prevalence 0.10):

- **Accuracy / which relatives matter:** corr(est, true g) is 0.373 at
  degree 3 (first cousins) vs 0.364 at degree 1 (first-degree only), with
  calibration slopes near 1 (0.96 / 1.03). The degree-3 advantage is modest
  at this ~3% effective case rate -- the LT-FGRS effect, in the realistic
  direction. (corr ~0.37 is itself the honest accuracy at this case rate and
  age structure; the pedigree benchmark's 0.53 used a 10% rate with a uniform
  threshold.)
- **CIP estimated from the register itself:** 0.3728 vs 0.3733 with the
  oracle curve. This run detected no loss from estimating the CIP from
  follow-up records at this register size; it does not establish zero cost
  without repeated-run uncertainty (consistent with section 19).
- **Prospective prediction** (diagnosis after index age 40): the honest
  familywise-censored score reaches corr 0.169 with the future outcome;
  adding relatives' post-index events gives 0.148, and leaking the proband's
  own future outcome inflates to 0.399. A single run retains no replicate
  uncertainty, so the relatives'-events contrast does not establish that
  those events help or harm at this sample size; the proband's-own-outcome
  leakage is unambiguous and large -- honest censoring costs real accuracy,
  and leaking the proband's own future buys plenty.
- **Throughput:** 266–359 probands/s over the two stored runs (400 in
  1.1–1.5 s), per-proband extraction plus a small dense kinship covariance
  each.

## 22. Tetrachoric correlations (`bench_tetrachoric.py`)

`ltpred.tetrachoric` estimates the latent liability correlation from 2x2
case/control tables by maximum likelihood (thresholds from the marginals,
bounded scalar optimisation over rho, SciPy's bivariate-normal CDF with
requested absolute and relative tolerances of `1e-10`, and SEs from the
observed information). The requested integration tolerance is not a proven
error bound. On families simulated under the liability-threshold
model (h2 = 0.5, prevalence 0.1, 5 replicates of 20,000 families):

| pair | expected h2*A | tetrachoric | latent corr |
|---|---|---|---|
| o-m | 0.250 | 0.258 +/- 0.008 | 0.251 |
| o-f | 0.250 | 0.240 +/- 0.009 | 0.252 |
| o-s1 | 0.250 | 0.247 +/- 0.010 | 0.248 |
| m-s1 | 0.250 | 0.267 +/- 0.006 | 0.253 |
| o-mgm | 0.125 | 0.115 +/- 0.005 | 0.127 |
| o-mau1 | 0.125 | 0.127 +/- 0.007 | 0.128 |
| m-f (mates) | 0.000 | 0.016 +/- 0.005 | 0.000 |

The pairwise means broadly track h2 * A and the latent Pearson correlations,
but several cells differ from the nominal target by more than one reported SE
(notably mother-sibling and the mate pair). Five replicates are too few to turn
that pattern into a calibrated equivalence claim. The checked-in SEs also used
the population-SD convention; the source now uses sample SD (`ddof=1`) and
requires a rerun for corrected numerical SEs. The Falconer heritability estimate
h2 ~ 2 x tetrachoric(first-degree) gives 0.497 +/- 0.014 (truth 0.5) from
binary relative pairs alone, agreeing with `fit_heritability` on the same
families (0.515 +/- 0.028). `tetrachoric_matrix` produces the expected h2*A
block for multi-variable status matrices.

Use it as a fast diagnostic and cross-check of the family model: liability
correlations straight from relative-pair statuses, before any fitting.

## 23. Liability-scale transformations (`bench_liability_scale.py`)

`ltpred.liability_scale` implements the Lee et al. (2011) observed/liability
heritability bridge and probit estimation of residual-scale genetic variance
`q` (the probit model IS the liability-threshold model:
per-SNP `q` is the identity 2 f (1-f) beta²; the z-statistic route is Lee &
Wray 2013 with the master factor). The default `q` is not a total-liability
fraction; aggregate it before applying `q / (1 + q)`.

**Historical-default warning:** the table below was generated with raw `z²`
second moments, equivalent to current `subtract_null=False`. The source
benchmark now uses the null-adjusted default `(z² - 1) / N`; these saved numbers
must not be presented as validation of that default. Point estimates below are
retained only as a record of the prior run.

On a polygenic disease simulated on
the probit convention (liab = X beta + eps, Var(X beta) = 0.5, prevalence
0.1):

| route | estimate | target |
|---|---|---|
| (a) joint probit fit -> sum 2f(1-f)beta² | 0.510 +/- 0.014 | 0.500 (exact) |
| (a') marginal probit fits (GWAS practice) | 0.352 +/- 0.006 | ~1/(1+V_bg) attenuated |
| (b) historical raw probit z², Lee & Wray 2013 factor | 0.335 +/- 0.020 | prior `subtract_null=False` |
| (c) OLS observed-scale total | 0.119 +/- 0.009 | (observed scale) |
| (c) Lee-2011 bridged to liability | 0.349 +/- 0.026 | 1/3 (Lee fraction) |

For the historical routes, the joint probit fit recovers the
probit residual-scale total (0.5) exactly; marginal per-SNP fits attenuate by
the polygenic background in their residuals (~1/(1+V_bg), material at h²=0.5);
and the OLS/Lee-2011 route recovers the *fraction-of-total-liability* form
(1/3), which is the McKelvey-Zavoina R² of Lee et al. 2012 (Genet Epidemiol,
eq. 9) -- implemented as `probit_liability_r2(..., fraction=True)`.

Paper-fixture checks in the tests cover the Lee 2011 Discussion factors
(observed 0.18/0.54/0.91 -> liability 0.1/0.3/0.5 at K=0.01, P=0.5) and the
Crohn's Table 3 fixture (0.61 -> 0.22).

## 24. Environment components in estimation (`bench_env_components.py`)

The sibship (C) and couple (M) shared-environment components fitted by
`fit_variance_components` are now wired into liability estimation:
`construct_covmat_single(..., c2=..., m2=...)` and the supported estimator
front doors (`estimate_liability`, `estimate_liability_pa_arrays`, and
`estimate_liability_gibbs_arrays`) accept them. The genetic target stays
coupled to relatives only through h2 * A (it shares no environment); the
components only change how relatives are conditioned. On families simulated
with true h2 = 0.4, sibship c2 = 0.15 and couple m2 = 0.15 (5 replicates of
4,000 families):

| arm | corr(g) | slope(g) | corr(o) |
|---|---|---|---|
| additive-only (misspecified) | 0.454 | 0.927 | 0.284 |
| oracle-wired (c2/m2 at truth) | 0.455 | 0.986 | 0.297 |
| fitted-wired (fit then wire) | 0.455 | 0.983 | 0.296 |

- **Calibration restored**: the additive-only model over-credits environmental
  clustering to genetics (slope 0.93, over-dispersed); wiring recalibrates to
  0.99 -- the algorithm doc's "sharper genetic estimate" promise, measured.
- **Full-liability prediction sharpens**: corr(E[l_o | family], truth) rises
  0.284 -> 0.297 when the environment is modelled.
- **The fit->wire loop works end to end**: fitted components (A 0.45, C 0.13,
  M 0.17 vs truth 0.4/0.15/0.15) give the same calibration as the oracle.
- Ranking is untouched (corr(g) flat), consistent with every other benchmark.

## 25. Onset-age-structured genetic correlation (`bench_aod_decay.py`)

`fit_genetic_correlation_decay` fits a genetic correlation that **decays with the
onset-age difference** between relatives,
`Cov(g_i^p, g_j^q) = A_ij sqrt(h2_p h2_q) rho_g K(|a_ip - a_jq|; lam)`, by a
Monte-Carlo EM (a likelihood M-step -- a Haseman-Elston moment step is confounded
by age-dependent ascertainment truncation; see the algorithm notes). Two-trait
nuclear families (h2 = 0.5/0.5, r_p = 0.3, prevalence 0.2, onset ages iid
U(15, 65) -- a pessimistic choice that maximises the amplitude-decay ridge),
fitted with `n_em = 45`, `n_draw = 100`, 3 replicates per cell:

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
  `lam ~ 0.001` -- no manufactured decay, exactly the scalar-model limit.
- **No spurious correlation at the `r_g = 0` null** (`0.009 +/- 0.089`).
- **The model is data-hungry**: at `n_fam = 1000` both `r_g` and `lam` run high
  (0.58 / 0.063) along the amplitude-decay ridge, converging to the truth as `n`
  grows into the thousands. Below that, estimates are ridge-dominated; prefer the
  scalar `fit_genetic_correlation`.
- *Estimator correctness*: the analytic cross-trait gradient of the M-step is
  pinned against finite differences
  (`research/tests/test_decay.py::GradientTests`);
  a symmetrisation bug in it was found and fixed in review, and the numbers
  above are from the corrected estimator.
- **Sampling variability is large**: even at `n_fam = 2500` the across-replicate
  SD of `r_g` is ~0.11-0.18, so any single estimate carries a wide interval;
  use `bootstrap_fit` for sampling uncertainty.

### Robustness (adversarial probes, `bench_aod_decay_robustness.py`)

The main grid is *circular* (the simulator draws from exactly the fitted
covariance), so it cannot reveal misspecification. Two adversarial arms probe it
(`n_fam = 2000`, `n_em = 35`, 2 reps):

- **Wrong kernel is benign**: truth = Gaussian decay, fit = OU gives
  `r_g ~ 0.55` (true 0.5) -- the amplitude is robust to the kernel shape.
- **Unmodelled shared family environment biases r_g DOWN**: adding a
  cross-relative environmental correlation (`c2 = 0.10`) the model cannot
  represent drops `r_g` from `0.50` to `~0.33`. The extra same-trait familial
  covariance is attributed to genetics, inflating `h2`, and since
  `r_g = G / sqrt(h2_0 h2_1)` the inflated denominator attenuates `r_g`. The
  model has **no shared-family environmental component across relatives**, so
  traits with household effects will be mis-estimated -- the most important
  caveat for application.

## 26. Sex in the covariance vs sex in the thresholds (`bench_sex_limitation.py`)

`construct_covmat_sex_limited` lets sex enter `Sigma` (sex-specific `h2`, a
cross-sex genetic correlation `rg`) rather than only the thresholds. The algebra
guarantees the BLUP weights change; it does not guarantee the score improves.
This benchmark measures it against a known truth.

Families (proband + both parents + one sibling) are drawn from the **true**
sex-limited covariance, so the proband's genetic liability is known exactly.
Proband and sibling sexes are balanced across the four cells. `K_female = 0.05`,
`K_male = 0.10`; 3 replicates of 2,000 families. Four arms, all scored by
`corr(estimate, true g)` and the calibration slope `regress(true on estimate)`:
the stored `±` values are the original normal 95% half-widths. The script now
uses t half-widths; rerun it before quoting its intervals as current.

| arm | thresholds | covariance |
|---|---|---|
| pooled | one pooled `K` | scalar pooled `h2` |
| sex thresholds | sex-specific `K` | scalar pooled `h2` |
| sex in Sigma (true) | sex-specific `K` | true `(h2_F, h2_M, rg)` |
| sex in Sigma (rg=1) | sex-specific `K` | true `h2` pair, `rg` wrongly 1 |

**(a) Sweeping the true `rg`** at `h2_F = 0.6`, `h2_M = 0.2`. `eff-N` is the
squared-correlation effective-N proxy for the covariance fix, thresholds held
fixed:

| true rg | sex thresholds | sex in Sigma | gain | eff-N proxy |
|---:|---:|---:|---:|---:|
| 1.0 | 0.3735 | 0.4149 | +0.0414 ± 0.0213 | 1.235x ± 0.126 |
| 0.8 | 0.3604 | 0.4023 | +0.0419 ± 0.0289 | 1.249x ± 0.199 |
| 0.6 | 0.3550 | 0.4013 | +0.0462 ± 0.0237 | 1.278x ± 0.151 |
| 0.4 | 0.3348 | 0.3937 | +0.0589 ± 0.0106 | 1.384x ± 0.064 |
| 0.2 | 0.3306 | 0.3921 | +0.0614 ± 0.0030 | 1.407x ± 0.046 |

**(b) Sweeping the heritability gap** at `rg = 1`, mean `h2 = 0.4`:

| gap | sex thresholds | sex in Sigma | gain | eff-N proxy |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4001 | 0.4001 | +0.0000 ± 0.0000 | 1.000x ± 0.000 |
| 0.2 | 0.3837 | 0.3948 | +0.0111 ± 0.0026 | 1.059x ± 0.023 |
| 0.4 | 0.3537 | 0.3942 | +0.0405 ± 0.0144 | 1.241x ± 0.070 |
| 0.6 | 0.3395 | 0.4335 | +0.0940 ± 0.0272 | 1.639x ± 0.334 |

At gap 0 with `rg = 1` the sex-limited covariance **is** the scalar covariance,
so the arms are bit-identical and the gain is exactly zero -- the benchmark's
own sanity check.

The gain is real but conditional, and modest until the sex difference is large:
`1.06x` at a heritability gap of 0.2, `1.24x` at 0.4, `1.64x` at 0.6.
**A sex-limited model with no sex difference to find buys nothing**, which is
the case a practitioner should expect by default. Mis-specifying `rg` as 1 when
it is truly 0.2 costs about a quarter of the gain: worth getting approximately
right, not catastrophic to get wrong.

**(c) Mechanism.** `rg` discounts cross-sex pairs and nothing else, so its
effect must vanish in an all-same-sex family and concentrate in an all-cross-sex
one. Two matched compositions -- proband + mother + sister, versus proband +
father + brother -- give the proband exactly two first-degree relatives each,
so they are matched on relatedness and differ *only* in sex configuration.
Heritability is equal for both sexes here (`h2 = 0.5`), so scalar limitation
cannot contribute and every bit of any gain is attributable to `rg`:

| true rg | composition | sex thresholds | sex in Sigma | gain |
|---:|---|---:|---:|---:|
| 1.0 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 0.6 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 0.2 | same-sex | 0.4105 | 0.4105 | +0.0000 ± 0.0000 |
| 1.0 | cross-sex | 0.4271 | 0.4271 | +0.0000 ± 0.0000 |
| 0.6 | cross-sex | 0.3740 | 0.3871 | +0.0131 ± 0.0096 |
| 0.2 | cross-sex | 0.3035 | 0.3538 | +0.0502 ± 0.0342 |

The effect appears in exactly the cells the model predicts and nowhere else:
identically zero across every same-sex row regardless of `rg`, zero in the
cross-sex family when `rg = 1`, and monotone in the remaining two. This is a
mechanistic confirmation rather than an association -- the parameter acts on
the pairs it is defined to act on. It also shows where the model is worth
reaching for: pedigrees rich in opposite-sex relatives, not same-sex ones.

**Calibration is the cleaner signal.** Only the sex-limited covariance is
calibrated; the slope sits at `0.945-1.019` across every cell. The
threshold-only arm degrades as the gap grows (`0.9609` at gap 0 down to
`0.8390` at gap 0.6), and the pooled arm sits between them.

**An unexpected result worth stating plainly.** When the architecture is
genuinely sex-limited but the covariance is left sex-blind, personalising the
*thresholds* by sex made the score **worse** than ignoring sex entirely --
`0.3395` vs `0.3906` in correlation at gap 0.6, and worse on calibration too
(`0.8390` vs `0.9469`). This is not an argument for sex-blind thresholds. It is
consistent with the factorisation: thresholds set `mu`, but the estimate is
`w' mu`, and sharpening `mu` for a subgroup whose weights `w` are wrong need not
improve the product. Two mis-specifications partially cancel in the pooled arm.
The reading is that sex-specific thresholds and a sex-specific covariance are
not substitutes -- under true sex limitation, only fixing `Sigma` fixed both
ranking and calibration. Verified as a real effect rather than noise: at gap 0
the pooled/threshold difference is `-0.0051` over the stored 3 replicates of
2,000 families, within the CI, and it appears only once the gap is nonzero.

Caveats: one pedigree shape, a balanced sex design, PA inference, and supplied
rather than fitted parameters. Nothing here estimates `h2_F`, `h2_M` or `rg`,
so the "sex in Sigma (true)" arm is a ceiling that a real analysis reaches only
as well as its parameter estimates allow.

## 27. Ignoring genetic nurture (`bench_nurture.py`)

`construct_covmat_nurture` separates a proband's **direct** additive value from
the **indirect** path by which the parents' genotypes shape the rearing
environment. This benchmark asks what a nurture-blind analysis costs.

Families (proband + both parents + one full sib) are drawn from the **true**
path model, so the direct value `A_o` is known exactly. `h2 = 0.4` (direct),
`K = 0.05`; 5 replicates of 3,000 families. The stored `±` values are the
original normal 95% half-widths; the script now uses t half-widths and requires
a rerun for current intervals. Four estimators are scored against `A_o`:

| arm | covariance |
|---|---|
| additive, true h2 | ordinary, at the true *direct* `h2` |
| additive, moment h2 | ordinary, at the `h2` a moment fitter reads off parent-offspring pairs |
| A + C matched to sibs | `c2` chosen to reproduce the nurture sib-sib covariance exactly |
| nurture | the true path model -- the ceiling |

**(a) Ranking barely moves.**

| true n | additive (moment h2) | A + C matched | nurture | cost |
|---:|---:|---:|---:|---:|
| 0.0 | 0.3805 | 0.3805 | 0.3805 | +0.0000 ± 0.0000 |
| 0.1 | 0.3952 | 0.3949 | 0.3958 | +0.0006 ± 0.0008 |
| 0.2 | 0.4237 | 0.4221 | 0.4253 | +0.0017 ± 0.0010 |
| 0.3 | 0.4602 | 0.4550 | 0.4645 | +0.0043 ± 0.0025 |

At `n = 0` every arm is identical and the cost is exactly zero -- the sanity
check. Correlation *rises* with `n` in all arms, because a stronger shared
parental contribution makes the relatives more informative about the parental
genetic values, which are themselves correlated with `A_o`. The **relative**
cost of ignoring nurture stays small throughout.

**(b) Calibration and heritability are where it bites.**

| true n | c2 matched | h2 from parent-offspring | h2 from sibs | A+C slope | nurture slope |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | 0.4000 | 0.4000 | 1.0157 | 1.0157 |
| 0.1 | 0.0880 | 0.4800 | 0.5760 | 1.0693 | 1.0000 |
| 0.2 | 0.1920 | 0.5600 | 0.7840 | 1.1458 | 1.0077 |
| 0.3 | 0.3120 | 0.6400 | 1.0000 | 1.1957 | 1.0005 |

The nurture model is calibrated at every setting (slope `1.000-1.016`). The
A+C model -- which reproduces the sibling covariance **exactly** -- drifts to
`1.196`, under-predicting the true direct effect by about a fifth.

The heritability column is the more alarming one. The true *direct* `h2` is
`0.4` throughout, but a nurture-blind moment fitter returns `0.64` from
parent-offspring pairs and `1.000` from sibs at `n = 0.3` -- the latter pinned
at the boundary, an impossible heritability. **The two estimates disagree by up
to 0.36, and that disagreement is the diagnostic**: no single additive `h2` can
satisfy both relative types when an indirect path is present.

**Reading.** In this setting genetic nurture is not primarily a threat to the
score's *ranking*; a nurture-blind score still orders probands almost as well.
It is a threat to the **scale** of the score and, far more seriously, to any
heritability estimated from the same families. An analyst who sees
parent-offspring and sibling heritabilities disagree should suspect an indirect
path rather than average them.

Caveats: one pedigree shape, PA inference, supplied rather than fitted `n`, and
the moment heritabilities are computed analytically from the true covariance
rather than by running `fit_heritability`, so they show what a moment estimator
targets without its finite-sample noise. Nuclear families only, as the
constructor requires.

## Historical report changes

- Replicated the accuracy and calibration grids across five independent
  seeds (every cell now reports an across-seed mean and SE), replacing the
  single-seed diagnostic values.
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
- Re-derived every quantitative table from the stored 2026-07-31/08-01 artifacts
  (the 19 committed `bench_*.csv` files and the `run_logs/` run logs) on
  2026-08-09, after an independent review (`docs/EXTERNAL_REVIEW.md`, 2026-08-07)
  found doc/artifact transcription skew in roughly a third of the tables.
  Qualitative text changed in §7 (genetic-correlation null-bias sign), §8
  (removed the false claim that the stored rows predate MCSE instrumentation --
  the CSV carries per-model MCSE-max/nonfinite/unconverged columns), §14
  (fold-order p95 ordering), §20 (pedigree-payoff magnitude), §21 (the
  relatives'-events leakage contrast: stored direction is a decrease, so the
  contrast is now unresolved at this n rather than asserted), and §26
  (replicate-count correction, 5 to 3, with the affected cells refreshed). No
  qualitative conclusion changed except §21's leakage-contrast direction.

## Remaining limitations

- The accuracy and calibration grids (sections 1 and 12) are now replicated
  across five independent seeds; every cell reports an across-seed mean and
  SE. The PA stress grid (section 14) remains single-seed — treat small
  differences there as descriptive.
- Three- or four-replicate panels give useful SEs but still estimate tail
  uncertainty coarsely. The integrated main panel now uses ten replicates and
  its sex isolation uses five; tail calibration remains noisier than paired
  score contrasts.
- Component test Type-I error, bootstrap coverage, and MCEM SEs were
  calibrated at coarse resolution (section 17; R = 25 bounds the
  resolution -- read as 'no gross miscalibration').
- The HAPNEST path was not executed here. The PA-FGRS censoring-mixture
  benchmark now exists (section 16); its censoring correction is small at
  the tested settings, and case encoding dominates calibration there.
- The lightweight GWAS helper uses the large-sample `n * r²` score statistic.
  A finite-sample regression test would use residual degrees of freedom and the
  `r² / (1-r²)` correction; the matched NCP ratios are robust to this
  small approximation, but genome-wide discovery counts are only illustrative.
- Most benchmark scripts still write canonical artifacts unconditionally. The
  integrated-personalization script accepts `--output-prefix`, and the common
  `run_benchmark.py` wrapper now records source/environment provenance and
  hashes changed top-level `bench_*.{csv,png}` artifacts. Custom-path outputs
  must be preserved and hashed separately. Historical artifacts predating that
  wrapper remain incompletely attributable.
