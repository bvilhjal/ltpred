# ltpred benchmark results

The [9 September 2026 time/memory rerun (§31)](#31-time-and-memory-v061-versus-v060)
compares clean v0.6.1 with v0.6.0 on seven matched workloads. It records warm
speedups, first-call time, memory and exact output agreement; the measured
source remains v0.6.1 in this v0.6.2 documentation update.

Sections 1–30 contain historical results from the local benchmark scripts (26
scripts since the 2026-08 consolidation; the retired standalone scripts are
reported as subsections or merged panels of their targets -- see the notes in
sections 3, 11, 24, 25, 26 and 27). Bounds
distinguish non-personalised, personalised pinned and interval-case encodings;
family-history inclusion distinguishes LT-FH++ (with relatives) from ADuLT
(index person only). Gibbs and Pearson–Aitken (PA) are alternative inference
engines for those bounds. PA-FGRS is a separate PA-specific specification; its
censoring mixture is not included in the PA–Gibbs comparisons below.

- **Snapshot:** the stored artifacts were regenerated in place by a full-suite
  rerun of all 26 scripts on 2026-08-20 at v0.4.0 (every script completed
  cleanly), in the environment recorded below, with Numba at 8 threads except
  `bench_scaling.py` and `bench_ltfhplus_compare.py` at 4. The report was
  previously assembled from focused runs on 2026-07-31, 2026-08-01 and
  2026-08-14/15, each recorded at the time in a run manifest (removed in the
  2026-08 lean-down); the first 14-section campaign was generated on
  2026-07-14, and the 2026-08-14/15 reruns used a different interpreter
  (Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3). Sections whose numbers the
  2026-08-20 rerun reproduced within sampling error were left as recorded;
  the per-section notes and the change log at the end mark the exceptions.
  The R-package lock and PGS-comparison artifacts were subsequently refreshed
  for v0.4.1; the historical provenance limit below applies to those runs too.
- **Recorded environment for the stored artifacts:** Python 3.14.6
  (free-threading), NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, 10 logical cores
  (Apple M2 Pro, arm64).
- **Provenance limit:** these checked-in values are historical artifacts. They
  do not automatically validate later source changes, including a dirty working
  tree. Claims should be refreshed after numerical or benchmark-source changes.
  The per-run provenance manifest and captured logs were removed in the
  2026-08 lean-down, so the exact source state of older runs is unrecoverable.
- **Verification, 2026-09-14:** the suite was re-run at v0.6.2 in the recorded
  environment to check that the stored artifacts still describe current
  behaviour. `bench_tetrachoric`, `bench_liability_scale`, `bench_calibration`
  and `bench_pa_robustness` reproduced their committed CSVs byte for byte.
  `bench_accuracy` and `bench_pedigree_inference` reproduced every statistic,
  differing only in wall-clock columns. `bench_register_pipeline` moved by
  ~1e-3 per row with every published replicate mean unchanged. Only
  `bench_cip_estimation` moved a published number (section 19), and its
  artifact is regenerated here. Timing-derived artifacts were deliberately not
  refreshed: the one-minute load average was 13-28 throughout, which the
  manifest records, and a fold time or a throughput is only meaningful from an
  otherwise-quiet machine.
- **Future runs:** use
  `python benchmarks/run_benchmark.py --artifact bench_<name>.csv bench_<name>.py -- [ARGS]`.
  Repeat `--artifact` for every retained CSV/PNG.
  The wrapper appends the clean source commit, exact command,
  environment/thread settings, machine profile, exit status, and
  declared-artifact hashes to `run_manifest.jsonl`.
  Commit the corresponding row with any regenerated artifact.
  Declare external data with repeatable `--input FILE` so its content hash
  travels with the result.
  Set `NUMBA_NUM_THREADS` and, where needed, `OMP_NUM_THREADS` explicitly.
- **External-data exception:** the HAPNEST real-LD path was not run because it
  requires an external multi-GB dataset; it remains opt-in in
  [`hapnest/README.md`](hapnest/README.md).

Unless stated otherwise, `±` denotes the standard error of a mean across
independent simulated cohorts. Tables labelled SD instead report the empirical
across-cohort standard deviation; tables labelled 95% CI report a half-width.
Some diagnostic grids remain single-seed illustrations; those are identified
rather than dressed up as certainty. The saved sex-limitation and nurture
tables previously held normal `1.96 × SE` half-widths; the 2026-08-20 rerun
regenerated them with the scripts' small-sample t half-widths.

Core fitter benchmarks (sections 5-9) use unascertained, population-sampled
simulated families and characterise the fitters only under that supported
contract. Section 29 measures the opposite case directly -- what those fitters
return on ascertained samples -- and is the basis for the marginal case-rate
screen applied under `sampling="population"`.

Metric names matter here. **NCP ratio** means a ratio of causal-SNP chi-square
noncentrality components. **Eff-N proxy** means a squared-correlation ratio to
case/control. The latter is useful for prediction comparisons but is not an
observed GWAS noncentrality ratio, so the two magnitudes are not interchangeable.
Every checked-in causal-SNP NCP and detection-power result below used independent
SNPs, non-overlapping simulated families, and the lightweight marginal score
statistic in `_common.py`. It is therefore evidence for that simulation estimand,
not for a real-LD, related-sample mixed-model GWAS. The HAPNEST input path was not
run for these artifacts.

## Headline findings

- **PA is the right default for the tested single-trait, no-mixture work.** Across the 27-cell accuracy
  grid (five seeds per cell), mean corr(PA, Gibbs) is 0.9995–1.0000. The
  stressful-pedigree benchmark remains
  at least 0.9991 (worst single-seed 0.99916). PA and Gibbs also give
  indistinguishable downstream independent-SNP causal-NCP results. In the
  4-thread timing run, the PA object path is **392–518× faster**
  than grouped Gibbs across the tested sizes and pedigrees. The ratio is not
  thread-count-free: Gibbs is the parallel engine while the PA object path is
  largely serial, so fewer threads inflate the speed-up. Quote it with its
  thread count or not at all.
- **ltpred Gibbs matches public LTFHPlus on the same families.** Against
  LTFHPlus 2.2.0, corr = 0.9999 and RMSE = 0.0041 ± 0.0001 (200 nuclear
  families, three cohorts, the R package's own Gibbs settings).
- **Public PA is LTFGRS, not LTFHPlus.** LTFHPlus 2.2.0 is Gibbs-only.
  ltpred PA and LTFGRS 1.0.1 `method="PA"` agree at corr = 1.0000
  (RMSE 0.000087 ± 0.000008). Same-algorithm fold times on this
  machine were **6.786 ± 0.079×** (LTFHPlus Gibbs / ltpred Gibbs) and
  **1418 ± 30×** (LTFGRS PA / ltpred PA). Mixing algorithms, LTFHPlus /
  ltpred PA is **8086 ± 250×**. Totals: 1-worker LTFHPlus took
  10.27 ± 0.06 s (51.33 ± 0.28 ms/family) versus 1.513 ± 0.012 s
  (7.57 ± 0.06 ms/family) for 4-thread ltpred Gibbs, 1.802 ± 0.011 s
  (9.01 ± 0.05 ms/family) for LTFGRS PA, and 0.00127 ± 0.00003 s
  (0.00636 ± 0.00017 ms/family) for ltpred PA. Isolated-process peak
  RSS was 440.7 ± 0.7, 259.8 ± 0.7, 180.1 ± 0.1 and 178.7 ± 0.3 MiB —
  LTFHPlus, LTFGRS PA, ltpred Gibbs and ltpred PA respectively
  (interpreter included). The 8086× figure is not
  "LT-FH++, but faster."
- **Classic LT-FH improves simulated independent-SNP causal NCP without average null inflation.**
  Across three genotype/effect/cohort replicates on **independent SNPs**, the same
  classic LT-FH model inferred by either PA or Gibbs delivers a causal-SNP NCP
  ratio of **1.47 ± 0.04×** over case/control. These NCP ratios are not a
  real-LD result; HAPNEST remains opt-in and unrun.
- **Full personalised LT-FH++ adds independent-SNP NCP beyond matched ADuLT.**
  With identical age/sex/cohort proband bounds, ADuLT reaches **1.049 ± 0.004×**
  adjusted causal-SNP NCP ratio over case/control and full LT-FH++ reaches
  **1.194 ± 0.006×** (same independent-SNP design). The paired LT-FH++ minus ADuLT increment is
  **+0.1454 ± 0.0154** NCP-ratio units (95% CI half-width). Full LT-FH++ has
  adjusted calibration slope **0.995 ± 0.015** and PA/Gibbs agreement 0.99996.
- **Cohort-blind family thresholds inflate stratified-null λ_GC; cohort-aware thresholds do not.**
  At a 4× lifetime-prevalence trend per 30 birth years, single-K family
  thresholds reach **16.460 ± 0.399**; the same family model with cohort-specific
  K stays at **1.007 ± 0.040**. Independent null SNPs stay near 1 for every
  method. Personalised CIP is a confounding-control device, not only a power
  tweak.
- **Sex-specific CIP improves stratum calibration, not proven adjusted
  independent-SNP NCP.**
  In a prespecified sex-only scenario, correct sex curves close the female-minus-
  male mean-score-error gap by **0.05092 ± 0.00096** (paired 95% CI), while the
  adjusted NCP-ratio increment is **0.0040 ± 0.0046** and remains unresolved.
- **A correctly specified posterior mean is self-calibrating; a wrong h² is not.**
  On the classic-LT-FH grid, calibration slope sits on 1 (e.g. 1.006 ± 0.008 at
  K=0.05). Assumed h² from 0.2 to 0.8 sweeps the slope from 2.240 ± 0.015 to
  0.678 ± 0.006 while ranking stays in 0.429–0.431. Use the score as a ranker
  or GWAS phenotype under a roughly right h²; do not treat its scale as
  E[g | family] unless h² is the one it was computed at.
- **A PGS and the family-history score are complementary.**
  On a 50/50 train/test split, test R² against held-out g is
  **0.236 ± 0.009** (PGS), **0.170 ± 0.003** (classic LT-FH) and
  **0.338 ± 0.009** (two-fold cross-fitted OLS on both). Observed corr(PGS,
  LT-FH) is 0.2009 ± 0.0046
  against the `a·b·√p` prediction 0.2003 ± 0.0050 (`p = 1` by construction).
- **The PA-FGRS mixture has a detectable but practically negligible ranking effect; case encoding dominates calibration.**
  Three of ten paired Δcorr 95% CIs exclude zero, but the largest shift is
  only −0.00034 (95% CI ± 0.00011). Under threshold crossing, pinned cases stay near
  slope 1. When onset only *tends* to track liability (ρ = 0.6), pinning
  over-conditions (slope 0.92 under heavy censoring). The lifetime interval
  under-conditions (slope up to 1.20); an age-specific case interval
  over-disperses (slope ≈ 0.83).
- **Phenotype ascertainment pins the heritability fitter at the clamp.**
  At true h² = 0 every phenotype-selected design returns h² = 1.0. Inverse-
  probability weighting recovers a 50/50 case/control cohort from 1.000 to
  **0.468** (truth 0.5; weights up to 19) and a 20%-enriched cohort to
  **0.521** in the current 3,000-family grid. Designs with a zero inclusion
  probability in some stratum cannot
  be reweighted.
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
| parents | 0.382 ± 0.009 | 0.382 ± 0.009 | 1.36 ± 0.03× | 0.9999 |
| parents + 2 siblings | 0.437 ± 0.009 | 0.437 ± 0.009 | 1.66 ± 0.07× | 0.9999 |
| extended | 0.420 ± 0.014 | 0.420 ± 0.014 | 1.71 ± 0.09× | 0.9999 |

Across all 27 cells, the mean PA–Gibbs agreement is 0.9995–1.0000 with
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
arm64), h²=0.5, K=0.05, and 25,000 Gibbs draws. This rerun includes the collapsed-genetic
Gibbs path (untruncated `g` integrated out of the sweep). No other benchmark ran
concurrently, but the machine was not otherwise idle: the recorded 1-minute
load average moved from 2.56 to 5.86 during the run. `benchmarks/run_manifest.jsonl`
used to record the machine, resolved thread count and load for every run, so a
timing taken under load was identifiable rather than silently slow; the
manifest was removed in the 2026-08 lean-down.

These supersede an earlier 10-thread reference (315–510×) taken on different
hardware and a different stack, which does not reproduce on this machine at any
thread count. Four threads is the operating point this project now baselines on.

### Scaling with number of families (parents + one sibling)

| families | Gibbs families/s | PA object families/s | PA array families/s | object speed-up |
|---:|---:|---:|---:|---:|
| 500 | 504 | 207,980 | 2.63 M | 412× |
| 1,000 | 512 | 215,713 | 3.74 M | 421× |
| 2,000 | 503 | 218,190 | 5.05 M | 434× |
| 4,000 | 506 | 220,534 | 5.53 M | 436× |
| 8,000 | 512 | 219,743 | 6.29 M | 429× |

The object path includes `Family`/`Member` bounds assembly and grouping. The
array path receives already aligned, repeatedly reused arrays; its 2.63–6.29
million families/s is therefore a hot-kernel measurement, not end-to-end input
preparation. Its very short calls also make cache and scheduler effects visible,
so use the CSV IQRs rather than interpreting the non-monotone point rates.

### Family size at 2,000 families

| relatives | structure | Gibbs time | PA object time | PA array time | object speed-up |
|---:|---|---:|---:|---:|---:|
| 2 | parents | 2.91 s | 0.00741 s | 0.00034 s | 392× |
| 3 | + sibling | 3.92 s | 0.00919 s | 0.00042 s | 427× |
| 5 | + two grandparents | 5.95 s | 0.0128 s | 0.00058 s | 463× |
| 7 | extended | 8.02 s | 0.0166 s | 0.00073 s | 483× |
| 10 | extended + aunts | 11.24 s | 0.0217 s | 0.00099 s | 518× |

The small PA times are not strictly monotone; five repeats quantify timing
variation but do not abolish operating-system noise, and this run carried
background load. The defensible claim is the observed **392–518×** object-path
speed-up *at four threads on this machine* — not a universal hardware-independent
constant, and not transferable to another thread count, since raising the thread
count speeds Gibbs up far more than the largely serial PA object path.

This is the only benchmark used for performance claims. It warms all paths,
records five timings per point with median/IQR, uses `perf_counter`, records the
Numba thread count and simulation configuration, and measures PA's object and
array APIs separately.

## 3. Age-of-onset information (`bench_fh_prediction.py` panel (e), formerly `bench_age_onset.py`)

*Script merged into `bench_fh_prediction.py` panel (e) in the 2026-08
consolidation; the numbers below are from the v0.4.0 rerun of the merged
panel. The retired script's standalone `bench_age_onset.{csv,png}` are no
longer retained -- no current script could regenerate them, and the panel's
rows live in `bench_fh_prediction.csv`.*

Three independent cohorts per cell, 3,000 families, eight
relatives. Gibbs is a first-replicate cross-check only (`gibbs_reps=1`).
The same simulated families are scored three ways: classic LT-FH (lifetime
case interval), interval (`[T(onset), ∞)`), and pin (`T(onset)` as a
point). Onset is the CIP inverse of true liability.

| h² | K | classic | pin | interval | pin / classic | interval / classic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.431 ± 0.001 | 0.436 ± 0.001 | 0.436 ± 0.001 | 1.024 ± 0.009× | 1.023 ± 0.007× |
| 0.5 | 0.30 | 0.642 ± 0.006 | 0.669 ± 0.005 | 0.665 ± 0.005 | 1.086 ± 0.006× | 1.074 ± 0.004× |
| 0.8 | 0.30 | 0.747 ± 0.001 | 0.779 ± 0.001 | 0.772 ± 0.000 | 1.086 ± 0.006× | 1.069 ± 0.003× |

All three columns use PA and condition on the same family; Gibbs is only the
first-replicate agreement check on the pin. This is an LT-FH++ age-component
ablation over classic LT-FH, not ADuLT and not raw case/control. Mean pin /
classic squared-correlation eff-N proxy over the eight-cell grid is 1.039×;
interval / classic is 1.034×. At low prevalence pin and interval are
indistinguishable. At K = 0.30 the pin adds a further ~0.01–0.02× over the
interval: most of the onset increment is knowing the case is at least that
extreme, not the pin-equals-liability identity. Minimum first-replicate
PA/Gibbs agreement is 0.99971.

## 4. Replicated classic-LT-FH independent-SNP association benchmark (`bench_gwas_power.py`)

Three independent genotype/effect/cohort replicates; each has 10,000 probands,
5,000 independent SNPs, 30 causal SNPs, h²=0.5, K=0.05, and parents plus one
sibling. The NCP ratio uses `(mean causal chi² - 1)`, not raw mean chi².

| Phenotype | mean causal chi² | causal-SNP NCP ratio / c-c | power at 5e-8 | lambda GC |
|---|---:|---:|---:|---:|
| case/control | 39.63 ± 2.19 | 1.00× | 41.1 ± 2.2% | 1.011 ± 0.009 |
| classic LT-FH (Gibbs) | 57.53 ± 1.87 | 1.468 ± 0.040× | 48.9 ± 4.0% | 1.012 ± 0.019 |
| classic LT-FH (PA) | 57.60 ± 1.91 | 1.469 ± 0.039× | 48.9 ± 4.0% | 1.007 ± 0.021 |
| oracle true g | 337.37 ± 2.07 | 8.77 ± 0.57× | 75.6 ± 1.1% | 1.047 ± 0.015 |

The former 1.53× headline was one seed; 1.47 ± 0.04× is the replicated NCP
ratio. Both rows are the same classic LT-FH model with different inference
engines. In real-LD mode, variants with r² >= 0.1 to any causal SNP are excluded
from lambda/QQ calibration by default because causal proxies are associated, not
null. The committed table did not run that mode, and neither mode is a
related-sample mixed-model analysis.

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
| 0.0 | 0.53276 | 0.53264 | -0.00012 ± 0.00040 | 0.502 | 0.492 |
| 0.1 | 0.49920 | 0.50017 | +0.00097 ± 0.00063 | 0.550 | 0.468 |
| 0.2 | 0.48888 | 0.49246 | +0.00358 ± 0.00146 | 0.643 | 0.469 |
| 0.3 | 0.47083 | 0.47671 | +0.00588 ± 0.00236 | 0.748 | 0.478 |

At c²=0.3, the paired gains with 2, 4, and 6 full siblings are respectively
+0.00265 ± 0.00346, +0.00627 ± 0.00564, and +0.00814 ± 0.00616 (95% CI
half-widths). More relatives help identify C, but four replicates are too few to
claim monotone gain with sibship size.

The benchmark retains per-family Gibbs MCSEs and warns when estimates miss the
requested tolerance, and the stored rows carry that instrumentation: across the
four fitted models the per-model MCSE maxima span 0.0048–0.0120, with zero
nonfinite and zero unconverged estimates in every cell. The canonical run took
542 s (recorded in the since-removed run manifest), not the ~30 minutes an
earlier transcription of this paragraph claimed.

The merged script's panel (c) repeats the §24 public-API C/M wiring check:
fitted A/C/M = 0.427/0.142/0.150 against truth 0.4/0.15/0.15, calibration
slope restored from 0.927 to 0.986, and corr(o) ≈ 0.60 under the current
`estimate_liability(out="full")` semantics (§24).

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

## 11. Genetic factor diagnostic (`bench_genetic_correlation.py` panel (c), formerly `bench_genetic_factor.py`)

*Script merged into `bench_genetic_correlation.py` panel (c) in the 2026-08
consolidation; the v0.4.0 rerun of the merged panel reproduces every number
below unchanged. The retired script's standalone `bench_genetic_factor.{csv,png}`
are no longer retained -- no current script could regenerate them, and the
panel's rows live in `bench_genetic_correlation.csv`.*

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
of 0 (Gibbs and PA slope means agreed within 0.01 cell for cell in the
historical run; the panel is now PA-only, with Gibbs-agreement evidence in
§1 and §14). The former
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
| 1 | 0.960 ± 0.012 | 0.960 ± 0.012 | 0.899 ± 0.034 |
| 2 | 0.964 ± 0.054 | 4.539 ± 0.253 | 2.008 ± 0.257 |
| 3 | 0.922 ± 0.057 | 10.275 ± 0.078 | 4.616 ± 0.220 |
| 4 | 1.007 ± 0.040 | 16.460 ± 0.399 | 7.666 ± 0.094 |

This benchmark isolates the cohort component in a family model; it is not the
full age/sex/cohort LT-FH++ design. At R=1 there is no trend-driven inflation;
the methods need not have identical finite-sample lambda values. Under strong
trends, cohort-aware thresholds remove the induced stratified-SNP inflation on
average.

## 14. PA robustness (`bench_pa_robustness.py`)

Three independent seeds per grid cell; values are the across-seed mean ± SE
(per-seed rows in `bench_pa_robustness.csv`):

| regime | corr(PA, Gibbs) | corr(PA, true g) | corr(Gibbs, true g) |
|---|---:|---:|---:|
| baseline | 0.999912 ± 0.000004 | 0.438 ± 0.005 | 0.438 ± 0.005 |
| large pedigree | 0.999879 ± 0.0000003 | 0.465 ± 0.011 | 0.465 ± 0.011 |
| rare, K=0.005 | 0.999504 ± 0.000025 | 0.216 ± 0.007 | 0.216 ± 0.007 |
| densely affected | 0.999189 ± 0.000020 | 0.486 ± 0.002 | 0.487 ± 0.002 |

The worst single-seed PA–Gibbs agreement is 0.99916 (densely affected).

Fold-order spread as a percentage of the between-proband score SD
(across-seed mean ± SE):

| pedigree | median | p95 |
|---|---:|---:|
| trio | 0.081% ± 0.011 | 1.90% ± 0.03 |
| parents + 2 siblings | 0.099% ± 0.010 | 2.25% ± 0.09 |
| extended | 0.083% ± 0.011 | 2.34% ± 0.15 |
| large | 0.069% ± 0.006 | 3.24% ± 0.05 |

The typical order effect is tiny. The p95 values are small and rise with
pedigree size in this grid — the largest pedigree has the highest p95 in all
three seeds — while the medians are not monotone. This script
does not compare fold-order spread with Gibbs Monte Carlo noise and does not
directly measure rank changes, so it makes neither claim.

## 15. Integrated personalised LT-FH++ independent-SNP association benchmark (`bench_ltfhpp_personalization.py`)

### Integrated panel

Ten paired replicates, 4,000 ascertained probands each, 1,200 SNPs (30 causal,
300 sex/cohort-stratified null), age-, sex-, and cohort-dependent CIP, coherent
onset/follow-up, and sex-dependent competing mortality. Accuracy, slope, and
independent-SNP marginal-association values below are adjusted for proband sex
and birth year; `±` is replicate SE. The final column shows the stratified-null
lambda before and after the same standard covariate adjustment.

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
age and cohort gains, but no conditional sex-specific NCP gain in this
independent-SNP design.

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
Δcorr = +0.00097 ± 0.00111 and ΔNCP ratio = +0.0040 ± 0.0046
(paired 95% CIs).
The calibration benefit is decisive because the paired errors are highly
correlated: adding the correct sex curve shifts female error by
-0.03691 ± 0.00040 and male error by +0.01401 ± 0.00064, closing the
female-minus-male error gap by **0.05092 ± 0.00096**.

On the first two 300-family, no-mixture main-panel cross-checks, PA/Gibbs agreement is
0.999964. Gibbs reaches the requested MCSE tolerance for every score (maximum
MCSE 0.0049 at tolerance 0.03); PA-vs-Gibbs normalized RMSE is 0.0132 score SD,
the Gibbs-on-PA slope is 0.990, and the mean difference is -0.0018.

## 16. PA-FGRS censoring-mixture validation (`bench_pafgrs_mixture.py`)

The PA-FGRS age-censored-control mixture previously had unit tests but no
generative benchmark. This one simulates families (proband + parents + sib)
under the liability-threshold model with a logistic CIP (h2 = 0.5, lifetime
prevalence 0.10, mid-point 60), censors honestly (a case is observed only if
its onset precedes the current age), and scores the estimate against the true
genetic liability over 5 replicates of 20,000 families. Two censoring regimes
(mid-life, heavy; old, light) and three observation models: the LT-FH++
threshold-crossing convention; a stochastic-onset model in which onset age
is drawn from the CIP independent of liability (the mixture's native model);
and a liability-dependent copula (ρ = 0.6) in which higher liability
advances onset with residual noise — the assumption-stress arm, because
the mixture treats future cases as a random draw from the above-threshold
tail.
Slope = regress(true on estimate); 1.0 is a calibrated posterior mean. `±` is
the across-replicate SE. Every replicate's corr/slope is retained in
`bench_pafgrs_mixture.csv` (long format: one row per model x regime x rep x
arm, plus the paired-contrast rows quoted below).

Threshold-crossing observation model:

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base (lifetime case) + no-mixture | 0.3281 ± 0.0047 | 1.1992 ± 0.0151 | 0.4693 ± 0.0030 | 1.0293 ± 0.0069 |
| base + mixture | 0.3277 ± 0.0047 | 1.1738 ± 0.0149 | 0.4691 ± 0.0030 | 1.0209 ± 0.0068 |
| interval case + mixture | 0.3334 ± 0.0047 | 0.8444 ± 0.0097 | 0.4766 ± 0.0033 | 0.8263 ± 0.0050 |
| pinned case + no-mixture | 0.3337 ± 0.0048 | 0.9977 ± 0.0109 | 0.4770 ± 0.0034 | 0.9885 ± 0.0057 |
| pinned case + mixture | 0.3337 ± 0.0048 | 0.9766 ± 0.0109 | 0.4769 ± 0.0034 | 0.9811 ± 0.0057 |

(Gibbs cross-check on base + no-mixture: corr(PA, Gibbs) 0.9998 / 0.9999,
matching slopes 1.2192 / 1.0217 -- the MID slope > 1 is the case encoding's
information loss, not a PA artifact. Under the stochastic-onset model
corr(PA, Gibbs) is 0.9998 / 0.9999 with matching slopes 1.0685 / 0.9845.
Under liability-dependent onset, 0.9998 / 0.9999 with slopes 1.1419 / 1.0127.)

Stochastic-onset observation model:

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base + no-mixture | 0.2781 ± 0.0018 | 1.0267 ± 0.0069 | 0.4536 ± 0.0037 | 0.9990 ± 0.0082 |
| base + mixture | 0.2782 ± 0.0018 | 1.0057 ± 0.0067 | 0.4537 ± 0.0037 | 0.9911 ± 0.0082 |

Liability-dependent onset (ρ = 0.6):

| arm | corr MID | slope MID | corr OLD | slope OLD |
|---|---|---|---|---|
| base + no-mixture | 0.3089 ± 0.0020 | 1.1321 ± 0.0103 | 0.4638 ± 0.0040 | 1.0162 ± 0.0083 |
| base + mixture | 0.3088 ± 0.0020 | 1.1088 ± 0.0101 | 0.4638 ± 0.0040 | 1.0081 ± 0.0083 |
| pinned + no-mixture | 0.3082 ± 0.0021 | 0.9199 ± 0.0088 | 0.4642 ± 0.0041 | 0.9631 ± 0.0075 |
| pinned + mixture | 0.3083 ± 0.0021 | 0.9008 ± 0.0086 | 0.4642 ± 0.0042 | 0.9559 ± 0.0075 |

Paired mixture-minus-no-mixture contrasts (per-replicate differences on
identical cohorts; mean ± SE with t-based 95% CI half-width, 5 replicates):

| cell | case encoding | Δcorr | Δslope |
|---|---|---|---|
| crossing MID | lifetime interval | -0.00034 ± 0.00004 (CI ± 0.00011) | -0.02537 ± 0.00032 (CI ± 0.00088) |
| crossing MID | pinned | -0.00002 ± 0.00004 (CI ± 0.00010) | -0.02106 ± 0.00032 (CI ± 0.00089) |
| crossing OLD | lifetime interval | -0.00017 ± 0.00005 (CI ± 0.00013) | -0.00844 ± 0.00013 (CI ± 0.00036) |
| crossing OLD | pinned | -0.00008 ± 0.00004 (CI ± 0.00011) | -0.00743 ± 0.00012 (CI ± 0.00034) |
| stochastic MID | lifetime interval | +0.00012 ± 0.00005 (CI ± 0.00013) | -0.02101 ± 0.00030 (CI ± 0.00083) |
| stochastic OLD | lifetime interval | +0.00008 ± 0.00005 (CI ± 0.00015) | -0.00785 ± 0.00016 (CI ± 0.00046) |
| dependent MID | lifetime interval | -0.00013 ± 0.00003 (CI ± 0.00008) | -0.02336 ± 0.00024 (CI ± 0.00066) |
| dependent MID | pinned | +0.00005 ± 0.00005 (CI ± 0.00014) | -0.01913 ± 0.00030 (CI ± 0.00083) |
| dependent OLD | lifetime interval | -0.00007 ± 0.00004 (CI ± 0.00012) | -0.00812 ± 0.00014 (CI ± 0.00038) |
| dependent OLD | pinned | -0.00006 ± 0.00005 (CI ± 0.00014) | -0.00724 ± 0.00016 (CI ± 0.00044) |

### Verdict: behaves as intended here; detectable but negligible correlation shifts; case encoding dominates; pinning is not a free lunch

- **The correlation shifts are measurable but practically negligible.**
  Three of ten Δcorr CIs exclude zero; the largest shift is
  -0.00034 with a 95% CI half-width of 0.00011, about 0.1% of the
  correlation level. The mixture therefore changes ranking slightly in some
  cells, including when its independence assumption is false.
- **Pinning is calibrated only under threshold crossing.** At ρ = 0.6 the
  pinned slope is 0.92 (MID) / 0.96 (OLD): onset is no longer the CIP
  inverse of liability, so a point mass at T(onset) over-conditions.
  The lifetime interval still under-conditions (MID slope 1.13), sitting
  between the crossing (1.20) and stochastic (1.03) extremes. Use the pin
  when the crossing model is believed; do not treat it as robust to noisy
  or only partly liability-dependent onset.
- **The calibration shifts are real and directionally consistent.** Every
  Δslope CI excludes zero; the mixture always lowers the slope, by 0.007-0.025
  and most strongly under heavy censoring. Where the no-mixture encoding
  under-conditions (lifetime interval, MID: slopes 1.20, 1.03, 1.13 across
  the three onset models) this moves calibration toward 1, including under
  the mixture's native stochastic-onset model (1.027 -> 1.006). Where the
  naive encoding is already calibrated it tilts slightly past (crossing
  pinned MID 0.998 -> 0.977): under threshold crossing the plain age
  truncation is already exact for censored controls, so the mixture has no
  correct work to do, and that small tilt is the price of applying it anyway.
- **Case encoding dominates calibration, but the right encoding depends on
  the onset model.** Under threshold crossing, pinned cases give slope
  0.98-1.00; the lifetime interval loses onset-age information (slope up
  to 1.20); the age-specific interval over-disperses (slope 0.83). Under
  liability-dependent onset the pin is the *wrong* encoding (slope 0.92).
  Guidance: pin cases at their onset threshold only where the crossing
  model is believed; use the lifetime interval when onset ages are
  unreliable or only partly liability-dependent.

## 17. Inference-machinery calibration (`bench_inference_calibration.py`)

The inferential layer (as opposed to point-estimate bias) validated over R = 25
independent datasets per panel (n_boot = 50, reduced fit iterations). The first
three panels use A-only datasets (400 families, proband + parents + sib,
prevalence 0.1, true h2 = 0.5, C = 0):

- **Type-I of `test_variance_component("C")`:** 0/25 rejections at 0.05
  (Clopper–Pearson two-sided 95% CI 0.00–0.14); p-values mean 0.428, median 0.392, min
  0.059 -- no anti-conservatism; if anything mildly conservative at this
  resolution (a uniform null would give mean 0.5, min ~0.04).
- **`bootstrap_fit` interval coverage:** 24/25 = 96% coverage of the true
  h2 = 0.5 at nominal 95% (Clopper–Pearson 95% CI 0.796–0.999). The bootstrap SE is
  mildly conservative: mean 0.168 vs across-dataset SD 0.132 (ratio 1.27),
  consistent with the slight over-coverage. The internal `h2_se` is a
  within-dataset Monte-Carlo diagnostic and substantially understates the
  across-dataset sampling SD (`ltpred/fit.py` `FitResult`; the >5x gap is
  locked by `test_bootstrap_fit_scalar_and_calibration`), so the bootstrap
  remains the right route.
- **MCEM OPG SE:** mean reported SE 0.127 vs across-dataset SD 0.136 (ratio
  0.93); point estimate mean A = 0.502 (truth 0.5). The approximate
  information SE is close to the sampling SD at this design, slightly narrow.
- **Type-I of `test_genetic_correlation`:** under a true r_g = 0 null with a
  nonzero phenotypic correlation (two traits, h2 = (0.5, 0.4), r_p = 0.2,
  prevalence 0.1, parents + 2 sibs — the null cell of section 7's design at
  the same R, n_fam, n_boot, and fit settings): 0/25 rejections at 0.05
  (Clopper–Pearson two-sided 95% CI 0.00–0.14); p-values mean 0.577, median
  0.569, min 0.059 -- no anti-conservatism, and the phenotypic correlation is
  not mistaken for a genetic one.

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
familial resemblance over-credited to genes). The ranking cost is small but not
uniformly negligible: assortative 0.007 and sibship 0.005, but heavy-tail
0.016 (0.4983 +/- 0.0045 vs 0.4820 +/- 0.0024) -- about 3% of the control
correlation, and roughly 3 SE, so it is a real if minor cost rather than none. The dominant calibration risk is a wrong prevalence/
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
  `estimate_liability` on a family cohort gives calibration slope 1.0023 vs
  1.0069 with the oracle curve, with identical correlation (0.3861). The
  scores were nearly identical in this run; repeated-run uncertainty was not
  retained.

This matches the LT-FH++ construction (Pedersen et al. 2022: Aalen-Johansen
with death and emigration as competing events, sex x birth-year strata).
Rerun 2026-09-14 at v0.6.2 in the recorded environment. The four
curve-recovery rows are bit-identical to the 2026-08-14 run; only the two
end-to-end calibration slopes moved (estimated 0.9969 -> 1.0023, oracle
1.0015 -> 1.0069). That is the expected signature of the v0.6.0 change to
condition pins jointly before approximating any remaining intervals: it
shifts the liability scale slightly and leaves the ranking alone, and the
correlation indeed moved only in the seventh decimal. Grid containment
uses the current SE implementation. A person-level bootstrap remains an
option for repeated-sample coverage. Artifacts: `bench_cip_estimation.csv`.

## 20. Pedigree inference from trio records (`bench_pedigree_inference.py`)

> **Evidence status (2026-09-03):** regenerated with the supported observation
> contract -- ancestors added solely for ancestral closure keep uninformative
> bounds, so the payoff arm measures relatives reached within `max_degree`
> only. Run under the provenance wrapper on a clean tree (commit `bad41ad`;
> free-threaded CPython 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0 at 4
> threads, BLAS pinned to 1; manifest `run_manifest.jsonl`). The historical
> pre-closure-masking values are superseded, not retained, since the old arm
> conditioned on out-of-scope diagnoses.

`ltpred.pedigree` discovers a proband's relatives from population
parent-offspring records (the Pedersen et al. 2025 graph-extraction niche,
with full-sibling edges giving the standard relationship-degree scale:
parents/siblings degree 1, grandparents/half-sibs/aunts degree 2, first
cousins degree 3). Exactness and scale use one simulated 3-generation
population (2,683 persons, remarriages and cousin links); the payoff is
replicated over five independent populations of the same simulator (2,529 to
2,719 persons). `±` is the across-replicate SE (sd/sqrt(R)); the paired
contrast carries a t-based 95% CI (section 15 convention). Per-replicate
values are in `bench_pedigree_inference.csv` (long format: rep, metric,
value; rep 0 marks the single-run parts).

- **Exactness:** with a full ancestral closure (every recorded ancestor of the
  extracted set included), the extracted sub-pedigree's kinship equals the
  full-population kinship restricted to the members -- max abs diff **0.0**
  over 300 probands (single run). (Without the closure, boundary members who
  are actually siblings were split into unrelated founders, diff 0.5; the
  closure is what makes the extracted pedigree safe to estimate from. The
  degree limit truncates only *which relatives* are included, never the
  kinship among them.)
- **Payoff (the LT-FGRS point):** estimating genetic liability on 300
  probands per replicate, corr(est, truth) is **0.569 ± 0.013** using all
  relatives up to third degree vs **0.498 ± 0.016** with the fixed named-role
  subset the grammar encodes (parents, full siblings, and grandparents); the
  paired all-minus-named-role contrast is **+0.0706 ± 0.0060**, 95% CI
  [+0.054, +0.087] -- the gain over the role grammar is real and stable
  across populations (ratio of means 1.14; the two scores correlate
  0.8987 ± 0.0062). Which relatives you include matters, consistent with the
  LT-FGRS package (Pedersen et al.).
- **Scale:** 3,000 extractions at degree 3 in 0.19 s (single-run timing,
  ~0.06 ms per proband); per-proband neighborhoods stay small (tens of
  nodes), so per-proband extraction plus a small dense kinship covariance is
  the right architecture.

Kinship here is the exact tabular method (inbreeding-aware), not the 2025
paper's path-counting approximation.

## 21. End-to-end register pipeline (`bench_register_pipeline.py`)

> **Evidence status (2026-09-03):** regenerated with the supported
> `ltpred.pipeline` driver -- explicit `birth_time`, calendar `index_time`
> per proband, and closure-only bounds uninformative by default. Same
> provenance as section 20 (commit `bad41ad`, free-threaded CPython 3.14.6,
> Numba at 4 threads, BLAS pinned to 1; manifest `run_manifest.jsonl`). The
> historical `research.pipeline` numbers (attained-age censoring shared across
> generations, closure diagnoses conditioned on) are superseded, not retained:
> they validated a different observation model. Note the simulator draws fresh
> liabilities per replicate, so cross-stack reruns are different datasets, not
> bit-identical ones; the claims below are replicate-mean claims, not
> seed-locked constants.

`ltpred.pipeline.estimate_liabilities` chains trio records -> pedigree
discovery -> per-stratum CIP thresholds -> per-proband scores. Over five
independent synthetic registers (3-generation populations of 2,529 to 2,719
with remarriages; one CONSISTENT liability field `G ~ N(0, h2 A)`, `L = G + E`
-- an earlier per-pedigree draw silently decorrelates probands' g from
relatives' statuses; the crossing model with a logistic CIP, lifetime
prevalence 0.10). `±` is the across-replicate SE (sd/sqrt(R)); paired
contrasts carry t-based 95% CIs (section 15 convention). Per-replicate
values are in `bench_register_pipeline.csv` (long format: rep, metric,
value; rep 0 marks the single-run throughput part).

- **Accuracy / which relatives matter:** corr(est, true g) is
  **0.574 ± 0.021** at degree 3 (first cousins) vs **0.539 ± 0.015** at
  degree 1 (first-degree only), with calibration slopes 1.08 ± 0.03 and
  1.08 ± 0.03. The paired degree-3-minus-degree-1 contrast,
  **+0.0354 ± 0.0115** with 95% CI [+0.003, +0.068], resolves the
  second/third-degree contribution as a small real gain -- the LT-FGRS
  effect, in the realistic direction. (corr ~0.57 is the accuracy at these
  registers' case rates and age structure under the supported driver's
  pinned-onset LT-FH++ bounds; the pedigree benchmark's 0.569 used a 10% rate
  with a uniform lifetime threshold.)
- **CIP estimated from the register itself:** 0.5728 ± 0.0214 vs
  0.5741 ± 0.0214 with the oracle curve. The paired estimated-minus-oracle
  contrast is -0.0013 ± 0.0007, 95% CI [-0.0031, +0.0005]: any cost of
  estimating the CIP from follow-up records at this register size is at most
  a few thousandths of a correlation point (consistent with section 19).
- **Prospective prediction** (diagnosis in (40, 70] after index age 40;
  observed future-case rate 0.072 ± 0.005): the honest familywise-censored
  score reaches corr 0.187 ± 0.042 with the future outcome and rank
  (Mann-Whitney) AUC **0.680 ± 0.039**, and is calibrated in the large --
  mean model-consistent predicted future-case risk 0.072 ± 0.002 vs the
  observed 0.072 (the prediction integrates the proband's posterior g and
  the residual E against the CIP; the linear calibration slope of future on
  score is 0.19 ± 0.04). Adding relatives' post-index events gives corr
  0.179 ± 0.040 and AUC 0.664 ± 0.039; leaking the proband's own future
  outcome inflates to corr 0.665 ± 0.015 and AUC 0.992 ± 0.001.
  - **The relatives'-events contrast is unresolved at R = 5, on both
    metrics.** Paired (b)-(a): Δcorr **-0.0082 ± 0.0206**, 95% CI
    [-0.066, +0.049]; ΔAUC **-0.0164 ± 0.0200**, 95% CI [-0.072, +0.039].
    The historical run had resolved a ΔAUC gain (+0.065 ± 0.014) for
    post-index relative events; under the supported driver's calendar
    censoring that direction does not reproduce, and neither direction is
    resolved here. Do not quote a relatives'-events payoff from this panel.
  - The proband's-own-outcome leakage is unambiguous and large: paired
    (c)-(a) Δcorr **+0.4778 ± 0.0384**, 95% CI [+0.371, +0.585]; ΔAUC
    +0.312 ± 0.039, 95% CI [+0.204, +0.419]. Honest censoring costs real
    accuracy, and leaking the proband's own future buys plenty.
- **Throughput:** 380 probands/s (400 probands in 1.1 s; single run, not
  replicated, on a 10-core Apple M2 Pro at load ~4-5 with Numba at 4
  threads), per-proband extraction plus a small dense kinship covariance
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
| o-m | 0.250 | 0.258 ± 0.009 | 0.251 |
| o-f | 0.250 | 0.240 ± 0.010 | 0.252 |
| o-s1 | 0.250 | 0.247 ± 0.011 | 0.248 |
| m-s1 | 0.250 | 0.267 ± 0.007 | 0.253 |
| o-mgm | 0.125 | 0.115 ± 0.006 | 0.127 |
| o-mau1 | 0.125 | 0.127 ± 0.008 | 0.128 |
| m-f (mates) | 0.000 | 0.016 ± 0.006 | 0.000 |

The pairwise means broadly track h2 * A and the latent Pearson correlations,
but several cells differ from the nominal target by more than one reported SE
(notably mother-sibling and the mate pair). Five replicates are too few to turn
that pattern into a calibrated equivalence claim. ± is the SE of the mean
across the five replicates (`np.std(..., ddof=1)/sqrt(REPS)`, rerun
2026-08-14) -- not the sample SD. The Falconer
heritability estimate h2 ~ 2 x tetrachoric(first-degree) gives 0.497 ± 0.016
(truth 0.5) from binary relative pairs alone, agreeing with
`fit_heritability` on 4,000-family subsets of those cohorts (0.515 ± 0.031).
The tetrachoric calculation uses all 20,000 families, so this SD comparison
does not establish greater statistical efficiency at equal sample size.
`tetrachoric_matrix` produces the expected h2*A block for multi-variable
status matrices. Artifacts: `bench_tetrachoric.csv`.

Use it as a fast diagnostic and cross-check of the family model: liability
correlations straight from relative-pair statuses, before any fitting.

## 23. Liability-scale transformations (`bench_liability_scale.py`)

`ltpred.liability_scale` implements the Lee et al. (2011) observed/liability
heritability bridge and probit estimation of residual-scale genetic variance
`q` (the probit model IS the liability-threshold model:
per-SNP `q` is the identity 2 f (1-f) beta²; the z-statistic route is Lee &
Wray 2013 with the master factor). The default `q` is not a total-liability
fraction; aggregate it before applying `q / (1 + q)`.

Rerun 2026-08-14 under the current null-adjusted default `(z² - 1) / N`.

On a polygenic disease simulated on
the probit convention (liab = X beta + eps, Var(X beta) = 0.5, prevalence
0.1):

| route | estimate | target |
|---|---|---|
| (a) joint probit fit -> sum 2f(1-f)beta² | 0.510 ± 0.017 | 0.500 (exact) |
| (a') marginal probit fits (GWAS practice) | 0.352 ± 0.007 | ~1/(1+V_bg) attenuated |
| (b) null-adjusted probit z², Lee & Wray 2013 factor | 0.320 ± 0.024 | same as (a'), attenuated |
| (c) OLS observed-scale total | 0.119 ± 0.011 | (observed scale) |
| (c) Lee-2011 bridged to liability | 0.334 ± 0.032 | 1/3 (Lee fraction) |

The joint probit fit recovers the probit residual-scale total (0.5);
marginal per-SNP fits attenuate by the polygenic background in their
residuals (~1/(1+V_bg), material at h²=0.5); the null-adjusted z route
agrees with that attenuated total; and the OLS/Lee-2011 route recovers
the *fraction-of-total-liability* form (1/3), which is the McKelvey-Zavoina
R² of Lee et al. 2012 (Genet Epidemiol, eq. 9) -- implemented as
`probit_liability_r2(..., fraction=True)`. Artifacts:
`bench_liability_scale.csv`.

Paper-fixture checks in the tests cover the Lee 2011 Discussion factors
(observed 0.18/0.54/0.91 -> liability 0.1/0.3/0.5 at K=0.01, P=0.5) and the
Crohn's Table 3 fixture (0.61 -> 0.22).

## 24. Environment components in estimation (`bench_shared_env.py` panel (c), formerly `bench_env_components.py`)

*Script merged into `bench_shared_env.py` panel (c) in the 2026-08
consolidation; the numbers below are from the v0.4.0 rerun of the merged
panel. corr(o) reads ≈0.60 rather than the 0.28–0.30 recorded before the
`estimate_liability(out="full")` semantics change (the proband's own bound is
now applied after the relative fold); the wiring-gain direction is unchanged.*

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
| additive-only (misspecified) | 0.454 | 0.927 | 0.605 |
| oracle-wired (c2/m2 at truth) | 0.455 | 0.986 | 0.608 |
| fitted-wired (fit then wire) | 0.455 | 0.983 | 0.607 |

- **Calibration restored**: the additive-only model over-credits environmental
  clustering to genetics (slope 0.93, over-dispersed); wiring recalibrates to
  0.99 -- the algorithm doc's "sharper genetic estimate" promise, measured.
- **Full-liability prediction sharpens**: corr(E[l_o | family], truth) rises
  0.605 -> 0.608 when the environment is modelled.
- **The fit->wire loop works end to end**: fitted components (A 0.43, C 0.14,
  M 0.15 vs truth 0.4/0.15/0.15) give the same calibration as the oracle.
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

### Robustness (adversarial probes, formerly `bench_aod_decay_robustness.py`, now `bench_aod_decay.py --robustness`)

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

## 26. Sex in the covariance vs sex in the thresholds (`bench_sex_limitation.py`, now panel (a) of `bench_covariance_extensions.py`)

`construct_covmat_sex_limited` lets sex enter `Sigma` (sex-specific `h2`, a
cross-sex genetic correlation `rg`) rather than only the thresholds. The algebra
guarantees the BLUP weights change; it does not guarantee the score improves.
This benchmark measures it against a known truth.

Families (proband + both parents + one sibling) are drawn from the **true**
sex-limited covariance, so the proband's genetic liability is known exactly.
Proband and sibling sexes are balanced across the four cells. `K_female = 0.05`,
`K_male = 0.10`; 3 replicates of 2,000 families. Four arms, all scored by
`corr(estimate, true g)` and the calibration slope `regress(true on estimate)`:
the stored `±` values are t-based 95% half-widths, reproduced unchanged by the
v0.4.0 rerun as panel (a) of `bench_covariance_extensions.py`.

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

## 27. Ignoring genetic nurture (`bench_nurture.py`, now panel (b) of `bench_covariance_extensions.py`)

`construct_covmat_nurture` separates a proband's **direct** additive value from
the **indirect** path by which the parents' genotypes shape the rearing
environment. This benchmark asks what a nurture-blind analysis costs.

Families (proband + both parents + one full sib) are drawn from the **true**
path model, so the direct value `A_o` is known exactly. `h2 = 0.4` (direct),
`K = 0.05`; 5 replicates of 3,000 families. The stored `±` values are t-based
95% half-widths, reproduced unchanged by the v0.4.0 rerun as panel (b) of
`bench_covariance_extensions.py`. Four estimators are scored against `A_o`:

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

## 28. PGS comparison and the joint model (`bench_pgs_comparison.py`)

The PGS-baseline and PGS + family-history joint analysis, with an honest
train/test split. Five independent genotype/effect/cohort replicates; each has
10,000 probands, 2,000 independent SNPs (30 causal), h²=0.5, K=0.05, and
parents plus one sibling — the `bench_gwas_power.py` generative framework. Each
replicate is split 50/50: the discovery GWAS and the PGS weight fit use only
the 5,000 train probands, and everything predictive is scored on the 5,000
held-out test probands. The two-score combiner is itself two-fold cross-fitted
within that test half: each subject is scored by coefficients fitted on the
opposite fold, so no subject is scored by a combiner trained on its own `g`.

The PGS is trained on the train case/control GWAS as marginal Z-scored
weights `w_j = √n · corr(x_j, y)` (the default self-contained numpy backend).
With independent SNPs there is no LD to shrink for, so this is the
LDpred-infinitesimal limit up to an overall scale; an optional
`--pgs-backend ldpred3` fits LDpred3-auto on the train summary statistics and
scores via saved weights (it agreed with the numpy backend to within 0.002
correlation on a shared replicate). The family-history score is the classic
LT-FH posterior mean from the Pearson–Aitken engine: this framework has no
age/sex/cohort structure, so personalised LT-FH++ thresholds would be identical
for every member and add nothing (section 15 covers the personalised case; PA
matches Gibbs to corr ≥ 0.997 per section 1). As elsewhere in this report, the
independent-SNP marginal-association panel below reports **causal-SNP NCP
ratios** (`mean chi² - 1`) while the
prediction panel reports **squared-correlation R²** against held-out true `g`;
the two magnitudes are not comparable. `±` is replicate SE throughout.

GWAS arms on the train cohort (as in section 4):

| Phenotype | mean causal chi² | causal-SNP NCP ratio / c-c | lambda GC |
|---|---:|---:|---:|
| case/control | 22.28 ± 1.11 | 1.000× | 0.999 ± 0.021 |
| LT-FH (PA) | 31.73 ± 1.43 | 1.446 ± 0.023× | 1.018 ± 0.023 |
| oracle true g | 168.38 ± 1.47 | 7.95 ± 0.43× | 1.017 ± 0.018 |

The LT-FH NCP ratio matches section 4's replicated 1.468 ± 0.040× (the ratio
is sample-size independent in expectation; only the discovery n differs).
Prediction arms on the test cohort, against the held-out true genetic value:

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
95% CI half-widths, section-15 style; the LT-FH-vs-case/control NCP-ratio
increment is +0.4464 ± 0.0638).

The two scores are weakly correlated and complementary, as the measurement
model in `docs/algorithm.md` predicts. The estimand: with
`p = h²_SNP / h²_total = 1` by construction here (the true `g` is built
entirely from the simulated SNPs), the prediction is
`Corr(PGS, FH) = a·b·√p = a·b` where `a = Corr(PGS, g)` and
`b = Corr(FH, g)` are measured on the test cohort — equivalently
`√(R²_pgs · R²_fh · p)` at `p = 1`. Observed corr(PGS, LT-FH) is
**0.2009 ± 0.0046** against a theory value of 0.2003 ± 0.0050; the paired
theory-minus-observed difference is **-0.0006 ± 0.0160** (95% CI), consistent
with zero. Conditional independence holds by construction in this design — the
PGS error is train-cohort sampling noise, the LT-FH error is posterior
uncertainty from the relatives, and the cohorts do not overlap — so this is a
verification of the identity, not evidence about its failure modes (ancestry,
assortment, selection), which the doc's caveats cover.

Caveats: independent SNPs only (no LD; the HAPNEST path remains blocked); the
joint regression targets the true `g`, an oracle evaluation rather than an
observed outcome; and at 2,000 SNPs the multiple-testing dilution is mild, so
the PGS-to-label gap is friendlier than a genome-wide setting — the joint
model's *increment* over the PGS is the portable number, not the absolute R².
Rerun: `python benchmarks/bench_pgs_comparison.py`.

## 29. Ascertainment: what the fitters do on selected samples (`bench_ascertainment.py`)

The other fitter benchmarks all draw unascertained population families, which is
the only design `fit_heritability` / `fit_variance_components` /
`fit_genetic_correlation` claim to support. This section measures what happens
when that contract is violated, because until 0.3.1 the `sampling="population"`
gate checked a string rather than the data, so an ascertained cohort passed
straight through.

Each replicate draws a **population** under a known model, applies one selection
rule, and fits the selected subset. Only the selection rule differs between
rows: model, analysed N, thresholds and fitter settings are held fixed. Five
replicates, N = 3,000 analysed families, K = 0.05, true h² = 0.5, n_iter = 800
-- the moderate default since the 2026-08 consolidation; the former campaign
grid (10 replicates, N = 10,000, n_iter = 1500) remains available under
`--full`.
`random_50` keeps half the population *independently of phenotype* and is the
negative control: it exercises the identical chunked accept/reject, redraw and
truncation path, so it isolates selection-on-phenotype from the harness.

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
(boundary fraction 1.00, across-replicate SD 0.000). The population and
negative-control arms are not. Case shares shown are the nuclear arm's; the
sibship arm realises the same shares except `family_history` (0.219).

**The strongest cell is true h² = 0** (`--h2 0.0`, 5 replicates, nuclear;
pre-consolidation grid -- the moderate default does not include this arm):
population returns 0.018 and `random_50` 0.025, while `proband_case`,
`case_control`, `enriched_20`, `family_history` and `fh_proband_control` **all
return 1.000**. Selecting families on phenotype makes the fitter report complete
heritability in a population that has none.

**A+C (`fit_variance_components`, sibship, true A = 0.4, C = 0.2).** The
multi-component fit does not pin at the elementwise clamp -- it renormalises onto
the simplex, so the diagnostic is the exhausted residual, not a component at
1-eps:

| scheme | A | C | residual | saturated |
|---|---:|---:|---:|---:|
| population | 0.428 | 0.203 | 0.369 | 0.00 |
| random_50 | 0.427 | 0.209 | 0.364 | 0.00 |
| proband_case | 0.500 | 0.500 | **0.0001** | 1.00 |
| case_control | 0.498 | 0.502 | **0.0001** | 1.00 |
| enriched_20 | 0.500 | 0.499 | **0.0001** | 1.00 |
| family_history | 0.594 | 0.406 | **0.0001** | 1.00 |
| fh_proband_control | 0.832 | 0.168 | **0.0001** | 1.00 |

A + C sums to 1 - eps under every ascertained scheme: the whole liability
variance is consumed. Reading the components alone would hide this, which is
why the CSV carries `residual`.

**Genetic correlation** (ascertained on trait 1, true r_g = 0.5) inherits it:
population stays unpinned at 0.381 (SD 0.033) with per-trait h² of 0.504/0.518;
`proband_case` and `case_control` return r_g = 1.000 with **both** per-trait h²
at 1.000. Since r_g = G/sqrt(h²₁h²₂), that 1.000 is a ratio of two pinned
denominators, not an estimate -- hence the `h2pin` column.

### It is bias, not noise, and not non-convergence

Three checks, because "the estimate is wrong" has several innocent explanations:

- **More data does not help.** Across N = 1,000 / 3,000 / 10,000 the population
  bias is -0.077 / -0.066 / +0.013 (SD 0.254 → 0.033) while
  `proband_case` holds at **+0.500 / +0.500 / +0.500**. Sampling noise shrinks
  as 1/√N; this does not.
- **It has converged.** `population` gives 0.477 / 0.450 / 0.468 at n_iter =
  250 / 800 / 2000; `proband_case` gives 0.9999 at all three with a trace-tail
  slope of 0. Decisively, multi-start: `proband_case` reaches 0.9999 from
  h2_init = 0.05, 0.5 and 0.95 alike (spread 0.0) -- it *climbs* to the ceiling
  from below rather than failing to leave a high start.
- **It is not the harness.** `random_50` selects half the population through the
  identical code path and recovers h² within this grid's noise (+0.102 nuclear,
  +0.043 sibship; across-replicate SD 0.139 / 0.049).

### How wrong, and how little it takes

The clamp censors the magnitude: every ascertained cell reports 0.9999 whatever
the truth. The Haseman-Elston moment evaluated on the selected families' own
liabilities is uncensored (nuclear):

| scheme | HE moment | × truth | centered |
|---|---:|---:|---:|
| population | 0.507 | 1.01 | 0.507 |
| random_50 | 0.517 | 1.03 | 0.517 |
| proband_case | 1.688 | **3.38** | 0.202 |
| case_control | 1.048 | 2.10 | 0.725 |
| enriched_20 | 0.685 | 1.37 | 0.652 |
| family_history | 1.094 | 2.19 | -0.047 |
| fh_proband_control | 0.828 | 1.66 | 0.010 |

**This is a decomposition, not the mechanism, and must not be read as one.** At
true h² = 0 (pre-consolidation `--h2 0.0` arm) every ascertained scheme still
fits 1.000 while this statistic sits at ~0 (`proband_case` +0.006) or negative
(`family_history` -0.107, and -0.562
once centered -- centering makes it *worse*). The fitter never sees these
liabilities; it sees truncated-MVN draws conditional on the selected status
pattern, so under proband ascertainment every augmented proband is redrawn above
threshold, relatives are pulled with it, and the cross-products stay positive
whatever the truth. The runaway is augmentation feedback, which a complete-data
statistic cannot see.

The tolerance is far tighter than intuition suggests (arm H; nuclear,
N = 2,000, 2 replicates, realised shares against an assumed K = 0.05):

| enrichment | 0.95× | 1.51× | 1.96× | 2.92× | 4.10× |
|---|---:|---:|---:|---:|---:|
| fitted h² | 0.420 | 0.984 | 0.998 | 1.000 | 1.000 |
| bias | -0.080 | **+0.484** | **+0.498** | +0.500 | +0.500 |
| across-rep SD | 0.070 | 0.001 | 0.001 | 0.000 | 0.000 |

A 7.6% case rate against an assumed 5.0% already inflates h² by +0.48, and by
2× the estimate is at the clamp. The curve is steep, and on this grid the
enriched cells saturate in both replicates, so read the shape, not any single
cell: mild enrichment is already damaging. The unenriched cell sits at 0.420
rather than 0.5, which is this arm's small-N noise floor (N = 2,000, 2
replicates, SD 0.070), not a bias of the fitter; the N = 10,000 population arm
above recovers 0.513.

### The correction, where one exists

Selection breaks the **mix** of families, not the per-family augmentation --
given a family's statuses, its truncated-MVN draw is already the correct
conditional law. So re-mixing by inverse probability of inclusion is a
better-matched remedy than any scale transform. `sampling="ipw"` with
per-family `weights = 1/P(sampled)` weights both the numerator and the
denominator of the Haseman-Elston ratio. Same grid, 5 replicates, N = 3,000:

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
of 0.5. Two limits bound this:

- **Positivity.** The three *undefined* rows use deterministic rules that sample
  no families at all from some complete status-pattern stratum, so no weight
  reconstructs what was never observed. The benchmark identifies that from the
  known selection rule and does not call the weighted fitter for those rows.
  They are examples of zero-probability designs, not a claim that every form of
  family-history selection is unweightable.
- **Efficiency.** Weights reach 19× at K = 0.05, and the IPW SD (0.099) exceeds
  the population arm's (0.088). IPW buys accuracy with precision, and the cost
  grows as the enrichment does.

A Lee et al. observed→liability factor cannot substitute. It is a multiplicative
function of (K, P) alone, while at fixed K a true h² of 0.5 and of 0.0 **both**
produce 1.000 -- no invertible constant maps both back. It also transforms an
*observed-scale* estimate, and this fitter is already on the liability scale.
Measured directly (arm J), an observed-scale HE + Lee pipeline on the same
ascertained cohorts lands at **0.224** (50/50) and **0.332** (20% enriched)
against a truth of 0.5, and is itself +0.23 off under clean population sampling
-- so it is not a drop-in replacement for the population-sampled fitter either.
This arm standardises each family role by its sample case rate, pools
multi-relative HE moments, and applies the proband's sample fraction in the Lee
factor. It is an in-repository diagnostic, not an implementation or empirical
evaluation of conventional unrelated-sample LDSC or GREML.

### What changed as a result

`sampling="population"` is now screened for gross marginal inconsistency rather
than taken entirely on trust (`ltpred.fit._assert_population_case_rate`). The thresholds assert a
prevalence, and under population sampling each role's case count is
Binomial(n_families, K), so a binomial z-test applies per role. All five
phenotype-selected schemes above raise at z = +67 to +436 (computed on the
pre-consolidation N = 10,000 campaign grid; the same schemes give +43 to +276
at N = 4,000). Arm I is the
specificity half: 6 unascertained cohorts across N ∈ {500 … 10,000} and
K ∈ {0.02 … 0.20}, **0 false positives**.
The bar is deliberately conservative (z ≥ 6 and a ratio outside [1/1.15, 1.15]):
`bootstrap_fit` resamples are centred on the cohort's rate rather than on K, so
their z carries the cohort's own sampling error as an offset, and a z ≥ 4 bar
fired on a legitimate 1,500-family cohort. The cost is power at small N.
Detectable enrichment is the ratio at which |z| reaches the bar,
`1 + 6·√((1−K)/(K·n))`, so it depends on the prevalence as well as N: at this
section's K = 0.05 it is **~1.48× at this section's N = 3,000**, ~1.67× at
N = 1,500 and ~1.26× at N = 10,000 (at
K = 0.10 it would be ~1.46× and ~1.18×). So the check catches the catastrophic
designs and does **not** certify population sampling. Mild enrichment on a small cohort still passes, and the dose-response
above shows that is not harmless.

With `sampling="ipw"` the same test is applied to the **weighted** counts, using
Kish's effective sample size so heavy weights widen the tolerance rather than
manufacturing significance. A failure falsifies the proposed weights or
sampling contract. Passing only establishes compatible role-wise case
marginals; it does not certify joint family-pattern positivity or weight
correctness.

## 30. Locked comparison to R LTFHPlus and LTFGRS (`bench_ltfhplus_compare.py`)

Same classic LT-FH families and bounds, three independent cohorts of 200
nuclear pedigrees (parents + one sibling), h² = 0.5, K = 0.05, tol = 0.01,
n_sim = 100,000, burn_in = 1,000 — LTFHPlus 2.2.0's own Gibbs settings.
LTFHPlus 2.2.0 is Gibbs-only. Public Pearson–Aitken is LTFGRS 1.0.1
(`estimate_liability(..., method="PA", useMixture=FALSE)`). `±` is the
across-replicate SE. Numba used 4 threads; both R packages used 1
`future` worker (sequential plan).
The CSV records LTFHPlus's distinct seed for each replicate, R/Python/NumPy/
Numba versions, and the resolved R/Numba/OMP/OpenBLAS thread settings.

Wall-clock is the estimator call after in-process warmup, as a cohort
total and as milliseconds per family (total / 200). Peak RSS is the
isolated child via `wait4` (ldpred3's inherited-floor launcher
`_peak_launcher.py`): each arm is a fresh process, so the high-water
mark is that process, not the fat driver. Those peaks include the
interpreter and packages.

| Estimator | corr vs LTFHPlus | RMSE vs LTFHPlus | total s / 200 fam. | ms / family | peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| LTFHPlus Gibbs | — | — | 10.27 ± 0.06 | 51.33 ± 0.28 | 440.7 ± 0.7 |
| LTFGRS PA | 0.9999 ± 0.0000 | 0.0046 ± 0.0003 | 1.802 ± 0.011 | 9.01 ± 0.05 | 259.8 ± 0.7 |
| ltpred Gibbs | 0.9999 ± 0.0000 | 0.0041 ± 0.0002 | 1.513 ± 0.012 | 7.57 ± 0.06 | 180.1 ± 0.1 |
| ltpred PA | 0.9999 ± 0.0000 | 0.0046 ± 0.0003 | 0.00127 ± 0.00003 | 0.00636 ± 0.00017 | 178.7 ± 0.3 |

The Gibbs scores agree at the scale of the Monte Carlo error (LTFHPlus
reports `genetic_se` ≈ 0.004). The PA scores agree with each other much
more tightly: corr(ltpred PA, LTFGRS PA) = 1.0000, RMSE = 0.000087 ±
0.000008 (max abs ≈ 0.0007). That is the same sequential two-moment
update, not two different approximations.

Per-family times are the cohort total divided by 200 equal nuclear
pedigrees. They are not a sweep over pedigree size.

Fold times are the mean ± SE of the three per-replicate ratios
(not the ratio of the mean times):

| Comparison | fold | same algorithm? |
|---|---:|:---|
| LTFHPlus Gibbs / ltpred Gibbs | 6.786 ± 0.079× | yes |
| LTFGRS PA / ltpred PA | 1418 ± 30× | yes |
| LTFHPlus Gibbs / ltpred PA | 8086 ± 250× | no |
| LTFGRS PA / ltpred Gibbs | 1.19 ± 0.01× | no |

The first row is Gibbs versus Gibbs across language and
parallelisation. The second is the same sequential PA update in R
versus compiled Python. The 8086× row mixes algorithms. None of
these is a hardware-independent constant.

**Parallelisation is not equally distributed across those rows**, so the
whole comparison was re-run with both sides at one thread
(`bench_ltfhplus_compare_1thread.csv`, same three cohorts, R workers = 1,
Numba threads = 1):

| Comparison | fold at 4 threads | fold at 1 thread |
|---|---:|---:|
| LTFHPlus Gibbs / ltpred Gibbs | 6.786 ± 0.079× | **1.751 ± 0.012×** |
| LTFGRS PA / ltpred PA | 1418 ± 30× | **1425 ± 22×** |

The 1-thread column is 12 independently seeded replicates
(`bench_ltfhplus_compare_1thread.csv`). Its `±` is across-replicate scatter,
not a universal measurement uncertainty. All arms are single-threaded there,
so the ratio is more portable than the absolute times, but it remains a
machine-specific measurement.
Per-replicate load was not recorded for this rerun, so the digits cannot be
attributed to a particular load profile.

ltpred's Gibbs is `prange`-parallel over families, so most of the 6.79× is
the 4:1 resource asymmetry rather than implementation: matched at one
thread the gap is 1.75×. PA is effectively serial, and its fold is essentially
unchanged at one thread (1425×) and four (1418×).
Both quantities are legitimate and they answer different questions — the
4-thread rows are what a user gets at each package's defaults, since the
`future` plan is sequential by default, while the 1-thread rows isolate the
implementation. Quote whichever, with its thread count.
Peak RSS at this n is mostly runtime: LTFHPlus Gibbs sits higher
because the sampler retains 10⁵ draws; the two PA processes differ
mainly in R versus Python heaps. Opt-in: requires R and LTFHPlus; the
script exits 2 if they are missing. LTFGRS is the PA arm.

## 31. Time and memory: v0.6.1 versus v0.6.0

The [clean-source rerun](results/2026-09-09-time-memory-v061-rerun/README.md)
on 9 September 2026 compares v0.6.1 (`b516271`) with v0.6.0 (`52dec52`)
with the driver and workload settings unchanged. All 28 workers completed:
seven cases, two versions, and separate runtime/RSS and allocation processes.
The candidate commit and all captured package/driver/support-script hashes
were stable before and after measurement.

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

PA compares complete observation-mask byte strings, reuses input bounds when
the target is already first, and centers owned float64 reduction buffers in
place. Parent-graph construction builds ordered sibling lists directly instead
of allocating temporary per-person sets. These changes account for the reduced
sorting and copying work. All four PA batches contain 200,000 families and
17 coordinates with a shared positive-definite covariance; they exercise
numerical paths rather than simulate a population. Graphs have half founders
and two children per parent pair. The register case has 300 probands sharing a
paternal chain of depth 60 and uses degree-one observation selection.

Every PA mean and variance, every reported register score and diagnostic, and
both graph adjacency digests matched exactly. Mixed-mask PA and the
million-record graph show the larger gains. The
small register case changes by about 3% and its allocation peak is unchanged;
this does not establish full-register scaling or a general end-to-end speedup.
No new statistical calibration, Gibbs, quadrature or fitting comparison was
performed in this rerun.

The runtime was Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3 and Numba 0.67.0 on
arm64 macOS with 10 logical CPUs. Each worker verified one Numba thread;
BLAS/OpenMP limits were requested at one, without an independent runtime BLAS
pool query. AC power and Low Power Mode off were checked before and after.
Worker-start one-minute load averages ranged from 2.46 to 3.29. These repeated
measurements on one machine are not hardware replications or confidence
intervals. The capsule includes first-call timings (with reached compilation),
all warm repetitions, raw memory measurements, provenance and reproduction
instructions. v0.6.2 changes documentation and its checks; the numerical
implementation is unchanged from the measured v0.6.1 source.

## 32. Pairwise composite-likelihood recovery (`bench_pairwise_recovery.py`)

> **Evidence status (2026-09-10):** commit `b6065ff`, CPython 3.10.20
> (conda-forge), numpy 1.26.4, scipy 1.15.3, ltpred 0.6.2, Apple M2 Pro
> (10 logical CPUs), 6 worker processes with the BLAS pool pinned to one
> thread each; 806 s; manifest `run_manifest.jsonl`. This is a statistical
> benchmark, so the machine load is not part of the claim. `--jobs` sets only
> the worker count: per-replicate seeds make the output identical at any
> value, verified for 1 against 6.

11,364 independent, unascertained population cohorts of `o, m, f, s1, s2`
families at a common threshold, 10% prevalence, truths `A = 0.30`,
`C = 0.15`, `M = 0.10`, totalling 43,974,000 simulated families. Replicates
are allocated proportional to `1/N`, so SE(bias) is 0.0009, 0.0007 and 0.0009
for A, C and M at *every* cohort size and the sizes are directly comparable.

| N | replicates | bias A | bias C | bias M | SE/SD A | coverage A | pinned |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 5,866 | -0.0032 | -0.0005 | -0.0019 | 1.01 | 0.948 | 10.3% |
| 3,000 | 2,933 | -0.0014 | -0.0009 | -0.0024 | 1.00 | 0.952 | 2.9% |
| 6,000 | 1,466 | -0.0005 | -0.0009 | -0.0007 | 1.00 | 0.953 | 0.4% |
| 12,000 | 733 | +0.0002 | +0.0005 | +0.0008 | 0.96 | 0.944 | 0.0% |
| 24,000 | 366 | -0.0009 | +0.0007 | +0.0002 | 0.97 | 0.940 | 0.0% |

- **Consistent, with a small negative `1/N` finite-sample bias.** Weighted
  least squares on `bias(N) = b0 + b1/N` puts the asymptotic bias `b0` at
  +0.00005 for A (95% CI -0.00113 to +0.00123), +0.00020 for C (-0.00066 to
  +0.00107) and +0.00024 for M (-0.00099 to +0.00147) -- all centred within
  0.0002 of zero. The whole signal is the `1/N` term, with `b1` of -4.67,
  -1.57 and -3.96, and the fit is adequate (chi2/df 0.42 to 1.21, p 0.30 to
  0.74). The largest deviation, A at N = 1,500, is -0.0032, about 1% of the
  parameter, and it decays.
- **The negative sign is predicted, not incidental.** In the reducible case of
  one component and one parent-offspring pair per family at threshold zero the
  estimator is exactly `2 sin(pi (p_hat - 1/2))` in the concordant fraction,
  which is concave for `A > 0` (`g''(p) = -pi^2 A`), so Jensen pulls the mean
  below the truth. Enumerating that case's binomial sampling distribution gives
  `N x bias -> -0.3667` at `A = 0.30`, converging from 0.975 to 1.000 of the
  delta-method constant over N = 50 to 5,000. The `b1 = -4.67` fitted here has
  the same sign and is about thirteen times larger, the direction the harder
  design should move it: ten dependent pairs, three components, and a 10%
  rather than 50% case rate.
- **Sandwich SEs are calibrated and intervals cover.** Mean reported SE over
  across-replicate SD lies between 0.96 and 1.10 across all components and
  sizes, and realised coverage of the nominal 95% normal interval between 0.940
  and 0.975. No fit failed and none was discarded at any size.
- **Boundary pinning is confined to the component nearest zero.** `M`, at truth
  0.10 with an across-replicate SD of 0.065 at N = 1,500, is pinned at the
  non-negativity boundary in 10.3% of replicates there, falling to 2.9%, 0.4%
  and then nil by N = 12,000. `A` is never pinned. Reported SEs are unavailable
  by design at the boundary, so those replicates are excluded from the coverage
  column and counted in the last one instead.

Scope: one design under the supported `sampling="population"` contract with a
known threshold. Other prevalences, relationship structures, and IPW weighting
are not covered, so the 0.6.0 caveat that efficiency and interval coverage need
broader validation is narrowed by this section, not retired.

## Historical report changes

- Replicated the accuracy and calibration grids across five independent
  seeds (every cell now reports an across-seed mean and SE), replacing the
  single-seed diagnostic values.
- Added independent-replicate SEs to association NCP/detection, age-onset, family-history,
  confounding, and cohort-span panels.
- Corrected marginal-association NCP ratios to use chi-square noncentrality rather than
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
- Added replication (five independent populations/registers) and long-format
  CSV artifacts to §20 (`bench_pedigree_inference.csv`) and §21
  (`bench_register_pipeline.csv`) on 2026-08-10, plus prospective
  discrimination metrics (rank AUC, calibration-in-the-large); the §21
  relatives'-events leakage contrast is now resolved by paired t CIs (helps
  rank discrimination; corr contrast unresolved).
- §16 gained paired replicate uncertainty (`bench_pafgrs_mixture`, now with a
  long-format CSV) and §17 gained the `test_genetic_correlation` r_g = 0 null
  panel (0/25 rejections); §14 is now 3-seed with across-seed mean ± SE
  (2026-08-10).
- Corrected the §13 confounding table to ordinary rounding against
  `bench_confounding.csv` (2026-08-14). The previous last-digit offsets
  (e.g. 16.461 ± 0.397 vs stored 16.460 ± 0.399) were unguarded
  transcription. The qualitative claim is unchanged.
- Corrected a docs/report transcription that quoted IPW recovery as 0.456 /
  0.481 for 50/50 and 20%-enriched cohorts. Those figures are the
  `random_50` IPW mean and a neighbour that is not in the committed CSV;
  the `ipw` arm of `bench_ascertainment.csv` gives **0.495** and **0.473**.
  RESULTS.md §29 already had the artifact values.
- 2026-08-14: added a liability-dependent onset arm (ρ = 0.6) to §16;
  compared pin vs interval encodings on the same families in §3; reran
  §19/§22/§23 under the current SE and `subtract_null` conventions and
  wrote `bench_{cip_estimation,tetrachoric,liability_scale}.csv`.
  Stochastic-onset §16 numbers shifted slightly because the cohort
  generator now shares `ltpred.simulate._onset_times` (vectorised
  per-role draws); crossing cells were unchanged.
- 2026-08-15: reran §2 scaling and the §30 LTFHPlus/LTFGRS lock after
  collapsing untruncated genetic coordinates out of the Gibbs sweep.
- 2026-08-15: locked LTFHPlus 2.2.0 Gibbs and LTFGRS 1.0.1 PA (§30),
  with isolated-process peak RSS and per-family times.
- 2026-08-20: full-suite rerun of all 26 scripts at v0.4.0 in the recorded
  environment (Numba at 8 threads; `bench_scaling.py` and
  `bench_ltfhplus_compare.py` at 4), with the committed CSVs regenerated in
  place. Sections not mentioned here or above reproduced their recorded
  values within sampling error and were left untouched. Refreshed: the §1
  and §14 PA–Gibbs agreement figures and the §15/§16 Gibbs cross-checks,
  which had been recorded before the 2026-08-15 collapsed-genetic Gibbs
  sweep and all moved up (§14's worst single seed is now in the
  densely-affected regime); and the §30 timing, peak-RSS and fold fields,
  which now come from the recorded Python 3.14.6 free-threading interpreter
  (the accuracy lock is unchanged; the ltpred-arm RSS rose with the
  interpreter). The merged-panel sections (§3, §8, §11, §24–§27, §29) were
  re-derived from the same artifacts, and the §26–§27 interval fields are
  now t-based, closing the earlier rerun caveat.
- 2026-09-03: regenerated §§20–21 with the supported observation contract
  (closure-only bounds uninformative; calendar `index_time` per proband) under
  the provenance wrapper on a clean tree (commit `bad41ad`; free-threaded
  CPython 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0 at 4 threads, BLAS
  pinned to 1; manifest `run_manifest.jsonl`). The historical pre-closure /
  attained-age numbers validated a different observation model and are
  superseded. Cross-stack reruns draw fresh liabilities per replicate (NumPy's
  large-matrix `multivariate_normal` stream differs between 2.2 and 2.4), so
  the claims are replicate-mean claims, not seed-locked constants. The
  relatives'-events contrast is unresolved at R = 5 on both metrics and is
  not quoted as a payoff.

## Remaining limitations

- The accuracy and calibration grids (sections 1 and 12) are now replicated
  across five independent seeds; every cell reports an across-seed mean and
  SE. The PA stress grid (section 14) is now 3-seed with across-seed mean
  ± SE; treat finer seed-level detail there as descriptive.
- Three- or four-replicate panels give useful SEs but still estimate tail
  uncertainty coarsely. The integrated main panel now uses ten replicates and
  its sex isolation uses five; tail calibration remains noisier than paired
  score contrasts.
- Component test Type-I error, bootstrap coverage, and MCEM SEs were
  calibrated at coarse resolution (section 17; R = 25 bounds the
  resolution -- read as 'no gross miscalibration').
- The LTFHPlus / LTFGRS lock (§30) is classic no-mixture bounds only,
  200 equal nuclear families, LTFHPlus 2.2.0 and LTFGRS 1.0.1. It does
  not cover the PA-FGRS mixture, personalised CIP, or a pedigree-size
  sweep; per-family times are the cohort total divided by 200.
- The HAPNEST path was not executed here. The PA-FGRS censoring-mixture
  benchmark (section 16) now includes a liability-dependent onset arm;
  its censoring correction is small at the tested settings, and case
  encoding dominates calibration there.
- Sex-limitation and nurture interval fields (§26–27) were refreshed in the
  v0.4.0 rerun of `bench_covariance_extensions.py`, which emits t-based 95%
  half-widths; the tabulated values reproduce the stored ones unchanged.
- The lightweight GWAS helper uses the large-sample `n * r²` score statistic.
  A finite-sample regression test would use residual degrees of freedom and the
  `r² / (1-r²)` correction; the matched NCP ratios are robust to this
  small approximation, but genome-wide discovery counts are only illustrative.
- Most benchmark scripts still write canonical artifacts unconditionally. The
  integrated-personalization script accepts `--output-prefix`; custom-path
  outputs must be preserved separately. Historical artifacts remain
  incompletely attributable.
