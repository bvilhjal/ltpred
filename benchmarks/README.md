# ltpred benchmarks

Benchmarks for the two ltpred inference engines — the **Gibbs sampler** and
deterministic **Pearson–Aitken (PA)** — across LT-FH, LT-FH++ and ADuLT inputs.
The PA-FGRS censoring mixture has unit tests and a dedicated generative
benchmark (`bench_pafgrs_mixture.py`).
Bounds define the observation encoding; relative rows distinguish
LT-FH++ from family-free ADuLT. Everything simulates
its own data, so the true genetic liability is known and the benchmarks run
locally with no downloads (the one exception, real-LD genotypes, is an opt-in
[HAPNEST](hapnest/README.md) step).

This project baselines on **four threads**, which is what the timings in
`RESULTS.md` were measured at. Pin the thread count explicitly rather than
letting Numba take every core: a speed-up is only interpretable alongside the
thread count it was measured at, because Gibbs is parallel while the PA object
path is largely serial.

```bash
python benchmarks/bench_accuracy.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 python benchmarks/bench_scaling.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 python benchmarks/bench_gwas_power.py
```

Timing runs also want an otherwise-quiet machine. The manifest records the load
average, so check it before quoting a number: `bench_scaling`'s current figures
were taken at 1-minute load ~7–11 on a 10-core box and are correspondingly
conservative.

Most scripts write a `.csv` and, if matplotlib is present, a `.png`. The
following scripts instead print focused diagnostics to stdout:
`bench_cip_estimation.py`, `bench_env_components.py`,
`bench_inference_calibration.py`, `bench_liability_scale.py`,
`bench_misspecification.py`, `bench_pafgrs_mixture.py`,
`bench_pedigree_inference.py`, `bench_register_pipeline.py`, and
`bench_tetrachoric.py`.

For a provenance record, run a benchmark through the lightweight wrapper:

```bash
python benchmarks/run_benchmark.py bench_accuracy.py -- --reps 5
```

The wrapper appends one JSON object to `benchmarks/run_manifest.jsonl` with the
exact command, UTC times, exit status, Git commit, tracked-diff hash, content
hashes for untracked source files, package versions, and thread settings. It
also records the machine the run happened on — architecture, CPU model, logical
and physical core counts — and the thread count Numba actually resolved to, so
timings taken on different hardware or a different Numba can be told apart
rather than compared blind. The `ltpred` version comes from the checkout the
benchmarks import, not from installed distribution metadata, which can be a
stale `egg-info` left over from an earlier build. It
captures stdout and stderr in run-specific logs and records their SHA-256
hashes alongside hashes of changed top-level `bench_*.{csv,png}` artifacts.
Custom output paths are not discovered automatically. The manifest is
generated at run time; it records a run but does not make results valid after
their source or numerical dependencies change.

Two gain metrics appear below. Genotype-GWAS panels report **causal-SNP NCP
ratios**, based on `mean chi² - 1`. Prediction-only panels report
**squared-correlation effective-N proxies**. These answer related but different
questions and their magnitudes should not be compared as if they were the same
statistic.

**Runtime depends on your machine.** These reference numbers were taken on 10
cores with Numba installed (`pip install -e ".[fast]"`); the first call in each
script pays a one-off JIT compile. Expect substantially slower runs on fewer
cores, on a cold JIT cache, or without Numba (the pure-Python fallback is
numerically identical, just slower). Each script takes CLI flags (`--reps`,
`--n-fam`, …) to trade runtime for precision.

## Scripts

Rows marked **unsupported research** import checkout-only APIs from
`research/`; those APIs are not installed as part of `ltpred`.

| Script | What it measures |
|--------|------------------|
| `bench_accuracy.py` | corr(estimated classic LT-FH liability, true g) across heritability × prevalence × family structure — Gibbs vs PA accuracy, calibration slope, RMSE and squared-correlation effective-N proxy over case/control (→ `bench_accuracy.{csv,png}`) |
| `bench_scaling.py` | wall-clock scaling of both methods with #families and family size; families/second and speed-up (→ `bench_scaling.{csv,png}`) |
| `bench_age_onset.py` | LT-FH++ age component with the same relatives on both sides: classic binary LT-FH vs FH + onset-pinned cases, both fit with PA (and Gibbs as an agreement check); squared-correlation effective-N proxy across prevalence (→ `bench_age_onset.{csv,png}`) |
| `bench_gwas_power.py` | replicated genotype-based GWAS power for classic LT-FH: case/control vs the same LT-FH model inferred by Gibbs or PA vs an oracle — causal-SNP NCP ratio, detection power, SEs, and λ_GC (→ `bench_gwas_power.{csv,png}`). The default family is parents plus one sibling. Pass `--plink PREFIX` for **real-LD** HAPNEST genotypes; causal LD proxies are excluded from its calibration set (opt-in; see [`hapnest/README.md`](hapnest/README.md)) |
| `bench_ltfhpp_personalization.py` | **integrated LT-FH++ genotype GWAS** with age-, sex-, and cohort-dependent CIP, coherent family onset/follow-up, competing mortality, ascertainment, and demographically stratified null SNPs. A matched ADuLT arm uses the identical personalised proband bounds with all relatives removed, directly isolating the LT-FH++ family-history increment. The 10-replicate main panel reports causal-SNP NCP ratios and paired CIs; a prespecified 5-replicate sex-isolation panel compares age-only with age+sex family bounds. PA is primary and Gibbs diagnostics cover the first two main replicates (→ `bench_ltfhpp_personalization.{csv,png}`) |
| `bench_fit_heritability.py` | variance-component inference: bias and across-dataset precision of `fit_heritability` vs true h², vs #families and family structure, and the calibration of the reported `h2_se` (→ `bench_fit_heritability.{csv,png}`) |
| `bench_variance_components.py` | multi-component inference: recovery of additive `A` and common-environment `C` by `fit_variance_components` (bias & across-dataset SD), the constrained C estimate at the zero boundary, and precision vs #families (→ `bench_variance_components.{csv,png}`) |
| `bench_genetic_correlation.py` | **unsupported research:** genetic-correlation inference: bias & across-dataset SD of `r_g` from `fit_genetic_correlation` vs the true value — including the null (`r_g=0` with non-zero phenotypic correlation) — and precision vs #families (→ `bench_genetic_correlation.{csv,png}`) |
| `bench_genetic_factor.py` | **unsupported research:** genetic factor model `r_g ≈ ΛΛ' + Ψ` (`fit_genetic_factor`): end-to-end recovery of planted single-factor loadings from family case/control data, plus `srmr` as an in-sample diagnostic for a planted two-factor misspecification (not a calibrated factor-number test) (→ `bench_genetic_factor.{csv,png}`) |
| `bench_calibration.py` | **calibration** of the genetic-liability estimate, not just its ranking: the calibration slope/intercept and decile calibration curve of true `g` on the estimate (a correctly-specified posterior mean is self-calibrating, slope ≈ 1), and how a **wrong assumed `h²`** leaves the ranking (`corr`) robust but tilts the scale (slope) (→ `bench_calibration.{csv,png}`) |
| `bench_confounding.py` | **LT-FH++ cohort-component confounding**: a secular prevalence trend plus birth-cohort-correlated null SNPs, showing cohort-blind family thresholds inflate `λ_GC` (to ~16.5 at the strongest trend) while cohort-aware family thresholds remain near 1; this isolates cohort, not full age/sex/cohort LT-FH++ (→ `bench_confounding.{csv,png}`) |
| `bench_pa_robustness.py` | **PA robustness**: agreement with the Gibbs posterior mean (corr ≥ 0.998) on stressful pedigrees — large, rare (K=0.005), densely affected — and deterministic fold-in ordering sensitivity (→ `bench_pa_robustness.{csv,png}`) |
| `bench_shared_env.py` | value of modelling shared environment `C`: corr(genetic-liability estimate, true genetic liability) when families are simulated under `A+C+E`, comparing ignore-C (additive) vs fit-`A+C` vs oracle, swept over `c²` and sib-ship size (→ `bench_shared_env.{csv,png}`) |
| `bench_couple_env.py` | the couple/spousal environment `M`: recovery of `A+M` (bias & across-dataset SD, including constrained-boundary behaviour at `m²=0`), and the identifiability contrast — ignoring a real `C` inflates additive-only `Â` much more than ignoring a real `M` (mates have `A=0`) (→ `bench_couple_env.{csv,png}`) |
| `bench_fh_prediction.py` | registry simulation with age, cohort and competing mortality: PA fits both classic LT-FH and age/cohort-personalised family bounds, compared with squared-correlation effective-N proxies; a separate own-onset cohort-span panel is explicitly family-free **ADuLT** and tests cohort-aware vs cohort-blind ranking (→ `bench_fh_prediction.{csv,png}`) |
| `bench_pafgrs_mixture.py` | generative validation of the PA-FGRS censored-control mixture under threshold-crossing and stochastic-onset observation models |
| `bench_inference_calibration.py` | **unsupported research:** coarse repeated-sample checks of component-test Type-I error, family-bootstrap interval containment, and MCEM information SEs |
| `bench_misspecification.py` | calibration and ranking under heavy tails, assortative mating, unmodelled shared environment, and wrong prevalence |
| `bench_cip_estimation.py` | Kaplan–Meier/Aalen–Johansen curve recovery, delayed entry, competing mortality, and an end-to-end CIP-to-score check |
| `bench_pedigree_inference.py` | pedigree extraction fidelity, arbitrary-kinship scoring versus the fixed named-role grammar, and extraction throughput |
| `bench_register_pipeline.py` | **unsupported research:** pipeline from trio records through pedigree extraction and CIP thresholds to liability scores |
| `bench_tetrachoric.py` | tetrachoric relative-pair correlations, latent-correlation comparison, and the Falconer diagnostic |
| `bench_liability_scale.py` | probit residual-scale variance and observed-to-liability transformations, including null-adjusted summary-statistic estimates |
| `bench_env_components.py` | end-to-end use of fitted `A`, `C`, and `M` components in liability estimation |
| `bench_aod_decay.py` | **unsupported research:** onset-age-dependent genetic-correlation fitting under the fitted covariance model |
| `bench_aod_decay_robustness.py` | **unsupported research:** adversarial onset-age-decay probes for wrong kernels and unmodelled shared family environment |
| `bench_sex_limitation.py` | **unsupported research:** sex-specific covariance versus sex-specific thresholds, including matched same-sex/cross-sex mechanisms |
| `bench_nurture.py` | **unsupported research:** direct-effect scoring and moment-heritability distortion under genetic nurture |

`_common.py` holds the shared simulation, estimation, GWAS and plotting helpers,
plus a minimal PLINK `.bed` reader for the HAPNEST path.

## How the data are simulated

* **Family-only benchmarks** (accuracy, scaling, age-of-onset) draw genetic `g`,
  full liability `o`, and relatives jointly from the liability-threshold family
  covariance. The bounds and retained relative rows define classic LT-FH or an
  LT-FH++ component ablation; none of these rows is ADuLT.
  Because `g` is retained, accuracy is corr(estimate, `g`).
* **Fitter benchmarks** draw unascertained, population-sampled simulated
  families and pass `sampling="population"` explicitly. They do not validate
  fitting after case/control or family-history ascertainment.
* **The classic GWAS benchmark** uses parents plus one sibling and builds each proband's genetic liability from simulated
  causal-SNP genotypes, then draws the relatives' liabilities *conditional on that
  value* from the same covariance. This gives a genotype matrix to associate
  against and a correctly correlated family history to estimate from. LD is not
  needed for the power comparison (which turns on each phenotype's correlation to
  the true genetic value); pass `--plink` for real-LD HAPNEST genotypes if you
  want realistic multiple-testing structure.
* **The personalised LT-FH++ GWAS benchmark** adds coherent age, onset,
  competing mortality, birth-cohort effects, ascertainment, and stratified null
  variants. Its matched ADuLT row retains the full proband CIP but removes family
  history. The integrated panel measures the combined `++` design; its separate
  sex-isolation panel removes cohort and mortality-sex effects before comparing
  age-only with age+sex thresholds. The CSV identifies replicate, paired-contrast,
  and CIP-curve rows and records the material simulation and Gibbs configuration.

## Caveats

- These are **stochastic** benchmarks. Some cells are single simulated cohorts;
  others report means across independent cohorts. Re-running shifts both by
  sampling noise, so read the reported replicate counts and uncertainty where
  available.
- The Gibbs timings use the structure-grouped, Numba-parallel path. Set
  `NUMBA_NUM_THREADS` to fix Numba's worker count; optionally set
  `OMP_NUM_THREADS` too so linked numerical libraries use the same limit.
- Checked-in CSVs and [`RESULTS.md`](RESULTS.md) are historical artifacts.
  Their claims apply only to their recorded designs and provenance, not
  automatically to the current source tree.
