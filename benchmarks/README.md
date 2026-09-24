# ltpred benchmarks

Benchmarks for the two ltpred inference engines — the **Gibbs sampler** and
deterministic **Pearson–Aitken (PA)** — across LT-FH, LT-FH++ and ADuLT inputs.
The PA-FGRS censoring mixture has unit tests and a dedicated generative
benchmark (`bench_pafgrs_mixture.py`).
Bounds define the observation encoding; relative rows distinguish
LT-FH++ from family-free ADuLT. Statistical benchmarks simulate
their own data, so the true genetic liability is known; computational benchmarks
use matched synthetic workloads. Both run locally with no downloads (real-LD genotypes are an opt-in
[HAPNEST](hapnest/README.md) step).

Historical scaling and R-package comparisons use **four threads**; other
campaigns record their own thread counts. The time/memory comparison below uses
**one thread**. Pin the thread count explicitly rather than
letting Numba take every core: a speed-up is only interpretable alongside the
thread count it was measured at, because Gibbs is parallel while the PA object
path is largely serial.

Use the provenance wrapper for retained CSV/PNG benchmark outputs. It
appends one JSON object to `run_manifest.jsonl` with the clean source commit,
exact command, runtime stack, thread settings, machine profile, exit status, and
hashes of declared CSV/PNG artifacts:

```bash
python benchmarks/run_benchmark.py --artifact bench_accuracy.csv \
  --artifact bench_accuracy.png bench_accuracy.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python benchmarks/run_benchmark.py --artifact bench_scaling.csv \
    --artifact bench_scaling.png bench_scaling.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python benchmarks/run_benchmark.py --artifact bench_gwas_power.csv \
    --artifact bench_gwas_power.png bench_gwas_power.py
```

The wrapper ignores existing benchmark outputs when checking source cleanliness,
so several scripts can be rerun before committing one campaign. It refuses all
other source changes: commit those first, or use direct script invocation for an
exploratory run whose output will not be retained. Declare every retained output
with repeatable `--artifact`; this records its hash even when a deterministic
rerun reproduces the existing bytes exactly.

For external inputs, add repeatable `--input FILE` arguments; the wrapper hashes
them before the run and rejects a run if they change while it is executing. A
PLINK prefix therefore needs three declarations (`.bed`, `.bim`, and `.fam`).

Timing runs also want an otherwise-quiet machine. Record the load average beside
the command before quoting a number: `bench_scaling`'s current figures were
taken while the 1-minute load moved from 2.56 to 5.86 on a 10-core box. The lean
wrapper restores prospective provenance without the removed log/source archives.

Most scripts write a `.csv` and, if matplotlib is present, a `.png`. The
following scripts print focused diagnostics to stdout and also write a
summary CSV: `bench_cip_estimation.py`, `bench_liability_scale.py`, and
`bench_tetrachoric.py`. These still print only:
`bench_inference_calibration.py` and
`bench_misspecification.py`. `bench_pedigree_inference.py` and
`bench_register_pipeline.py` print the same style of diagnostics but also
write long-format `rep, metric, value` CSVs (`--reps` independent
populations/registers; rep 0 marks single-run parts).

Two gain metrics appear below. Independent-SNP marginal-association panels
report **causal-SNP NCP ratios**, based on `mean chi² - 1`. Prediction-only panels report
**squared-correlation effective-N proxies**. These answer related but different
questions and their magnitudes should not be compared as if they were the same
statistic. Every checked-in genotype-association result used independent SNPs,
non-overlapping simulated families, and the lightweight marginal score statistic
in `_common.py`; none is evidence from a real-LD, related-sample mixed-model GWAS.
The HAPNEST input path is opt-in and has not been run for the committed results.

**Runtime depends on your machine.** These reference numbers were taken on 10
cores with Numba installed (`pip install -e ".[fast]"`); the first call in each
script pays a one-off JIT compile. Expect substantially slower runs on fewer
cores, on a cold JIT cache, or without Numba (the pure-Python fallback is
numerically identical, just slower). Each script takes CLI flags (`--reps`,
`--n-fam`, …) to trade runtime for precision.

## Time and memory between versions

The latest committed comparison is [v0.7.1 versus v0.7.0](results/2026-09-23-time-memory-v071/README.md),
covering matched graph, PA and register-scoring workloads. These are
historical source-bound measurements, not timings of the current tree.

The earlier [v0.6.1 rerun against v0.6.0](results/2026-09-09-time-memory-v061-rerun/README.md)
covers two parent-graph sizes, four PA batches of 200,000 families, and a small
register-scoring workload. Mixed-mask PA and million-record graph construction
gained the most on warm calls, with lower RSS; the small register case barely
moved. All measured outputs agree exactly. The figures and measurement limits
are in [RESULTS.md §31](RESULTS.md#31-time-and-memory-v061-versus-v060).

Run the standalone JSON driver from the source version to be measured, using
a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
.venv/bin/python benchmarks/bench_time_memory.py \
  --baseline-ref 52dec5294c101d4730c86d3307091be75dc47a5e \
  --reps 5 --output /tmp/ltpred-time-memory
```

The driver uses separate runtime/RSS and allocation processes for each case and
version, archives both package sources, and checks output agreement and source
stability. First-call times include reached JIT compilation; warm medians use
five later calls. Allocation tracing runs separately after warmup, outside the
timing process. The AC-power/Low-Power-Mode guard must pass. This JSON campaign
does not use the CSV/PNG wrapper: retain its aggregate `results.json`, run log,
and before/after source provenance together, as in the linked capsule. That
capsule measured the clean v0.6.1 commit `b516271`; the documentation patch
v0.6.2 leaves the numerical implementation unchanged.

## Environments

- **`ltpred314`** — the CSV/PNG campaigns, and package verification (pytest,
  ruff, `mkdocs build --strict`). This is the free-threaded Python 3.14.6 build
  with NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0 and matplotlib, which is the
  environment [`RESULTS.md`](RESULTS.md) records for the stored artifacts.
  Reruns meant to be comparable with them belong here.
- **`.venv`** (the checkout's own) — the time/memory driver only, whose capsule
  records Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3 and Numba 0.67.0. Do not
  use it for the statistical campaigns: it has no matplotlib, so every figure
  is silently skipped, and its NumPy draws a different `multivariate_normal`
  stream, which shifts sampling-based benchmarks in the third decimal.
- **`ldpred3`** — the optional ldpred3 PGS backend used by the
  PGS-comparison arm.

## Scripts

Rows marked *research* import checkout-only APIs from `research/`; those APIs
are not installed as part of `ltpred`. Each script's module docstring gives its
design, arms and CLI flags. The tables group scripts by the question they
answer; the RESULTS.md section order is historical and does not follow these
groups.

**Score accuracy, calibration and robustness**

| Script | Measures | Output |
|---|---|---|
| `bench_accuracy.py` | Gibbs vs PA corr(estimate, true g), calibration slope, RMSE and effective-N proxy over h² × prevalence × structure | `bench_accuracy.{csv,png}` |
| `bench_calibration.py` | calibration slope/intercept and decile curve; effect of a wrong assumed h² | `bench_calibration.{csv,png}` |
| `bench_misspecification.py` | calibration and ranking under heavy tails, assortative mating, unmodelled C, wrong prevalence | stdout |
| `bench_pa_robustness.py` | PA–Gibbs agreement on stressful pedigrees and fold-order sensitivity, three seeds per cell | `bench_pa_robustness.{csv,png}` |
| `bench_pafgrs_mixture.py` | PA-FGRS mixture under threshold-crossing, stochastic and liability-dependent onset; paired contrasts | `bench_pafgrs_mixture.csv` |

**GWAS association gains**

| Script | Measures | Output |
|---|---|---|
| `bench_gwas_power.py` | classic LT-FH independent-SNP causal-NCP ratio, power and λ_GC vs case/control (`--plink` for HAPNEST real LD) | `bench_gwas_power.{csv,png}` |
| `bench_ltfhpp_personalization.py` | integrated LT-FH++ association simulation (age/sex/cohort CIP, mortality, ascertainment) with a matched ADuLT arm and a sex-isolation panel | `bench_ltfhpp_personalization.{csv,png}` |
| `bench_confounding.py` | λ_GC under a secular prevalence trend, cohort-blind vs cohort-aware thresholds | `bench_confounding.{csv,png}` |
| `bench_pgs_comparison.py` | PGS baseline and the PGS + LT-FH joint model with a train/test split and cross-fitted OLS (`--pgs-backend ldpred3` optional) | `bench_pgs_comparison.{csv,png}` |

**Fitting h² and variance components**

| Script | Measures | Output |
|---|---|---|
| `bench_fit_heritability.py` | bias, precision and `h2_se` calibration of `fit_heritability` | `bench_fit_heritability.{csv,png}` |
| `bench_variance_components.py` | recovery of A and C by `fit_variance_components`, boundary behaviour, precision vs N | `bench_variance_components.{csv,png}` |
| `bench_pairwise_multi.py` | joint h²/rg/re recovery, cluster-SE coverage, shared C/M, MCAR, IPW and correlation nulls; [development evidence](results/2026-09-16-joint-pairwise/README.md) | source snapshot + replicate/summary JSON in `--output` directory |
| `bench_pairwise_recovery.py` | `fit_pairwise` recovery of A/C/M, sandwich-SE calibration, coverage, boundary pinning | `bench_pairwise_recovery.{csv,png}` |
| `bench_shared_env.py` | value of modelling C (ignore-C vs fit A+C vs oracle); panel (c) end-to-end `c2`/`m2` wiring | `bench_shared_env.{csv,png}` |
| `bench_couple_env.py` | recovery of A+M and the C-vs-M omission contrast | `bench_couple_env.{csv,png}` |
| `bench_ascertainment.py` | what the moment fitters return on selected samples and what IPW recovers (`--h2 0 --tag _h2null --arms h2 mechanism --structures nuclear --n-fam 10000 --n-iter 1500 --burn-in 500` for the h² = 0 cell) | `bench_ascertainment*.{csv,png}` |
| `bench_inference_calibration.py` | *research:* component-test Type-I error, bootstrap coverage, MCEM SEs, `test_genetic_correlation` null (`--parts`) | stdout |

**Research-model fits**

| Script | Measures | Output |
|---|---|---|
| `bench_genetic_correlation.py` | *research:* `fit_genetic_correlation` bias and SD including the null; panel (c) `fit_genetic_factor` loadings and `srmr` | `bench_genetic_correlation.{csv,png}` |
| `bench_aod_decay.py` | *research:* onset-age-dependent r_g fitting; `--robustness` adds wrong kernels and unmodelled C | `bench_aod_decay.{csv,png}` |
| `bench_covariance_extensions.py` | *research:* sex-limited covariance vs sex-specific thresholds (a); direct-effect scoring under genetic nurture (b) | `bench_sex_limitation.{csv,png}`, `bench_nurture.{csv,png}` |

**End-to-end pipelines and input handling**

| Script | Measures | Output |
|---|---|---|
| `bench_cip_estimation.py` | KM/AJ curve recovery, delayed entry, competing mortality, CIP-to-score check | `bench_cip_estimation.csv` |
| `bench_tetrachoric.py` | tetrachoric relative-pair correlations and the Falconer diagnostic | `bench_tetrachoric.csv` |
| `bench_liability_scale.py` | probit residual-scale variance and observed ↔ liability transformations | `bench_liability_scale.csv` |
| `bench_pedigree_inference.py` | pedigree extraction fidelity, kinship vs role-grammar scoring, extraction throughput | `bench_pedigree_inference.csv` |
| `bench_register_pipeline.py` | public trio-register pipeline: accuracy, CIP and calendar-prospective arms, AUC, calibration, throughput | `bench_register_pipeline.csv` |
| `bench_fh_prediction.py` | registry simulation: classic vs personalised family bounds, family-free ADuLT cohort span, onset-encoding ablation (panel (e)) | `bench_fh_prediction.{csv,png}` |

**Cross-package and computational**

| Script | Measures | Output |
|---|---|---|
| `bench_ltfhplus_compare.py` | **opt-in** lock against LTFHPlus Gibbs and LTFGRS PA: scores, per-family time, fold times, isolated peak RSS (exits 2 without R) | `bench_ltfhplus_compare*.csv` |
| `bench_scaling.py` | wall-clock scaling with #families and family size; families/s and speed-up | `bench_scaling.{csv,png}` |
| `bench_time_memory.py` | matched package versions: first-call and warm runtime, process RSS and call-allocation peaks for graph construction, PA batching and register scoring, with exact output agreement | JSON capsule (`--output`) |

`_common.py` holds the shared simulation, estimation, GWAS, replicate-summary,
CSV-writing and plotting helpers, plus a minimal PLINK `.bed` reader for the
HAPNEST path.

## How the data are simulated

* **Family-only benchmarks** (accuracy, scaling, age-of-onset) draw genetic `g`,
  full liability `o`, and relatives jointly from the liability-threshold family
  covariance. The bounds and retained relative rows define classic LT-FH or an
  LT-FH++ component ablation; none of these rows is ADuLT.
  Because `g` is retained, accuracy is corr(estimate, `g`). The age-of-onset
  script scores classic, interval, and pinned encodings on the same families.
  The mixture script draws onset under threshold-crossing, stochastic, or
  liability-dependent (`onset_rho=0.6`) models.
* **Fitter benchmarks** draw unascertained, population-sampled simulated
  families and pass `sampling="population"` explicitly. `bench_ascertainment.py`
  is the exception and the complement: it measures what those fitters return
  when the contract is violated, and what `sampling="ipw"` recovers.
* **The classic independent-SNP association benchmark** uses parents plus one sibling and builds each proband's genetic liability from simulated
  causal-SNP genotypes, then draws the relatives' liabilities *conditional on that
  value* from the same covariance. This gives a genotype matrix to associate
  against and a correctly correlated family history to estimate from. LD is not
  needed for the power comparison (which turns on each phenotype's correlation to
  the true genetic value); pass `--plink` for real-LD HAPNEST genotypes if you
  want realistic multiple-testing structure. The association helper remains a
  marginal score calculation, and the simulation does not represent overlapping
  extracted pedigrees.
* **The personalised LT-FH++ independent-SNP association benchmark** adds coherent age, onset,
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
  automatically to the current source tree. `scripts/check_evidence.py`
  reconciles the release-defining scaling, IPW, R-lock and PGS claims with their
  stored artifacts; verify other numbers against their CSVs when quoting them.
- CI never re-runs a benchmark, by design: they are minutes to hours of
  stochastic work whose value is the recorded provenance. The `docs` job runs
  `scripts/check_evidence.py`, which checks the committed artifacts against the
  prose, not against the current code. Re-run a script locally before quoting
  its number as current.
