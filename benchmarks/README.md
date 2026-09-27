# ltpred benchmarks

Benchmarks for the two ltpred inference engines — the **Gibbs sampler** and
deterministic **Pearson–Aitken (PA)** — across LT-FH, LT-FH++ and ADuLT inputs,
plus the fitters, the register pipeline and the PA-FGRS censoring mixture.
Statistical benchmarks simulate their own data, so the true genetic liability
is known; computational benchmarks use matched synthetic workloads. Both run
locally with no downloads (real-LD genotypes are an opt-in
[HAPNEST](hapnest/README.md) step). Results are in [RESULTS.md](RESULTS.md).

Pin the thread count explicitly rather than letting Numba take every core: a
speed-up is only interpretable alongside its thread count, because Gibbs is
parallel while the PA object path is largely serial. The scaling and R-package
comparisons use **four threads**, the time/memory driver **one**; other
campaigns record their own.

Use the provenance wrapper for retained benchmark CSVs. It appends one JSON
object to `run_manifest.jsonl` with the clean source commit, exact command,
runtime stack, thread settings, machine profile, exit status, and hashes of the
declared CSV artifacts; commit that row with the regenerated artifact:

```bash
python benchmarks/run_benchmark.py --artifact bench_accuracy.csv bench_accuracy.py
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python benchmarks/run_benchmark.py --artifact bench_scaling.csv bench_scaling.py
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
Timing runs also want an otherwise-quiet machine: record the load average
beside the command before quoting a number.

Each script's committed artifact is the `.csv` in the Output column of the
[Scripts](#scripts) tables. With matplotlib present most scripts also draw a
`.png` figure; figures stay local (`benchmarks/*.png` is gitignored) and are
regenerated from a rerun, so the older `run_manifest.jsonl` rows that hash a PNG
describe files no longer in the repository. The two gain metrics
(causal-SNP NCP ratio versus squared-correlation eff-N proxy) and the
independent-SNP scope of every association result are defined once, in the
[RESULTS.md preamble](RESULTS.md).

## Time and memory between versions

`bench_time_memory.py` compares two package versions on matched graph, PA and
register-scoring workloads. Its capsules are historical source-bound
measurements, not timings of the current tree. **Known gap:** no time/memory
capsule spans v0.7.1 → v0.7.4.

**Table 1. Every committed capsule in [`results/`](results/)** (JSON
campaigns, statistical ones included). Each README gives the design,
environment, limits and reproduction command.

| capsule | measures | retained files | cited in |
|---|---|---|---|
| [2026-09-09-time-memory-v061-rerun](results/2026-09-09-time-memory-v061-rerun/README.md) | driver: v0.6.1 `b516271` vs v0.6.0 `52dec52` | `results.json`, `provenance.json`, `run.log` | [RESULTS.md §31](RESULTS.md#31-time-and-memory-v061-versus-v060); pinned by `scripts/check_evidence.py` |
| [2026-09-16-joint-pairwise](results/2026-09-16-joint-pairwise/README.md) and [-nulls](results/2026-09-16-joint-pairwise-nulls/README.md) | `bench_pairwise_multi.py` statistical evidence, development checkout (v0.7.0 `d4aa91e` sources) | `manifest.json`, `replicates.json`, `summary.json`; main run also `bench_pairwise_multi-vs-d4aa91e.diff` | RESULTS.md §33 |
| [2026-09-23-lean-v072](results/2026-09-23-lean-v072/README.md) | chunked-kernel allocation on synthetic probes, v0.7.2 candidate vs v0.7.1 `8926f25`; no general speedup claim | `measure.py`, `change.patch`, `results.json`, `baseline.npz`, before/after time and memory JSONs | — |
| [2026-09-23-time-memory-v071](results/2026-09-23-time-memory-v071/README.md) | driver: v0.7.1 `5b42b13` vs `ac78ba3` (v0.7.0 plus documentation) | `results.json`, `provenance.json`, `run.log` | RESULTS.md §31a |
| [2026-09-24-review](results/2026-09-24-review/README.md) | v0.7.3 microbenchmarks (quadrature moments, bound-row dedup, pid normalisation, Mendelian simulation), `db2377f` vs `6686541` + patch; shared host, not a speed ranking or an end-to-end register measurement | `probe.py`, before/after JSONs, `comparison.json`, `mendelian-{400,8000}.json`, `after-vs-6686541.patch` | — |

The design rationale and measured folds behind `_selected_kinship`,
`pairwise.py` and `quadrature.py` come from an external efficiency review of
2026-09-05 whose capsule is not in this repository.

Run the driver from the source version to be measured, with that version's
interpreter (the v0.7.1 capsule used `ltpred314`) and a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python benchmarks/bench_time_memory.py \
  --baseline-ref <baseline commit SHA> \
  --reps 5 --output /tmp/ltpred-time-memory
```

The driver uses separate runtime/RSS and allocation processes per case and
version, saves both package sources, checks output agreement and source
stability, and on macOS requires the AC-power/Low-Power-Mode guard to pass
(elsewhere it records `not_applicable`); the timing
definitions are in each capsule README. It does not use the CSV wrapper:
retain `results.json`, the provenance record and the run log together.

## Environments

- **`ltpred314`** — the CSV campaigns, the time/memory driver since v0.7.1,
  and package verification (pytest, ruff, `mkdocs build --strict`). This is the
  free-threaded Python 3.14.6 build with NumPy 2.4.6, SciPy 1.18.0, Numba
  0.66.0 and matplotlib, the environment [`RESULTS.md`](RESULTS.md) records for
  the stored artifacts. Reruns meant to be comparable with them belong here.
- **`.venv`** (the checkout's own) — measured the v0.6.1 time/memory capsule
  (Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, Numba 0.67.0) but no longer
  imports SciPy on the reference macOS host, so the v0.7.1 capsule used
  `ltpred314`. Never use it for the statistical campaigns: it has no
  matplotlib (every figure is silently skipped), and its NumPy draws a
  different `multivariate_normal` stream, which shifts sampling-based
  benchmarks in the third decimal.
- **`ldpred3`** — the optional ldpred3 PGS backend used by the
  PGS-comparison arm.

## Scripts

Rows marked *research* import checkout-only APIs from `research/`; those APIs
are not installed as part of `ltpred`. Each script's module docstring gives its
design, arms and CLI flags. The tables group scripts by the question they
answer; the RESULTS.md section order is historical and does not follow these
groups. Scripts whose Output is `stdout` archive no artifact.


**Score accuracy, calibration and robustness**

| Script | Measures | Output |
|---|---|---|
| `bench_accuracy.py` | Gibbs vs PA corr(estimate, true g), calibration slope, RMSE and effective-N proxy over h² × prevalence × structure | `bench_accuracy.csv` |
| `bench_calibration.py` | calibration slope/intercept and decile curve; effect of a wrong assumed h² | `bench_calibration.csv` |
| `bench_misspecification.py` | calibration and ranking under heavy tails, assortative mating, unmodelled C, wrong prevalence | stdout |
| `bench_pa_robustness.py` | PA–Gibbs agreement on stressful pedigrees and fold-order sensitivity, three seeds per cell | `bench_pa_robustness.csv` |
| `bench_pafgrs_mixture.py` | PA-FGRS mixture under threshold-crossing, stochastic and liability-dependent onset; paired contrasts | `bench_pafgrs_mixture.csv` |

**GWAS association gains**

| Script | Measures | Output |
|---|---|---|
| `bench_gwas_power.py` | classic LT-FH independent-SNP causal-NCP ratio, power and λ_GC vs case/control (`--plink` for HAPNEST real LD) | `bench_gwas_power.csv` |
| `bench_ltfhpp_personalization.py` | integrated LT-FH++ association simulation (age/sex/cohort CIP, mortality, ascertainment) with a matched ADuLT arm and a sex-isolation panel | `bench_ltfhpp_personalization.csv` |
| `bench_confounding.py` | λ_GC under a secular prevalence trend, cohort-blind vs cohort-aware thresholds | `bench_confounding.csv` |
| `bench_pgs_comparison.py` | PGS baseline and the PGS + LT-FH joint model with a train/test split and cross-fitted OLS (`--pgs-backend ldpred3` optional) | `bench_pgs_comparison.csv` |

**Fitting h² and variance components**

| Script | Measures | Output |
|---|---|---|
| `bench_fit_heritability.py` | bias, precision and `h2_se` calibration of `fit_heritability` | `bench_fit_heritability.csv` |
| `bench_variance_components.py` | recovery of A and C by `fit_variance_components`, boundary behaviour, precision vs N | `bench_variance_components.csv` |
| `bench_pairwise_multi.py` | joint h²/rg/re recovery, cluster-SE coverage, shared C/M, MCAR, IPW and correlation nulls; [development evidence](results/2026-09-16-joint-pairwise/README.md) | source snapshot + replicate/summary JSON in `--output` directory |
| `bench_pairwise_recovery.py` | `fit_pairwise` recovery of A/C/M, sandwich-SE calibration, coverage, boundary pinning | `bench_pairwise_recovery.csv` |
| `bench_shared_env.py` | value of modelling C (ignore-C vs fit A+C vs oracle); panel (c) end-to-end `c2`/`m2` wiring | `bench_shared_env.csv` |
| `bench_couple_env.py` | recovery of A+M and the C-vs-M omission contrast | `bench_couple_env.csv` |
| `bench_ascertainment.py` | *research (arm C):* what the moment fitters return on selected samples and what IPW recovers (`--h2 0 --tag _h2null --arms h2 mechanism --structures nuclear --n-fam 10000 --n-iter 1500 --burn-in 500` for the h² = 0 cell) | `bench_ascertainment*.csv` |
| `bench_inference_calibration.py` | *research:* component-test Type-I error, bootstrap coverage, MCEM SEs, `test_genetic_correlation` null (`--parts`) | stdout |

**Research-model fits**

| Script | Measures | Output |
|---|---|---|
| `bench_genetic_correlation.py` | *research:* `fit_genetic_correlation` bias and SD including the null; panel (c) `fit_genetic_factor` loadings and `srmr` | `bench_genetic_correlation.csv` |
| `bench_aod_decay.py` | *research:* onset-age-dependent r_g fitting; panel (d), on by default (`--no-robustness` skips it), adds a wrong-kernel arm and unmodelled C | `bench_aod_decay.csv` |
| `bench_covariance_extensions.py` | *research:* sex-limited covariance vs sex-specific thresholds (a); direct-effect scoring under genetic nurture (b) | `bench_sex_limitation.csv`, `bench_nurture.csv` |

**End-to-end pipelines and input handling**

| Script | Measures | Output |
|---|---|---|
| `bench_cip_estimation.py` | KM/AJ curve recovery, delayed entry, competing mortality, CIP-to-score check | `bench_cip_estimation.csv` |
| `bench_tetrachoric.py` | tetrachoric relative-pair correlations and the Falconer diagnostic | `bench_tetrachoric.csv` |
| `bench_liability_scale.py` | probit residual-scale variance and observed ↔ liability transformations | `bench_liability_scale.csv` |
| `bench_pedigree_inference.py` | pedigree extraction fidelity, kinship vs role-grammar scoring, extraction throughput | `bench_pedigree_inference.csv` |
| `bench_register_pipeline.py` | public trio-register pipeline: accuracy, CIP and calendar-prospective arms, AUC, calibration, throughput | `bench_register_pipeline.csv` |
| `bench_fh_prediction.py` | registry simulation: classic vs personalised family bounds, family-free ADuLT cohort span, onset-encoding ablation (panel (e)) | `bench_fh_prediction.csv` |

**Cross-package and computational**

| Script | Measures | Output |
|---|---|---|
| `bench_ltfhplus_compare.py` | **opt-in** lock against LTFHPlus Gibbs and LTFGRS PA: scores, per-family time, fold times, isolated peak RSS (exits 2 without R or LTFHPlus) | `bench_ltfhplus_compare*.csv` |
| `bench_scaling.py` | wall-clock scaling with #families and family size; families/s and speed-up | `bench_scaling.csv` |
| `bench_time_memory.py` | matched package versions: first-call and warm runtime, process RSS and call-allocation peaks for graph construction, PA batching and register scoring, with exact output agreement | JSON capsule (`--output`) |

`_common.py` holds the shared simulation, estimation, GWAS, replicate-summary,
CSV-writing and plotting helpers, plus a minimal PLINK `.bed` reader for the
HAPNEST path.

## How the data are simulated

* **Family-only benchmarks** (accuracy, scaling, the `bench_fh_prediction.py`
  onset-encoding panel (e), PA-FGRS mixture)
  draw genetic `g`, full liability `o` and relatives jointly from the
  liability-threshold family covariance; because `g` is retained, accuracy is
  corr(estimate, `g`). None of these rows is ADuLT.
* **Fitter benchmarks** draw unascertained, population-sampled families and
  pass `sampling="population"` explicitly; `bench_ascertainment.py` measures
  what those fitters return when that contract is violated, and what
  `sampling="ipw"` recovers.
* **Association benchmarks** build each proband's genetic liability from
  simulated independent causal-SNP genotypes and draw the relatives
  *conditional on that value*, giving both a genotype matrix and a correctly
  correlated family history. `--plink` substitutes real-LD HAPNEST genotypes;
  the association helper remains a marginal score calculation, and the
  simulation does not represent overlapping extracted pedigrees. The
  personalised LT-FH++ benchmark adds coherent age, onset, competing mortality,
  birth-cohort effects, ascertainment and stratified null variants.

## Caveats

- These are **stochastic** benchmarks. Some cells are single simulated cohorts;
  others report means across independent cohorts. Read the reported replicate
  counts and uncertainty; a rerun shifts both by sampling noise.
- **Runtime depends on the machine.** The reference numbers were taken on 10
  cores with Numba installed (`pip install -e ".[fast]"`); the first call in
  each script pays a one-off JIT compile. Expect slower runs on fewer cores, on
  a cold JIT cache, or without Numba (the pure-Python fallback is numerically
  identical). CLI flags (`--reps`, `--n-fam`, …) trade runtime for precision.
- Gibbs timings use the structure-grouped, Numba-parallel path: set
  `NUMBA_NUM_THREADS`, and `OMP_NUM_THREADS` for linked numerical libraries.
- Checked-in CSVs and [`RESULTS.md`](RESULTS.md) are historical artifacts whose
  claims apply to their recorded designs and provenance, not automatically to
  the current source tree. CI never reruns a benchmark; the `docs` job runs
  `scripts/check_evidence.py`, which checks committed artifacts against prose,
  not against the current code. It pins the scaling tables (and the CSV's
  content hash, plus the array-API prose in `docs/estimation.md`), the PA
  robustness floor, the IPW recovery, the R-package lock and its thread
  provenance, the PGS joint model, every cell of the `gwas_power`,
  `confounding` and `fit_heritability` paper tables, the v0.6.1 time/memory
  capsule (revisions, source hashes and its seven rows), the SHA-256,
  one-thread and power-guard records of every `results.json` capsule that
  carries them (the two time/memory driver capsules; `2026-09-23-lean-v072`
  records none), and the
  tracked report PDF. Verify any other number against its CSV, and rerun the
  script locally before quoting it as current.
