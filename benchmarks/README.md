# ltpred benchmarks

Benchmarks for the two ltpred fitting methods — the **Gibbs sampler** (LT-FH++)
and the deterministic **Pearson–Aitken** estimator (PA-FGRS) — inspired by the
comparisons made in the LT-FH++, ADuLT and PA-FGRS papers. Everything simulates
its own data, so the true genetic liability is known and the benchmarks run
locally with no downloads (the one exception, real-LD genotypes, is an opt-in
[HAPNEST](hapnest/README.md) step).

Run single-core-ish for reproducible timings, or let Numba use all cores for
speed:

```bash
python benchmarks/bench_accuracy.py
python benchmarks/bench_scaling.py
OMP_NUM_THREADS=10 python benchmarks/bench_gwas_power.py
```

Each script writes a `.csv` and (if matplotlib is present) a `.png`.

**Runtime depends on your machine.** These reference numbers were taken on 10
cores with Numba installed (`pip install -e ".[fast]"`); the first call in each
script pays a one-off JIT compile. Expect substantially slower runs on fewer
cores, on a cold JIT cache, or without Numba (the pure-Python fallback is
numerically identical, just slower). Each script takes CLI flags (`--reps`,
`--n-fam`, …) to trade runtime for precision.

## Scripts

| Script | What it measures |
|--------|------------------|
| `bench_accuracy.py` | corr(estimated genetic liability, true g) across heritability × prevalence × family structure — Gibbs vs PA-FGRS accuracy, calibration slope, RMSE and the effective-N gain over case/control (→ `bench_accuracy.{csv,png}`) |
| `bench_scaling.py` | wall-clock scaling of both methods with #families and family size; families/second and speed-up (→ `bench_scaling.{csv,png}`) |
| `bench_age_onset.py` | value of the liability→age-of-onset map (ADuLT/LT-FH++): plain case/control vs onset-pinned cases, fit with PA (and Gibbs, to confirm agreement); accuracy and eff-N gain vs prevalence (→ `bench_age_onset.{csv,png}`) |
| `bench_gwas_power.py` | genotype-based GWAS power: case/control vs LT-FH++ vs PA-FGRS vs an oracle — mean χ² at causal SNPs (effective N), detection power, and λ_GC calibration at nulls (→ `bench_gwas_power.{csv,png}`). Pass `--plink PREFIX` for **real-LD** HAPNEST genotypes (opt-in; see [`hapnest/README.md`](hapnest/README.md)) |
| `bench_fit_heritability.py` | variance-component inference: bias and across-dataset precision of `fit_heritability` vs true h², vs #families and family structure, and the calibration of the reported `h2_se` (→ `bench_fit_heritability.{csv,png}`) |
| `bench_variance_components.py` | multi-component inference: recovery of additive `A` and common-environment `C` by `fit_variance_components` (bias & across-dataset SD), the false-positive `C` on purely additive data, and precision vs #families (→ `bench_variance_components.{csv,png}`) |
| `bench_genetic_correlation.py` | genetic-correlation inference: bias & across-dataset SD of `r_g` from `fit_genetic_correlation` vs the true value — including the null (`r_g=0` with non-zero phenotypic correlation) — and precision vs #families (→ `bench_genetic_correlation.{csv,png}`) |

`_common.py` holds the shared simulation, estimation, GWAS and plotting helpers,
plus a minimal PLINK `.bed` reader for the HAPNEST path.

## How the data are simulated

* **Family-only benchmarks** (accuracy, scaling, age-of-onset) draw whole
  families straight from the LT-FH++ covariance — genetic `g`, full `o` and
  relatives jointly multivariate normal — and threshold them. `g` is the ground
  truth, so accuracy is just corr(estimate, `g`).
* **The GWAS benchmark** builds each proband's genetic liability from simulated
  causal-SNP genotypes, then draws the relatives' liabilities *conditional on that
  value* from the same covariance. This gives a genotype matrix to associate
  against and a correctly correlated family history to estimate from. LD is not
  needed for the power comparison (which turns on each phenotype's correlation to
  the true genetic value); pass `--plink` for real-LD HAPNEST genotypes if you
  want realistic multiple-testing structure.

## Caveats

- These are **stochastic** benchmarks: every number is one Monte-Carlo draw, so
  re-running shifts values by sampling noise. The qualitative conclusions in
  [`RESULTS.md`](RESULTS.md) are stable.
- The Gibbs timings use the structure-grouped, Numba-parallel path; set
  `OMP_NUM_THREADS` to fix the core count for comparable numbers.
