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
| `bench_genetic_factor.py` | genetic factor model `r_g ≈ ΛΛ' + Ψ` (`fit_genetic_factor`): end-to-end recovery of planted single-factor loadings from family case/control data, and whether the `srmr` fit index flags a one-factor model as too few when the truth has two genetic factors (→ `bench_genetic_factor.{csv,png}`) |
| `bench_calibration.py` | **calibration** of the genetic-liability estimate, not just its ranking: the calibration slope/intercept and decile calibration curve of true `g` on the estimate (a correctly-specified posterior mean is self-calibrating, slope ≈ 1), and how a **wrong assumed `h²`** leaves the ranking (`corr`) robust but tilts the scale (slope) — the complement to `liability_sensitivity` (→ `bench_calibration.{csv,png}`) |
| `bench_shared_env.py` | value of modelling shared environment `C`: corr(genetic-liability estimate, true genetic liability) when families are simulated under `A+C+E`, comparing ignore-C (additive) vs fit-`A+C` vs oracle, swept over `c²` and sib-ship size (→ `bench_shared_env.{csv,png}`) |
| `bench_couple_env.py` | the couple/spousal environment `M`: recovery of `A+M` (bias & across-dataset SD, no spurious `M` at `m²=0`), and the identifiability contrast — ignoring a real `C` inflates additive-only `Â` while ignoring a real `M` leaves it essentially unbiased (mates have `A=0`) (→ `bench_couple_env.{csv,png}`) |
| `bench_fh_prediction.py` | the LT-FH++ GWAS phenotype `E[g \| self+family(+age+cohort)]` vs case/control, on an **age-, cohort-, and mortality-consistent registry** simulation (onset when liability crosses the age-declining, birth-cohort-shifted CIP threshold; **death a competing risk** so relatives are observed up to `min(death, now)`; generationally-consistent ages), swept over **ascertainment**, **heritability**, and **secular prevalence trend**; family-history gain over case/control is largest for rare observed disease (2.4× eff-N), age-of-onset gain grows with ascertainment, and the birth-cohort correction helps both calibration (removes a bias) and ranking/power (up to ~1.6× eff-N when cases span wide cohorts) — the Pedersen 2022/2023 directions (→ `bench_fh_prediction.{csv,png}`) |

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
