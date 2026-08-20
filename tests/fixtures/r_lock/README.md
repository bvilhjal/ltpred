# Locked R-package reference scores

`input_tbl.csv` holds 48 simulated classic LT-FH families (fam_vec `m`, `f`,
`s1`; h2 = 0.5, K = 0.05; `simulate_under_LTM_single(..., seed=20260816)`).
The two score files are the R packages' own outputs on that table, so a
numerical regression in the Python port fails in pytest instead of waiting
for the next opt-in benchmark rerun:

- `ltfhplus_gibbs.csv` — LTFHPlus 2.2.0 `estimate_liability` (Gibbs, tol
  0.01, 1 worker, R seed 20260816); the fixture also records the R version and
  RNG kind used to generate it;
- `ltfgrs_pa.csv` — LTFGRS 1.0.1 `estimate_liability(method = "PA",
  useMixture = FALSE)`.

Regenerate from the repository root (needs R with both packages installed):

```bash
Rscript benchmarks/ltfhplus_compare.R tests/fixtures/r_lock/input_tbl.csv \
    tests/fixtures/r_lock/ltfhplus_gibbs.csv 0.5 0.01 1 20260816
Rscript benchmarks/ltfgrs_compare.R tests/fixtures/r_lock/input_tbl.csv \
    tests/fixtures/r_lock/ltfgrs_pa.csv 0.5 0.01 1 PA
```
