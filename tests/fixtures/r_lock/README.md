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

## LT-FH++ (age-of-onset) cohort

`input_tbl_age_pin.csv` and `input_tbl_age_interval.csv` hold the *same* 48
simulated families (fam_vec `m`, `f`, `s1`; h2 = 0.5, K = 0.20;
`simulate_under_LTM_single(..., use_age=True, onset_resolution=None,
seed=20260912)`), differing only in how the 16 observed cases are written:

- `_pin`: `lower == upper == T(onset)`, the point pin that `age_thresholds`
  and `thresholds_from_cip(case_mode="pin")` emit -- LTFHPlus's
  `prepare_LTFHPlus_input(use_fixed_case_thr = TRUE)`;
- `_interval`: `(T(onset), Inf)`, LTFHPlus's default
  (`use_fixed_case_thr = FALSE`) and `thresholds_from_cip(case_mode="interval")`.

The classic table above has no pinned row and six case rows, so it exercised
neither age encoding. Reference outputs, same R packages and settings as
above (LTFHPlus seed 20260912):

```bash
for enc in pin interval; do
  Rscript benchmarks/ltfhplus_compare.R tests/fixtures/r_lock/input_tbl_age_$enc.csv \
      tests/fixtures/r_lock/ltfhplus_gibbs_age_$enc.csv 0.5 0.01 1 20260912
  Rscript benchmarks/ltfgrs_compare.R tests/fixtures/r_lock/input_tbl_age_$enc.csv \
      tests/fixtures/r_lock/ltfgrs_pa_age_$enc.csv 0.5 0.01 1 PA
done
```

Agreement at generation (ltpred 0.6.2, R 4.6.0): PA vs LTFGRS corr 0.999998 /
RMSE 6.7e-4 (pin; the residual is ltpred's joint pin conditioning versus
LTFGRS's sequential fold) and 0.9999998 / 1.8e-4 (interval); Gibbs vs
LTFHPlus corr 0.9999 / RMSE 3.6e-3 (pin) and 3.5e-3 (interval).
