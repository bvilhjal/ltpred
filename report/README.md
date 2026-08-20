# Technical report

`ltpred_methods.pdf` is a methods note for colleagues and PhD
students who already know GWAS, pedigree \(h^2\), and Falconer's
threshold model. It fixes the estimand, the observation-model
menu, the PA exactness boundary, the implementation (grouping, collapsed untruncated coordinates,
streaming batch-means, object vs array path, `var` vs `se`), and
the load-bearing simulation tables — agreement and speed, the
LTFHPlus / LTFGRS lock (total and per-family time, fold times versus
both R packages, isolated-process peak RSS), GWAS NCP, personalisation,
cohort confounding, calibration,
PGS complementarity, the censoring mixture (including
liability-dependent onset), pin versus interval encodings, and
ascertainment.

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd report
tectonic -X compile ltpred_methods.tex
cd ..
python scripts/check_evidence.py
```

Six load-bearing report tables are static LaTeX inputs derived from the
committed benchmark CSVs (the full generator was removed in the 2026-08
lean-down). The compact evidence check covers the release-defining rows and
also rejects a report source or included table newer than the tracked PDF.

The PDF is tracked as repository documentation. The lean source distribution
omits the report because it is not a runtime dependency of the installed
package.
