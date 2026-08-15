# Technical report

`ltpred_methods.pdf` is a methods note for colleagues and PhD
students who already know GWAS, pedigree \(h^2\), and Falconer's
threshold model. It fixes the estimand, the observation-model
menu, the PA exactness boundary, the implementation (grouping,
streaming batch-means, object vs array path, `var` vs `se`), and
the load-bearing simulation tables — agreement and speed, GWAS
NCP, personalisation, cohort confounding, calibration, PGS
complementarity, the censoring mixture (including liability-dependent
onset), pin versus interval encodings, and ascertainment.

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
python scripts/make_results.py
cd report
tectonic -X compile ltpred_methods.tex
```

Six load-bearing report tables are LaTeX inputs generated directly from the
committed benchmark CSVs. `tests/test_check_results.py` fails if those inputs
are stale, so their cells should not be transcribed into the report by hand.

The PDF is tracked as repository documentation. The lean source distribution
omits the report because it is not a runtime dependency of the installed
package.
