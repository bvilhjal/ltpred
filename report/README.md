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
cd report
tectonic -X compile ltpred_methods.tex
```

The PDF is included in the source distribution (`MANIFEST.in`). It is
documentation, not a runtime dependency of the installed package.
