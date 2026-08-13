# Technical report

`ltpred_methods.pdf` is a methods note for colleagues and PhD
students who already know GWAS, pedigree \(h^2\), and Falconer's
threshold model. It fixes the estimand, the observation-model
menu, the PA exactness boundary, and what the simulations actually
show.

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd report
tectonic -X compile ltpred_methods.tex
```

The PDF is included in the source distribution (`MANIFEST.in`). It is
documentation, not a runtime dependency of the installed package.
