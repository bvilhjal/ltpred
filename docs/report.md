# Technical report

A methods note aimed at colleagues who already work with GWAS and
the Falconer threshold model. It is tracked in the repository, not in the lean
source distribution or this HTML site:

- PDF: [`report/ltpred_methods.pdf`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
- LaTeX source: [`report/ltpred_methods.tex`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.tex)

Rebuild with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
python scripts/make_results.py
cd report
tectonic -X compile ltpred_methods.tex
```

The cross-benchmark headline, GWAS, integrated-personalisation, confounding, PGS
and heritability tables are generated LaTeX inputs from the committed benchmark
CSVs. A repository test checks that the inputs are current.

The report states the estimand and assumptions first, then the
liability-threshold model and BLUP identity, both inference engines,
the implementation (grouping, collapsed untruncated coordinates,
streaming batch-means, object vs array path, posterior `var` vs
estimator `se`), the population/IPW sampling
contracts, and the load-bearing simulation tables — agreement
and speed, GWAS NCP, personalisation, cohort confounding, calibration,
PGS complementarity, the censoring mixture (including
liability-dependent onset), pin versus interval encodings, and
ascertainment.
Those tables are historical snapshots; see
[`benchmarks/RESULTS.md`](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for provenance.
