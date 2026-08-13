# Technical report

A methods note aimed at colleagues who already work with GWAS and
the Falconer threshold model. It is in the repository (and in the
source distribution), not in this HTML site:

- PDF: [`report/ltpred_methods.pdf`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
- LaTeX source: [`report/ltpred_methods.tex`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.tex)

Rebuild with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd report
tectonic -X compile ltpred_methods.tex
```

The report states the estimand and assumptions first, then the
liability-threshold model and BLUP identity, both inference engines,
the population-sampling fitting contract, architecture notes, and
the headline benchmark tables. Those tables are historical snapshots;
see [`benchmarks/RESULTS.md`](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for provenance.
