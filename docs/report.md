# Technical report

A methods note aimed at colleagues who already work with GWAS and
the Falconer threshold model. It uses the same three uses as the
[vignette](vignette.md): family-history risk prediction, a
quantitative GWAS phenotype, and architecture / relationships /
aetiology from $h^2$, $r_g$ and/or the CIP. It is tracked in the
repository, not in the lean source distribution or this HTML site:

- PDF: [`report/ltpred_methods.pdf`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
- LaTeX source: [`report/ltpred_methods.tex`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.tex)

Rebuild with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd report
tectonic -X compile ltpred_methods.tex
cd ..
python scripts/check_evidence.py
```

The cross-benchmark headline, independent-SNP marginal-association,
integrated-personalisation, confounding, PGS and heritability tables are generated
LaTeX inputs from the committed benchmark CSVs (the full generator was removed in
the 2026-08 lean-down; the tables are now static snapshots). The NCP tables are
not real-LD, related-sample mixed-model GWAS evidence. A compact release check
pins the headline rows and rejects a report source or included table newer than
the tracked PDF.

The "Time and memory between versions" subsection reports the
[9 September 2026 efficiency rerun](https://github.com/bvilhjal/ltpred/tree/main/benchmarks/results/2026-09-09-time-memory-v061-rerun)
of v0.6.1 against v0.6.0, with matched workloads, warm runtimes, process RSS
and exact output agreement. The evidence check binds its table to the saved
JSON measurements and source provenance. v0.6.2 updates the documentation;
its numerical implementation is unchanged from the measured v0.6.1 source.

The report states the estimand and assumptions first, then the
liability-threshold model and BLUP identity, Algorithms G
(Gibbs), P (Pearson–Aitken) and M (mixture),
the implementation (grouping, collapsed untruncated coordinates,
streaming batch-means, object vs array vs register path, posterior `var` vs
estimator `se`), the standardized inbred target, calendar-time censoring
and closure-only observation masking, cumulative versus incident risk,
the population/IPW sampling
contracts, and the load-bearing simulation tables — agreement
and speed, independent-SNP causal NCP, personalisation, cohort confounding, calibration,
PGS complementarity, the censoring mixture (including
liability-dependent onset), pin versus interval encodings, and
ascertainment.
Those tables are historical snapshots; see
[`benchmarks/RESULTS.md`](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for provenance.
The old register-depth and prospective figures are no longer presented as
current evidence: the corrected driver needs a clean-source benchmark rerun.
The runnable vignette's small register example checks the input contract,
not predictive performance.
