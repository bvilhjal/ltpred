# Technical report

A methods note aimed at colleagues who already work with GWAS and
the Falconer threshold model. It uses the same three uses as the
[vignette](vignette.md): family-history risk prediction, a
quantitative GWAS phenotype, and architecture / relationships /
aetiology from $h^2$, $r_g$ and/or the CIP. It is tracked in the
repository, not in the lean source distribution or this HTML site:

- PDF: [`report/ltpred_methods.pdf`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
- LaTeX source: [`report/ltpred_methods.tex`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.tex)

The tracked report describes v0.7.2 (dated 23 September 2026), including joint heritability and
genetic/residual environmental covariance fitting. For its input contract,
replicated evidence and runnable code, see the
[Inference guide](inference.md#joint-heritability-and-cross-trait-correlations)
and [vignette example](vignette.md#joint-heritability-and-geneticenvironmental-correlation).

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
JSON measurements and source provenance. The newer
[v0.7.1 versus v0.7.0 rerun](https://github.com/bvilhjal/ltpred/tree/main/benchmarks/results/2026-09-23-time-memory-v071)
(RESULTS §31a) is not yet in the report.

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
The report's register-depth and prospective section still asks for a rerun
of the corrected driver; that rerun is in RESULTS §§20–21 (regenerated
2026-09-21 and 2026-09-23), which the [vignette](vignette.md) quotes, and
the report has not yet been updated to it. The runnable vignette's small
register example checks the input contract, not predictive performance.
