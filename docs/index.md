# ltpred

LTpred estimates posterior mean genetic liability from disease records,
age of onset and family relationships. It supports LT-FH, LT-FH++, ADuLT and
PA-FGRS, plus fitting heritability and covariance parameters from family data.
The score can be a GWAS phenotype or a family-history predictor; it is not a SNP
polygenic score or an absolute disease probability.

## Start here

1. **[Getting started](quickstart.md)** — choose the analysis, install, and score
   six rows from start to finish.
2. **[Tutorial](tutorial.md)** — work through one simulated register, from
   incidence estimation to scoring and prospective prediction, then fit two traits.
   Tests execute every block and check the printed results.
3. **Use your own data:** [prepare inputs](data-preparation.md),
   [score liabilities](estimation.md), or [fit model parameters](inference.md).
   Read the [analysis checklist](assumptions.md) before interpreting a real cohort.

## Reference and evidence

- [Model and algorithms](algorithm.md) explains the estimand and links to the
  detailed methods PDF. [API reference](api.md) documents functions and results.
- [Numerical checks](validation.md) explains the reproducible simulation checks.
  [Benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
  is the evidence ledger for accuracy, runtime and memory. Measurements retain
  their source versions; the latest full time/memory comparison is v0.7.1 versus
  v0.7.0, not a measurement of subsequent changes.
- [Research extensions](research.md) describes unsupported, checkout-only models.

Project maintenance lives in the repository:
[changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md),
[roadmap](https://github.com/bvilhjal/ltpred/blob/main/docs/ROADMAP.md),
[paper plan](https://github.com/bvilhjal/ltpred/blob/main/docs/PAPER_PLAN.md),
[release instructions](https://github.com/bvilhjal/ltpred/blob/main/docs/RELEASING.md)
and [historical reviews](https://github.com/bvilhjal/ltpred/tree/main/docs/reviews).

Developed with the support of [SMARTbiomed](https://smartbiomed.dk/).
