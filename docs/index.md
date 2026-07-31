# ltpred

ltpred is a Python toolkit for estimating genetic liability under the
liability-threshold model. It implements LT-FH, LT-FH++, ADuLT and PA-FGRS for
family-history, age-of-onset and registry-scale analyses.

## Start here

- **[Choose a method](guide.md)** — identify the appropriate model, bounds and
  inference engine.
- **[Quickstart](quickstart.md)** — run the estimator on a small tutorial dataset.
- **[Workflow pages](data-preparation.md)** — prepare data, estimate scores and
  understand the population-sampling-only fitting contract.
- **[API reference](api.md)** — inspect the public workflow and advanced APIs.

For the statistical model and implementation details, see
[Algorithm & model](algorithm.md). The [assumptions checklist](assumptions.md)
summarises the checks to make before a real-data analysis.

Unsupported experimental fitters and the register pipeline live in the source
checkout's
[`research/` directory](https://github.com/bvilhjal/ltpred/tree/main/research);
they are not installed with ltpred.
