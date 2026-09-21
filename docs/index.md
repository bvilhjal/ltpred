# ltpred

ltpred is a Python toolkit for estimating genetic liability under the
liability-threshold model. It implements LT-FH, LT-FH++, ADuLT and PA-FGRS for
family-history, age-of-onset and registry-scale analyses. The same core has
three uses: family-history **risk prediction** (optionally with a PGS), a
quantitative **GWAS phenotype**, and **architecture / relationships /
aetiology** from $h^2$, genetic/environmental covariances and/or the CIP — see the
[vignette](vignette.md).

## Start here

- **[Tutorial](tutorial.md)** — the lab: five questions on a simulated cohort
  whose genetic liability is known. Steps 1–4 are one disease; step 5 is two
  traits. Every block runs in order and a test checks the printed output.
- **[Choose a method](guide.md)** — after the lab, which bounds and estimator
  a student's own table needs.
- **[Quickstart](quickstart.md)** — six hand-typed rows of that table.
- **[Vignette](vignette.md)** — how to run ltpred: three uses (prediction,
  GWAS, aetiology), then $h^2$, pedigree, CIP, family history. Live page:
  [bvilhjal.github.io/ltpred/vignette](https://bvilhjal.github.io/ltpred/vignette/).
- **[Workflow pages](data-preparation.md)** — prepare data, estimate scores and
  understand the population/positive-IPW fitting contract. For joint $h^2$,
  $r_g$ and residual $r_e$, start with the [vignette example](vignette.md#joint-heritability-and-geneticenvironmental-correlation).
- **[API reference](api.md)** — inspect the public workflow and advanced APIs.

For the statistical model and implementation details, see
[Algorithm & model](algorithm.md). The [assumptions checklist](assumptions.md)
summarises the checks to make before a real-data analysis.
A typeset technical report (theory, implementation, and committed simulation
numbers) ships in the repository as
[`report/ltpred_methods.pdf`](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf).

For measured runtime and memory changes, see the
[v0.6.1 versus v0.6.0 benchmark rerun](https://github.com/bvilhjal/ltpred/tree/main/benchmarks/results/2026-09-09-time-memory-v061-rerun)
and the [PA memory notes](estimation.md#scaling-to-large-cohorts).

Unsupported experimental fitters and covariance extensions live in the source
checkout's [`research/` directory](https://github.com/bvilhjal/ltpred/tree/main/research);
they are not installed with ltpred and are documented on the
[research extensions](research.md) page.

Developed with the support of [SMARTbiomed](https://smartbiomed.dk/).
