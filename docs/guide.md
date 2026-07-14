# ltpred user guide

ltpred estimates an individual's **genetic liability** to a disease from their
own and their relatives' case/control status and ages, under the
liability-threshold model (LT-FH++ / ADuLT / PA-FGRS). The estimate is a
continuous score you use in place of the 0/1 case-control label — most often as
the phenotype in a GWAS, where it recovers power, but also directly as a
family-based risk score.

The guide is split into short, task-focused pages:

| page | what's in it |
|---|---|
| **[Quickstart](quickstart.md)** | one complete runnable analysis, start to finish |
| **[Data preparation](data-preparation.md)** | inputs, role grammar, arbitrary pedigrees, threshold builders, CIPs, getting `h²` |
| **[Estimation](estimation.md)** | running the estimator, reading the result, Gibbs vs PA, scaling, multi-trait, GWAS export |
| **[Inference](inference.md)** | fitting `h²`, variance components (A/C/M), genetic correlation, factor models, significance tests |
| **[Assumptions & checklist](assumptions.md)** | modelling assumptions, real-data checklist, pitfalls |
| **[API reference](api.md)** | every public function, with signatures and docstrings |

See [algorithm.md](algorithm.md) for the model and the estimators, and
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md) for how the two methods
compare. For runnable end-to-end scripts see
[`../examples/registry_pipeline.py`](../examples/registry_pipeline.py) (a
status/age table → GWAS phenotype template) and
[`../examples/ltfh_power_demo.py`](../examples/ltfh_power_demo.py).

## When to use ltpred

Use ltpred when you have, per proband:

- a binary disease **status** (and ideally an **age** — age of onset for cases,
  age at last follow-up for controls), and
- the same for some **relatives** of known relationship (parents, siblings,
  grandparents, half-sibs, aunts/uncles, children), and
- a **population prevalence** and a **liability-scale heritability** `h²` for the
  disease.

The output is the posterior mean genetic liability of each proband. Feeding it to
a linear-regression GWAS is the canonical use (LT-FH++ / ADuLT); it also stands
alone as a pedigree-based genetic risk score (PA-FGRS).

ltpred does not build LD or run the GWAS itself — those are upstream/downstream
steps. It **does** estimate `h²` from the family data (`fit_heritability`) when you
don't have an external value.

Families can be supplied two ways: the compact **role grammar** (`o`, `m`, `f`,
`s1`, …) for common nuclear/extended structures, or a **kinship / relationship-
matrix** path (`kinship_from_pedigree`, `estimate_liability_from_kinship`) for
arbitrary pedigrees — deeper trees, cousins, inbreeding, non-standard structures.
The role grammar is fastest and simplest; the kinship path is the general case.
ltpred does not yet include an igraph-style pedigree-object interface or the
LTFHPlus plotting utilities.

**Which path to run:**

| use case | bounds builder | estimator | notes |
|---|---|---|---|
| toy / simulation, no age | `prevalence_thresholds` | PA or Gibbs | classic LT-FH |
| age-of-onset (LT-FH++ / ADuLT) | `age_thresholds`, or `thresholds_from_cip(…, case_mode="pin")` | Gibbs (PA without mixture as an approximation) | cases pinned at the onset threshold |
| PA-FGRS with censoring | `pa_thresholds`, or `thresholds_from_cip(…, case_mode="interval")` | `method="pearson-aitken"`, `use_mixture=True` | conservative case intervals; censored-control mixture |
| real register analysis | `thresholds_from_cip` (empirical, per stratum) | usually PA (the default) | **not** the logistic demo CIP |
| multiple traits | vector `h2` + `genetic_corrmat` + `full_corrmat` | Gibbs only (auto-selected) | PA multi-trait not implemented |

New here? Start with the **[quickstart](quickstart.md)**.
