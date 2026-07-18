# ltpred user guide

ltpred estimates an individual's **genetic liability** to a disease under the
liability-threshold model. The model names describe different inputs, not
different inference engines:

- **LT-FH** uses family history with non-personalised prevalence thresholds.
- **LT-FH++** adds age-, birth-year- and sex-dependent prevalence for the
  proband and relatives.
- **ADuLT** uses the same personalised construction for the proband alone,
  without family history.
- **PA-FGRS** uses its interval-case and censored-control-mixture encoding.

The resulting continuous score can replace the 0/1 case-control label in a GWAS.
Pearson–Aitken (PA, the single-trait default) and Gibbs are inference engines for
these inputs; neither engine determines the model name.

The guide is split into short, task-focused pages:

| page | what's in it |
|---|---|
| **[Quickstart](quickstart.md)** | one complete runnable analysis, start to finish |
| **[Data preparation](data-preparation.md)** | inputs, role grammar, arbitrary pedigrees, threshold builders, CIPs, getting `h²` |
| **[Estimation](estimation.md)** | running the estimator, reading the result, Gibbs vs PA, scaling, multi-trait, GWAS export |
| **[Inference](inference.md)** | fitting `h²`, variance components (A/C/M), genetic correlation, factor models, significance tests |
| **[Assumptions & checklist](assumptions.md)** | modelling assumptions, real-data checklist, pitfalls |
| **[API reference](api.md)** | the exported workflow plus advanced module APIs, with signatures and docstrings |

See [algorithm.md](algorithm.md) for the model and the estimators, and the
[benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for model and engine comparisons. For runnable end-to-end scripts see
[`examples/registry_pipeline.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/registry_pipeline.py) (a
status/age table → GWAS phenotype template) and
[`examples/ltfh_power_demo.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/ltfh_power_demo.py).

## When to use ltpred

Use ltpred when you have, per proband:

- a binary disease **status** (and ideally an **age** — age of onset for cases,
  age at last follow-up for controls), and
- for LT-FH/LT-FH++, the same for some **relatives** of known relationship
  (parents, siblings, grandparents, half-sibs, aunts/uncles, children), and
- a **population prevalence** and a **liability-scale heritability** `h²` for the
  disease.

The output targets the posterior mean genetic liability of each proband (Gibbs by
Monte Carlo; PA by a sequential-moment approximation). Feeding it to a
linear-regression GWAS is the canonical use. With relatives and personalised
CIPs the analysis is LT-FH++; with the proband only it is ADuLT.

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

| use case | bounds and observed-person records | estimator | model |
|---|---|---|---|
| no age, with relatives | `prevalence_thresholds`; include `o` + relatives | PA or Gibbs | classic LT-FH |
| personalised CIP, with relatives | `thresholds_from_cip(…, case_mode="pin")`; include `o` + relatives | PA (default); Gibbs reference | LT-FH++ |
| personalised CIP, no relatives | same pinned bounds; include role `o` only | PA (default); Gibbs reference | ADuLT |
| logistic tutorial | `age_thresholds` with either row pattern above | PA or Gibbs | age-only demonstration, not full LT-FH++ |
| PA-FGRS with censoring | `pa_thresholds`, or `thresholds_from_cip(…, case_mode="interval")` | PA with `use_mixture=True` | PA-FGRS |
| multiple traits | vector `h2` + correlation matrices | Gibbs only | model still follows bounds + rows |

New here? Start with the **[quickstart](quickstart.md)**.
