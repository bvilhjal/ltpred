# ltpred user guide

ltpred estimates an individual's **genetic liability** to a disease under the
liability-threshold model. For LT-FH, LT-FH++ and ADuLT, the names describe the
observation and family inputs rather than the inference engine:

- **LT-FH** uses family history with non-personalised prevalence thresholds
  ([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6)).
- **LT-FH++** adds age-, birth-year- and sex-dependent prevalence for the
  proband and relatives
  ([Pedersen et al. 2022](https://doi.org/10.1016/j.ajhg.2022.01.009)).
- **ADuLT** uses the same personalised construction for the proband alone,
  without family history
  ([Pedersen et al. 2023](https://doi.org/10.1038/s41467-023-41210-z)).
- **PA-FGRS** is the exception: its published specification couples the
  deterministic Pearson–Aitken approximation to
  lifetime-threshold case intervals and an age-censored-control mixture
  ([Dybdahl Krebs et al. 2024](https://doi.org/10.1016/j.ajhg.2024.09.009)).

Here **PA-FGRS** means the Pearson–Aitken score. It is not the distinct,
register-standardised FGRS of
[Kendler et al. (2021)](https://doi.org/10.1001/jamapsychiatry.2021.0336),
which ltpred does not implement.

The resulting continuous score can replace the 0/1 case-control label in a GWAS,
the use introduced for LT-FH by
[Hujoel et al. (2020)](https://doi.org/10.1038/s41588-020-0613-6).
Pearson–Aitken (PA, the single-trait default) and Gibbs can both infer LT-FH,
LT-FH++ and ADuLT inputs. The PA-FGRS name includes PA, and ltpred's censoring
mixture is available only in the PA engine.

The guide is split into short, task-focused pages:

| page | what's in it |
|---|---|
| **[Quickstart](quickstart.md)** | one complete runnable analysis, start to finish |
| **[Vignette](vignette.md)** | how to run ltpred: $h^2$, pedigree, CIP, family history, estimate, GWAS |
| **[Data preparation](data-preparation.md)** | inputs, role grammar, arbitrary pedigrees, threshold builders, CIPs, getting `h²` |
| **[CIP estimation](cip-estimation.md)** | estimating cumulative incidence from follow-up records (Kaplan-Meier, Aalen-Johansen), estimands, stratification |
| **[Estimation](estimation.md)** | running the estimator, reading the result, Gibbs vs PA, scaling, multi-trait, GWAS export |
| **[Inference](inference.md)** | population-sampled fitting of `h²` and variance components (A/C/M), plus family-cluster bootstrap |
| **[Assumptions & checklist](assumptions.md)** | modelling assumptions, real-data checklist, pitfalls |
| **[API reference](api.md)** | the exported workflow plus advanced module APIs, with signatures and docstrings |

Unsupported experimental fitters and the register pipeline live in the
checkout-only [`research/` package](https://github.com/bvilhjal/ltpred/tree/main/research);
they are not installed with ltpred.

See [algorithm.md](algorithm.md) for the model and the estimators, and the
[benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
for model and engine comparisons. For how to run the pipeline see the [vignette](vignette.md)
([`examples/vignette.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/vignette.py)).
Other runnable scripts:
[`examples/registry_pipeline.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/registry_pipeline.py)
(status/age table to a score) and
[`examples/ltfh_power_demo.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/ltfh_power_demo.py).

## When to use ltpred

Use ltpred when you have, per proband:

- a binary disease **status** (and ideally an **age** — age of onset for cases,
  age at last follow-up for controls), and
- for LT-FH/LT-FH++, the same for some **relatives** of known relationship
  (parents, siblings, grandparents, half-sibs, aunts/uncles, children), and
- a **population prevalence** and a **liability-scale heritability** `h²` for the
  disease — preferably external, or cross-checked with tetrachoric correlations
  (`ltpred.tetrachoric`, the Falconer route `h² ~ 2 ×` first-degree
  tetrachoric).

!!! warning "Family-data fitting needs a declared sampling design"

    `fit_heritability` and `fit_variance_components` assume independent,
    non-overlapping families under one of two contracts (a `pid` that appears
    in more than one family is rejected):
    `sampling="population"` for an unascertained sample — screened for gross
    marginal case-rate inconsistency, though a passing screen is not proof of
    population sampling — or `sampling="ipw"` with
    per-family `weights = 1 / P(family sampled)` when selection was on observed
    status with a known, strictly positive probability (case/control cohorts,
    biobank case enrichment).

    Selection on **family history** is reweightable only when every complete
    observed family pattern has a known, strictly positive inclusion
    probability. Any design that samples no families from some stratum
    (ascertainment through an affected proband) cannot be reweighted. See
    [Inference](inference.md#ascertained-samples).

The output targets the posterior mean genetic liability of each proband (Gibbs by
Monte Carlo; PA by a sequential-moment approximation). Feeding it to a
linear-regression GWAS is the canonical use. With relatives and personalised
CIPs the analysis is LT-FH++; with the proband only it is ADuLT.

The proband's own status is an **optional conditioning observation**. Include role
`o` when intentionally constructing a GWAS phenotype from the diagnosis, as in
LT-FH/LT-FH++/ADuLT association analyses. For prospective disease prediction or
classification, omit `o` or set its bounds to `(-inf, inf)` so the outcome being
predicted does not leak into its predictor. An ADuLT input with `o` unbound has no
remaining observation and is therefore uninformative.

ltpred does not build LD or run the GWAS itself — those are upstream/downstream
steps. Its family-data fitters are optional and restricted by the
population-sampling contract above.

Families can be supplied two ways: the compact **role grammar** (`o`, `m`, `f`,
`s1`, …) for common nuclear/extended structures, or a **kinship / relationship-
matrix** path (`kinship_from_pedigree`, `estimate_liability_from_kinship`) for
arbitrary pedigrees — deeper trees, cousins, inbreeding, non-standard structures.
The role grammar is fastest and simplest; the kinship path is the general case.
The high-level kinship estimator accepts ordinary `lower`/`upper` bounds and,
on the PA engine, `use_mixture=True` with per-member `K_i`/`K_pop` for the
PA-FGRS censored-control mixture. Gibbs still has no mixture implementation.
Pedigrees can be discovered from trio records with `ltpred.pedigree`
(`build_parent_graph`, `extract_pedigree`, `ParentGraph`, `Pedigree`), which feeds
`kinship_from_pedigree`. ltpred does not wrap igraph the way LTFHPlus does, and
ships no plotting utilities.

**Which path to run:**

| use case | bounds and observed-person records | estimator | model |
|---|---|---|---|
| no age, with relatives | `prevalence_thresholds`; relatives + optional `o` | PA or Gibbs | classic LT-FH |
| personalised CIP, with relatives | `thresholds_from_cip(…, case_mode="pin")`; relatives + optional `o` | PA (default); Gibbs reference | LT-FH++ |
| personalised CIP, no relatives | same pinned bounds; include role `o` only | PA (default); Gibbs reference | ADuLT |
| logistic tutorial | `age_thresholds` with either row pattern above | PA or Gibbs | age-only demonstration, not full LT-FH++ |
| published base PA-FGRS | role/object or kinship API; lifetime bounds from `prevalence_thresholds`; add control-specific `K_i`/`K_pop` from the CIP | PA with `use_mixture=True` | PA-FGRS |
| age-dependent interval/mixture encoding | `pa_thresholds`, or `thresholds_from_cip(…, case_mode="interval")` | PA with `use_mixture=True` | PA-FGRS-style variant; neither base PA-FGRS nor exact PA-FGRS_ADT |
| multiple traits | vector `h2` + correlation matrices | Gibbs only | model still follows bounds + rows |

In base PA-FGRS, every observed case uses
`[Φ⁻¹(1 − K_pop), ∞)`. Age-specific incidence enters through the mixture weights
for censored controls, not through the case threshold. The current convenience
helpers instead put cases above their age-of-onset threshold; use them only when
that age-dependent variant is the intended observation model. Do not label this
helper encoding PA-FGRS_ADT: it is not a faithful implementation of that published
specification. The
[data-preparation page](data-preparation.md#getting-lowerupper-from-status-and-age)
shows the base recipe explicitly.

New here? Start with the **[quickstart](quickstart.md)**.
