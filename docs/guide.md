# Choose a method

Read the [tutorial](tutorial.md) before this table. The six typed rows are
the [quickstart](quickstart.md). The contract is the
[vignette](vignette.md#three-uses).

When the table is your own, pick a row below. PA is the default
for one trait. Gibbs is the sampling check. The PA-FGRS censored-control
mixture (`use_mixture=True`) is PA-only, and it is not the
register-standardised FGRS of
[Kendler et al. (2021)](https://doi.org/10.1001/jamapsychiatry.2021.0336).
Use I (risk prediction, own status out) and use II (GWAS phenotype, own
status in) need a binary status, an age when you have one (onset for cases,
last follow-up for controls), relatives of known relationship for
LT-FH/LT-FH++, a population prevalence, and a liability-scale `h²` for that
disease. Prefer an external `h²`, or cross-check it with `ltpred.tetrachoric`
(`h² ≈ 2 ×` the parent–offspring tetrachoric correlation, if you assume no
shared environment). Use III (architecture: `h²`, correlations, CIP) can stop
at those parameters. Unsupported fitters stay in
[`research/`](research.md).

!!! warning "Family-data fitting needs a declared sampling design"

    `fit_heritability`, `fit_variance_components`, `fit_pairwise` and
    `fit_pairwise_multi` assume independent,
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

    These fitters require a common case/control threshold per trait and
    identifying contrasts among jointly observed relatives. Personalised
    CIP/onset bounds remain scoring inputs; unobserved relatives cannot
    identify a fitted component.

**Table 1. Scoring paths by observation model.**

| use case | bounds and observed-person records | estimator | model |
|---|---|---|---|
| no age, with relatives | `prevalence_thresholds`; relatives + optional `o` | PA or Gibbs | classic LT-FH |
| additive, noninbred nuclear family; no C/M or mixture | roles `o/m/f/s1/s2/…`, `0 <= h2 < 1` | quadrature, with refinement diagnostics | same no-mixture posterior as Gibbs, integrated numerically |
| personalised CIP, with relatives | `thresholds_from_cip(…, case_mode="pin")`; relatives + optional `o` | PA (default); Gibbs reference | LT-FH++ |
| personalised CIP, no relatives | same pinned bounds; include role `o` only | PA (default); Gibbs reference | ADuLT |
| logistic demo ([quickstart](quickstart.md)) | `age_thresholds` with either row pattern above | PA or Gibbs | age-only demonstration, not full LT-FH++ |
| population register (`ids`/`father`/`mother`) | `estimate_liabilities(…, use="gwas"` or `"prediction")` builds pinned CIP bounds itself | PA | LT-FH++ |
| arbitrary pedigree kinship | `kinship_from_pedigree` + `estimate_liability_from_kinship` | PA or Gibbs | LT-FH / LT-FH++ |
| published base PA-FGRS | role/object or kinship API; lifetime bounds from `prevalence_thresholds`; add control-specific `K_i`/`K_pop` from the CIP | PA with `use_mixture=True` | PA-FGRS |
| age-dependent interval/mixture encoding | `pa_thresholds`, or `thresholds_from_cip(…, case_mode="interval")` | PA with `use_mixture=True` | PA-FGRS-style variant; neither base PA-FGRS nor exact PA-FGRS_ADT |
| score multiple traits | vector `h2` + genetic and full-liability correlation matrices | Gibbs only | A+E covariance; C/M fitting does not extend this scorer |

In base PA-FGRS, every observed case uses
`[Φ⁻¹(1 − K_pop), ∞)`. Age-specific incidence enters through the mixture weights
for censored controls, not through the case threshold. The current convenience
helpers (`pa_thresholds`, `thresholds_from_cip(case_mode="interval")`) instead
put cases above their age-of-onset threshold; use them only when
that age-dependent variant is the intended observation model. Do not label this
helper encoding PA-FGRS_ADT: it is not a faithful implementation of that published
specification. The
[data-preparation page](data-preparation.md#getting-lowerupper-from-status-and-age)
shows the base recipe explicitly.

Worked rows for your own data start at the
**[quickstart](quickstart.md)**.

## Choosing a fitter

**Table 2. Fitters for independent families with common binary thresholds.**

| fitter | use when | computation and uncertainty |
|---|---|---|
| `fit_heritability` | estimate additive h² with no shared-environment component | iterative stochastic moment fit; `bootstrap_fit` supplies family-resampling uncertainty |
| `fit_variance_components` | distinguish additive, sibship and/or couple components using identifying relationships | iterative stochastic moment fit; family bootstrap for uncertainty |
| `fit_pairwise` | deterministic A/C/M fitting from binary pair patterns is appropriate | composite likelihood; conditional cluster-sandwich covariance for identifiable interior fits |
| `fit_pairwise_multi` | jointly estimate trait h², genetic/residual correlations and optional C/M covariance | multivariate pairwise composite likelihood; same identification, sampling and boundary caveats |

All four need the sampling contract above. PA is the fast approximate scoring
default. Quadrature is a deterministic cross-check for its restricted additive
nuclear-family model; Gibbs covers the wider and multi-trait scoring models.
Set `seed` explicitly for reproducible Gibbs estimates; multi-trait requests
with `method=None` select Gibbs. Non-default sampler controls warn when supplied
to deterministic engines.
