# ltpred

**ltpred** is a Python implementation of **LT-FH++**, extending the original
**LT-FH** family-history phenotype
([Hujoel et al. 2020, *Nature Genetics*](https://doi.org/10.1038/s41588-020-0613-6))
with age of onset (sex and cohort effects enter through the sex/cohort-specific
thresholds or cumulative-incidence proportions (CIPs) you supply). It is a faithful
port of the R package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus)
([Pedersen et al. 2022, AJHG](https://doi.org/10.1016/j.ajhg.2022.01.009); the
family-free age-dependent variant, ADuLT, in
[Pedersen et al. 2023, Nat Commun](https://www.nature.com/articles/s41467-023-41210-z)),
and it also ships a deterministic **Pearson–Aitken (PA)** inference engine plus the
PA-specific, optional PA-FGRS censoring model
([Dybdahl Krebs et al. 2024, AJHG](https://doi.org/10.1016/j.ajhg.2024.09.009)).
The published base PA-FGRS model uses lifetime-threshold case intervals and an
age-censored-control mixture; the age-specific case-interval convenience helpers
are documented separately as a PA-FGRS-style variant, not as an exact
implementation of the paper's PA-FGRS_ADT specification.

Given each individual's case/control status, age and their relatives' statuses,
ltpred estimates with Gibbs—or sequentially approximates with PA—the **posterior
mean genetic liability**. This continuous phenotype can recover association power
relative to a plain case/control label when the liability model and inputs are
appropriate. Conceptually it is a
**liability-threshold, age-aware, family-history
analogue of BLUP / selection-index prediction**: binary and censored disease
observations are treated as intervals on latent liabilities, which are projected
onto the proband's additive genetic value the way a breeding value is predicted
from relatives (see
[algorithm.md](docs/algorithm.md#connection-to-selection-index-and-blup)).

The result is not itself a SNP polygenic score. Combining family-derived and
genotype-derived predictors is a separate downstream model; see
[Hujoel et al. 2022, *Cell Genomics*](https://doi.org/10.1016/j.xgen.2022.100152)
and the direct PA-FGRS/PGS analysis and theory in
[Dybdahl Krebs et al. 2026, *AJHG*](https://doi.org/10.1016/j.ajhg.2025.11.016).
Conditioning on the proband's own diagnosis is appropriate when constructing a
GWAS phenotype from that diagnosis. For prospective disease prediction or
classification, omit the proband's role `o` or give it uninformative
`(-inf, inf)` bounds; otherwise the outcome being predicted leaks into the score.

## Documentation

- **User guide** — the [quickstart](docs/quickstart.md) (a complete run), then
  [data preparation](docs/data-preparation.md), [CIP estimation](docs/cip-estimation.md),
  [estimation](docs/estimation.md),
  [inference](docs/inference.md), [assumptions & checklist](docs/assumptions.md),
  and the [API reference](docs/api.md). ([Overview & when-to-use](docs/guide.md).)
- **[Algorithm & model](docs/algorithm.md)** — the liability-threshold model, both
  estimators, the Pearson–Aitken selection formula, the censoring mixture, and the
  implementation/performance notes.
- **[Methods note](report/ltpred_methods.pdf)** — estimand, observation models,
  PA exactness, and simulation evidence, written for colleagues
  (`report/ltpred_methods.tex`).
- **[Benchmarks](benchmarks/RESULTS.md)** — accuracy, speed and GWAS-power
  comparison of the two methods.

## How it works

For each proband, ltpred conditions on the whole family under the
liability-threshold model:

1. **Covariance** — a liability splits into a genetic part `l_g ~ N(0, h²)` and an
   environmental part, summing to a full liability `l_o ~ N(0, 1)`. Two relatives'
   genetic parts correlate by the fraction of DNA they share, so every covariance
   entry is `shared_DNA × h²`.
2. **Thresholds and family context** — classic LT-FH uses non-personalised
   case/control bounds with family history. LT-FH++ uses age-, birth-year- and
   sex-specific prevalence for the proband and relatives. ADuLT uses the same
   personalised construction for the proband alone, without family history.
3. **Inference** — a **Gibbs** sampler (the sampling-based reference) or the deterministic
   **Pearson–Aitken** (PA) engine turns the covariance and
   intervals into an estimate of the posterior mean of the proband's genetic (`g`)
   and/or full (`o`) liability. PA has no Monte-Carlo error, but retains sequential
   approximation error, and ran 203–492× faster in the controlled 4-thread,
   **no-mixture** benchmark; PA and Gibbs `genetic` posterior-mean estimates had
   correlation ≥ 0.997 on those benchmarked structures. The PA-FGRS censoring
   mixture is PA-only and was not part of that comparison.

See [estimation](docs/estimation.md#choosing-gibbs-vs-pearsonaitken) for how to choose,
and [algorithm.md](docs/algorithm.md) for the math and the performance internals
(Numba JIT, thread-parallel per-structure kernels, and the biobank-scale array API).

## Installation

Not yet on PyPI. From a local checkout:

```bash
git clone https://github.com/bvilhjal/ltpred.git
cd ltpred
pip install -e ".[fast,test]"
pytest -q          # optional: confirm the install
```

`numpy` and `scipy` are required. `[fast]` adds an optional **Numba** JIT for the
Gibbs sweep and is strongly recommended — the pure-Python fallback is numerically
identical, just slower. For biobank-scale runs see the guide's
[scaling section](docs/estimation.md#scaling-to-large-cohorts).

## Quickstart

```python
from ltpred import simulate_under_LTM_single, estimate_liability

# simulate families (proband + mother, father, one sibling) under h²=0.5
sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=0.5, pop_prev=0.05, n_sim=2000,
    use_age=True, seed=1,
)

# PA approximation to posterior mean genetic liability per proband. This is an
# age-only, no-mixture family example for GWAS-phenotype construction; full
# LT-FH++ uses sex/birth-cohort-stratified CIPs.
# The deterministic, fast PA inference engine is the single-trait default.
pa = estimate_liability(sim.families, h2=0.5)
pa.est["genetic"]      # (n_families,) PA approximations to posterior means
pa.var["genetic"]      # PA moment approximations to conditional variances
                        # (pa.se is 0 — deterministic, not zero approximation error)

# ...or the Gibbs truncated-MVN sampler as a sampling-based cross-check.
gibbs = estimate_liability(sim.families[:200], h2=0.5, method="gibbs",
                           tol=0.03, n_sim=25_000, burn_in=800, seed=1)
gibbs.est["genetic"]   # agrees with PA to ~1e-2 in this no-mixture example
```

### Bring your own data

Build families from flat, tibble-style columns:

```python
import numpy as np
from ltpred import families_from_columns, age_thresholds, estimate_liability

# status (1=case) and age (age of onset for cases, current age otherwise)
lower, upper = age_thresholds(status, age, pop_prev=0.05)   # shared personalised bounds

families = families_from_columns(
    fam_id=fam_id,       # groups rows into families
    role=role,           # "o" = proband, "m"/"f"/"s1"/"mgm"/... = relatives
    lower=lower, upper=upper,
)
# Relative rows make this a family-history analysis. With real stratum-specific
# CIPs it is LT-FH++; keep only role="o" for family-free ADuLT.
res = estimate_liability(families, h2=0.5)          # PA is the single-trait default
score = res.genetic
```

Roles follow the LTFHPlus grammar (`o` proband, `m`/`f` parents, `s1`/`s2` sibs,
grandparents, half-sibs, aunts/uncles, children); the genetic row `g` is added
automatically. For **age-censored controls**, **multiple correlated traits**,
choosing between the methods, and using the score in a GWAS, see the
**[user guide](docs/guide.md)**.

## Scope

ltpred covers the core prediction and fitting APIs: role-based *and* arbitrary-
pedigree (`kinship_from_pedigree`) covariance construction, the threshold/age/CIP
conversions plus CIP estimation from follow-up records (Kaplan-Meier and
Aalen-Johansen, `ltpred.cip`), both inference engines (Gibbs and PA),
single- and multi-trait
`estimate_liability`, simulation, and **model fitting** — heritability
(`fit_heritability`), variance components A + C + M (`fit_variance_components`) — feedable back
into single-trait role-based estimation via `c2`/`m2`, approximate iid-family
cluster percentile intervals
(`bootstrap_fit`), liability-scale transformations (`ltpred.liability_scale`),
and tetrachoric-correlation diagnostics
(`ltpred.tetrachoric`) for liability correlations straight from 2x2
case/control tables.

Demoted research machinery lives in the unsupported **`research/` package** at
the repository root (importable as `research.<module>` from a checkout; it is
not part of the installed distribution): the end-to-end register pipeline
(`research.pipeline`), and in `research.advanced_fitting` the genetic-correlation,
onset-age-decay, common-factor and genetic-nurture fits, the MCEM
variance-component fit, and the parametric-bootstrap significance tests, plus
the sex-limited and genetic-nurture covariance constructors in
`research.covariance_extensions`.

The high-level arbitrary-kinship estimator accepts `A`, `lower`/`upper`, and
(PA only) `use_mixture=True` with per-member `K_i`/`K_pop` for the PA-FGRS
censored-control mixture. Gibbs still has no mixture implementation.

Beyond the additive `A` covariance, the fitted sibship (`C`) and couple (`M`)
components feed back into prediction through the `c2`/`m2` arguments on the
single-trait role/object and array estimators (`estimate_liability` with scalar
`h2` included). Multi-trait and arbitrary-kinship analyses require an explicitly
assembled covariance; the high-level multi-trait route rejects `c2`/`m2` rather
than silently ignoring them. Arbitrary user-supplied kernels still require the
covariance-level APIs.

Not included: an igraph-style pedigree-object interface, plotting utilities, and
the xgboost heritability helpers from LTFHPlus; and ltpred does not build LD or run
the GWAS itself (it produces the family-history liability phenotype you feed to
one).

## Benchmarks

[`benchmarks/`](benchmarks/) compares model encodings and inference engines on
simulated data. Its checked-in results are a historical snapshot, not an
automatic validation of later source changes; see the provenance header and
rerun instructions in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). The
integrated LT-FH++ benchmark includes age-, sex-, and
cohort-dependent CIP, coherent onset/censoring, ascertainment, and a genotype
GWAS. A matched ADuLT arm keeps the same personalised proband bounds but removes
relatives: it reaches a 1.049 ± 0.004× adjusted causal-SNP NCP ratio, versus
1.194 ± 0.006× for full LT-FH++; the paired family-history increment is
+0.1454 ± 0.0154 NCP-ratio units
(95% CI half-width). Full LT-FH++ has calibration slope 0.995 ± 0.015. A
prespecified sex-CIP panel separately shows removal of a
0.05091 ± 0.00096 female–male score-error gap, while its adjusted power increment
remains unresolved. In the replicated classic-LT-FH GWAS, PA and Gibbs both
reach a 1.47 ± 0.04× causal-SNP NCP ratio. Across matched **no-mixture** bounds,
their posterior-mean estimates had correlation ≥ 0.997. These PA–Gibbs claims do
not validate the PA-only censoring mixture. See
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); real-LD runs use
[HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in).

## License

[MIT](LICENSE). Please cite the method(s) used and this repository; see
[CITATION.cff](CITATION.cff) for curated method and related-work references.
