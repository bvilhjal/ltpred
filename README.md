# ltpred

**[Vignette (HTML, with equations)](https://bvilhjal.github.io/ltpred/vignette/)**
· [markdown source](docs/vignette.md)

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
age-censored-control mixture. ltpred supports both LT-FH++ case encodings that
LTFHPlus emits — the onset pin (`use_fixed_case_thr = TRUE`, ltpred's default)
and the onset interval (`use_fixed_case_thr = FALSE`, the R default); pairing the
interval with the mixture is a PA-FGRS-style variant, not the paper's
PA-FGRS_ADT specification.

The score is the **posterior mean genetic liability**: conceptually a
liability-threshold, age-aware, family-history analogue of BLUP /
selection-index prediction, projecting binary and censored disease observations
onto the proband's additive genetic value the way a breeding value is predicted
from relatives (see
[algorithm.md](docs/algorithm.md#connection-to-selection-index-and-blup)) —
not a SNP polygenic score.

## Installation

PyPI publication is pending (see the [roadmap](docs/ROADMAP.md) and
[release checklist](docs/RELEASING.md)). Until then, installation from a source
checkout is the supported path:

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
# h2 is required: the liability-scale heritability of *this* disease, with no
# disease-independent default (docs/data-preparation.md, "Which h²?").
pa = estimate_liability(sim.families, h2=0.5)
pa.est["genetic"]      # (n_families,) PA approximations to posterior means
pa.var["genetic"]      # posterior variance Var(g | family) — how uncertain this
                        # proband is; both engines report it (PA as a moment
                        # approximation). pa.se is 0: deterministic, which is
                        # not the same as zero approximation error.

# ...or the Gibbs truncated-MVN sampler as a sampling-based cross-check.
gibbs = estimate_liability(sim.families[:200], h2=0.5, method="gibbs",
                           tol=0.03, n_sim=25_000, burn_in=800, seed=1)
gibbs.est["genetic"]   # agrees with PA to ~1e-2 in this no-mixture example
```

For additive nuclear families, `method="quadrature"` integrates at most two
parental factors, regardless of the number of siblings. It reports convergence
diagnostics for both posterior moments and raises if refinement does not settle;
see [estimation](docs/estimation.md). Ordinary PA now conditions pins jointly
before approximating any remaining intervals. The [methods report](report/README.md)
derives both reductions and their exactness boundaries.

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
choosing between the methods, and using the score in a GWAS, see
**[choose a method](docs/guide.md)**.

## Three uses

One rule decides the job: if the proband's own diagnosis is in the input,
`estimate_liability` conditions on it (a GWAS phenotype); to *predict* that
diagnosis from family history, leave it out — keep the row with uninformative
bounds (dropping the row also works, but `pids` then falls back to the family
ID). [`examples/vignette.py`](examples/vignette.py) shows both. The three uses
(the [vignette](https://bvilhjal.github.io/ltpred/vignette/) is the run-book):

- **I. Risk prediction** from family history (own status *out*). Same
  `estimate_liability` call; leave their own diagnosis out of the input
  (the example script does this). Combining
  family-derived and genotype-derived predictors is a separate downstream
  model; see
  [Hujoel et al. 2022, *Cell Genomics*](https://doi.org/10.1016/j.xgen.2022.100152)
  and
  [Dybdahl Krebs et al. 2026, *AJHG*](https://doi.org/10.1016/j.ajhg.2025.11.016).
- **II. A quantitative GWAS phenotype** in place of the 0/1 label (own status
  *in*), the LT-FH association use. This is what you get if you pass their
  diagnosis in the usual way. ADuLT skips relatives.
- **III. Architecture, relationships, aetiology** from liability-scale h² and
  genetic/environmental covariances and/or the CIP. Pedigree scoring is optional.

## How it works

For each proband, ltpred conditions on the whole family under the
liability-threshold model:

1. **Covariance** — a liability splits into a genetic part `l_g ~ N(0, h²)` and an
   environmental part, summing to a full liability `l_o ~ N(0, 1)`. Two relatives'
   genetic parts correlate by the fraction of DNA they share, so an
   *off-diagonal* entry is `shared_DNA × h²` while every full liability keeps
   unit variance: `Sigma = h²A + (1 - h²)I`. Optional sibship and couple
   components add their own kernels off the diagonal —
   `Sigma = h²A + c²C + m²M + e²I` with `e² = 1 - h² - c² - m²` — and the
   residual absorbs them, so the diagonal is still 1.
2. **Thresholds and family context** — classic LT-FH uses non-personalised
   case/control bounds with family history. LT-FH++ uses age-, birth-year- and
   sex-specific prevalence for the proband and relatives. ADuLT uses the same
   personalised construction for the proband alone, without family history.
3. **Inference** — a **Gibbs** sampler (the sampling-based reference;
   untruncated genetic coordinates are integrated out of the sweep) or the
   deterministic **Pearson–Aitken** (PA) engine turns the covariance and
   intervals into an estimate of the posterior mean of the proband's genetic (`g`)
   and/or full (`o`) liability. PA has no Monte-Carlo error, but retains
   sequential approximation error. The PA-FGRS censoring mixture is PA-only.

See [estimation](docs/estimation.md#choosing-gibbs-vs-pearsonaitken) for how to choose,
and [algorithm.md](docs/algorithm.md) for the math and the performance internals
(Numba JIT, thread-parallel per-structure kernels, and the biobank-scale array API).

## Benchmarks

[`benchmarks/`](benchmarks/) compares model encodings and inference engines on
simulated data, and [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) is the
evidence ledger: PA versus Gibbs agreement and speed, the locked comparison to
the public R packages LTFHPlus and LTFGRS, independent-SNP marginal-association
gains for classic LT-FH and personalised LT-FH++ (with a matched family-free
ADuLT arm), calibration, cohort confounding, the PA-FGRS censoring mixture,
fitter recovery and ascertainment, and matched time/memory comparisons between
package versions. Its provenance header says which numbers are historical
snapshots and how to rerun them, and `scripts/check_evidence.py` reconciles
the release-defining claims in that ledger and in the
[methods report](report/ltpred_methods.pdf) with the committed artifacts.
Real-LD runs use [HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in;
not run for the committed results).

## Scope

ltpred covers the core prediction and fitting APIs: role-based and arbitrary-
pedigree (`kinship_from_pedigree`) covariance construction, an installed
population-trio register driver (`estimate_liabilities`), threshold/age/CIP
conversions plus CIP estimation from follow-up records (Kaplan-Meier and
Aalen-Johansen, `ltpred.cip`), Gibbs, PA and nuclear-family quadrature, single-
and multi-trait `estimate_liability`, simulation, and **model fitting** —
heritability (`fit_heritability`), variance components A + C + M
(`fit_variance_components`), optional common-threshold pairwise likelihood
(`fit_pairwise`), joint multi-trait h²/rg/re and optional sibship/couple
covariance (`fit_pairwise_multi`), approximate iid-family cluster percentile intervals
(`bootstrap_fit`), liability-scale transformations (`ltpred.liability_scale`),
and tetrachoric-correlation diagnostics
(`ltpred.tetrachoric`) for liability correlations straight from 2x2
case/control tables.

The fitted sibship (`C`) and couple (`M`) components feed back into prediction
through `c2`/`m2` on the single-trait role/object and array estimators
(`estimate_liability` with scalar `h2` included); on an arbitrary pedigree,
pass the aligned `C` and `M` kernels too, since `A` alone cannot tell a
full-sib pair from parent--offspring, or mates from strangers. The
arbitrary-kinship estimator also accepts caller-supplied kernels via
`c2`/`c_kernel` and `m2`/`m_kernel`, and (PA only) `use_mixture=True` with
per-member `K_i`/`K_pop` for the PA-FGRS censored-control mixture; Gibbs still
has no mixture implementation. The fitters remain role-based and require
independent, non-overlapping families, and the multi-trait scoring route rejects
`c2`/`m2` rather than silently ignoring them.

Unsupported research code lives in the checkout-only **`research/` package**
(importable as `research.<module>` from a checkout; not installed with the
wheel, not part of the public API, interfaces may change): the
HE genetic-correlation, onset-age-decay, common-factor and genetic-nurture fits,
the MCEM variance-component fit and the parametric-bootstrap significance tests
in `research.advanced_fitting`, plus the sex-limited and genetic-nurture
covariance constructors in `research.covariance_extensions`. It has its own
test suite (`research/tests`, run by CI) and five benchmark scripts import it;
the models are documented in [docs/research.md](docs/research.md).

Not included: an igraph-style pedigree-object interface, plotting utilities, and
the xgboost heritability helpers from LTFHPlus; and ltpred does not build LD or run
the GWAS itself (it produces the family-history liability phenotype you feed to
one).

## Documentation

- **User guide** — the [quickstart](docs/quickstart.md) (a complete run), then
  the [vignette](https://bvilhjal.github.io/ltpred/vignette/) (how to run:
  three uses — prediction, GWAS, aetiology — then h², pedigree, CIP,
  family history; source [docs/vignette.md](docs/vignette.md)),
  [data preparation](docs/data-preparation.md), [CIP estimation](docs/cip-estimation.md),
  [estimation](docs/estimation.md),
  [inference](docs/inference.md), [assumptions & checklist](docs/assumptions.md),
  and the [API reference](docs/api.md). ([Choose a method](docs/guide.md).)
- **[Algorithm & model](docs/algorithm.md)** — the liability-threshold model, both
  estimators, the Pearson–Aitken selection formula, the censoring mixture, and the
  implementation/performance notes.
- **[Methods note](report/ltpred_methods.pdf)** — estimand, three uses,
  observation models, PA exactness, and simulation evidence, written
  for colleagues (`report/ltpred_methods.tex`).
- **[Benchmarks](benchmarks/RESULTS.md)** — accuracy, speed and independent-SNP
  causal-NCP evidence, plus matched runtime and memory comparisons between
  package versions.

## Development

The code and documentation were written together with AI. Developed with the
support of [SMARTbiomed](https://smartbiomed.dk/).

## License

[MIT](LICENSE). Please cite the method(s) used and this repository; see
[CITATION.cff](CITATION.cff) for curated method and related-work references.
