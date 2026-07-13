# ltpred

**ltpred** is a Python implementation of **LT-FH++**, the liability-threshold
model conditioned on family history and age of onset (sex and cohort effects enter
through the sex/cohort-specific thresholds or CIPs you supply). It is a faithful
port of the R package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus)
([Pedersen et al. 2022, AJHG](https://doi.org/10.1016/j.ajhg.2022.01.009); the
family-free age-dependent variant, ADuLT, in
[Pedersen et al. 2023, Nat Commun](https://www.nature.com/articles/s41467-023-41210-z)),
and it also ships the deterministic **PA-FGRS** estimator
([Krebs et al. 2024, AJHG](https://pubmed.ncbi.nlm.nih.gov/39471805/)).

Given each individual's case/control status, age and their relatives' statuses,
ltpred estimates the **posterior mean genetic liability** — a continuous phenotype
that, used in a linear GWAS, recovers association power a plain case/control label
throws away. Conceptually it is a **liability-threshold, age-aware, family-history
analogue of BLUP / selection-index prediction**: binary and censored disease
observations are treated as intervals on latent liabilities, which are projected
onto the proband's additive genetic value the way a breeding value is predicted
from relatives (see
[algorithm.md](docs/algorithm.md#connection-to-selection-index-and-blup)).

## Documentation

- **[User guide](docs/guide.md)** — inputs, role grammar, threshold builders,
  choosing a method, reading results, GWAS use, options, and the full
  [function reference](docs/guide.md#function-reference).
- **[Algorithm & model](docs/algorithm.md)** — the liability-threshold model, both
  estimators, the Pearson–Aitken selection formula, the censoring mixture, and the
  implementation/performance notes.
- **[Benchmarks](benchmarks/RESULTS.md)** — accuracy, speed and GWAS-power
  comparison of the two methods.

## How it works

For each proband, ltpred conditions on the whole family under the
liability-threshold model:

1. **Covariance** — a liability splits into a genetic part `l_g ~ N(0, h²)` and an
   environmental part, summing to a full liability `l_o ~ N(0, 1)`. Two relatives'
   genetic parts correlate by the fraction of DNA they share, so every covariance
   entry is `shared_DNA × h²`.
2. **Thresholds** — each person's status and age become a liability interval:
   classic LT-FH puts a case in `(T, ∞)` and a control in `(-∞, T)`; LT-FH++/ADuLT
   instead pin a case at the threshold of its age of onset.
3. **Fitting** — a **Gibbs** sampler (the exact LT-FH++ reference) or the
   deterministic **Pearson–Aitken** (PA-FGRS) estimator turns the covariance and
   intervals into the posterior mean of the proband's genetic (`g`) and/or full
   (`o`) liability. PA has no Monte-Carlo error and runs ~100–360× faster; the two
   agree on the `genetic` score to corr ≥ 0.997 on the benchmarked structures.

See the [guide](docs/guide.md#choosing-gibbs-vs-pearsonaitken) for how to choose,
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
[scaling section](docs/guide.md#scaling-to-large-cohorts).

## Quickstart

```python
from ltpred import simulate_under_LTM_single, estimate_liability

# simulate families (proband + mother, father, one sibling) under h²=0.5
sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=0.5, pop_prev=0.05, n_sim=2000, seed=1,
)

# posterior mean genetic liability per proband — the LT-FH++ GWAS phenotype.
# PA-FGRS is deterministic and fast; use it by default.
pa = estimate_liability(sim.families, h2=0.5, method="pearson-aitken")
pa.est["genetic"]      # (n_families,) posterior means
pa.var["genetic"]      # posterior variances (pa.se is 0 — deterministic)

# ...or the Gibbs sampler (the exact LT-FH++ reference) as a cross-check.
gibbs = estimate_liability(sim.families[:200], h2=0.5, method="gibbs",
                           tol=0.03, n_sim=25_000, burn_in=800, seed=1)
gibbs.est["genetic"]   # agrees with PA to ~1e-2
```

### Bring your own data

Build families from flat, tibble-style columns:

```python
import numpy as np
from ltpred import families_from_columns, age_thresholds, estimate_liability

# status (1=case) and age (age of onset for cases, current age otherwise)
lower, upper = age_thresholds(status, age, pop_prev=0.05)   # LT-FH++/ADuLT bounds

families = families_from_columns(
    fam_id=fam_id,       # groups rows into families
    role=role,           # "o" = proband, "m"/"f"/"s1"/"mgm"/... = relatives
    lower=lower, upper=upper,
)
res = estimate_liability(families, h2=0.05, out=("genetic", "full"))
```

Roles follow the LTFHPlus grammar (`o` proband, `m`/`f` parents, `s1`/`s2` sibs,
grandparents, half-sibs, aunts/uncles, children); the genetic row `g` is added
automatically. For **age-censored controls**, **multiple correlated traits**,
choosing between the methods, and using the score in a GWAS, see the
**[user guide](docs/guide.md)**.

## Scope

ltpred covers the **statistical engine end to end**: role-based *and* arbitrary-
pedigree (`kinship_from_pedigree`) covariance construction, the threshold/age/CIP
conversions, both estimators (Gibbs and PA-FGRS), single- and multi-trait
`estimate_liability`, simulation, and **model fitting** — heritability
(`fit_heritability`), variance components A + C + M (`fit_variance_components`),
genetic correlation (`fit_genetic_correlation`) and its common-factor model
(`fit_genetic_factor`), bootstrap CIs (`bootstrap_fit`) and h² sensitivity
(`liability_sensitivity`).

Not included: an igraph-style pedigree-object interface, plotting utilities, and
the xgboost heritability helpers from LTFHPlus; and ltpred does not build LD or run
the GWAS itself (it produces the family-history liability phenotype you feed to
one).

## Benchmarks

[`benchmarks/`](benchmarks/) compares the two methods on simulated data (accuracy,
runtime scaling, age-of-onset information, and genotype-based GWAS power), inspired
by the LT-FH++, ADuLT and PA-FGRS papers. Headline: PA-FGRS matches the Gibbs
LT-FH++ posterior mean to corr ≥ 0.997 with the same 1.52× GWAS effective-N gain
over case/control — while running **100–360× faster**. See
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); real-LD runs use
[HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in).

## License

[GPL-3.0-or-later](LICENSE). Please cite the LT-FH++, ADuLT and PA-FGRS papers
(linked above) and this repository; see [CITATION.cff](CITATION.cff).
