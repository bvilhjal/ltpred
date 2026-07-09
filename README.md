# ltpred

**ltpred** is a Python implementation of **LT-FH++**, the liability-threshold
model conditioned on family history and age of onset (sex and cohort effects
enter through the sex/cohort-specific thresholds or CIPs you supply — the
built-in age-CIP helper is not itself sex-stratified). It is a faithful port
of the R package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus)
([Pedersen et al. 2022, AJHG](https://doi.org/10.1016/j.ajhg.2022.01.009);
the family-free age-dependent variant, ADuLT, in
[Pedersen et al. 2023, Nat Commun](https://www.nature.com/articles/s41467-023-41210-z)).

Given each individual's case/control status, age and their relatives' statuses,
ltpred estimates the **posterior mean genetic liability** — a continuous
phenotype that, used in a linear GWAS, recovers association power that a plain
case/control label throws away.

Conceptually it is a **liability-threshold, age-aware, family-history analogue of
BLUP / selection-index prediction**: binary and censored disease observations are
treated as intervals on latent liabilities, which are then projected onto the
proband's additive genetic value the same way a breeding value is predicted from
relatives' phenotypes (see
[algorithm.md](docs/algorithm.md#connection-to-selection-index-and-blup)).

## Documentation

- **[User guide](docs/guide.md)** — inputs, role grammar, threshold builders,
  choosing a method, reading results, GWAS use, options and pitfalls.
- **[Algorithm & model](docs/algorithm.md)** — the liability-threshold model, the
  Gibbs sampler, the Pearson–Aitken selection formula and the censoring mixture.
- **[Benchmarks](benchmarks/RESULTS.md)** — accuracy, speed and GWAS-power
  comparison of the two methods.

## How it works

For each proband, ltpred conditions on the whole family under the
liability-threshold model:

1. **Covariance** — a person's liability splits into a genetic part
   `l_g ~ N(0, h²)` and an environmental part, summing to a full liability
   `l_o ~ N(0, 1)`. Two relatives' genetic parts correlate by the fraction of DNA
   they share, so every covariance entry is `shared_DNA × h²`
   (`construct_covmat`).
2. **Thresholds** — each person's status and age become a liability interval:
   classic LT-FH puts a case in `(T, ∞)` and a control in `(-∞, T)`
   (`prevalence_thresholds`); LT-FH++/ADuLT instead pins a case at the threshold
   of its age of onset and bounds a control below the threshold for its current
   age (`age_thresholds`).
3. **Fitting** — two back-ends turn the covariance and intervals into the
   posterior mean of the proband's `g` (genetic) and/or `o` (full) liability
   (they agree closely on the `genetic` score; see the
   [guide](docs/guide.md#choosing-gibbs-vs-pearsonaitken) for the differences):
   - **Gibbs** (`method="gibbs"`, default) — `rtmvnorm_gibbs` draws from the
     truncated multivariate normal, re-running until every estimate's batch-means
     Monte-Carlo standard error is below `tol`.
   - **Pearson–Aitken** (`method="pearson-aitken"`) — the deterministic **PA-FGRS**
     estimator ([Krebs et al. 2024, AJHG](https://pubmed.ncbi.nlm.nih.gov/39471805/)).
     It folds relatives in one at a time with the Pearson–Aitken selection formula
     (analytic truncated-normal moments + rank-1 mean/covariance updates), so there
     is **no sampling and no Monte-Carlo error** — exact for a single truncation, a
     close approximation for a family, and ~100× faster than Gibbs. Set
     `use_mixture=True` for its age-censored-control correction (a censored control
     is modelled as a mixture of a true control and a not-yet-onset future case).

### Performance

The Gibbs sweep is JIT-compiled with **Numba** (`[fast]` extra). Because families
are independent, the estimator groups those that share a family structure
(identical roles → identical covariance) and samples the whole group in one
compiled, **thread-parallel** (`prange`) kernel, accumulating the posterior mean
and the batch-means SE *online* — no per-family Python overhead and no full
`(n_sim × n_out)` sample array. Gibbs keeps only streaming batch-mean *summaries*
(sum and sum-of-squares), so its Monte-Carlo-SE memory is `O(families)` regardless
of `n_sim`; the conditional-regression factorisation uses the precision matrix
(one inverse, not `d` solves). On a 10-core machine this runs the reference demo
(2000 trios, 25 000 draws each) in ~5 s, versus ~27 s single-threaded. Without
Numba installed the exact same code runs serially in pure Python. Note the
**first** call to each method includes one-time JIT compilation (~1–2 s); the
benchmark and scaling numbers are measured after warm-up, so a single tiny run
will show a smaller apparent speed-up.

For biobank-scale runs, the **array API** (`estimate_liability_pa_arrays`,
`estimate_liability_gibbs_arrays`) bypasses the `Family`/`Member` objects and their
per-call bounds assembly — measured ~100× faster than the object PA path at large
`N` (200k trios; see [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md#2-runtime-scaling-bench_scalingpy)
and the [guide](docs/guide.md#scaling-to-large-cohorts)).

## Installation

Not yet on PyPI. From a local checkout:

```bash
git clone https://github.com/bvilhjal/ltpred.git
cd ltpred
pip install -e ".[fast,test]"
pytest -q          # optional: confirm the install
```

`numpy` and `scipy` are required. `[fast]` adds an optional **Numba** JIT for the
Gibbs sweep and is strongly recommended (the pure-Python fallback is numerically
identical, just slower).

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
# Sampling is slower, so cross-check on a subset with a looser tolerance:
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

Roles follow the LTFHPlus grammar: `o` (proband full liability), `m`/`f`
(parents), `s1`,`s2` (full siblings), `mgm`/`mgf`/`pgm`/`pgf` (grandparents),
`mhs*`/`phs*` (maternal/paternal half-sibs), `mau*`/`pau*` (aunts/uncles), `c*`
(children). The genetic row `g` is added automatically.

For **age-censored controls** (PA-FGRS mixture), **multiple correlated traits**,
choosing between the two methods, and using the score in a GWAS, see the
**[user guide](docs/guide.md)**.

## Public API

| Function | Purpose |
|---|---|
| `estimate_liability` | end-to-end estimator (`method=` gibbs / pearson-aitken; trait dispatch) |
| `liability_sensitivity` | sweep the assumed h² and report how stable the score is |
| `estimate_liability_pa` | deterministic PA-FGRS estimator |
| `estimate_liability_pa_arrays` / `_gibbs_arrays` | array API — skip `Family` objects for biobank scale |
| `fit_heritability` | **fit** liability-scale h² from family data (data-augmentation Gibbs) |
| `fit_variance_components` | **fit** additive `A` + common-environment `C` proportions (multiple HE regression) |
| `fit_genetic_correlation` | **fit** the genetic correlation `r_g` between traits (cross-trait HE regression) |
| `bootstrap_fit` | family-resampling bootstrap SE / CI for any of the fitters (honest uncertainty) |
| `test_variance_component` / `test_genetic_correlation` | parametric-bootstrap significance test (is `C` / `r_g` non-zero?) |
| `set_num_threads` | set the Numba-parallel thread count |
| `pa_algorithm` / `pa_estimate_batched` | Pearson–Aitken selection updates |
| `tnorm_moments` / `tnorm_mixture_conditional` | truncated-normal moments (+ censoring mixture) |
| `construct_covmat` / `_single` / `_multi` | family covariance from relatedness |
| `get_relatedness` | shared-DNA × h² for a pair of roles |
| `kinship_from_pedigree` | additive relationship matrix `A` from a pedigree (`id`, `father`, `mother`) — arbitrary pedigrees |
| `construct_covmat_from_kinship` | liability covariance from a kinship/`A` matrix (generalises the role grammar) |
| `estimate_liability_from_kinship` | estimate a target's liability from a pedigree `A` + per-member bounds |
| `rtmvnorm_gibbs` | truncated-MVN Gibbs sampler |
| `prevalence_thresholds` / `age_thresholds` / `pa_thresholds` | status (+age) → liability bounds |
| `thresholds_from_cip` | bounds from an empirical (population) CIP curve — for real data |
| `convert_age_to_cir` / `convert_age_to_thresh` / … | age ↔ incidence ↔ threshold |
| `convert_observed_to_liability_scale` | observed → liability-scale h² (Lee et al.) |
| `simulate_under_LTM_single` | simulate families for testing/benchmarking |
| `families_from_columns` | build family inputs from flat columns |

## Scope

ltpred covers the **statistical engine end to end**: role-based *and* arbitrary-
pedigree (`kinship_from_pedigree`) covariance construction, the threshold/age/CIP
conversions, both estimators (Gibbs and PA-FGRS), single- and multi-trait
`estimate_liability`, simulation, and **model fitting** — heritability
(`fit_heritability`), variance components A+C (`fit_variance_components`), genetic
correlation (`fit_genetic_correlation`), bootstrap CIs (`bootstrap_fit`) and h²
sensitivity (`liability_sensitivity`).

Not included: an igraph-style pedigree-object interface, plotting utilities, and
the xgboost heritability helpers from LTFHPlus; and ltpred does not build LD or run
the GWAS itself (it produces the family-history liability phenotype you feed to
one).

## Benchmarks

[`benchmarks/`](benchmarks/) compares the two fitting methods on simulated data
(accuracy, runtime scaling, age-of-onset information, and genotype-based GWAS
power), inspired by the LT-FH++, ADuLT and PA-FGRS papers. Headline: PA-FGRS is a
deterministic moment approximation (exact for a single truncation) that, on the
benchmarked family structures, matches the Gibbs LT-FH++ posterior mean to
corr ≥ 0.997 with the same 1.52× GWAS effective-N gain over case/control — while
running **100–360× faster**. See
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); real-LD runs use
[HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in).

## Tests

```bash
pytest
```

The suite checks the sampler against closed-form truncated-normal means, the
relatedness table, the age/threshold inversions, and that the estimated genetic
liability recovers the simulated truth end-to-end.
