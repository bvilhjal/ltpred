# ltpred

**ltpred** is a Python implementation of **LT-FH++**, the liability-threshold
model conditioned on family history, age of onset and sex. It is a faithful port
of the R package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus)
([Pedersen et al. 2022, AJHG](https://doi.org/10.1016/j.ajhg.2022.01.009);
the family-free age-dependent variant, ADuLT, in
[Pedersen et al. 2023, Nat Commun](https://www.nature.com/articles/s41467-023-41210-z)).

Given each individual's case/control status, age and their relatives' statuses,
ltpred estimates the **posterior mean genetic liability** — a continuous
phenotype that, used in a linear GWAS, recovers association power that a plain
case/control label throws away.

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
3. **Fitting** — two interchangeable back-ends turn the covariance and intervals
   into the posterior mean of the proband's `g` (genetic) and/or `o` (full)
   liability:
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
`(n_sim × n_out)` sample array. On a 10-core machine this runs the reference demo
(2000 trios, 25 000 draws each) in ~5 s, versus ~27 s single-threaded. Without
Numba installed the exact same code runs serially in pure Python.

## Installation

Not yet on PyPI. From a local checkout:

```bash
pip install -e ."[fast,test]"
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

# posterior mean genetic liability per proband — the LT-FH++ GWAS phenotype
res = estimate_liability(sim.families, h2=0.5, out=("genetic",))
res.est["genetic"]     # (n_families,) posterior means
res.se["genetic"]      # matching Monte-Carlo standard errors

# ...or the deterministic PA-FGRS estimator (no sampling, ~100x faster)
pa = estimate_liability(sim.families, h2=0.5, method="pearson-aitken")
pa.est["genetic"]      # posterior means (agree with Gibbs to ~1e-2)
pa.var["genetic"]      # posterior variances; pa.se is 0 (deterministic)
```

### Age-censored controls (PA-FGRS mixture)

```python
from ltpred import pa_thresholds, families_from_columns, estimate_liability

# cases pinned at onset; controls get their cumulative incidence K_i + prevalence K_pop
lower, upper, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)
families = families_from_columns(fam_id, role, lower, upper, K_i=K_i, K_pop=K_pop)
res = estimate_liability(families, h2=0.05, method="pearson-aitken", use_mixture=True)
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

### Multiple correlated traits

```python
import numpy as np
res = estimate_liability(
    families,                     # each member's lower/upper is length n_pheno
    h2=[0.5, 0.3],
    genetic_corrmat=np.array([[1, 0.4], [0.4, 1]]),
    full_corrmat=np.array([[1, 0.5], [0.5, 1]]),
    phen_names=["A", "B"],
    out=("genetic",),
)
res.est["genetic_A"], res.est["genetic_B"]
```

## Public API

| Function | Purpose |
|---|---|
| `estimate_liability` | end-to-end estimator (`method=` gibbs / pearson-aitken; trait dispatch) |
| `estimate_liability_single` / `_multi` | the per-flavour Gibbs estimators |
| `estimate_liability_pa` | deterministic PA-FGRS estimator |
| `pa_algorithm` / `pa_estimate_batched` | Pearson–Aitken selection updates |
| `tnorm_moments` / `tnorm_mixture_conditional` | truncated-normal moments (+ censoring mixture) |
| `construct_covmat` / `_single` / `_multi` | family covariance from relatedness |
| `get_relatedness` | shared-DNA × h² for a pair of roles |
| `rtmvnorm_gibbs` | truncated-MVN Gibbs sampler |
| `prevalence_thresholds` / `age_thresholds` / `pa_thresholds` | status (+age) → liability bounds |
| `convert_age_to_cir` / `convert_age_to_thresh` / … | age ↔ incidence ↔ threshold |
| `convert_observed_to_liability_scale` | observed → liability-scale h² (Lee et al.) |
| `simulate_under_LTM_single` | simulate families for testing/benchmarking |
| `families_from_columns` | build family inputs from flat columns |

## Scope

This first port covers the **core statistical engine**: role-based covariance
construction, the threshold/age conversions, the Gibbs sampler, single- and
multi-trait `estimate_liability`, and simulation. The peripheral LTFHPlus
machinery (igraph-based flexible family-graph construction, ggplot plotting,
xgboost heritability helpers) is not included.

## Benchmarks

[`benchmarks/`](benchmarks/) compares the two fitting methods on simulated data
(accuracy, runtime scaling, age-of-onset information, and genotype-based GWAS
power), inspired by the LT-FH++, ADuLT and PA-FGRS papers. Headline: PA-FGRS
reproduces the Gibbs LT-FH++ posterior mean (corr ≥ 0.997, identical 1.52× GWAS
effective-N gain over case/control) while running **100–360× faster**. See
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); real-LD runs use
[HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in).

## Tests

```bash
pytest
```

The suite checks the sampler against closed-form truncated-normal means, the
relatedness table, the age/threshold inversions, and that the estimated genetic
liability recovers the simulated truth end-to-end.
