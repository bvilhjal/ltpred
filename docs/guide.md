# ltpred user guide

ltpred estimates an individual's **genetic liability** to a disease from their
own and their relatives' case/control status and ages, under the
liability-threshold model (LT-FH++ / ADuLT / PA-FGRS). The estimate is a
continuous score you use in place of the 0/1 case-control label — most often as
the phenotype in a GWAS, where it recovers power, but also directly as a
family-based risk score.

This page is the day-to-day usage guide. See [algorithm.md](algorithm.md) for the
model and the estimators, and [../benchmarks/RESULTS.md](../benchmarks/RESULTS.md)
for how the two methods compare. For runnable end-to-end scripts see
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

Do **not** expect ltpred to build LD, run the GWAS, or estimate `h²` for you —
those are upstream/downstream steps. It also does not (yet) read pedigree graphs;
you describe each family by a fixed vocabulary of relationship **roles** (below).

## Inputs

You describe the data as a flat table with one row per (proband, relative),
grouped into families. Each row needs:

| column | meaning |
|---|---|
| `fam_id` | groups rows into one family (one proband + relatives) |
| `role` | the relationship of this row to the proband (see grammar below) |
| `lower`, `upper` | the person's liability interval, from status (+age) |
| `pid` | *(optional)* a personal identifier |
| `K_i`, `K_pop` | *(optional)* cumulative incidence + prevalence, PA mixture only |

You do not provide the proband's genetic-liability row (`g`) — the estimator adds
it automatically, and it is what you get back.

### Role grammar

Roles use the LTFHPlus abbreviations. Each is relative to the proband:

| role | who |
|---|---|
| `o` | the proband's own status (the "offspring"/index person) |
| `m`, `f` | mother, father |
| `s1`, `s2`, … | full siblings (numbered) |
| `mgm`, `mgf`, `pgm`, `pgf` | maternal/paternal grand-mother/-father |
| `mhs1`…, `phs1`… | maternal/paternal half-siblings |
| `mau1`…, `pau1`… | maternal/paternal aunts/uncles |
| `c1.1`, `c1.2`, … | children (partner-group `.` child index) |

A family is any subset of these plus `o`. Two relatives of the same kind must be
numbered (`s1`, `s2`). Relatedness (and hence covariance) is derived from the role
labels — see `get_relatedness`.

### Getting `lower`/`upper` from status and age

Three helpers turn status (+age) into the truncation bounds, matching the three
model variants. Pick one:

```python
import numpy as np
from ltpred import prevalence_thresholds, age_thresholds, pa_thresholds

status = np.array([1, 0, 1, 0])          # 1 = case
age    = np.array([45, 70, 52, 33])      # onset age for cases, follow-up age for controls

# (1) classic LT-FH — one prevalence threshold, no age
lower, upper = prevalence_thresholds(status, pop_prev=0.05)

# (2) LT-FH++ / ADuLT — case pinned at onset threshold, control below its age threshold
lower, upper = age_thresholds(status, age, pop_prev=0.05)

# (3) PA-FGRS — like (1)/(2) but also emits K_i, K_pop for the censoring mixture
lower, upper, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)
```

- `prevalence_thresholds` — a case is `(T, ∞)`, a control `(-∞, T)` with
  `T = Φ⁻¹(1 − K)`. Use when you have no age information.
- `age_thresholds` — a case is **pinned** at `thresh(age_of_onset)` (younger
  onset ⇒ more extreme liability); a control is `(-∞, thresh(current_age))`
  (older healthy ⇒ lower liability). Use when you have ages.
- `pa_thresholds` — the same bounds plus the per-person cumulative incidence
  `K_i` and lifetime prevalence `K_pop`, needed only if you turn on the PA
  censored-control mixture (`use_mixture=True`).

You can also build the bounds yourself: any `(lower, upper)` interval per person
is valid (`lower == upper` pins a liability exactly; `(-inf, inf)` is
uninformative).

### Getting `h²` on the liability scale

Heritability must be on the **liability** scale. If you only have an
observed/case-control-scale estimate, convert it (Lee et al. 2011):

```python
from ltpred import convert_observed_to_liability_scale
h2_liab = convert_observed_to_liability_scale(obs_h2=0.15, pop_prev=0.05, prop_cases=0.5)
```

## Building families

From flat columns (the common path):

```python
from ltpred import families_from_columns
families = families_from_columns(
    fam_id=fam_id,          # e.g. ["A","A","A","B","B", ...]
    role=role,              # e.g. ["o","m","f","o","s1", ...]
    lower=lower, upper=upper,
    pid=pid,                # optional
    # K_i=K_i, K_pop=K_pop, # only for the PA mixture
)
```

Rows sharing a `fam_id` become one family; family order follows first appearance,
and the results come back in that order. You can also build `Family`/`Member`
objects directly if you prefer.

## Running the estimator

```python
from ltpred import estimate_liability
res = estimate_liability(families, h2=0.05, out=("genetic",))
```

- `h2` — liability-scale heritability (scalar; a vector selects the multi-trait
  model, below).
- `method` — `"gibbs"` (default, the LT-FH++ sampler) or `"pearson-aitken"`
  (aliases `"pa"`, `"pa-fgrs"`, the deterministic PA-FGRS estimator).
- `out` — which liabilities to return: `"genetic"` (the proband's `g`), `"full"`
  (the proband's `o`), or both.
- `use_mixture` — PA only: turn on the age-censored-control mixture (needs
  `K_i`/`K_pop`).

## Reading `LiabilityResult`

| field | meaning |
|---|---|
| `res.fam_ids` | family id per row (aligned with the input order) |
| `res.pids` | proband id (the `o` member's `pid`, else the `fam_id`) |
| `res.est["genetic"]` | posterior mean genetic liability per proband — **the score** |
| `res.est["full"]` | posterior mean full liability (if requested) |
| `res.se["genetic"]` | Gibbs: Monte-Carlo standard error of the mean; PA: `0` |
| `res.var["genetic"]` | PA only: posterior variance of the estimate (`None` for Gibbs) |

Multi-trait columns are suffixed with the phenotype name, e.g.
`res.est["genetic_height"]`.

```python
score = res.est["genetic"]      # use this as your GWAS phenotype / risk score
```

## Choosing Gibbs vs Pearson–Aitken

Both estimate the same quantity and agree to corr ≥ 0.997 on realistic families.

| | Gibbs (`"gibbs"`) | Pearson–Aitken (`"pearson-aitken"`) |
|---|---|---|
| kind | Monte-Carlo (truncated-MVN sampler) | deterministic closed-form sweep |
| error | batch-means MC SE (`res.se`) | none; gives posterior variance (`res.var`) |
| exactness | exact in the limit of infinite draws | exact for 1 truncation, close approx for families |
| speed | ~570 families/s (10 cores) | ~150 000 families/s — **100–360× faster** |
| censoring mixture | not implemented | `use_mixture=True` |

**Rule of thumb:** use **Pearson–Aitken** for biobank-scale runs (millions of
probands) and for the age-censoring mixture; use **Gibbs** when you want
posterior draws, a sampling-based cross-check, or the exact LT-FH++ reference
behaviour. They are interchangeable in downstream GWAS power.

## Multiple correlated traits

Pass a vector `h2` with genetic and full correlation matrices; each member's
`lower`/`upper` must then be length-`n_pheno` (one interval per trait):

```python
import numpy as np
res = estimate_liability(
    families,
    h2=[0.5, 0.3],
    genetic_corrmat=np.array([[1, 0.4], [0.4, 1]]),
    full_corrmat=np.array([[1, 0.5], [0.5, 1]]),
    phen_names=["A", "B"],
    out=("genetic",),
)
res.est["genetic_A"], res.est["genetic_B"]
```

Multi-trait borrows strength across genetically correlated diseases. It is
currently Gibbs-only (Pearson–Aitken here raises `NotImplementedError`).

## Using the estimate in a GWAS

The genetic-liability estimate is a quantitative phenotype — run an ordinary
linear-regression GWAS of it on standardized genotypes:

```python
# Xs: (n_indiv, m_snp) column-standardized genotypes, aligned to res.fam_ids
y = res.est["genetic"]
y = (y - y.mean()) / y.std()
chi2 = len(y) * ((Xs.T @ y) / len(y)) ** 2      # 1-df association statistic per SNP
```

Because the phenotype is continuous and mean-zero, it stays well calibrated
(λ_GC ≈ 1) while lifting the association signal at causal variants — a ~1.5×
effective-sample-size gain over the case/control label in the benchmarks.

## Options reference

| option | default | use |
|---|---:|---|
| `method` | `"gibbs"` | `"gibbs"` or `"pearson-aitken"` |
| `h2` | `0.5` | liability-scale heritability (scalar, or vector for multi-trait) |
| `out` | `("genetic",)` | `"genetic"`, `"full"`, or both |
| `use_mixture` | `False` | PA age-censored-control mixture (needs `K_i`/`K_pop`) |
| `tol` | `0.01` | Gibbs: batch-means SE convergence target |
| `n_sim`, `burn_in` | `100_000`, `1000` | Gibbs: draws kept / discarded per round |
| `max_rounds` | `100` | Gibbs: cap on convergence rounds |
| `seed` | `None` | Gibbs: RNG seed (per-family, deterministic) |
| `genetic_corrmat`, `full_corrmat`, `phen_names` | `None` | multi-trait only |

`n_sim`/`tol` trade speed for Monte-Carlo precision; the defaults converge for
typical families. PA ignores all Gibbs options.

## Pitfalls

- **`h²` must be liability-scale.** Convert observed-scale estimates first.
- **Ages mean different things by status.** For `age_thresholds`/`pa_thresholds`,
  `age` is the **age of onset** for cases and the **age at last follow-up** for
  controls.
- **Number repeated relatives** (`s1`, `s2`) — an unnumbered duplicate role
  collides.
- **The proband is role `o`**, not `g`. `g` (what you estimate) is added for you.
- **Prevalence and CIPs should match the population** the thresholds refer to;
  stratify by sex/birth-year if your incidence differs across strata (pass the
  per-person `K_i`).
- **Very low prevalence + tiny families** carry little information; the estimate
  approaches the population mean and the gain over case/control shrinks.
- **Install `[fast]`** (Numba) for large runs; the pure-Python fallback is
  numerically identical but much slower.
