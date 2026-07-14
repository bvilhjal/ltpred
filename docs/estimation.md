# Estimation

Running the estimator, reading the result, choosing between the two back-ends, and
scaling up. Assumes you already have families with liability bounds — see
[data preparation](data-preparation.md).

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
res = estimate_liability(families, h2=0.5, out=("genetic",))
```

- `h2` — liability-scale heritability (scalar; a vector selects the multi-trait
  model, below).
- `method` — the **default** is the deterministic **Pearson–Aitken** (PA)
  inference engine for a single trait (fast, matches Gibbs to ~1e-2), falling back to
  **Gibbs** for the multi-trait model. Pass `"gibbs"` to force the sampler (needed
  for multiple traits, a Monte-Carlo SE, or posterior draws), or `"pearson-aitken"`
  (aliases `"pa"`, `"aitken"`) to force PA. Bounds determine the observation
  encoding; inclusion of relatives distinguishes LT-FH++ from ADuLT. The engine
  is orthogonal to both.
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
| `res.est["full"]` | posterior mean full liability (if requested; see caveat below) |
| `res.se["genetic"]` | Gibbs: Monte-Carlo standard error of the mean; PA: `0` |
| `res.var["genetic"]` | PA only: conditional variance of the latent genetic liability, `Var(G_i \| family)` — residual uncertainty about `G_i`, **not** a standard error of the estimate (`None` for Gibbs) |

`res.genetic` is shorthand for `res.est["genetic"]` (the usual single-trait output).
Multi-trait columns are suffixed with the phenotype name, e.g.
`res.est["genetic_height"]`.

```python
score = res.genetic             # use this as your GWAS phenotype / risk score
```

> **`out="full"` differs between the methods.** For **Gibbs**, `full` is
> `E[l_o | own status, relatives]` — the proband's full liability conditioned on
> everything, including their own interval. For **Pearson–Aitken**, the full
> liability is the *target* of the sweep and so is **not** conditioned on its own
> observed interval; `full` there means "the proband's full liability predicted
> from the relatives". They are not the same quantity (e.g. a case with no
> relatives gives a positive Gibbs `full` but a `0` PA `full`). The canonical GWAS
> phenotype is `out="genetic"`, where the two agree; if you specifically want the
> own-status-conditioned full liability, use Gibbs.

### What the score is — and is not

```text
Estimand:  mu_i = E[ additive genetic liability of proband i
                     | statuses, ages, family structure, h2, CIP/prevalence model ]
```

`res.est["genetic"]` is the **posterior mean additive genetic liability** under
the specified liability-threshold model — a family-history-derived *latent*
phenotype on the standardized liability scale. It is the threshold-model,
family-history analogue of a BLUP / selection-index breeding value (see
[algorithm.md](algorithm.md#connection-to-selection-index-and-blup)). Concretely:

- **Not a SNP polygenic score.** No marker effects are used to build it; it comes
  from relatives' phenotypes and the assumed relationship matrix.
- **Not an absolute disease risk.** It lives on the liability scale; turning it
  into a risk needs the threshold/CIP model on top.
- **A conditional estimate.** the liability estimators *condition* on an assumed
  `h2`, prevalence/CIP model and family covariance; they do not estimate the CIPs
  internally. (`h2` itself can optionally be fit from the family data with
  `fit_heritability` — see [Inference](inference.md).)
- **A GWAS phenotype.** Used in a GWAS, a SNP association tests whether the SNP
  predicts *inferred additive genetic liability*, not merely the observed 0/1
  diagnosis — that is where the power gain comes from.

## Choosing Gibbs vs Pearson–Aitken

Both target the same posterior-liability idea, but Pearson–Aitken is a
deterministic *approximation*: it is exact for a single observed truncation and,
in the benchmarked family structures, matches the Gibbs `genetic` estimate to
corr ≥ 0.997. For unusual pedigrees — very large, densely affected, or heavily
truncated — treat Gibbs as the reference and cross-check.

| | Gibbs (`"gibbs"`) | Pearson–Aitken (`"pearson-aitken"`) |
|---|---|---|
| kind | Monte-Carlo (truncated-MVN sampler) | deterministic sequential-selection approximation |
| error | batch-means MC SE (`res.se`) | no Monte-Carlo error, but a non-zero sequential moment-approximation error; gives `Var(G_i \| family)` in `res.var` |
| exactness | exact in the limit of infinite draws | exact for 1 truncation, close approx for families |
| speed | 246–803 families/s (10 threads) | 138k–298k families/s — **323–569× faster** across tested sizes/structures |
| censoring mixture | not implemented | `use_mixture=True` |

Those rates are machine-specific medians from five warmed timings per point;
`benchmarks/RESULTS.md` reports the configuration and IQR-backed grid. Treat the
range as evidence about scale, not a hardware promise.

**Rule of thumb:** the default already picks **Pearson–Aitken** for single-trait
runs — keep it for biobank-scale cohorts and the age-censoring mixture; pass
`method="gibbs"` when you want posterior draws, a sampling-based cross-check, or
the exact truncated-MVN reference behaviour. For the `genetic` score they agree closely and
give the same downstream GWAS power on the benchmarked structures.

## Scaling to large cohorts

For millions of probands, the `Family`/`Member` objects and their per-call bounds
assembly become the bottleneck (the PA math is already sub-second for millions).
Skip the objects with the **array API**, which takes already-aligned bounds:

```python
from ltpred import estimate_liability_pa_arrays

# roles shared by the cohort (o + relatives; g is added). lower/upper are
# (n_families, len(roles)) aligned to `roles` — build them straight from columns.
est, var = estimate_liability_pa_arrays(
    roles=["o", "m", "f", "s1"], lower=lower, upper=upper, h2=0.5,
    out="genetic",                       # or use_mixture=True with K_i=, K_pop=
)
```

This runs the covariance construction once and the parallel PA kernel directly —
7–31× faster than the object path in the current warmed timing grid, at
1.6–9.4 million already-aligned families/s (and
`estimate_liability_gibbs_arrays` does the same for Gibbs, returning `(est, se)`).
Control the thread count with `ltpred.set_num_threads(n)`, and warm up once (the
first call JIT-compiles) before timing. Different family structures still need
separate array calls (one covariance each); the object API groups them for you.

**Shape contract.** `roles` is a length-`k` list (`"o"` + relatives; `g` is added
internally); `lower` and `upper` are both `(n_families, k)`, column `j` aligned to
`roles[j]`. A relative that is **absent or uninformative** for a given family is
encoded as the full real line — `lower = -np.inf`, `upper = np.inf` — so every row
carries the same `k` columns even when some relatives are missing:

```python
roles = ["o", "m", "f", "s1"]
assert lower.shape == upper.shape == (n_families, len(roles))
lower[i, 2], upper[i, 2] = -np.inf, np.inf     # family i's father unobserved
```

**Memory: `dtype=np.float32`.** The memory that scales at biobank size is the
per-family `(n_families, len(roles))` bounds (`lower`/`upper`, and `K_i`/`K_pop`),
not the tiny per-structure covariance. Store them in single precision to halve
that footprint — pass `dtype=np.float32` to `estimate_liability` (and the
`_single`/`_multi`/`_pa` variants), or simply hand the array API `float32` bound
arrays; it keeps them float32. The covariance, conditional-regression factors and
Monte-Carlo accumulators stay float64, so the estimates match the float64 result
to ~1e-5 (float32 rounding of the thresholds only). Quantising the *covariance*
itself (à la ldpred3's int8 LD) would not help here — it is a small `d×d` matrix
shared per structure, kilobytes total, and it is used in a matrix inverse.

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
Gibbs-only — the default picks Gibbs automatically for multiple traits, and an
explicit `method="pearson-aitken"` here raises `NotImplementedError`. To estimate
the genetic correlation itself, see [Inference](inference.md#genetic-correlation-between-traits).

## Using the estimate in a GWAS

The genetic-liability estimate is a quantitative phenotype — feed it to any
continuous-outcome GWAS. As with any quantitative GWAS, **residualize the
phenotype (and adjust) for covariates** — sex, birth year, genotyping batch,
ancestry principal components, and any ascertainment/design covariates — or use a
linear mixed model; probands who are themselves relatives should be handled by a
mixed model or by pruning. The estimate is centered on the population mean, but in
an ascertained sample it may not be mean-zero until you center/residualize.

```python
# Xs: (n_indiv, m_snp) column-standardized genotypes, aligned to res.fam_ids
# In practice regress out covariates first (or fit an LMM); simple sketch:
y = res.est["genetic"]
y = (y - y.mean()) / y.std()                     # center + scale (after covariate residualization)
chi2 = len(y) * ((Xs.T @ y) / len(y)) ** 2       # 1-df association statistic per SNP
```

After centering/residualization the phenotype is continuous and is near
λ_GC = 1 on average in the benchmark replicates while lifting the association
signal at causal variants — a 1.47 ± 0.04× effective-sample-size gain over the
case/control label in the replicated classic-LT-FH GWAS.
Personalising the thresholds by birth cohort (see [data preparation](data-preparation.md#getting-lowerupper-from-status-and-age))
is what keeps `λ_GC` valid under a secular prevalence trend.

## Options reference

| option | default | use |
|---|---:|---|
| `method` | `None` → PA (single-trait), Gibbs (multi-trait) | `"pearson-aitken"` or `"gibbs"` to force |
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
