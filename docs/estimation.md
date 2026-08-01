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
objects directly if you prefer. Role `o` is optional: when absent, the estimator
inserts an uninformative proband-status coordinate. Include `o` for a deliberately
diagnosis-derived GWAS phenotype; omit or unbind it when predicting/classifying
that same diagnosis, or the outcome leaks into the score.

## Running the estimator

```python
from ltpred import estimate_liability
res = estimate_liability(families, h2=0.5, out=("genetic",))
```

- `h2` — liability-scale heritability (scalar; a vector selects the multi-trait
  model, below).
- `method` — the **default** is the deterministic **Pearson–Aitken** (PA)
  inference engine for a single trait (PA–Gibbs posterior-mean correlation ≥ 0.997
  in the tested no-mixture structures), falling back to **Gibbs** for the
  multi-trait model. Pass
  `"gibbs"` to force the sampler (needed
  for multiple traits, a Monte-Carlo SE, or a sampling-based cross-check), or
  `"pearson-aitken"` (aliases `"pa"`, `"aitken"`) to force PA. Bounds determine the observation
  encoding; inclusion of relatives distinguishes LT-FH++ from ADuLT. The engine
  is orthogonal to both. PA-FGRS is the exception: its published name includes PA,
  and ltpred's censoring mixture is PA-only.
- `out` — which liabilities to return: `"genetic"` (the proband's `g`), `"full"`
  (the proband's `o`), or both.
- `use_mixture` — PA only: turn on the age-censored-control mixture (needs
  `K_i`/`K_pop`).
- `c2`, `m2` — optional sibship (`C`) and couple (`M`) shared-environment
  variance components; see below.

## Reading `LiabilityResult`

| field | meaning |
|---|---|
| `res.fam_ids` | one family id per result, in first-appearance family order |
| `res.pids` | one proband id per result (the `o` member's `pid`, else the `fam_id`) |
| `res.est["genetic"]` | genetic-liability estimate per proband — a Gibbs Monte-Carlo estimate or PA sequential-moment approximation to the posterior mean; **the score** |
| `res.est["full"]` | full-liability estimate (if requested), with the same Gibbs/PA interpretation as above; see caveat below |
| `res.se["genetic"]` | Gibbs: Monte-Carlo standard error of the mean; PA: `0` |
| `res.var["genetic"]` | PA only: sequential-moment approximation to the conditional variance `Var(G_i \| family)` — residual uncertainty about `G_i`, **not** a standard error of the estimate (`None` for Gibbs) |

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

`res.est["genetic"]` targets the **posterior mean additive genetic liability** under
the specified liability-threshold model—by Monte Carlo for Gibbs and a
sequential-moment approximation for PA—and is a family-history-derived *latent*
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

That last use deliberately allows the proband's observed status into the
phenotype construction. It is **not** a leakage-free disease predictor. When the
same diagnosis is the prediction/classification outcome, omit role `o` or set its
bounds to `(-inf, inf)` and estimate from family history alone.

Combining this family-derived score with a SNP polygenic score is a **separate
downstream prediction model**, not an operation performed by ltpred. Hujoel et al.
found that a target-population-fitted PRS-plus-family-history model improved disease
prediction across the UK Biobank target populations they studied, and preferred a
logistic combination when clinical covariates were included
([2022, *Cell Genomics*](https://doi.org/10.1016/j.xgen.2022.100152)). For five
psychiatric disorders, Dybdahl Krebs et al. found PA-FGRS and PGS to be weakly
correlated but complementary; their theory explains this as two noisy estimates of
the same additive genetic liability, not necessarily two different constructs
([2026, *AJHG*](https://doi.org/10.1016/j.ajhg.2025.11.016)). Fit and validate any
combination in the target population rather than adding the two scores uncalibrated.

## Shared-environment components (`C` and `M`)

The default family covariance is additive-genetic only. When families cluster
for environmental reasons — a shared sibship environment (`C`, loading on
full-sib pairs) or a couple/spousal environment (`M`, loading on mate pairs
such as `m`/`f` or `mgm`/`mgf`) — modelling those components improves the
estimate in two ways: the genetic liability `g` is not inflated by
environmental resemblance (better calibration), and full-liability prediction
`E[l_o | family]` sharpens (see [algorithm.md](algorithm.md#adding-environmental-covariance-to-improve-prediction)).
The components enter the relatives' covariance as

```text
Cov(l_i, l_j) = h2 * A_ij + c2 * C_ij + m2 * M_ij   (i != j)
Var(l_i)      = h2 + c2 + m2 + e2 = 1               (residual e2 absorbs)
```

so `h2 + c2 + m2 <= 1` must hold. The genetic target still couples to
relatives only through `h2 * A` — `g` remains a *genetic* liability.

Only if the families are independent, non-overlapping and unascertained
population samples, fit the components and wire them back in:

```python
from ltpred import estimate_liability, fit_variance_components

fit = fit_variance_components(
    families, ("A", "C", "M"), sampling="population"
)
res = estimate_liability(families, h2=fit.components["A"],
                         c2=fit.components.get("C", 0.0),
                         m2=fit.components.get("M", 0.0))
```

Or pass known values directly: `estimate_liability(families, h2=0.4, c2=0.15,
m2=0.1)`. The `c2`/`m2` arguments are supported by the single-trait role/object
and array entry points (`estimate_liability` with scalar `h2`,
`estimate_liability_pa_arrays`, `estimate_liability_gibbs_arrays`). The
high-level multi-trait route rejects nonzero components until their cross-trait
covariance is defined; the kinship/arbitrary-pedigree path likewise requires
you to assemble its covariance explicitly. Validated in
[benchmarks/RESULTS.md](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md) (section 24): wiring
recalibrates the genetic estimate (slope 0.93 -> 0.99) and sharpens
full-liability prediction on environmentally clustered families.

For case/control-enriched or family-history-selected samples, use externally
estimated components or a fitter that models the sampling design; the built-in
moment fitter and family bootstrap do not correct ascertainment bias.

## Choosing Gibbs vs Pearson–Aitken

Both target the same posterior-liability idea, but Pearson–Aitken is a
deterministic *approximation*: it is exact for a single observed truncation. In
the benchmarked **no-mixture** family structures, PA and Gibbs posterior-mean
`genetic` estimates had correlation ≥ 0.997. For unusual pedigrees — very large, densely
affected, or heavily truncated — treat Gibbs as the reference and cross-check.

Hujoel et al.'s original LT-FH study reported a less favourable Pearson–Aitken
comparison for UK Biobank's aggregate sibling question (at least one sibling
affected). That union event is not equivalent to separately observed per-sibling
intervals. ltpred requires separate member intervals, so its PA–Gibbs benchmarks
test a different, no-mixture observation model
([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6)).

| | Gibbs (`"gibbs"`) | Pearson–Aitken (`"pearson-aitken"`) |
|---|---|---|
| kind | Monte-Carlo (truncated-MVN sampler) | deterministic sequential-selection approximation |
| error | batch-means MC SE (`res.se`) | no Monte-Carlo error, but a non-zero sequential moment-approximation error; gives `Var(G_i \| family)` in `res.var` |
| exactness | exact in the limit of infinite draws | exact for 1 truncation, close approx for families |
| speed | ~117–363 families/s (4 threads) | ~57k–78k families/s — **203–492× faster** across tested sizes/structures, at the same 4 threads |
| censoring mixture | not implemented | `use_mixture=True` |

The speed and agreement comparisons use ordinary bounds without the censoring
mixture. Those rates are machine-specific medians from five warmed timings per point;
`benchmarks/RESULTS.md` reports the configuration and IQR-backed grid. Treat the
range as evidence about scale, not a hardware promise.

**Rule of thumb:** the default already picks **Pearson–Aitken** for single-trait
runs — keep it for biobank-scale cohorts and the age-censoring mixture; pass
`method="gibbs"` for a sampling-based truncated-MVN cross-check. The high-level
estimator returns posterior-mean estimates and
Monte-Carlo SEs, not retained draws; use the low-level `rtmvnorm_gibbs` function
when you need the sampled TMVN coordinates themselves. For the `genetic` score,
PA and Gibbs agree closely and gave the same downstream GWAS power on the
benchmarked no-mixture structures. These comparisons do not validate the PA-only
censoring mixture.

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
21–52× faster than the object path in the current warmed timing grid, at
1.2–3.8 million already-aligned families/s (and
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
that footprint — pass `dtype=np.float32` to `estimate_liability`, or hand either
array API `float32` bound arrays; they remain float32. The covariance,
conditional-regression factors and
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

The three covariance inputs must define one coherent model. With
`D = diag(sqrt(h2))`, the genetic covariance `G = D @ genetic_corrmat @ D` and
the residual covariance `E = full_corrmat - G` must both be positive
semi-definite; both correlation matrices must also be symmetric with unit
diagonal. Incoherent inputs now raise instead of being silently changed.

Multi-trait borrows strength across genetically correlated diseases. It is
Gibbs-only — the default picks Gibbs automatically for multiple traits, and an
explicit `method="pearson-aitken"` here raises `NotImplementedError`.
Experimental genetic-correlation fitting is available only in the checkout's
unsupported `research/` package; see
[Inference](inference.md#unsupported-research-prototypes).

## Using the estimate in a GWAS

The genetic-liability estimate is a quantitative phenotype — feed it to any
continuous-outcome GWAS. As with any quantitative GWAS, **residualize the
phenotype (and adjust) for covariates** — sex, birth year, genotyping batch,
ancestry principal components, and any ascertainment/design covariates — or use a
linear mixed model. Prefer pruning or non-overlapping family definitions when
target probands share relatives. If related targets are retained, an ordinary LMM
is not a blanket calibration guarantee: family-history phenotypes can carry extra
dependence, while the combination of severe case-control imbalance with
low-frequency variants can distort Gaussian-tail tests. Verify calibration under
the actual design and use an association method that represents those features
when needed
([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6);
[Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)). The estimate
is centered on the population mean, but in an ascertained sample it may not be
mean-zero until you center/residualize.

```python
# Xs: (n_indiv, m_snp) column-standardized genotypes, aligned to res.pids
# In practice regress out covariates first (or fit an LMM); simple sketch:
y = res.est["genetic"]
y = (y - y.mean()) / y.std()                     # center + scale (after covariate residualization)
chi2 = len(y) * ((Xs.T @ y) / len(y)) ** 2       # 1-df association statistic per SNP
```

Join genotype rows to `res.pids` explicitly; `res.fam_ids` identifies family
groups and need not be the genotyped proband identifier.

After centering/residualization the phenotype is continuous. In the replicated
classic-LT-FH benchmark, PA and Gibbs produced a `1.47 ± 0.04×` **causal-SNP
noncentrality ratio** relative to case/control. That is not the separate
squared-correlation effective-sample-size proxy. In the tested secular-trend
simulation, cohort-specific thresholds removed the genomic-control inflation
caused by the deliberately misspecified single-threshold analysis; this does not
replace ordinary GWAS covariate adjustment or guarantee calibration under other
misspecification.

## Options reference

| option | default | use |
|---|---:|---|
| `method` | `None` → PA (single-trait), Gibbs (multi-trait) | `"pearson-aitken"` or `"gibbs"` to force |
| `h2` | `0.5` | liability-scale heritability (scalar, or vector for multi-trait) |
| `out` | `("genetic",)` | `"genetic"`, `"full"`, or both |
| `use_mixture` | `False` | PA age-censored-control mixture (needs `K_i`/`K_pop`) |
| `c2`, `m2` | `None` (0) | single-trait only: sibship (`C`) / couple (`M`) shared-environment components; `h2 + c2 + m2 <= 1` |
| `tol` | `0.01` | Gibbs: batch-means SE convergence target |
| `n_sim`, `burn_in` | `100_000`, `1000` | Gibbs: draws kept / discarded per round |
| `max_rounds` | `100` | Gibbs: cap on convergence rounds |
| `seed` | `None` | Gibbs: integer RNG seed in `[0, 2**32 - 1]` (booleans rejected; per-family, deterministic) |
| `genetic_corrmat`, `full_corrmat`, `phen_names` | `None` | multi-trait only |

`n_sim`/`tol` trade speed for Monte-Carlo precision; the defaults converge for
typical families. PA ignores all Gibbs options.
