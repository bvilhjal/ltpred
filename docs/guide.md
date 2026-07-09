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

ltpred does not build LD or run the GWAS itself — those are upstream/downstream
steps. It **does** estimate `h²` from the family data (`fit_heritability`) when you
don't have an external value.

Families can be supplied two ways: the compact **role grammar** (`o`, `m`, `f`,
`s1`, …) for common nuclear/extended structures, or a **kinship / relationship-
matrix** path (`kinship_from_pedigree`, `estimate_liability_from_kinship`) for
arbitrary pedigrees — deeper trees, cousins, inbreeding, non-standard structures.
The role grammar is fastest and simplest; the kinship path is the general case.
ltpred does not yet include an igraph-style pedigree-object interface or the
LTFHPlus plotting utilities.

**Which path to run:**

| use case | bounds builder | estimator | notes |
|---|---|---|---|
| toy / simulation, no age | `prevalence_thresholds` | PA or Gibbs | classic LT-FH |
| age-of-onset (LT-FH++ / ADuLT) | `age_thresholds`, or `thresholds_from_cip(…, case_mode="pin")` | Gibbs (PA without mixture as an approximation) | cases pinned at the onset threshold |
| PA-FGRS with censoring | `pa_thresholds`, or `thresholds_from_cip(…, case_mode="interval")` | `method="pearson-aitken"`, `use_mixture=True` | conservative case intervals; censored-control mixture |
| real register analysis | `thresholds_from_cip` (empirical, per stratum) | usually PA | **not** the logistic demo CIP |
| multiple traits | vector `h2` + `genetic_corrmat` + `full_corrmat` | Gibbs only | PA multi-trait not implemented |

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

### Beyond the role grammar: arbitrary pedigrees

When your relatives don't fit the fixed roles — deeper pedigrees, cousins,
multiple marriages, inbreeding — describe the pedigree by **who each person's
parents are** instead. `kinship_from_pedigree` turns `(id, father, mother)` columns
into the additive relationship matrix `A`, and `estimate_liability_from_kinship`
estimates the target's liability from `A` and per-individual bounds:

```python
from ltpred import kinship_from_pedigree, estimate_liability_from_kinship
ids    = ["o", "m", "f", "s1", "mgm", "mgf"]      # target first
father = ["f", "mgf", None, "f", None, None]      # None / unlisted = unknown founder
mother = ["m", "mgm", None, "m", None, None]
_, A = kinship_from_pedigree(ids, father, mother)
# lower/upper are (n_families, n_individuals) in `ids` order (from a threshold builder)
gen, se = estimate_liability_from_kinship(A, lower, upper, h2=0.5, target=0)
```

For a pedigree that *does* fit the role grammar the two paths give identical
results (same covariance); the pedigree path additionally handles half-sibs of any
degree, cousins, and inbred pedigrees (where a self-relationship can exceed 1).
Build the covariance alone with `construct_covmat_from_kinship(A, h2, target)`.

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

The three builders differ mainly in **how they encode a case** — this is the
distinction between the LT-FH, ADuLT and PA-FGRS variants, so pick deliberately:

| builder | case encoding | control encoding | extra outputs | model |
|---|---|---|---|---|
| `prevalence_thresholds` | `(T, ∞)` | `(-∞, T)` | — | classic **LT-FH** (no age) |
| `age_thresholds` | **pinned** `[thresh(onset), thresh(onset)]` | `(-∞, thresh(age))` | — | **ADuLT / LT-FH++** point-mass onset |
| `pa_thresholds` | interval `(thresh(onset), ∞)` — **not** pinned | `(-∞, thresh(age))` | `K_i`, `K_pop` | **PA-FGRS** (optionally with the mixture) |

`T = Φ⁻¹(1 − K)`. Note the two age-aware builders are **not** interchangeable:
`age_thresholds` *pins* a case's liability at its onset threshold (a point mass —
the deterministic age-of-onset map), whereas `pa_thresholds` bounds it *above*
that threshold (an interval) and adds the per-person cumulative incidence `K_i`
and lifetime prevalence `K_pop` used by the optional censored-control mixture
(`use_mixture=True`). Pinning a case at `thresh(age_of_onset)` assumes a
**deterministic monotone mapping** from onset age to liability — it treats onset
age as strictly stronger information than merely being affected. That is powerful,
but should be checked when diagnosis age is noisy, delayed, or shaped by
health-care access; the interval encoding is more conservative there.

You can also build the bounds yourself: any `(lower, upper)` interval per person
is valid (`lower == upper` pins a liability exactly; `(-inf, inf)` is
uninformative).

> **CIPs and sex/cohort.** ltpred has no explicit `sex` argument — sex, birth
> year and cohort enter only through the thresholds/CIPs you supply. The built-in
> `convert_age_to_cir` is a **logistic placeholder for simulation and demos**; for
> real analyses use externally estimated, **population-representative** cumulative
> incidence curves stratified by sex, birth year/cohort, ancestry and calendar
> period. `thresholds_from_cip(status, age, cip_ages, cip_values, ...)` takes such
> a curve directly (call it once per stratum) and returns `lower, upper, K_i,
> K_pop` — use it instead of the logistic builders for real data. CIPs from an
> ascertained biobank sample, or that ignore competing risks (death, emigration),
> can bias the estimate.

#### A real register-data recipe

The end-to-end path for register data uses your **own** cumulative-incidence curve
(not the logistic demo), one call to `thresholds_from_cip` per stratum, and PA:

```python
from ltpred import thresholds_from_cip, families_from_columns, estimate_liability

# Per sex / birth-cohort / ancestry stratum, with that stratum's CIP curve
# (cip_ages ascending, cip_values the cumulative incidence at each age):
lower, upper, K_i, K_pop = thresholds_from_cip(
    status=status, age=age,               # 1=case; onset age (cases) / follow-up age (controls)
    cip_ages=cip_ages, cip_values=cip_values,
    k_pop=lifetime_prevalence,            # stratum lifetime prevalence (defaults to max CIP)
    case_mode="interval",                 # PA-FGRS case encoding ("pin" = ADuLT/LT-FH++)
)
families = families_from_columns(
    fam_id=fam_id, role=role, lower=lower, upper=upper,
    K_i=K_i, K_pop=K_pop,                 # carry the mixture inputs
)
res = estimate_liability(families, h2=0.05, method="pearson-aitken",
                         use_mixture=True, out=("genetic",))
```

Key points:

- `case_mode="pin"` is the ADuLT / LT-FH++ onset-pinned encoding; `case_mode="interval"`
  (default) is the conservative PA-FGRS interval encoding.
- `use_mixture=True` only makes sense when `K_i` / `K_pop` are supplied (from
  `pa_thresholds` or `thresholds_from_cip`); it is the age-censored-control correction.
- Estimate the **CIP outside ltpred** from population-representative register data,
  stratified by sex, birth cohort, ancestry and calendar period, and accounting for
  competing risks — then pass one stratum's curve per call.

### Getting `h²` on the liability scale

Heritability must be on the **liability** scale. If you only have an
observed/case-control-scale estimate, convert it (Lee et al. 2011):

```python
from ltpred import convert_observed_to_liability_scale
h2_liab = convert_observed_to_liability_scale(obs_h2=0.15, pop_prev=0.05, prop_cases=0.5)
```

**Which `h²`?** The right value is the additive genetic variance component you want
the family covariance to represent — the model is additive-genetic only (see
[algorithm.md](algorithm.md#connection-to-selection-index-and-blup)). A pedigree /
twin **narrow-sense** estimate captures more of the family-history signal but can
be inflated by shared environment, assortative mating or indirect genetic effects
if those are not separately modelled; a **SNP-heritability** estimate is smaller
but better aligned with a downstream molecular GWAS. Neither is uniquely "correct",
so run a **sensitivity analysis** over plausible `h²` values (and prevalence/CIP)
and check how much the score and downstream results move.

### Fitting `h²` from the family data

If you don't have an external `h²`, you can **estimate it from the families
themselves** — `fit_heritability` fits the liability-scale heritability from the
relatives' case/control (and age-of-onset) statuses with a data-augmentation Gibbs
sampler (see [algorithm.md](algorithm.md#fitting-the-covariance-heritability)):

```python
from ltpred import fit_heritability
fit = fit_heritability(families)     # families with member bounds (from a threshold builder)
fit.h2, fit.h2_se                    # fitted liability-scale heritability (+ Monte-Carlo SE)
```

It needs relatives (lone probands carry no information and raise). `fit.h2_se` is
the *within-dataset* Monte-Carlo error — the spread across datasets is ~20–30×
larger — so for a real confidence interval use `bootstrap_fit`, which resamples
the families with replacement and refits:

```python
from ltpred import bootstrap_fit
bs = bootstrap_fit(families, lambda f: fit_heritability(f, seed=1).h2, n_boot=100)
bs.estimate, bs.se, (bs.ci_low, bs.ci_high)   # point, honest SE, 95% percentile CI
```

The same helper wraps any of the fitters (pass a `lambda` that returns the
quantity of interest, e.g. `fit_variance_components(f, ("A","C"), seed=1).components["C"]`
or `fit_genetic_correlation(f, seed=1).rg[0,1]`); fix the estimator's `seed` so the
spread reflects family sampling, not sampler noise. It costs `n_boot`+1 fits.
Feed the point estimate back in as `h2=fit.h2` (or, better, run the sensitivity
analysis around it).

### Sensitivity to the assumed `h²`

Because no single `h²` is uniquely correct, check how much the score actually
depends on it. `liability_sensitivity` sweeps a grid and reports the cross-setting
correlation of the estimates:

```python
from ltpred import liability_sensitivity
sens = liability_sensitivity(families, [0.3, 0.4, 0.5, 0.6, 0.7], method="pa")
sens.min_corr          # worst-case correlation of the score across the grid
sens.mean, sens.sd     # how the scale shifts with h²
```

In practice `min_corr` is very high (≈0.97 across `h² 0.2–0.8` for a typical
pedigree): the assumed `h²` mostly **rescales** the liability, barely changing the
*ranking* — so a linear GWAS on it is nearly invariant to the choice. A low
`min_corr` is the signal to pin `h²` down (with `fit_heritability`). Prevalence/CIP
sensitivity changes the truncation bounds rather than the covariance, so probe it
by rebuilding the families under each prevalence and comparing.

To separate additive heritability from a shared **common-environment** component
`C` (e.g. a full-sib effect that inflates familial resemblance beyond genetics),
`fit_variance_components` fits both at once by a multiple Haseman–Elston
regression:

```python
from ltpred import fit_variance_components
vc = fit_variance_components(families, ("A", "C"))
vc.components["A"], vc.components["C"], vc.residual   # proportions of liability variance
```

`C` is identified only from **full-sib pairs**, so the families must contain them
(otherwise the fit raises). The same `h2_se` caveat applies — use `bootstrap_fit` for a CI.
Dominance is intentionally not offered (it needs MZ/DZ twin contrasts). With
`("A",)` alone the result matches `fit_heritability`.

For **several traits**, `fit_genetic_correlation` estimates the genetic
correlation `r_g` between them (and each trait's `h²`). Each member must carry one
interval per trait (length-`n_pheno` `lower`/`upper`, as for
`estimate_liability_multi`):

```python
from ltpred import fit_genetic_correlation
gc = fit_genetic_correlation(families, phen_names=["adhd", "depression"])
gc.rg          # (P, P) genetic-correlation matrix (the headline)
gc.re          # (P, P) environmental correlation (phenotypic corr not from shared genes)
gc.h2, gc.rp   # per-trait heritabilities; phenotypic (full-liability) correlations
```

The phenotypic correlation splits into genetic and environmental parts —
`gc.rp` corresponds to `gc.genetic_cov + gc.env_cov` — so you get both `r_g` and
`r_e`. It needs related pairs (the genetic correlation is carried by the
cross-relative, cross-trait resemblance). It is ~unbiased near the null and mildly
attenuated at large `|r_g|`; use `bootstrap_fit` for a CI.

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
| `res.est["full"]` | posterior mean full liability (if requested; see caveat below) |
| `res.se["genetic"]` | Gibbs: Monte-Carlo standard error of the mean; PA: `0` |
| `res.var["genetic"]` | PA only: posterior variance of the estimate (`None` for Gibbs) |

Multi-trait columns are suffixed with the phenotype name, e.g.
`res.est["genetic_height"]`.

```python
score = res.est["genetic"]      # use this as your GWAS phenotype / risk score
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
  `fit_heritability` — see above.)
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
| error | batch-means MC SE (`res.se`) | none; gives posterior variance (`res.var`) |
| exactness | exact in the limit of infinite draws | exact for 1 truncation, close approx for families |
| speed | ~570 families/s (10 cores) | ~150 000 families/s — **100–360× faster** |
| censoring mixture | not implemented | `use_mixture=True` |

**Rule of thumb:** use **Pearson–Aitken** for biobank-scale runs (millions of
probands) and for the age-censoring mixture; use **Gibbs** when you want
posterior draws, a sampling-based cross-check, or the exact LT-FH++ reference
behaviour. For the `genetic` score they agree closely and give the same
downstream GWAS power on the benchmarked structures.

## Scaling to large cohorts

For millions of probands, the `Family`/`Member` objects and their per-call bounds
assembly become the bottleneck (the PA math is already sub-second for millions).
Skip the objects with the **array API**, which takes already-aligned bounds:

```python
from ltpred import estimate_liability_pa_arrays

# roles shared by the cohort (o + relatives; g is added). lower/upper are
# (n_families, len(roles)) aligned to `roles` — build them straight from columns.
est, var = estimate_liability_pa_arrays(
    roles=["o", "m", "f", "s1"], lower=lower, upper=upper, h2=0.05,
    out="genetic",                       # or use_mixture=True with K_i=, K_pop=
)
```

This runs the covariance construction once and the parallel PA kernel directly —
in practice ~100× faster than the object path at large `N` (and
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
currently Gibbs-only (Pearson–Aitken here raises `NotImplementedError`).

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

After centering/residualization the phenotype is continuous and stays well
calibrated (λ_GC ≈ 1 in the benchmarks) while lifting the association signal at
causal variants — a ~1.5× effective-sample-size gain over the case/control label.

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

## Modelling assumptions

The family covariance models **additive genetic sharing only**: every off-diagonal
entry is `shared_DNA × h²`. Not modelled are shared environment, household/cultural
transmission, assortative mating (parents are assumed genetically unrelated —
`m`–`f` covariance is 0), dominance/epistasis, and indirect genetic effects. When
these contribute to familial aggregation — common for psychiatric, reproductive,
metabolic and social traits — read the output as the **additive-genetic-model
projection of the family history**, not a pure causal genetic value, and expect
some over- or under-statement of "genetic" liability. The estimate is also
conditional on the assumed `h²`, prevalence and CIPs; treat those as inputs whose
uncertainty propagates (see the checklist).

The covariance is modular, though: you can **add environmental covariance**
(shared environment `c²`, maternal effects, assortative mating) to the
between-relative covariance to separate genetic from shared-environmental
resemblance and improve prediction. `construct_covmat` ships only the
additive-genetic table, but the covariance-level entry points (`rtmvnorm_gibbs`,
`pa_algorithm`, `pa_estimate_batched`) accept an arbitrary covariance — see
[algorithm.md](algorithm.md#adding-environmental-covariance-to-improve-prediction).

## Real-data checklist

Before running a production analysis:

1. Obtain **population-representative CIPs** (cumulative incidence by age),
   ideally from a register or other representative source — not the logistic
   default and not an ascertained biobank sample.
2. **Stratify** CIPs by sex, birth year/cohort, ancestry and calendar period
   where incidence differs; use a censoring-aware / competing-risk estimator
   (Kaplan–Meier, Aalen–Johansen) if death/emigration/competing diagnoses matter.
3. Convert `h²` to the **liability scale** (`convert_observed_to_liability_scale`).
4. Check **sensitivity** of the score to `h²` and to prevalence/CIP choices,
   especially for rare traits and dense pedigrees.
5. **Validate roles**: valid abbreviations, no duplicate roles within a family
   (the estimator now raises on duplicates).
6. Decide **case encoding** — pinned (`age_thresholds`) vs interval
   (`pa_thresholds`) — and record it.
7. Choose **Gibbs vs Pearson–Aitken**; for unusual pedigrees cross-check PA
   against Gibbs.
8. **Residualize** the phenotype for covariates (sex, cohort, PCs, batch) and
   handle related probands (LMM / pruning) before the GWAS.

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
