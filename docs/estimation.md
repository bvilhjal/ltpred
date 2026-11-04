# Scoring genetic liabilities

This page covers estimator choice, outputs and scaling. Build bounds first with
[data preparation](data-preparation.md); the [model reference](algorithm.md)
defines the estimand.

## Choose a scoring model

**Table 1. Scoring paths by observation model.**

| use case | records and bounds | engine | model |
|---|---|---|---|
| no age, with relatives | `prevalence_thresholds`; identified `o` plus relatives | PA or Gibbs | classic LT-FH |
| additive, noninbred nuclear family | roles `o/m/f/s1/s2/…`; no C/M or mixture; `0 <= h2 < 1` | quadrature | numerical cross-check of the same posterior |
| personalised CIP, with relatives | `thresholds_from_cip(…, case_mode="pin")` | PA (default) or Gibbs | LT-FH++ |
| personalised CIP, no relatives | the same bounds for role `o` only | PA (default) or Gibbs | ADuLT |
| logistic demonstration | `age_thresholds` | PA or Gibbs | age-only demonstration |
| population register | `estimate_liabilities(…, use="gwas"` or `"prediction")` | PA | pinned-onset LT-FH++ |
| arbitrary pedigree | `kinship_from_pedigree` + `estimate_liability_from_kinship` | PA or Gibbs | LT-FH / LT-FH++ |
| base PA-FGRS | lifetime bounds from `prevalence_thresholds`; control-specific `K_i`/`K_pop` | PA with `use_mixture=True` | PA-FGRS |
| age-dependent interval/mixture encoding | `pa_thresholds` or `thresholds_from_cip(…, case_mode="interval")` | PA with `use_mixture=True` | PA-FGRS-style variant |
| multiple traits | vector `h2`, genetic and full-liability correlation matrices | Gibbs | A+E covariance |

In base PA-FGRS, every observed case uses
`[Φ⁻¹(1 − K_pop), ∞)`. Age-specific incidence enters the censored-control
mixture, not the case threshold. The convenience interval helpers instead put
cases above their onset threshold. That is a different observation model, not
base PA-FGRS or a faithful PA-FGRS_ADT implementation; see the explicit
[base recipe](data-preparation.md#getting-lowerupper-from-status-and-age).

## Build and run

```python
from ltpred import families_from_columns, estimate_liability

families = families_from_columns(
    fam_id=fam_id, role=role, lower=lower, upper=upper,
    pid=pid,                    # optional personal join keys
    # K_i=K_i, K_pop=K_pop,     # PA mixture only
)
res = estimate_liability(families, h2=0.5, out=("genetic",))
score = res.genetic             # shorthand for res.est["genetic"]
```

Rows sharing a `fam_id` become one family. Include role `o` with its diagnosis
for a GWAS phenotype. For prediction, retain `o` and its `pid` but use
`(-inf, inf)` bounds. Only wholly pid-free families may omit that row and fall
back to `fam_id` as the result key. An identified relative without an identified
proband raises rather than silently losing the join.

Single-trait estimation defaults to deterministic **Pearson–Aitken (PA)**;
multi-trait estimation defaults to **Gibbs**. Use `method="gibbs"` for a
sampling cross-check or Monte-Carlo SE, `"pa"` to force PA, and
`"quadrature"` for the restricted numerical integration route. `dtype=np.float32`
halves bound storage while accumulators remain float64. The
[options reference](#options-reference) collects the remaining controls.

## Reading `LiabilityResult`

**Table 2. Estimates and numerical diagnostics.**

| field | meaning |
|---|---|
| `res.fam_ids`, `res.pids` | family ids and proband join keys, aligned to input families |
| `res.est["genetic"]` | posterior-mean genetic-liability score |
| `res.est["full"]` | full-liability estimate, if requested |
| `res.se["genetic"]` | Gibbs batch-means **Monte-Carlo** SE; zero for deterministic PA and quadrature |
| `res.var["genetic"]` | **posterior** conditional variance, not an SE that shrinks with numerical effort |
| `res.quadrature_error[...]` | largest mean/variance change over the last two refinements; not an error bound |
| `res.quadrature_nodes[...]` | nodes per active factor dimension; zero for an analytic answer |

`var` describes uncertainty about the proband, `se` describes sampling noise,
and `quadrature_error` describes grid refinement. Zero deterministic `se` does
not imply zero approximation or integration error. Multi-trait columns are
suffixed by phenotype, for example `res.est["genetic_height"]`.

`res.to_dict()` copies aligned estimate and diagnostic columns; `res.to_frame()`
returns them as a pandas DataFrame (pandas is optional). Repeated pids remain
separate rows, so validate join cardinality. `PopulationScores` and
`QuadratureResult` provide the same export pattern. Array tuple APIs return PA
`(est, var)`, Gibbs `(est, se)` or `(est, se, var)` with `return_var=True`, and
kinship `(est, se, var)`.

`out="full"` means `E[l_o | own interval and relatives]`. PA conditions exact
pins first and folds an unpinned target's interval after the remaining relative
intervals. The usual GWAS phenotype is still `out="genetic"`.

### What the score is — and is not

The target is the model-defined posterior mean additive genetic liability:

```text
mu_i = E[a_i | statuses, ages, family structure, h2, CIP/prevalence model]
```

It is a latent, standardised-liability phenotype—the family-history analogue
of a BLUP or selection-index breeding value—not a SNP polygenic score or an
absolute disease risk. Scoring conditions on `h2`, prevalence/CIPs and family
covariance; it does not estimate them as a side effect.

- **GWAS phenotype:** include the proband's diagnosis. The resulting phenotype
  contains that diagnosis and is not a leakage-free predictor.
- **Family-history prediction:** leave own status uninformative and censor every
  relative at the prediction landmark. A relative born at `b` is observed only
  to age `t - b`. Only `estimate_liabilities(use="prediction")` applies this
  calendar censoring for you.
- **Combining with a PGS:** fit a separate downstream model in the target
  population; do not add uncalibrated scores. Existing studies report
  complementarity between family-history and SNP predictors
  ([Hujoel et al. 2022](https://doi.org/10.1016/j.xgen.2022.100152),
  [Dybdahl Krebs et al. 2026](https://doi.org/10.1016/j.ajhg.2025.11.016)).

## Shared-environment components (`C` and `M`)

The default covariance is additive-genetic only. Optional full-sibship (`C`)
and mate-pair (`M`) components prevent environmental resemblance from being
attributed to the genetic target:

```text
Cov(l_i, l_j) = h2 * A_ij + c2 * C_ij + m2 * M_ij   (i != j)
Var(l_i)      = h2 + c2 + m2 + e2 = 1
```

Thus `h2 + c2 + m2 <= 1`. The genetic target remains coupled to relatives only
through `h2 * A`.

```python
from ltpred import estimate_liability, fit_variance_components

fit = fit_variance_components(families, ("A", "C", "M"), sampling="population")
res = estimate_liability(
    families, h2=fit.components["A"],
    c2=fit.components.get("C", 0.0), m2=fit.components.get("M", 0.0))
```

Known values can be supplied directly. The role/object and single-trait array
APIs accept `c2`/`m2`. The kinship route additionally requires aligned
`c_kernel`/`m_kernel` matrices (finite, symmetric PSD, unit diagonal), because
`A` cannot distinguish full siblings from parent–offspring or mates from
strangers:

```python
gen, se, var = estimate_liability_from_kinship(
    A, lower, upper, h2=0.4,
    c2=0.15, c_kernel=C, m2=0.10, m_kernel=M)
```

Component fitting requires independent, non-overlapping families, common
thresholds and a declared sampling design. It is not a fitter for overlapping
register pedigrees. For selected samples, `sampling="ipw"` is valid only with
known, strictly positive inclusion probabilities for every complete observed
family pattern; the family bootstrap does not correct ascertainment. See
[fitting and ascertainment](inference.md#ascertained-samples).

## Choosing Gibbs vs Pearson–Aitken

Both engines target the same posterior mean. Gibbs is exact in the limit of
infinite draws. No-mixture PA conditions pins jointly and marginalises
uninformative rows first; it is exact with zero or one remaining interval and
a sequential two-moment approximation otherwise. Its fold order is canonical
(sorted roles) for reproducibility, not an accuracy guarantee.

**Table 3. Inference engines.**

| | Gibbs | Pearson–Aitken |
|---|---|---|
| method | truncated-MVN Monte Carlo, with collapsed unbounded coordinates | deterministic sequential moments |
| error | batch-means MC SE | no MC error; approximation error outside exact cases |
| posterior variance | retained-draw estimate | sequential-moment approximation |
| exactness | exact in the limit | exact after pins with at most one interval |
| censoring mixture | not implemented | `use_mixture=True` |
| typical use | multi-trait and sampling cross-checks | single-trait production scoring |

On benchmarked no-mixture structures, PA and Gibbs genetic estimates had
correlation ≥ 0.997. Cross-check unusual, densely affected or heavily
truncated pedigrees against Gibbs. Fold-order stress results and locked
comparisons with LTFHPlus/LTFGRS are in
[RESULTS §§1, 2, 14 and 30](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
Those no-mixture comparisons do not validate the PA-only mixture. Hujoel et
al.'s less favourable aggregate-sibling comparison used a union event rather
than separately observed sibling intervals, so it is not the input model here
([2020](https://doi.org/10.1038/s41588-020-0613-6)).

Use `rtmvnorm_gibbs` directly only when retained TMVN draws are required; the
high-level estimator returns summaries. Batch means estimate MC precision, not
a proof of stationarity: difficult bounds still warrant independent seeds and
longer chains.

## Nuclear-family quadrature

For an additive, noninbred nuclear-family cross-check, use
`method="quadrature"` or the explicit array API. Conditioning on the two
parental breeding values leaves at most a two-dimensional integral; siblings
add likelihood factors, not dimensions.

```python
import numpy as np
from ltpred import estimate_liability_quadrature_arrays

q = estimate_liability_quadrature_arrays(
    roles=["o", "m", "f", "s1"],
    lower=[[2.0, -np.inf, 0.5, -np.inf]],
    upper=[[2.0, 1.5, np.inf, 1.5]],
    h2=0.5, out="genetic", atol=1e-8, max_nodes=128,
)
q.est, q.var              # posterior mean and variance
q.error, q.n_nodes        # refinement change and nodes per active dimension
```

Bounds have shape `(n_families, len(roles))`. Supported roles are unique `o`,
`m`, `f`, `s1`, `s2`, … with unrelated, noninbred parents and `0 <= h2 < 1`.
C/M, mixtures and other pedigrees are unsupported. Leave `o` uninformative for
relatives-only prediction.

Two successive refinements must meet `atol`; `max_nodes` is 64–512 per active
dimension. Nonconvergence raises with the failing row (under the object API,
its index within its role-set group). `q.error` is the larger of those changes,
not a certified error bound. Analytic answers use zero nodes. The object API
exposes the same diagnostics as `quadrature_error` and `quadrature_nodes`.

## Scaling to large cohorts

Array APIs avoid `Family`/`Member` objects and construct one covariance per
role set:

```python
from ltpred import estimate_liability_pa_arrays

est, var = estimate_liability_pa_arrays(
    roles=["o", "m", "f", "s1"], lower=lower, upper=upper, h2=0.5,
    out="genetic",                 # or use_mixture=True with K_i=, K_pop=
)
```

The warmed grid that shipped with v0.4.0 (four threads;
`benchmarks/bench_scaling.csv`, commit `1684fc5`) measured **13–29× faster than the object path at 2.02–6.29 million**
already-aligned families/s. It has not been re-measured; later object-path
improvements make the current ratio smaller. `estimate_liability_gibbs_arrays`
provides the corresponding Gibbs interface. Call `ltpred.set_num_threads(n)`
to control Numba workers and warm the JIT before timing.

The chunked and batch drivers bound kernel working memory:

```python
from ltpred import (estimate_liability_pa_chunked,
                    estimate_liability_pa_batches,
                    estimate_liability_gibbs_chunked)

est, var = estimate_liability_pa_chunked(
    roles, lower, upper, h2=0.5, chunk_size=65_536)
est, se = estimate_liability_gibbs_chunked(
    roles, lower, upper, h2=0.5, seed=1, chunk_size=4096)

est, var = estimate_liability_pa_batches(
    roles,
    ((lower_mm[i:i+65_536], upper_mm[i:i+65_536])
     for i in range(0, n_families, 65_536)),
    h2=0.5)
```

Gibbs seeds by global row position, so any chunking is bit-identical to the
one-shot call. PA agrees to rounding. An in-memory chunked array still holds
all bounds; use an iterator or memmap with `*_batches` to avoid that. Mixture
validation is global for `*_chunked` and per batch for `*_batches`.
`estimate_liability_gibbs_batches` uses the same global-row seed rule.

**Shape and memory.** `lower` and `upper` are `(n_families, len(roles))` and
column `j` matches `roles[j]`; `g` is added internally. Encode an absent or
uninformative relative as `lower[i,j] = -np.inf`, `upper[i,j] = np.inf`, not as
a control. Float32 bound/K arrays halve the cohort-scale input footprint while
covariances, conditional factors and accumulators stay float64. Quantising the
small shared covariance would save little and harm the solve.

PA does not modify inputs. Historical time/memory comparisons and their exact
output checks are in [RESULTS §31](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#31-time-and-memory-v061-versus-v060).
For large simulated registers, `simulate_register_liabilities(method="mendelian")`
avoids the dense relationship matrix and factorisation; it preserves the model,
including inbreeding, but not dense-method seeded draws.

## Multiple correlated traits

```python
import numpy as np
res = estimate_liability(
    families, h2=[0.5, 0.3],
    genetic_corrmat=np.array([[1, 0.4], [0.4, 1]]),
    full_corrmat=np.array([[1, 0.5], [0.5, 1]]),
    phen_names=["A", "B"], out=("genetic",))
res.est["genetic_A"], res.est["genetic_B"]
```

Every member needs one interval per trait. With `D = diag(sqrt(h2))`, both
`G = D @ genetic_corrmat @ D` and `full_corrmat - G` must be PSD. Multi-trait
scoring is Gibbs-only. Nonzero `c2`/`m2` raises because proportions alone do
not specify cross-trait environmental covariance.

For common-threshold data, `fit_pairwise_multi` jointly fits heritabilities and
correlations. An A+E fit supplies `joint.h2`, `joint.rg` and `joint.rp` for the
inputs above. A fit including C/M cannot be transferred to this A+E scorer;
see [joint inference](inference.md#joint-heritability-and-cross-trait-correlations).

## Using the estimate in a GWAS

Use the genetic estimate as a continuous phenotype. Join genotypes explicitly
to `res.pids`, residualize the phenotype and adjust for sex, cohort, ancestry
PCs, batch and design covariates, or use an appropriate linear mixed model.
Prefer non-overlapping target families. If related targets remain, validate
calibration under the actual relatedness, case-control imbalance and variant
frequency spectrum; an ordinary LMM is not a blanket guarantee
([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6);
[Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)).

```python
# Simple sketch after covariate residualization and checked pid alignment:
y = res.est["genetic"]
y = (y - y.mean()) / y.std()
chi2 = len(y) * ((Xs.T @ y) / len(y)) ** 2
```

In the replicated independent-SNP, non-overlapping-family benchmark, PA and
Gibbs gave a `1.47 ± 0.04×` causal-SNP noncentrality ratio over case/control.
That is not the squared-correlation effective-N proxy or a real-LD,
relatedness-aware mixed-model result. Cohort-specific thresholds removed the
inflation in one secular-trend simulation; they do not replace ordinary GWAS
covariate adjustment. See [RESULTS §§4 and 13](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).

## Options reference

**Table 4. High-level estimator options.**

| option | default | use |
|---|---:|---|
| `method` | `None` → PA (single), Gibbs (multi) | `"pearson-aitken"`, `"gibbs"`, `"quadrature"` |
| `h2` | required | liability-scale heritability; vector for multi-trait |
| `out` | `("genetic",)` | `"genetic"`, `"full"`, or both |
| `use_mixture` | `False` | PA censoring mixture (needs `K_i`/`K_pop`) |
| `c2`, `m2` | `None` (0) | single-trait C/M; sum with `h2` at most 1 |
| `tol` | `0.01` | Gibbs batch-means MC-SE target; not a mixing certificate |
| `n_sim`, `burn_in` | `100_000`, `1000` | Gibbs retained/discarded sweeps per round |
| `max_rounds` | `100` | cap on convergence rounds |
| `quadrature_atol` | `1e-8` | successive moment-change tolerance |
| `quadrature_max_nodes` | `128` | maximum nodes per active dimension (64–512) |
| `seed` | `None` | uint32 Gibbs seed; per-family deterministic streams |
| `genetic_corrmat`, `full_corrmat`, `phen_names` | `None` | multi-trait inputs |

Deterministic engines warn when Gibbs controls differ from their defaults.
For difficult samplers compare independent seeds, longer burn-in/chains and a
deterministic reference where available.

## Coming from LTFHPlus or LTFGRS

**Table 5. Migration equivalents.**

| R | ltpred |
|---|---|
| `estimate_liability(.tbl = df, …)` | `families_from_columns(...)`, then `estimate_liability(families, h2=…)` |
| column-name arguments | pass the columns themselves |
| `$genetic_est` | `res.genetic` |
| LTFGRS PA `$var` | `res.var["genetic"]`, not `res.se` (zero under deterministic PA) |
| `future::plan(multisession, workers = n)` | `ltpred.set_num_threads(n)` |

Role labels are unchanged except children: write `c<group>.<idx>`; children in
different partner groups are half sibs. ltpred ships no igraph-style pedigree
object or plotting utilities.
