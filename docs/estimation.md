# Estimation

Running the estimator, reading the result, choosing an inference engine, and
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
inserts an uninformative proband-status coordinate. Include `o` for **use II** (a diagnosis-derived GWAS phenotype); omit
or unbind it for **use I** (predicting/classifying that same diagnosis),
or the outcome leaks into the score. **Use III** may not call the
estimator at all; see the [vignette](vignette.md).

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
  `"pearson-aitken"` (aliases `"pa"`, `"aitken"`) to force PA. For additive
  nuclear families, `"quadrature"` selects the opt-in
  [numerical integration route](#nuclear-family-quadrature). Bounds determine the observation
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

**Table 1. Liability estimates and method-specific diagnostics.**

| field | meaning |
|---|---|
| `res.fam_ids` | one family id per result, in first-appearance family order |
| `res.pids` | one proband id per result (the `o` member's `pid`, else the `fam_id`) |
| `res.est["genetic"]` | posterior-mean genetic-liability score: estimated by Gibbs sampling, approximated by PA moments, or computed numerically by quadrature |
| `res.est["full"]` | corresponding full-liability estimate, if requested; see below |
| `res.se["genetic"]` | Gibbs batch-means **Monte-Carlo** SE; zero for deterministic PA and quadrature, which does not imply zero approximation or integration error |
| `res.var["genetic"]` | **posterior** conditional variance `Var(G_i \| family)`, estimated by the selected engine; uncertainty about the proband's liability, not an SE that shrinks with numerical effort |
| `res.quadrature_error["genetic"]` | quadrature only: largest mean/variance change over the last two refinements, not a certified error bound; the dictionary is `None` for other engines |
| `res.quadrature_nodes["genetic"]` | quadrature only: nodes per active factor dimension, or zero for an analytic answer; the dictionary is `None` for other engines |

These quantities answer different questions: `var` describes the proband,
`se` describes sampling noise, and `quadrature_error` describes numerical
refinement. On a seven-observation
family fold PA's `var` sits within 5% of the sampler's
(`tests/test_pearson_aitken.py::test_pa_conditional_variance_matches_gibbs_multi_truncation`).

`res.genetic` is shorthand for `res.est["genetic"]` (the usual single-trait output).
Multi-trait columns are suffixed with the phenotype name, e.g.
`res.est["genetic_height"]`.

```python
score = res.genetic             # use this as your GWAS phenotype / risk score
```

> **`out="full"` is E[l_o | own interval and relatives]**
> on each supported engine. No-mixture PA conditions exact pins jointly first;
> an unpinned target's interval is folded after the remaining relative intervals
> (an unbounded `g` is a no-op). A lone case therefore gives a
> positive PA `full`, matching Gibbs, not zero. Omit role `o` or set its
> bounds to `(-inf, inf)` when you want a relatives-only predictor — the
> same rule as for prospective prediction. The canonical GWAS phenotype is
> still `out="genetic"`.

### What the score is — and is not

```text
Estimand:  mu_i = E[ additive genetic liability of proband i
                     | statuses, ages, family structure, h2, CIP/prevalence model ]
```

`res.est["genetic"]` targets the **posterior mean additive genetic liability** under
the specified liability-threshold model—by Gibbs sampling, PA moment
approximation or nuclear-family quadrature—and is a family-history-derived *latent*
phenotype on the standardized liability scale. It is the threshold-model,
family-history analogue of a BLUP / selection-index breeding value (see
[algorithm.md](algorithm.md#connection-to-selection-index-and-blup)). Concretely:

- **Not a SNP polygenic score.** No marker effects are used to build it; it comes
  from relatives' phenotypes and the assumed relationship matrix.
- **Not an absolute disease risk.** It lives on the liability scale; turning it
  into a risk needs the threshold/CIP model on top.
- **A conditional estimate.** The liability estimators *condition* on an assumed
  `h2`, prevalence/CIP model and family covariance; they do not estimate the CIPs
  internally. (`h2` itself can optionally be fit from the family data with
  `fit_heritability` — see [Inference](inference.md).)
- **Use I — a relatives-only predictor.** Own status out of `D_F`. An optional
  PGS is combined afterwards, not by ltpred.
- **Use II — a GWAS phenotype.** Own status in. A SNP association tests whether
  the SNP predicts *inferred additive genetic liability*, not merely the
  observed 0/1 diagnosis — that is where the power gain comes from.
- **Use III does not need this score.** Liability-scale `h²` / `r_g` and the CIP
  can stand alone ([vignette](vignette.md) Table 1).

Use II deliberately allows the proband's observed status into the
phenotype construction. It is **not** a leakage-free disease predictor. When the
same diagnosis is the prediction/classification outcome (use I), omit role `o`
or set its bounds to `(-inf, inf)` and estimate from family history alone.

Combining this family-derived score with a SNP polygenic score is a **separate
downstream prediction model**, not an operation performed by ltpred. Hujoel et al.
found that a target-population-fitted PRS-plus-family-history model improved disease
prediction across the UK Biobank target populations they studied
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

If the families are independent and non-overlapping, and you can declare their
sampling design (unascertained, or selected on observed status with known
inclusion probabilities — see [Inference](inference.md#ascertained-samples)),
fit the components and wire them back in:

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
covariance is defined. For an arbitrary pedigree, the high-level kinship route
accepts the proportions only together with aligned relationship kernels:

```python
gen, se, var = estimate_liability_from_kinship(
    A, lower, upper, h2=0.4,
    c2=0.15, c_kernel=C,
    m2=0.10, m_kernel=M,
)
```

Every kernel must be finite, symmetric, positive semi-definite, and have unit
diagonal. It is deliberately caller-supplied: `A` cannot distinguish a full-sib
pair from parent--offspring, or a mate pair from two unrelated strangers. The
current component fitters remain role-based and assume independent,
non-overlapping families; this scoring API is not a component fitter for
overlapping extracted register pedigrees. Validated on the role path in
[benchmarks/RESULTS.md](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md) (section 24): wiring
recalibrates the genetic estimate (slope 0.93 -> 0.99) and sharpens
full-liability prediction on environmentally clustered families.

For **case/control-enriched** samples with known inclusion probabilities, pass
`sampling="ipw"` with per-family weights — see
[Inference](inference.md#ascertained-samples) for the worked example and its two
limits. A **family-history-selected** sample is reweightable only if every
complete observed family pattern has a known, strictly positive inclusion
probability. When probabilities are unknown or misspecified, or any stratum has
zero probability, use externally estimated components or a fitter that models
the sampling design. The family bootstrap does not correct ascertainment bias.

## Choosing Gibbs vs Pearson–Aitken

Both engines return an estimate of the same target
`μ_i = E[a_i | D_F]`
([algorithm.md](algorithm.md#the-estimand), equation (3)).
Write Algorithm G for the truncated-MVN Gibbs sampler and
Algorithm P for the Pearson–Aitken sequential-selection sweep
([algorithm.md](algorithm.md#inference-engine-1-gibbs-sampler)).
G is exact in the limit of infinite draws. No-mixture P conditions pins jointly
and marginalizes uninformative rows first. It is exact with zero or one remaining
interval, and a two-moment approximation for multiple remaining intervals.

In the benchmarked **no-mixture** family structures, PA and Gibbs
posterior-mean `genetic` estimates had correlation ≥ 0.997. For
unusual pedigrees — very large, densely affected, or heavily
truncated — treat Algorithm G as the reference and cross-check.
Algorithm P is fold-order dependent; the estimator canonicalizes
each family to a sorted role order before folding (a
reproducibility choice, not an accuracy one), and
`benchmarks/bench_pa_robustness.py` puts the spread across fold orders at a
median < 0.12% and p95 < 3.4% of the between-proband score SD on the stress
pedigrees.

Hujoel et al.'s original LT-FH study reported a less favourable Pearson–Aitken
comparison for UK Biobank's aggregate sibling question (at least one sibling
affected). That union event is not equivalent to separately observed per-sibling
intervals. ltpred requires separate member intervals, so its PA–Gibbs benchmarks
test a different, no-mixture observation model
([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6)).

**Table 2. Gibbs and PA in the benchmarked comparison.**

| | Gibbs (`"gibbs"`) | Pearson–Aitken (`"pearson-aitken"`) |
|---|---|---|
| kind | Monte-Carlo truncated-MVN sampler; untruncated genetic coordinates are collapsed out of the sweep | deterministic sequential-selection approximation |
| error | batch-means MC SE (`res.se`) | no Monte-Carlo error (`res.se` is `0`), but a non-zero sequential moment-approximation error |
| posterior variance | `Var(G_i \| family)` in `res.var`, from the retained draws | `Var(G_i \| family)` in `res.var`, as a sequential-moment approximation |
| exactness | exact in the limit of infinite draws | exact for 1 truncation, close approx for families |
| speed | ~180–690 families/s (4 threads) | ~92k–270k families/s — **392–518× faster than Gibbs in this package** across tested sizes/structures, at the same 4 threads. Versus the public R packages on the locked 200-family cohort, each at its default parallelism (ltpred 4 Numba threads, R 1 `future` worker): **6.79×** vs LTFHPlus Gibbs and **1418×** vs LTFGRS PA. Only the second is an implementation comparison. Re-run with both sides at one thread, the LTFHPlus/Gibbs fold is **1.75×** while LTFGRS/PA is **1425×** — Gibbs is `prange`-parallel so its fold tracks the thread count, PA is serial so its fold does not (RESULTS §30 for both columns and the load caveat) |
| censoring mixture | not implemented | `use_mixture=True` |

A locked comparison to R LTFHPlus 2.2.0 and LTFGRS 1.0.1 on the same
classic LT-FH families is in
[`RESULTS.md` §30](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md):
both engines had correlation 0.9999 with the R Gibbs scores; ltpred PA
and LTFGRS PA agree at RMSE 0.000087. Same-algorithm fold times were
6.79× versus LTFHPlus Gibbs and 1418× versus LTFGRS PA. Total and
per-family times, and isolated-process peak RSS, are in that section.

The intra-package speed and agreement comparisons use ordinary bounds without the censoring
mixture. Those rates are machine-specific medians from five warmed timings per point;
`benchmarks/RESULTS.md` reports the configuration and IQR-backed grid. Treat the
range as evidence about scale, not a hardware promise.

**Rule of thumb:** the default already picks **Pearson–Aitken** for single-trait
runs — keep it for biobank-scale cohorts and the age-censoring mixture; pass
`method="gibbs"` for a sampling-based truncated-MVN cross-check. The high-level
estimator returns posterior-mean estimates and
Monte-Carlo SEs, not retained draws; use the low-level `rtmvnorm_gibbs` function
when you need the sampled TMVN coordinates themselves. For the `genetic` score,
PA and Gibbs agree closely and gave the same downstream independent-SNP
causal-NCP result under the benchmark's marginal association calculation and
non-overlapping simulated families. That is not evidence from a real-LD,
related-sample mixed-model GWAS. These comparisons also do not validate the
PA-only censoring mixture.

## Nuclear-family quadrature

For a numerical posterior cross-check on an additive nuclear family, pass
`method="quadrature"` to `estimate_liability`, or call
`estimate_liability_quadrature_arrays` for aligned arrays. It conditions on the two
parental breeding values: adding siblings adds likelihood factors while the
remaining integral has at most two dimensions. Point observations and simple
conditional moments are handled analytically. The ordinary single-trait
`estimate_liability` default remains PA.

For `Family` inputs, numerical controls are `quadrature_atol` and
`quadrature_max_nodes`; the resulting `LiabilityResult` carries the dictionaries
in Table 1. The explicit array API uses `atol` and `max_nodes` and returns a
`QuadratureResult`:

```python
import numpy as np
from ltpred import estimate_liability_quadrature_arrays

q = estimate_liability_quadrature_arrays(
    roles=["o", "m", "f", "s1"],
    lower=[[2.0, -np.inf, 0.5, -np.inf]],
    upper=[[2.0, 1.5, np.inf, 1.5]],
    h2=0.5, out="genetic", atol=1e-8, max_nodes=128,
)
q.est, q.var              # posterior mean and posterior variance
q.error, q.n_nodes        # refinement diagnostic and nodes per active dimension
```

Bounds have shape `(n_families, len(roles))`. Supported roles are unique `o`,
`m`, `f`, `s1`, `s2`, ... with unrelated, noninbred parents and `0 <= h2 < 1`.
Other pedigrees, shared-environment components and the PA-FGRS censoring mixture
are outside this API. Finite intervals, onset pins and uninformative bounds are
supported. Omit `o` or leave its bounds uninformative for a relatives-only
prediction; `out="full"` instead targets the proband's full liability.

`q.var` measures posterior uncertainty. `q.error` measures the largest change
in the mean or variance across the last two quadrature refinements; it is
neither a certified numerical error bound nor a Monte-Carlo SE. Analytic
answers report zero nodes and zero refinement error. The `max_nodes` limit is
64–512 nodes **per active dimension**, and failure to meet the refinement
criterion raises `RuntimeError` identifying the family. No unchecked result
is returned on nonconvergence.

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
13–29× faster than the object path in the current warmed timing grid, at
2.02–6.29 million already-aligned families/s (and
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
Nonzero `c2`/`m2` also raise: component proportions alone do not specify the
cross-trait covariance of `C` or `M`. Separate single-trait liability estimates
are appropriate only when giving up cross-trait borrowing is intentional; a
joint extension needs explicit cross-trait environmental covariance inputs.
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
squared-correlation effective-sample-size proxy. The result used independent
SNPs, non-overlapping simulated families, and a lightweight marginal score
statistic; it is not a real-LD, relatedness-aware mixed-model GWAS result. In the
tested secular-trend simulation, cohort-specific thresholds removed the
genomic-control inflation caused by the deliberately misspecified
single-threshold analysis; this does not replace ordinary GWAS covariate
adjustment or guarantee calibration under other misspecification.

## Options reference

**Table 3. High-level estimator options.**

| option | default | use |
|---|---:|---|
| `method` | `None` → PA (single-trait), Gibbs (multi-trait) | `"pearson-aitken"`, `"gibbs"`, or opt-in nuclear-family `"quadrature"` |
| `h2` | `0.5` | liability-scale heritability (scalar, or vector for multi-trait) |
| `out` | `("genetic",)` | `"genetic"`, `"full"`, or both |
| `use_mixture` | `False` | PA age-censored-control mixture (needs `K_i`/`K_pop`) |
| `c2`, `m2` | `None` (0) | single-trait only: sibship (`C`) / couple (`M`) shared-environment components; `h2 + c2 + m2 <= 1` |
| `tol` | `0.01` | Gibbs: batch-means SE convergence target |
| `n_sim`, `burn_in` | `100_000`, `1000` | Gibbs: draws kept / discarded per round |
| `max_rounds` | `100` | Gibbs: cap on convergence rounds |
| `quadrature_atol` | `1e-8` | quadrature: successive mean/variance refinement tolerance, not a certified error bound |
| `quadrature_max_nodes` | `128` | quadrature: maximum nodes per active dimension; accepted range 64–512 |
| `seed` | `None` | Gibbs: integer RNG seed in `[0, 2**32 - 1]` (booleans rejected; per-family, deterministic) |
| `genetic_corrmat`, `full_corrmat`, `phen_names` | `None` | multi-trait only |

`n_sim`/`tol` trade speed for Monte-Carlo precision; the defaults converge for
typical families. PA ignores all Gibbs options.
