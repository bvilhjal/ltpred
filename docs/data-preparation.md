# Data preparation

Turn status, age and relationship records into liability bounds and families.
For a complete first run, use [Getting started](quickstart.md) or the
[register tutorial](tutorial.md). Scoring requires these inputs; fitting
heritability or estimating incidence can be a separate analysis.

## Inputs

You describe the data as a flat table with one row per observed person, grouped
by the target proband into families. Each record needs:

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
| `o` | *(optional)* the proband's own status (the "offspring"/index person) |
| `m`, `f` | mother, father |
| `s1`, `s2`, … | full siblings (numbered) |
| `mgm`, `mgf`, `pgm`, `pgf` | maternal/paternal grand-mother/-father |
| `mhs1`…, `phs1`… | maternal/paternal half-siblings |
| `mau1`…, `pau1`… | maternal/paternal aunts/uncles |
| `c1.1`, `c1.2`, … | children (partner-group `.` child index) |

A family can contain any subset of these. If `o` is absent, the estimator inserts
it with uninformative `(-inf, inf)` bounds. Two relatives of the same kind must be
numbered (`s1`, `s2`). Relatedness (and hence covariance) is derived from the role
labels — see `get_relatedness`.

One half-sib convention is worth knowing: following LTFHPlus, two **same-side**
half-sibs (`mhs1`/`mhs2`, or `phs1`/`phs2`) are related `0.5·h²` to each other,
not `0.25·h²` — the grammar implies they share the unrecorded second parent —
while a maternal/paternal half-sib pair is unrelated. If two same-side half-sibs
in fact have different second parents, the role labels cannot express that;
describe the pedigree through `kinship_from_pedigree` instead, where the pair's
relatedness comes from the recorded parents (`0.25·h²` through the single shared
parent, with no implied second-parent sharing).

Conditioning on `o` is an analysis choice, not a structural requirement. Include
it for **use II** (a GWAS phenotype from the proband's diagnosis). For **use I**
(prospective prediction/classification of that diagnosis), keep `o` and its `pid`
but make its bounds uninformative; otherwise the outcome leaks directly into the predictor.

Unnumbered `s` is a supported label for one sibling; use `s1`, `s2`, … for
multiple siblings. It is not an input error. Repeated labels still raise.

### Beyond the role grammar: arbitrary pedigrees

If the pedigree itself must be *discovered* from population parent-offspring
records, `ltpred.pedigree` does that first: `build_parent_graph(ids, father,
mother)` indexes the records and `extract_pedigree(graph, proband,
max_degree=3)` returns each proband's relatives up to third degree (parents,
siblings, grandparents, half-sibs, aunts/uncles, cousins) with all their
ancestors closed in, so the sub-pedigree's kinship is exact (see
`benchmarks/bench_pedigree_inference.py`). For the full chain -- trio records
-> pedigrees -> per-stratum CIP thresholds -> per-proband scores -- use the
supported `ltpred.estimate_liabilities` driver. Its required `use` argument
makes the observation design explicit: `"gwas"` conditions on the proband's
diagnosis, while `"prediction"` makes that diagnosis uninformative and
reconstructs every relative's record at a common calendar landmark:

```python
from ltpred import estimate_liabilities

gwas_scores = estimate_liabilities(
    ids, father, mother,
    probands=gwas_ids, status=status, age=age, use="gwas",
    strata=stratum, cip_by_stratum=cip_by_stratum,
    max_degree=3, h2=0.5,
)

prediction_scores = estimate_liabilities(
    ids, father, mother,
    probands=prediction_ids, status=status, age=age, use="prediction",
    birth_time=birth_time, index_time=index_time,
    strata=stratum, cip_by_stratum=cip_by_stratum,
    max_degree=3, h2=0.5,
)
```

Here `birth_time` and `strata` (a stratum label per person, the keys of
`cip_by_stratum`) are aligned to `ids`, and `index_time` to `probands`. They must
use one numeric calendar scale (for example decimal calendar year), and that
scale's unit must match the unit of `age`. At landmark `t`, a relative born at
`b` is censored at attained age `t - b`; assigning every generation the
proband's attained age is not familywise calendar censoring. A person born at
or after `t` has no follow-up and is uninformative. Prevalent cases and
probands whose follow-up ended before the landmark are not dropped but flagged
in `proband_state`; only `"disease_free_and_followed"` probands belong in a
prospective evaluation.

The extracted pedigree retains ancestors beyond `max_degree` when they are
needed for exact kinship, but marks them `Pedigree.closure_only`. The supported
driver leaves their diagnoses uninformative by default, so `max_degree` really
bounds the observation set. Set `condition_closure=True` only when those extra
diagnoses are deliberately part of the analysis. A non-null parent id that
matches no record is still treated as an unknown founder, but the driver counts
these per table: it reports `frac_records_with_unresolved_parents`, warns when the share is
implausibly high for a register boundary, and refuses when no non-null
reference resolves at all, which is an id-format or join mismatch rather than
missing history. The driver intentionally uses
pinned-onset LT-FH++ bounds and deterministic Pearson--Aitken inference; build
bounds and call the lower-level estimators directly for interval-case or
PA-FGRS mixture models.

When your relatives don't fit the fixed roles — deeper pedigrees, cousins,
multiple marriages, inbreeding — describe the pedigree by **who each person's
parents are** instead. `kinship_from_pedigree` turns `(ids, father, mother)` columns
into the additive relationship matrix `A`, and `estimate_liability_from_kinship`
estimates the target's liability from `A` and per-individual bounds:

```python
from ltpred import kinship_from_pedigree, estimate_liability_from_kinship
ids    = ["o", "m", "f", "s1", "mgm", "mgf"]      # target first
father = ["f", "mgf", None, "f", None, None]      # None / unlisted = unknown founder
mother = ["m", "mgm", None, "m", None, None]
_, A = kinship_from_pedigree(ids, father, mother)
# lower/upper are (n_families, n_individuals) in `ids` order (from a threshold builder)
gen, se, var = estimate_liability_from_kinship(A, lower, upper, h2=0.5, target=0)
# Pearson-Aitken is the default (its se is exactly 0 — no Monte-Carlo error);
# pass method="gibbs" for a sampler-based se. var is the posterior variance on both.
```

This high-level arbitrary-kinship function accepts `A`, `lower`/`upper`, and
on Pearson–Aitken `use_mixture=True` with per-member `K_i`/`K_pop` for the
PA-FGRS censored-control mixture. Pass `method="gibbs"` only for the
no-mixture sampler. Shared-environment scoring is also supported, but the
relationship classes must be explicit: pass `c2` with an aligned `c_kernel`
and/or `m2` with `m_kernel`. They cannot be recovered from `A` alone.

For a pedigree that *does* fit the role grammar the two paths use the identical
covariance (except for same-side half-sibs, below); Gibbs agrees to Monte-Carlo
error and PA up to its fold-order difference (about 0.1% of the score SD,
RESULTS §14); the pedigree path additionally handles half-sibs of any
degree, cousins, and inbred pedigrees (where a self-relationship can exceed 1).
For inbred pedigrees, the raw additive covariance is formed from `A` and then
standardised so every full liability has unit marginal variance; standard-normal
prevalence thresholds therefore retain their usual meaning. Build the covariance
alone with `construct_covmat_from_kinship(A, h2, target)` (and the same optional
component kernels).

Discovery from population registers is handled by `ltpred.pedigree` above, with
`max_degree` setting how far the traversal reaches. The graph-based extraction
approach is described in
[Pedersen et al. (2025)](https://doi.org/10.3389/fgene.2025.1708315), whose graph
utilities are implemented in the R package LTFHPlus. If you take that route
instead, translate `graph_to_trio` output into `ids`, `father` and `mother`, then
call ltpred's
`kinship_from_pedigree`. Do **not** pass LTFHPlus `get_kinship` output straight to
ltpred: it can already contain heritability scaling and target augmentation, while
ltpred's kinship-matrix API expects the unscaled additive relationship matrix
`A = 2φ` and applies `h2` itself.

### Getting `lower`/`upper` from status and age

Three helpers turn status (+age) into different truncation encodings. Pick the one
that matches the intended model:

```python
import numpy as np
from ltpred import prevalence_thresholds, age_thresholds, pa_thresholds

status = np.array([1, 0, 1, 0])          # 1 = case
age    = np.array([45, 70, 52, 33])      # onset age for cases, follow-up age for controls

# (1) classic LT-FH — one prevalence threshold, no age
lower, upper = prevalence_thresholds(status, pop_prev=0.05)

# (2) personalised pinned bounds — LT-FH++ with relatives, ADuLT without them
lower, upper = age_thresholds(status, age, pop_prev=0.05)

# (3) Age-dependent PA-FGRS-style variant — adds K_i, K_pop for the mixture
lower, upper, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)
```

The builders determine **how an observation is encoded**. They do not, by
themselves, distinguish LT-FH++ from ADuLT: personalised pinned bounds plus
relative rows are LT-FH++; the same bounds with only role `o` are ADuLT.

| builder | case encoding | control encoding | extra outputs | model |
|---|---|---|---|---|
| `prevalence_thresholds` | `(T, ∞)` | `(-∞, T)` | — | classic **LT-FH** (no age) |
| `age_thresholds` | **pinned** `[thresh(onset), thresh(onset)]` | `(-∞, thresh(age))` | — | personalised demo: **LT-FH++ with relatives; ADuLT without** |
| `pa_thresholds` | interval `(thresh(onset), ∞)` — **not** pinned | `(-∞, thresh(age))` | `K_i`, `K_pop` | age-dependent **PA-FGRS-style variant**, not base PA-FGRS |

`T = Φ⁻¹(1 − K)`. Note the two age-aware builders are **not** interchangeable:
`age_thresholds` *pins* a case's liability at its onset threshold (a point mass —
the deterministic age-of-onset map), whereas `pa_thresholds` bounds it *above*
that threshold (an interval) and adds the per-person cumulative incidence `K_i`
and lifetime prevalence `K_pop` used by the optional censored-control mixture
(`use_mixture=True`). Pinning a case at `thresh(age_of_onset)` assumes a
**deterministic monotone mapping** from onset age to liability — it treats onset
age as strictly stronger information than merely being affected. That is powerful,
but should be checked when diagnosis age is noisy, delayed, or shaped by
health-care access. The interval makes a weaker mechanistic assumption; it does
**not** necessarily produce a smaller liability score or a statistically more
conservative analysis.

The published **base PA-FGRS** model uses the lifetime threshold for every
observed case; age-specific incidence enters only through the censored-control
mixture. With the built-in logistic curve, construct those inputs as follows:

```python
# Paper-faithful base PA-FGRS: lifetime bounds plus control-specific mixture data.
import numpy as np
from ltpred import (
    estimate_liability, families_from_columns, pa_thresholds,
    prevalence_thresholds,
)

lower, upper = prevalence_thresholds(status, pop_prev=0.05)
_, _, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)

# This is an explicit analysis decision, not an estimate_liability argument.
include_proband_status = True   # True for GWAS-phenotype construction
if not include_proband_status: # use False when predicting/classifying this disease
    is_proband = np.asarray(role) == "o"
    lower, upper, K_i, K_pop = (x.copy() for x in (lower, upper, K_i, K_pop))
    lower[is_proband], upper[is_proband] = -np.inf, np.inf
    K_i[is_proband], K_pop[is_proband] = np.nan, np.nan

res = estimate_liability(
    families_from_columns(
        fam_id=fam_id, role=role, lower=lower, upper=upper,
        K_i=K_i, K_pop=K_pop,
    ),
    h2=0.5,
    use_mixture=True,
)
```

For real data, obtain `K_i` from the population CIP rather than the logistic demo,
retain lifetime bounds from `prevalence_thresholds` (using the matching stratum's
`K_pop`), and provide `K_i`/`K_pop` only for controls. Instead of unbinding `o` as
above, you may omit its row only when no member has a `pid`; the output then
uses `fam_id` as its join key. With personal IDs, retain `o` and its `pid` with
`(-inf, inf)` bounds. Missing or absent proband pids raise rather than silently
changing the join key.

You can also build the bounds yourself: any `(lower, upper)` interval per person
is valid (`lower == upper` pins a liability exactly; `(-inf, inf)` is
uninformative).

> **CIPs and sex/cohort.** ltpred has no explicit `sex` argument. In LT-FH++ and
> ADuLT, sex, birth year and cohort enter through the personalised thresholds you
> supply; in base PA-FGRS they enter a censored control's mixture weight through
> `K_i`. None of them changes the covariance model. The built-in
> `convert_age_to_cir` is a **logistic placeholder for simulation and demos**; for
> real analyses use externally estimated, **population-representative** cumulative
> incidence curves stratified by sex, birth year/cohort, ancestry and calendar
> period. `thresholds_from_cip(status, age, cip_ages, cip_values, ...)` takes such
> a curve directly (call it once per stratum) and returns `lower, upper, K_i,
> K_pop` — use it instead of the logistic builders for real data. CIPs from an
> ascertained biobank sample, or that ignore competing risks (death, emigration),
> can bias the estimate.
>
> `thresholds_from_cip` interpolates within the supplied age grid and holds the
> first or last CIP constant outside it; it does **not** extrapolate incidence.
> Make the grid cover every onset/follow-up age you will analyse. If `k_pop` is
> omitted, the helper uses `max(cip_values)` as lifetime prevalence, which is
> appropriate only when the curve reaches the intended lifetime horizon. Pass a
> separately justified lifetime prevalence otherwise.

#### Estimating the CIP from follow-up records

If the CIP itself must be estimated from health data, `ltpred.cip` provides the
two standard estimators, following the LT-FH++ construction (Pedersen et al.
2022: Aalen-Johansen with death and emigration as competing events, one curve
per sex x birth-year stratum). The full methodology -- input format, risk
sets, formulas, variance estimators, estimand choice, stratification, worked
example, edge cases -- is in [CIP estimation](cip-estimation.md):

```python
from ltpred import aalen_johansen_cip, kaplan_meier_cip

# per person: age at entry into follow-up (0, register start, or immigration),
# age at exit, and what happened at exit (0 censoring, 1 diagnosis, 2 death)
curve = aalen_johansen_cip(age_entry, age_exit, event_type)   # competing risks
# Values below 1 feed thresholds_from_cip; an exhausted terminal risk set can
# yield exactly 1, which has no finite probit threshold. curve.se is the
# finite-risk-set, tie-correct Aalen (1978) SE.
```

Use `aalen_johansen_cip` when the target is the **crude diagnosed
proportion** in a population where death can preclude diagnosis (the LT-FH++
convention; which curve the thresholds need is discussed under
[the estimand choice](cip-estimation.md#the-estimand-choice-the-most-important-decision-on-this-page)). Death remains
a competing event even if death and diagnosis are statistically independent.
Treating death as censoring asks for the different, hypothetical no-death
**net risk** and generally overestimates the diagnosed proportion (quantified
in `benchmarks/bench_cip_estimation.py`). Use `kaplan_meier_cip` only when net
risk is the intended estimand and non-event censoring is independent. Both
estimators handle left truncation (delayed entry) when entry is independent of
the event process conditional on the modelled strata and the risk sets overlap.
Ordinary right censoring must likewise be conditionally non-informative.
Stratify by calling the estimator once per stratum (sex, birth-year band) and
passing each group of relatives the curve of their stratum.

#### A real register-data recipe

The end-to-end path for register data uses your **own** cumulative-incidence curve
(not the logistic demo), one call to `thresholds_from_cip` per stratum, and PA:

```python
import numpy as np
from ltpred import thresholds_from_cip, families_from_columns, estimate_liability

# One label per observed family-member row, in the same order as status/age.
# Each mapping value is (CIP ages, CIP values, lifetime prevalence).
stratum = np.asarray(sex_birth_cohort_ancestry)
cip_by_stratum = {
    "F_1950_EUR": (ages_f50, cip_f50, lifetime_f50),
    "M_1950_EUR": (ages_m50, cip_m50, lifetime_m50),
    # ...all strata represented in `stratum`
}
status, age = np.asarray(status), np.asarray(age)
lower = np.empty(status.shape, dtype=float)
upper = np.empty(status.shape, dtype=float)
K_i = np.full(status.shape, np.nan)
K_pop = np.full(status.shape, np.nan)

case_mode = "pin"  # LT-FH++; "interval" selects the age-dependent PA-style variant
for label in np.unique(stratum):
    mask = stratum == label
    cip_ages, cip_values, lifetime_prevalence = cip_by_stratum[label]
    lo, hi, ki, kp = thresholds_from_cip(
        status=status[mask],
        age=age[mask],  # onset age for cases; last follow-up for controls
        cip_ages=cip_ages,
        cip_values=cip_values,
        k_pop=lifetime_prevalence,
        case_mode=case_mode,
    )
    lower[mask], upper[mask] = lo, hi
    K_i[mask], K_pop[mask] = ki, kp  # restore the original row order

families = families_from_columns(
    fam_id=fam_id, role=role, lower=lower, upper=upper,
    pid=pid, K_i=K_i, K_pop=K_pop,
)
res = estimate_liability(
    families, h2=0.5,
    use_mixture=(case_mode == "interval"),
)
```

Key points:

- `case_mode="pin"` (the default) is the personalised onset-pinned encoding:
  LT-FH++ with relative rows, ADuLT with role `o` only;
  `case_mode="interval"` is an age-dependent PA-FGRS-style interval encoding,
  not the published base model or an exact implementation of PA-FGRS_ADT.
- `use_mixture=True` only makes sense when `K_i` / `K_pop` are supplied (from
  `pa_thresholds` or `thresholds_from_cip`); it is the PA-FGRS
  age-censored-control correction and is supported by the PA engine only.
- Estimate the **CIP from population-representative data**, using
  [`ltpred.cip`](cip-estimation.md) or an external estimator. Stratify by sex,
  birth cohort, ancestry and calendar period, account for competing risks, and
  pass one stratum's curve per call.
- Ensure each stratum's age grid covers the analysed ages. Values outside the grid
  use the nearest endpoint rather than extrapolation, and `k_pop` should be passed
  explicitly unless the last CIP value is a defensible lifetime prevalence.

For **ADuLT**, use the same stratum-specific proband bounds but build one row per
person with `role="o"`; do not add relative rows. There is no separate ADuLT
inference switch—the absence of family history is the distinction. ADuLT is a
diagnosis-derived phenotype construction; unbinding its only `o` observation leaves
no information for disease prediction.

### Getting heritability on the liability scale

Heritability must be on the **liability** scale. If you only have an
observed/case-control-scale estimate, convert it (Lee et al. 2011):

```python
from ltpred import observed_to_liability_h2

# Use the case fraction in the same analysed sample that produced obs_h2.
study_case_fraction = n_cases / sample_size
h2_liab = observed_to_liability_h2(
    obs_h2=0.15,
    pop_prev=0.05,
    prop_cases=study_case_fraction,
)
```

`prop_cases` is the **observed study case fraction**, not the population
prevalence and not a generic `0.5`. Omit it only for a population-representative
sample; for an ascertained case-control study, pass the actual analysed
fraction.

The full transformation toolkit lives in `ltpred.liability_scale`: both
directions of the Lee et al. (2011) h² bridge
(`observed_to_liability_h2` / `liability_to_observed_h2`), and probit estimation
of residual-scale genetic variance (`probit_liability_r2`,
`liability_r2_from_z`). Despite their compatibility names, the default output
is `q = 2 f (1-f) beta²` in probit residual-variance units, not a fraction of
total liability variance. For independent or suitably LD-pruned variants, sum
`q` first; only then convert the aggregate to the total-variance fraction
`q / (1 + q)`.

**Which `h²`?** The right value is the additive genetic variance component you want
the family covariance to represent — the model is additive-genetic only (see
[algorithm.md](algorithm.md#connection-to-selection-index-and-blup)). A pedigree /
twin **narrow-sense** estimate captures more of the family-history signal but can
be inflated by shared environment, assortative mating or indirect genetic effects
if those are not separately modelled; a **SNP-heritability** estimate is smaller
and understates the total additive resemblance between relatives that the
`h²·A` covariance represents, so treat it as a conservative lower bound. Neither
is uniquely "correct",
so run a **sensitivity analysis** over plausible `h²` values (and prevalence/CIP)
and check how much the score and downstream results move — see
[Inference](inference.md#sensitivity-to-the-assumed-heritability). Don't have any external
value? [Fit `h²` from the families themselves](inference.md#fitting-heritability-from-the-family-data),
or get a fast, fitting-free cross-check from **tetrachoric correlations**
(`ltpred.tetrachoric`): the tetrachoric correlation between two relatives'
case/control statuses estimates their latent liability correlation directly
from the 2x2 table, and parent–offspring pairs give `h² / 2` under an additive
model without shared or transmitted environment (the classic Falconer route;
siblings also carry shared environment). The table's thresholds come from the
sample's own case rates, so the check assumes population sampling and
age-complete statuses — relatives still young and undiagnosed pull it down:

```python
from ltpred import tetrachoric
r = tetrachoric(proband_status, mother_status)   # -> r.rho, r.se
h2_falconer = 2 * r.rho
```

Pairwise tetrachorics are also a useful model *diagnostic*: compare them with
the `h² × A` the fitted covariance implies (benchmarked in
`benchmarks/bench_tetrachoric.py`).

### Preparing multiple traits for covariance fitting

For `fit_pairwise_multi`, use role-based, independent families and **one
common prevalence per trait**. Prepare an `(n_people, n_traits)` status
matrix in a fixed trait order, with `0`/`1` for observed diagnoses and `NaN`
for missing ones. The same person occupies one row across traits:

```python
import numpy as np
from ltpred import families_from_columns, prevalence_thresholds

# status: your person-by-trait matrix; prevalence: one population value per trait
lower = np.full(status.shape, -np.inf)
upper = np.full(status.shape, np.inf)
for p, k in enumerate(prevalence):
    observed = ~np.isnan(status[:, p])
    lower[observed, p], upper[observed, p] = prevalence_thresholds(
        status[observed, p], pop_prev=k,
    )
two_trait_families = families_from_columns(
    fam_id, role, lower, upper, pid=pid,
)
```

Missing cells remain `(-inf, inf)`; never recode them as controls.
Personalised CIP/onset bounds are for scoring and are rejected by this
fitter. Keep `phen_names` in column order and any IPW weights in family order
(the first appearance of each `fam_id`). Person IDs permit overlap checks;
independence remains a study-design requirement. Identification uses jointly
observed pairs, so entirely missing relatives add no information. Dropping
missing observations does not correct informative missingness. The
[tutorial example](tutorial.md#step-5-two-traits-at-once)
fits the resulting data; [Inference](inference.md#joint-heritability-and-cross-trait-correlations)
states the sampling contract and uncertainty limits.
