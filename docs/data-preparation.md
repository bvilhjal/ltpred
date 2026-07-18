# Data preparation

How to turn a registry-style status/age table into the inputs `estimate_liability`
needs: families of members with liability bounds. For the end-to-end flow first, see
the [quickstart](quickstart.md); this page is the reference for each piece.

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
gen, var = estimate_liability_from_kinship(A, lower, upper, h2=0.5, target=0)
# Pearson-Aitken is the default; pass method="gibbs" to receive Monte-Carlo SE instead.
```

For a pedigree that *does* fit the role grammar the two paths give identical
results (same covariance); the pedigree path additionally handles half-sibs of any
degree, cousins, and inbred pedigrees (where a self-relationship can exceed 1).
For inbred pedigrees, the raw additive covariance is formed from `A` and then
standardised so every full liability has unit marginal variance; standard-normal
prevalence thresholds therefore retain their usual meaning. Build the covariance
alone with `construct_covmat_from_kinship(A, h2, target)`.

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

# (2) personalised pinned bounds — LT-FH++ with relatives, ADuLT without them
lower, upper = age_thresholds(status, age, pop_prev=0.05)

# (3) PA-FGRS — like (1)/(2) but also emits K_i, K_pop for the censoring mixture
lower, upper, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)
```

The builders determine **how an observation is encoded**. They do not, by
themselves, distinguish LT-FH++ from ADuLT: personalised pinned bounds plus
relative rows are LT-FH++; the same bounds with only role `o` are ADuLT.

| builder | case encoding | control encoding | extra outputs | model |
|---|---|---|---|---|
| `prevalence_thresholds` | `(T, ∞)` | `(-∞, T)` | — | classic **LT-FH** (no age) |
| `age_thresholds` | **pinned** `[thresh(onset), thresh(onset)]` | `(-∞, thresh(age))` | — | personalised demo: **LT-FH++ with relatives; ADuLT without** |
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
>
> `thresholds_from_cip` interpolates within the supplied age grid and holds the
> first or last CIP constant outside it; it does **not** extrapolate incidence.
> Make the grid cover every onset/follow-up age you will analyse. If `k_pop` is
> omitted, the helper uses `max(cip_values)` as lifetime prevalence, which is
> appropriate only when the curve reaches the intended lifetime horizon. Pass a
> separately justified lifetime prevalence otherwise.

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

case_mode = "pin"  # LT-FH++ with relatives; change to "interval" for PA-FGRS
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
  `case_mode="interval"` is the conservative PA-FGRS interval encoding.
- `use_mixture=True` only makes sense when `K_i` / `K_pop` are supplied (from
  `pa_thresholds` or `thresholds_from_cip`); it is the PA-FGRS
  age-censored-control correction and is supported by the PA engine only.
- Estimate the **CIP outside ltpred** from population-representative register data,
  stratified by sex, birth cohort, ancestry and calendar period, and accounting for
  competing risks — then pass one stratum's curve per call.
- Ensure each stratum's age grid covers the analysed ages. Values outside the grid
  use the nearest endpoint rather than extrapolation, and `k_pop` should be passed
  explicitly unless the last CIP value is a defensible lifetime prevalence.

For **ADuLT**, use the same stratum-specific proband bounds but build one row per
person with `role="o"`; do not add relative rows. There is no separate ADuLT
inference switch—the absence of family history is the distinction.

### Getting heritability on the liability scale

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
and check how much the score and downstream results move — see
[Inference](inference.md#sensitivity-to-the-assumed-heritability). Don't have any external
value? [Fit `h²` from the families themselves](inference.md#fitting-heritability-from-the-family-data).
