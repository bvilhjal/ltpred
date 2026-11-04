# Data preparation

Turn status, age and relationships into liability bounds and families. For a
first run see [Getting started](quickstart.md) or the [register tutorial](tutorial.md).

## Inputs

Use one row per observed person, grouped by target proband:

| column | meaning |
|---|---|
| `fam_id` | family/proband group |
| `role` | relationship to the proband |
| `lower`, `upper` | liability interval, derived from status and optionally age |
| `pid` | *(optional)* personal identifier |
| `K_i`, `K_pop` | *(optional)* current and lifetime incidence for the PA mixture |

Do not supply genetic role `g`; the estimator adds and returns that coordinate.

### Role grammar

| role | who |
|---|---|
| `o` | *(optional)* proband's own status |
| `m`, `f` | mother, father |
| `s`, `s1`, `s2`, … | full sibling(s); number repeated roles |
| `mgm`, `mgf`, `pgm`, `pgf` | maternal/paternal grandparents |
| `mhs1`…, `phs1`… | maternal/paternal half-siblings |
| `mau1`…, `pau1`… | maternal/paternal aunts/uncles |
| `c1.1`, `c1.2`, … | children (partner-group `.` child index) |

Any subset is allowed. Duplicate roles raise. With no personal ids, missing `o`
is inserted as uninformative. With pids, retain `o` and its pid even when its
bounds are `(-inf, inf)`. Include own diagnosis for a GWAS; leave it
uninformative for prediction and censor relatives at the landmark.

Same-side half-sib sets (`mhs1`/`mhs2` or `phs1`/`phs2`) follow LTFHPlus's
shared-other-parent convention and are related `0.5·h²` to each other, not
`0.25·h²`. If their other parents differ, the role grammar cannot represent
that: use a pedigree matrix instead.

### Beyond the role grammar: arbitrary pedigrees

For population parent-pointer data, use `estimate_liabilities`. It discovers
each proband's pedigree, applies CIP bounds and scores in one call:

```python
from ltpred import estimate_liabilities

gwas_scores = estimate_liabilities(
    ids, father, mother, probands=gwas_ids, status=status, age=age,
    use="gwas", strata=stratum, cip_by_stratum=cip_by_stratum,
    max_degree=3, h2=0.5)

prediction_scores = estimate_liabilities(
    ids, father, mother, probands=prediction_ids, status=status, age=age,
    use="prediction", birth_time=birth_time, index_time=index_time,
    strata=stratum, cip_by_stratum=cip_by_stratum,
    max_degree=3, h2=0.5)
```

`birth_time` and `strata` align to `ids`; `index_time` aligns to `probands`.
Their calendar unit must match `age`. At landmark `t`, a relative born at `b`
is censored at age `t - b`; people not yet born are uninformative. Only
`proband_state == "disease_free_and_followed"` belongs in a prospective
incident-risk evaluation.

Pedigree extraction includes every recorded ancestor needed for exact kinship.
Ancestors beyond `max_degree` are marked `closure_only` and their diagnoses are
ignored unless `condition_closure=True`. Unlisted parents are founders, but
`frac_records_with_unresolved_parents` exposes a likely register boundary or
broken join; no resolved non-null parent reference raises. This driver uses
pinned-onset LT-FH++ and PA. Other observation models use the lower-level APIs.

For deeper pedigrees, cousins, multiple marriages or inbreeding, construct the
additive matrix directly:

```python
from ltpred import kinship_from_pedigree, estimate_liability_from_kinship

ids    = ["o", "m", "f", "s1", "mgm", "mgf"]      # target first
father = ["f", "mgf", None, "f", None, None]      # None / unlisted = founder
mother = ["m", "mgm", None, "m", None, None]
_, A = kinship_from_pedigree(ids, father, mother)
# lower/upper: (n_families, n_individuals), aligned to ids
gen, se, var = estimate_liability_from_kinship(A, lower, upper, h2=0.5, target=0)
```

PA is the default; use `method="gibbs"` for a sampler-based SE. `var` is
posterior variance on both. The PA-only mixture accepts aligned `K_i`/`K_pop`.
C/M scoring requires explicit aligned `c_kernel`/`m_kernel` matrices because
those environmental classes cannot be recovered from `A`. Inbred full
liabilities are standardised to unit variance, preserving the threshold scale.
`construct_covmat_from_kinship` exposes the covariance alone.

Role and pedigree paths agree when they encode the same relationships (apart
from the half-sib convention above and PA fold order). The graph-extraction
approach follows [Pedersen et al. 2025](https://doi.org/10.3389/fgene.2025.1708315).
Do not pass LTFHPlus `get_kinship` output directly: ltpred expects the unscaled
`A = 2φ` and applies `h2` itself.

### Getting `lower`/`upper` from status and age

```python
import numpy as np
from ltpred import prevalence_thresholds, age_thresholds, pa_thresholds

status = np.array([1, 0, 1, 0])          # 1 = case
age    = np.array([45, 70, 52, 33])      # onset for cases, last follow-up for controls

lower, upper = prevalence_thresholds(status, pop_prev=0.05)
lower, upper = age_thresholds(status, age, pop_prev=0.05)
lower, upper, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)
```

| builder | cases | controls | outputs and scope |
|---|---|---|---|
| `prevalence_thresholds` | `(T, ∞)` | `(-∞, T)` | bounds; classic LT-FH |
| `age_thresholds` | pinned at `T(onset)` | `(-∞, T(age))` | bounds; logistic demonstration |
| `pa_thresholds` | `(T(onset), ∞)` | `(-∞, T(age))` | bounds plus `K_i`, `K_pop`; age-dependent PA-FGRS-style variant |

Here `T = Φ⁻¹(1 − K)`. Personalised bounds with relatives define LT-FH++;
with role `o` only they define ADuLT. Pins assume a deterministic monotone
onset–liability map and can over-condition noisy diagnosis ages. Intervals
make a weaker assumption but are not automatically more conservative.

**Base PA-FGRS is different:** cases use the lifetime threshold and age enters
only the control mixture. Construct lifetime bounds and control mixture data
separately:

```python
from ltpred import estimate_liability, families_from_columns, pa_thresholds, prevalence_thresholds

lower, upper = prevalence_thresholds(status, pop_prev=0.05)
_, _, K_i, K_pop = pa_thresholds(status, age, pop_prev=0.05)

# For prediction, copy the arrays and set the proband's bounds to (-inf, inf)
# and K_i/K_pop to NaN. For a GWAS, include the proband's diagnosis.
res = estimate_liability(
    families_from_columns(fam_id=fam_id, role=role, lower=lower, upper=upper,
                          K_i=K_i, K_pop=K_pop),
    h2=0.5, use_mixture=True)
```

For real data, obtain `K_i` from an appropriate population CIP rather than the
logistic demo. Any explicit interval is also valid: `lower == upper` is a pin
and `(-inf, inf)` is uninformative.

> **CIPs and sex/cohort.** There is no `sex` argument. Sex, birth year and
> cohort enter personalised thresholds, or base PA-FGRS mixture weights through
> `K_i`; they do not change covariance. Use population-representative curves,
> not CIPs estimated from an ascertained biobank sample. The choice between net
> and competing-risk incidence is an [estimand decision](cip-estimation.md#the-estimand-choice-the-most-important-decision-on-this-page).
>
> `thresholds_from_cip` interpolates within its age grid and holds endpoint
> values constant outside it. Cover all analysed ages. Its default
> `k_pop = max(cip_values)` is appropriate only when the curve reaches the
> intended lifetime horizon; otherwise supply `k_pop` explicitly.

#### Estimating the CIP from follow-up records

`ltpred.cip` implements Kaplan–Meier and Aalen–Johansen estimation with delayed
entry and right censoring:

```python
from ltpred import aalen_johansen_cip, kaplan_meier_cip

# age_entry/age_exit per person; event: 0 censoring, 1 diagnosis, 2 death
curve = aalen_johansen_cip(age_entry, age_exit, event_type)
```

Call once per population stratum, then feed `curve.ages` and `curve.values` to
`thresholds_from_cip`. An exact terminal CIP of one has no finite probit
threshold. [CIP estimation](cip-estimation.md) owns the risk sets, equations,
variance estimators and examples.

#### A real register-data recipe

```python
import numpy as np
from ltpred import thresholds_from_cip, families_from_columns, estimate_liability

# One stratum label per observed member row; each value is
# (CIP ages, CIP values, lifetime prevalence).
stratum = np.asarray(sex_birth_cohort_ancestry)
cip_by_stratum = {
    "F_1950_EUR": (ages_f50, cip_f50, lifetime_f50),
    "M_1950_EUR": (ages_m50, cip_m50, lifetime_m50),
}
status, age = np.asarray(status), np.asarray(age)
lower = np.empty(status.shape, dtype=float)
upper = np.empty(status.shape, dtype=float)
K_i = np.full(status.shape, np.nan)
K_pop = np.full(status.shape, np.nan)

case_mode = "pin"  # or "interval" for the age-dependent PA-style variant
for label in np.unique(stratum):
    mask = stratum == label
    cip_ages, cip_values, lifetime_prevalence = cip_by_stratum[label]
    lo, hi, ki, kp = thresholds_from_cip(
        status[mask], age[mask], cip_ages, cip_values,
        k_pop=lifetime_prevalence, case_mode=case_mode)
    lower[mask], upper[mask] = lo, hi
    K_i[mask], K_pop[mask] = ki, kp

families = families_from_columns(
    fam_id, role, lower, upper, pid=pid, K_i=K_i, K_pop=K_pop)
res = estimate_liability(
    families, h2=0.5, use_mixture=(case_mode == "interval"))
```

`case_mode="pin"` is onset-pinned LT-FH++/ADuLT. `"interval"` with the mixture
is the age-dependent PA-FGRS-style variant, not base PA-FGRS or PA-FGRS_ADT.
Mixture input is PA-only and requires valid control K pairs. For ADuLT, use the
same personalised bounds but only role `o`; there is no separate engine switch.

### Getting heritability on the liability scale

Convert observed-scale estimates with the study's actual case fraction:

```python
from ltpred import observed_to_liability_h2

study_case_fraction = n_cases / sample_size
h2_liab = observed_to_liability_h2(
    obs_h2=0.15, pop_prev=0.05, prop_cases=study_case_fraction)
```

Omit `prop_cases` only for a population-representative sample. The related
helpers provide the inverse bridge and probit variance `q = 2 f (1-f) beta²`.
Sum independent/LD-pruned `q` values before converting to total-liability
fraction `q/(1+q)`.

Use the additive variance component you want the family covariance to
represent. Pedigree/twin narrow-sense estimates capture more family signal but
can include unmodelled shared or indirect effects; SNP heritability is usually
a conservative lower bound for the `h²·A` covariance. Run sensitivity analyses
over plausible `h²` and CIP values ([example](inference.md#sensitivity-to-the-assumed-heritability)).
When the sampling contract holds, [fit h² from the families](inference.md#fitting-heritability-from-the-family-data).
Tetrachoric relative-pair correlations are a fitting-free diagnostic:

```python
from ltpred import tetrachoric
r = tetrachoric(proband_status, mother_status)
h2_falconer = 2 * r.rho       # parent–offspring, additive A+E model
```

That shortcut assumes population sampling, age-complete statuses and no shared
or transmitted environment. Compare pairwise estimates with the `h² × A`
covariance you intend to use.

### Preparing multiple traits for covariance fitting

Use independent role families and one common prevalence per trait. Missing
phenotypes are `NaN` in the status matrix and remain `(-inf, inf)` bounds—not
controls:

```python
import numpy as np
from ltpred import families_from_columns, prevalence_thresholds

# status: (n_people, n_traits); prevalence: one value per trait
lower = np.full(status.shape, -np.inf)
upper = np.full(status.shape, np.inf)
for p, k in enumerate(prevalence):
    observed = ~np.isnan(status[:, p])
    lower[observed, p], upper[observed, p] = prevalence_thresholds(
        status[observed, p], pop_prev=k)
two_trait_families = families_from_columns(fam_id, role, lower, upper, pid=pid)
```

Keep `phen_names` in trait-column order and IPW weights in first-appearance
family order. Personalised CIP/onset bounds are rejected by this fitter.
Identification uses jointly observed pairs; entirely missing relatives add no
information and dropping missingness does not correct informative missingness.
See [joint inference](inference.md#joint-heritability-and-cross-trait-correlations)
and the [tutorial](tutorial.md#step-5-two-traits-at-once).
