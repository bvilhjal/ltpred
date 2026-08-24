# Estimating cumulative incidence (CIP) from health data

The personalised thresholds of LT-FH++, ADuLT and PA-FGRS all consume a
**cumulative-incidence proportion** — the fraction of people in a population
stratum who are diagnosed before a given age. For uses I and II of the
[vignette](vignette.md) that curve is an *input* to the thresholds. For use III
it can *be* the result: the age, sex and cohort pattern of incidence, without
scoring families.

```
CIP(age | stratum) = P(diagnosed at or before `age`),   T = Phi^-1(1 - CIP).
```

This page describes how to estimate that curve from registry-style health
data with `ltpred.cip`: the input format, the two estimators (Kaplan-Meier and
Aalen-Johansen) with their formulas and variance estimators, the estimand
choice that matters most in practice, stratification, worked examples, and the
edge cases. The construction follows the LT-FH++ paper (Pedersen et al. 2022)
and the Danish register-epidemiology literature (Andersen, Borgan, Gill &
Keiding 1993; Plana-Ripoll, Dalsgaard, Beck and colleagues).

## When you need this page

If you already have a published CIP (e.g. from the LT-FH++ supplementary
tables or a national registry report), skip to
[data preparation](data-preparation.md) and feed it to
`thresholds_from_cip`. Come here when the CIP must be estimated from raw
follow-up records: a study cohort, a hospital registry, or a new
population/phenotype without published curves.

## Input: follow-up records

Every estimator takes three 1-D arrays of equal length, one row per person:

| array | meaning | examples |
|---|---|---|
| `age_entry` | age when follow-up starts | `0` (birth), age at register start, age at immigration, a disorder-specific minimum onset age |
| `age_exit` | age when follow-up ends | age at diagnosis, death, emigration, or the administrative end date minus birth date |
| event code | what happened at exit | `0` censored (emigration, administrative end), `1` diagnosis, `2` death (or another competing event) |

Conventions (the counting-process convention of Andersen, Borgan, Gill &
Keiding 1993, and of the Danish register papers):

- **Age is the time scale**, not calendar time. Calendar effects enter only
  through the entry/exit ages and through stratification by birth cohort.
- A person is **at risk** at age `t` when `age_entry < t <= age_exit`.
  **Delayed entry** (left truncation — the register starts mid-life, or a
  person immigrates) is handled by `age_entry > 0`: they contribute risk only
  from entry. Identification requires entry to be independent of the event
  process conditional on the modelled strata, with overlapping risk-set
  support; risk-set bookkeeping cannot repair informative entry.
- Events and censorings happen at `age_exit`; persons with
  `age_exit == age_entry` contribute no follow-up. Ties are grouped on unique
  exit ages.
- Right censoring must be non-informative conditional on the modelled strata.
  Coding an outcome-related loss process as ordinary censoring biases either
  estimator.
- Persons with `age_exit < age_entry` are rejected, as are non-finite ages.

For `kaplan_meier_cip` the event array is boolean (`1` = event of interest,
`0` = censored); for `aalen_johansen_cip` it is an integer code with `0` =
censored, `cause` = event of interest (default `1`), and any other positive
code a competing event (e.g. `2` = death without diagnosis).

## Kaplan-Meier (`kaplan_meier_cip`)

The product-limit estimator. At each unique event age `t_j`, let `d_j` be the
number of events and `Y_j` the number at risk; then

```
S(t) = prod_{t_j <= t} (1 - d_j / Y_j),        CIP(t) = 1 - S(t)
```

with the Greenwood variance

```
Var(S(t)) = S(t)^2 * sum_{t_j <= t} d_j / (Y_j (Y_j - d_j)) .
```

**When it is appropriate:** when the target is **net risk** in a hypothetical
world without the censoring process, and censoring is independent of the event
process (for example, administrative end of follow-up).

**When it is not:** when the target is the proportion actually diagnosed in
the presence of death. Death then remains a competing event even if death and
diagnosis times are statistically independent. Censoring the dead asks a
different, hypothetical no-death question and `1 - KM` generally
**overestimates** the crude diagnosed proportion — increasingly at older ages
(measured directly in `benchmarks/bench_cip_estimation.py`: an absolute
overestimation of 0.022 at a 42% death share). Use Aalen-Johansen and put death
in a competing-event code.

## Aalen-Johansen (`aalen_johansen_cip`)

The competing-risks estimator. With `d_j` the number of events of *any* type
at `t_j`, `d_kj` the events of the cause of interest, and `S` the overall
event-free survival (any event),

```
S(t) = prod_{t_j <= t} (1 - d_j / Y_j),
F_k(t) = sum_{t_j <= t} S(t_j-) * d_kj / Y_j .
```

`F_k` is the **crude (marginal) cumulative incidence**: the probability of
being diagnosed by age `t` *in the presence of* death and emigration. This is
what LT-FH++ estimated for its CIPs ("the cumulative incidence function for
each disorder was estimated with the Aalen-Johansen approach considering death
and emigration as competing events", Pedersen et al. 2022, one curve per sex
and birth year), and it is the right estimand for the threshold construction:
a person who died undiagnosed cannot be diagnosed later, and the population
fraction *diagnosed* by an age is the crude one.

Pointwise standard errors use the finite-risk-set, tie-correct **Aalen (1978)
variance** reported by `cmprsk::cuminc`. The implementation follows its grouped
recurrence, because a closed form that collapses all causes into `d_j` is not
equivalent when target and competing causes are tied at the same age.

At age `t_j`, let `d_rj` be the count for group `r` (target `k` or all other
causes), `S_j+` the post-event survival, and define

```text
q_rj = 1                                   if d_rj = 1
        1 - (d_rj - 1) / (Y_j - 1)         otherwise,       (Equation 1)

a_rj = S(t_j-)^2 q_rj d_rj / Y_j^2 .                       (Equation 2)
```

Three running sums `(v1, v2, v3)` start at zero. For competing events use
`u = F_k(t_j) / S_j+`, `w = 1 / S_j+`; for target events use
`u = 1 + F_k(t_j) / S_j+`, `w = 1 / S_j+`. Each non-empty group updates

```text
v1 <- v1 + u^2 a_rj
v2 <- v2 + w u a_rj
v3 <- v3 + w^2 a_rj,                                      (Equation 3)

Var F_k(t_j) = v1 + F_k(t_j)^2 v3 - 2 F_k(t_j) v2.         (Equation 4)
```

When `S_j+ = 0`, the competing-group update is skipped. If target events are
present, their removable boundary uses `w = 0` and `u = 1`; this is the finite
`cmprsk` convention. The separate target/other updates are essential for tied
causes. Replacing them with `d_j / Y_j^2` gives only a large-risk-set
approximation. With a single event type the Aalen and Greenwood variances are
asymptotically equivalent, not finite-sample identities.

## The estimand choice (the most important decision on this page)

Three quantities are easily confused:

- **Net incidence** (`1 - Kaplan-Meier`): incidence in a hypothetical world
  without death. Almost never what a health registry means by the observed
  diagnosed proportion.
- **Crude / marginal cumulative incidence** (Aalen-Johansen): the
  real-population probability of being diagnosed by age `t` while death
  removes people. What LT-FH++ used, and what `thresholds_from_cip` expects.
- **Plain proportions by age** (`#diagnosed / #people of that age`): only
  valid for a birth cohort with essentially complete follow-up past the
  target age. On modern, heavily right-censored cohorts it under-counts late
  onsets and can bias badly; do not use it for iPSYCH-era birth years.

`benchmarks/bench_cip_estimation.py` measures all of this on a simulated
registry with a known curve: AJ recovers the crude curve to ~0.002 absolute
error at N = 50,000 (with or without delayed entry), while KM-as-censoring
overshoots the crude curve by ~0.022 at a 42% death share.

## Stratification

CIPs are stratum-specific: LT-FH++ uses one curve per **sex x birth year**.
Estimate each curve separately by calling the estimator once per stratum:

```python
from ltpred import aalen_johansen_cip, thresholds_from_cip

curves = {}
for (sex, cohort), idx in df.groupby(["sex", "birth_year"]).groups.items():
    sub = df.loc[idx]
    curves[(sex, cohort)] = aalen_johansen_cip(
        sub.age_entry.values, sub.age_exit.values, sub.event.values)

# then, per person, the curve of their stratum:
lower, upper, K_i, K_pop = thresholds_from_cip(
    status, age, curve.ages, curve.values, k_pop=k_pop_for_stratum)
```

Practical guidance:

- **Sparse strata.** A stratum with few events has a noisy, steppy curve.
  Pooling adjacent birth years (e.g. 5-year bands) trades resolution for
  stability; the estimator raises on a stratum with zero events rather than
  inventing a curve. `curve.n_events` and `curve.se` tell you how sparse you
  are.
- **`k_pop`.** `thresholds_from_cip` defaults `k_pop` to `max(cip_values)`,
  which is only the lifetime prevalence if the curve reaches the lifetime
  horizon. On a heavily censored young cohort it does not — pass a separately
  justified `k_pop` (e.g. from an older stratum or the literature) instead.
  An empirical terminal value of exactly one is valid when the risk set is
  exhausted, but it cannot define a finite probit threshold; use a justified
  non-degenerate horizon rather than clipping it silently.

## Worked example

```python
import numpy as np
from ltpred import aalen_johansen_cip

# 8 people: entry ages, exit ages, event codes (0 censor, 1 diagnosis, 2 death)
entry = np.array([0, 0, 0, 0, 5, 10, 0, 0])
exit_ = np.array([40, 52, 47, 60, 55, 61, 44, 57])
event = np.array([1, 2, 1, 0, 1, 2, 0, 1])

diag = aalen_johansen_cip(entry, exit_, event)
death = aalen_johansen_cip(entry, exit_, event, cause=2)
```

with the verified curve (computed by the package, checked by hand):

| age | at risk Y | event | F_diag | F_death |
|---|---|---|---|---|
| 40 | 8 | diag | 0.125 | 0.000 |
| 47 | 6 | diag | 0.271 | 0.000 |
| 52 | 5 | death | 0.271 | 0.146 |
| 55 | 4 | diag | 0.417 | 0.146 |
| 57 | 3 | diag | 0.563 | 0.146 |
| 61 | 1 | death | 0.563 | 0.438 |

Walking the first rows: at 40, one diagnosis among `Y = 8` at risk (the two
late entries, ages 5 and 10, are already at risk — delayed entry only excludes
people not yet entered), so `S = 7/8` and `F_diag = 1 * 1/8 = 0.125`. At 47,
the risk set is down to 6 (the person diagnosed at 40 has left), so
`F_diag = 0.125 + (7/8) * (1/6) = 0.271`. At 52 the death removes one from the
risk set without adding to `F_diag`: `F_death = (7/8 * 5/6) * (1/5) = 0.146`.
At the horizon, `F_diag + F_death = 0.563 + 0.438 = 1 = 1 - S`, as required.
`tests/test_cip.py` contains further hand-computed KM and AJ examples, and
`benchmarks/bench_cip_estimation.py` simulates a full registry.

## Edge cases and gotchas

- **Prevalent cases (washout).** If a person was diagnosed *before* their
  entry age, they are not an incident case: exclude them (the Danish
  practice), or left-truncate the whole analysis past a minimum onset age.
- **Zero-length follow-up** (`exit == entry`): a censored row contributes
  nothing and may be kept for bookkeeping; an event-coded row is rejected
  because that person was never in the risk set.
- **Curve does not reach 1 or the horizon:** normal under censoring; the
  curve simply stops at the last observed event age. `thresholds_from_cip`
  holds the last value constant beyond the grid (it does not extrapolate
  incidence).
- **Curve reaches exactly 1:** also possible when the final target event
  exhausts the risk set. The estimate is valid, but
  `thresholds_from_cip` rejects it because prevalence one has no finite
  probit threshold. Use an earlier or externally justified lifetime horizon.
- **Emigration:** can be coded either as censoring (`0`) or as a competing
  event (its own code). The difference is small (emigrants are few percent);
  LT-FH++ treated it as competing. Death, however, must be a competing event
  whenever it is common — never ordinary censoring.
- **Monotonicity:** guaranteed by construction (the estimator is a sum of
  nonnegative increments); `thresholds_from_cip` validates it.
- **Uncertainty propagation:** LT-FH++ used point estimates of the CIP and
  did not propagate CIP uncertainty into the thresholds. The returned SEs are
  for diagnosing the curve's quality, not inputs to the estimator.

## Validation

`tests/test_cip.py` reproduces hand-computed KM and AJ examples exactly
(including left truncation and the identity `F_diag + F_death = 1 - S`),
checks Greenwood, checks the finite-risk Aalen variance for tied and untied
events including an exhausted final risk set, and runs the curve into
`thresholds_from_cip`. `benchmarks/bench_cip_estimation.py`
(RESULTS.md section 19) simulates a 50,000-person registry with a known
curve, mortality, administrative censoring and a register-start year, and
shows exact recovery, the KM competing-risks bias, and the end-to-end cost of
using an estimated curve (calibration slope 0.9969 vs 1.0015 oracle,
identical correlation).

## References

- Pedersen et al. (2022), *Am J Hum Genet* — LT-FH++; its CIPs are
  Aalen-Johansen curves per sex x birth year with death and emigration
  competing ([10.1016/j.ajhg.2022.01.009](https://doi.org/10.1016/j.ajhg.2022.01.009)).
- Andersen, Borgan, Gill & Keiding (1993), *Statistical Models Based on
  Counting Processes* (Springer) — the risk-set/left-truncation framework and
  the AJ variance (sec. IV.4).
- Aalen (1978), *Ann Statist* 6:534-545 — the closed-form CIF variance.
- Plana-Ripoll et al. (2019), *JAMA Psychiatry* — register comorbidity with
  the same counting-process machinery (analysis code: NB-COMO).
- Beck et al. (2024), *Acta Psychiatr Scand* — age-of-onset CIPs across
  birth cohorts with delayed entry and competing risks
  ([PMC11065580](https://pmc.ncbi.nlm.nih.gov/articles/PMC11065580/)).
- Greenwood (1926) — the KM variance; Kaplan & Meier (1958) — the estimator.
