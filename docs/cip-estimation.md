# Estimating cumulative incidence (CIP) from health data

LT-FH++, ADuLT and PA-FGRS consume cumulative incidence—the population fraction
diagnosed by a given age:

```
CIP(age | stratum) = P(diagnosed at or before age),   T = Phi^-1(1 - CIP).
```

The curve can be the analysis result itself or an input to thresholds. If a
suitable published curve exists, skip to [data preparation](data-preparation.md).
This page estimates one from follow-up records.

## Input: follow-up records

Every estimator takes aligned 1-D arrays:

| array | meaning | examples |
|---|---|---|
| `age_entry` | start of follow-up | birth, register start, immigration, minimum onset age |
| `age_exit` | end of follow-up | diagnosis, death, emigration, administrative end |
| event code | exit reason | `0` censored, `1` diagnosis, `2` death or another competing event |

Age is the time scale; calendar effects enter through entry/exit ages and
stratification. A person is at risk at `t` when `age_entry < t <= age_exit`.
Delayed entry therefore contributes risk only after entry. Entry and censoring
must be non-informative conditional on modelled strata, with overlapping
risk-set support. Ties are grouped by exit age. `exit < entry` and non-finite
ages raise; zero-length follow-up contributes no risk and cannot carry an event.

For `kaplan_meier_cip`, the event array is boolean or exact 0/1. For
`aalen_johansen_cip`, it is an integer code: `0` censored, `cause` (default 1)
the target event, and other positive codes competing events.

## Kaplan-Meier (`kaplan_meier_cip`)

At each unique event age `t_j`, let `d_j` be target events and `Y_j` the number
at risk:

```
S(t) = prod_{t_j <= t} (1 - d_j / Y_j),        CIP(t) = 1 - S(t)
Var(S(t)) = S(t)^2 * sum_{t_j <= t} d_j / (Y_j (Y_j - d_j)) .
```

Under independent censoring this estimates **net risk** in a hypothetical world
without that censoring process. For the proportion actually diagnosed before
death, death is a competing event, not ordinary censoring—even when death and
diagnosis times are independent. Censoring the dead generally overestimates
crude diagnosed incidence; use Aalen–Johansen instead.

## Aalen-Johansen (`aalen_johansen_cip`)

Let `d_j` count events of any type, `d_kj` events of cause `k`, and `S` overall
event-free survival:

```
S(t) = prod_{t_j <= t} (1 - d_j / Y_j),
F_k(t) = sum_{t_j <= t} S(t_j-) * d_kj / Y_j .
```

`F_k` is **crude cumulative incidence** in the presence of competing events.
Published LT-FH++ used this approach per sex and birth-year stratum with death
and emigration as competing events.

Pointwise SEs use the finite-risk-set, tie-correct Aalen (1978) variance, in the
grouped recurrence used by `cmprsk::cuminc`. At age `t_j`, for cause group `r`
(target or all other causes), define

```text
q_rj = 1                                   if d_rj = 1
        1 - (d_rj - 1) / (Y_j - 1)         otherwise,       (Equation 1)
a_rj = S(t_j-)^2 q_rj d_rj / Y_j^2 .                       (Equation 2)
```

For competing groups use `u = F_k(t_j) / S_j+`; for target events use
`u = 1 + F_k(t_j) / S_j+`; both use `w = 1 / S_j+`. Update three running sums:

```text
v1 <- v1 + u^2 a_rj
v2 <- v2 + w u a_rj
v3 <- v3 + w^2 a_rj,                                      (Equation 3)
Var F_k(t_j) = v1 + F_k(t_j)^2 v3 - 2 F_k(t_j) v2.         (Equation 4)
```

Competing-group updates are skipped when `S_j+ = 0`; target events use the
removable boundary `w = 0`, `u = 1`. Separate cause/other updates are essential
for tied causes. With one event type, Aalen and Greenwood variances agree
asymptotically, not as finite-sample identities.

## The estimand choice (the most important decision on this page)

- **Net incidence** (`1 - Kaplan–Meier`, death censored): incidence in a
  hypothetical no-death world, not usually the registry's diagnosed proportion.
- **Crude incidence** (Aalen–Johansen): probability of diagnosis by age `t` in
  the real population with competing death. This is the published LT-FH++
  convention; `thresholds_from_cip` accepts either curve.
- **Plain age-specific proportions:** appropriate only with essentially complete
  follow-up beyond the target age. Recent, right-censored cohorts undercount
  late onset.

Threshold choice is also a model decision. If death is independent of
liability, threshold crossing among people alive at `t` implies the **net**
curve. If death is liability-related, or the target is the population diagnosed
fraction, the **crude** curve is natural. They differ little when death before
onset is rare. The end-to-end example used net incidence without mortality;
Aalen–Johansen thresholds with mortality remain untested.

`benchmarks/bench_cip_estimation.py` reports AJ recovery to about 0.002 absolute
error at N = 50,000 and KM-as-censoring overshoot up to 0.022 at a 42% death
share in its simulated registry.

## Stratification

Estimate one curve per population stratum (published LT-FH++ used sex × birth
year):

```python
import numpy as np
from ltpred import aalen_johansen_cip, thresholds_from_cip

# event: 0 censored, 1 diagnosis, 2 death
stratum = np.char.add(sex.astype(str), birth_year.astype(str))
lower = np.empty(len(status))
upper = np.empty(len(status))
for key in np.unique(stratum):
    m = stratum == key
    curve = aalen_johansen_cip(age_entry[m], age_exit[m], event[m])
    ages = np.r_[0.0, curve.ages]       # anchor at (0, 0)
    values = np.r_[0.0, curve.values]
    lower[m], upper[m], _, _ = thresholds_from_cip(
        status[m], age[m], ages, values)
```

Sparse strata produce noisy step curves; pooling adjacent birth years trades
resolution for stability. `curve.n_events` and `curve.se` expose sparsity. A
stratum with no events raises. `thresholds_from_cip` defaults `k_pop` to the
curve maximum, which is lifetime prevalence only if the curve reaches that
horizon; otherwise supply it from an older stratum or literature.

## Worked example

```python
import numpy as np
from ltpred import aalen_johansen_cip

entry = np.array([0, 0, 0, 0, 5, 10, 0, 0])
exit_ = np.array([40, 52, 47, 60, 55, 61, 44, 57])
event = np.array([1, 2, 1, 0, 1, 2, 0, 1])

diag = aalen_johansen_cip(entry, exit_, event)
death = aalen_johansen_cip(entry, exit_, event, cause=2)
```

| age | at risk Y | event | F_diag | F_death |
|---|---|---|---|---|
| 40 | 8 | diag | 0.125 | 0.000 |
| 47 | 6 | diag | 0.271 | 0.000 |
| 52 | 5 | death | 0.271 | 0.146 |
| 55 | 4 | diag | 0.417 | 0.146 |
| 57 | 3 | diag | 0.563 | 0.146 |
| 61 | 1 | death | 0.563 | 0.438 |

At 40, one diagnosis among eight gives `S = 7/8` and `F_diag = 1/8`. At 47,
the diagnosis and a censoring have left, so
`F_diag = 0.125 + (7/8)(1/6) = 0.271`. At 52, death adds
`(7/8)(5/6)(1/5) = 0.146` to `F_death` only. At the horizon,
`F_diag + F_death = 1 - S`, as required.

## Edge cases and gotchas

- **Prevalent cases:** exclude diagnoses before entry; they are not incident.
- **Zero-length follow-up:** censored rows contribute nothing; event-coded
  rows raise because no one was at risk.
- **Beyond the event grid:** incidence is not extrapolated. `thresholds_from_cip`
  holds the last value constant. Prepend `(0, 0)` as above or younger rows use
  the first event age's incidence.
- **Zero incidence:** values are clipped to `[min_cip, k_pop]` (default
  `min_cip=1e-5`), which can put an onset in a zero-CIP region at an arbitrary
  high threshold.
- **CIP exactly one:** valid when the last target event exhausts the risk set,
  but no finite probit threshold exists. Choose a justified earlier/lifetime
  horizon.
- **Emigration:** ordinary censoring under the liability model; LT-FH++ coded
  it as competing. Keep death as a separate event so net and crude curves can
  both be formed.
- **Uncertainty:** returned SEs diagnose curve precision; they are not
  propagated into thresholds.

## Validation and references

`tests/test_cip.py` checks hand-computed KM/AJ curves, delayed entry, Greenwood
and tied finite-risk Aalen variances, exhausted risk sets and threshold
integration. `benchmarks/bench_cip_estimation.py` measures curve recovery,
competing-risk bias and end-to-end scoring.

- Pedersen et al. (2022), *Am J Hum Genet* — LT-FH++ and its AJ curves
  ([10.1016/j.ajhg.2022.01.009](https://doi.org/10.1016/j.ajhg.2022.01.009)).
- Andersen, Borgan, Gill & Keiding (1993), *Statistical Models Based on
  Counting Processes* — risk sets and AJ variance (sec. IV.4).
- Aalen (1978), *Ann Statist* 6:534-545 — CIF variance.
- Beck et al. (2024), *Acta Psychiatr Scand* — delayed entry and competing
  risks ([PMC11065580](https://pmc.ncbi.nlm.nih.gov/articles/PMC11065580/)).
- Greenwood (1926), Kaplan & Meier (1958) — variance and product-limit estimator.
