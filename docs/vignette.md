---
title: Vignette
description: A working tour of ltpred — simulate families, estimate genetic liability, and map the same steps onto your own table.
---

# Vignette: from family history to a GWAS phenotype

This page is a working tour of **ltpred**, a methods package of the Pioneer
Centre for [SMARTbiomed](https://smartbiomed.dk/). It is longer than the
[quickstart](quickstart.md) and shorter than the reference pages. You will
simulate a small nuclear-family cohort, estimate each proband's posterior mean
genetic liability, see why that score is not the same thing as case/control
status, and then map the same steps onto a table you bring yourself.

On the documentation site this is a webpage
([/vignette/](https://bvilhjal.github.io/ltpred/vignette/)).
From a checkout:

```bash
pip install -e ".[docs,fast]"
mkdocs serve                 # http://127.0.0.1:8000/ltpred/vignette/
python examples/vignette.py  # the calculations quoted below
```

The numbers quoted below are what that script printed with the committed seed.
They are a teaching sample, not a benchmark. Real analyses need
population-representative cumulative incidence, a liability-scale $h^2$, and
the [checklist](assumptions.md).

??? example "Output of `python examples/vignette.py` (seed 1)"

    ```
    == 1. Simulate a classic LT-FH cohort ==
    families 800; liability-scale h2 0.5; prevalence 0.05
    proband case rate 0.055

    == 2. Posterior mean genetic liability (PA, the default) ==
    score mean 0.004  sd 0.297
    PA se is identically 0 (deterministic): 0.0
    one case    score +0.947  posterior var 0.271
    one control score -0.119  posterior var 0.430

    == 3. The score tracks true g better than 0/1 status ==
    corr(case/control, true g) 0.342
    corr(LT-FH PA,     true g) 0.419
    squared-corr eff-N proxy   1.50x

    == 4. Own status is a modelling choice, not a requirement ==
    corr(relatives-only, true g) 0.268

    == 5. ADuLT is the same bounds with no relative rows ==
    corr(ADuLT, true g) 0.342

    == 6. Gibbs agrees with PA on these rectangles ==
    corr(PA, Gibbs) on 80 families 0.9997
    Gibbs MC se (median) 0.0066

    == 7. Bring-your-own columns (same families, rebuilt) ==
    max |rebuilt - original| 0.00e+00

    == 8. Kinship path matches the role grammar on this pedigree ==
    max |kinship PA - role PA| 5.64e-04

    == 9. Liability-scale h2 (Falconer / Lee) ==
    parent-offspring tetrachoric 0.237  (Falconer 2*rho = 0.474; truth 0.5)
    Lee: observed-scale 0.20 at K=0.05 -> liability-scale 0.893

    == 10. Fit h2 only on independent, unascertained families ==
    fitted h2 0.363  (truth 0.5; within-dataset MC se 0.0060)

    == 11. Age-aware tutorial encoding (not full LT-FH++) ==
    corr(age-aware PA, true g) 0.261  (case/control 0.122)
    ```

## What is being estimated

Write $\ell$ for a person's full liability, $a$ for its additive genetic
part, and $e$ for an independent environmental part. On the unit-liability
scale

$$
\ell = a + e,
\qquad
a \sim N(0, h^2),
\qquad
e \sim N(0, 1-h^2),
\qquad
\ell \sim N(0, 1).
$$

A case is $\ell > T$. With lifetime prevalence $K$,

$$
T = \Phi^{-1}(1-K).
$$

Let $i$ be a designated proband, $A$ the additive relationship matrix of the
people whose records are used, and $K(\cdot)$ the prevalence or CIP map that
turns status and age into interval endpoints (or mixture weights). Write
$\ell_F$ for the vector of full liabilities and $D_F$ for the observation
model encoded by the records. The target is

$$
\mu_i = \mathbb{E}\bigl[a_i \mid D_F,\, A,\, h^2,\, K(\cdot)\bigr].
$$

If $\ell_F$ were observed continuously this would be the selection-index /
animal-model BLUP

$$
\mathbb{E}[a_i \mid \ell_F]
  = \mathrm{Cov}(a_i, \ell_F)\, \mathrm{Var}(\ell_F)^{-1} \ell_F.
$$

Disease records give a rectangle $\ell_F \in C_F$ (or a mixture of truncated
laws, for PA-FGRS censored controls). Under joint normality the BLUP weights
are unchanged and only the right-hand side is replaced by a truncated mean:

$$
\mathbb{E}[a_i \mid \ell_F \in C_F]
  = \mathrm{Cov}(a_i, \ell_F)\, \mathrm{Var}(\ell_F)^{-1}\,
    \mathbb{E}[\ell_F \mid \ell_F \in C_F].
$$

Gibbs estimates that truncated mean by sampling. Pearson–Aitken approximates
the same sequential moment updates. Neither output is a SNP polygenic score,
and neither is an absolute lifetime risk.

The names **LT-FH**, **LT-FH++**, **ADuLT** and **PA-FGRS** choose $D_F$
(which people are observed, and how ages enter). **Gibbs** and
**Pearson–Aitken (PA)** are the engines that compute $\mu_i$. PA is the
single-trait default.

| Model | Records you condition on | Typical helper |
|---|---|---|
| Classic LT-FH | Relatives + optional own status; one lifetime threshold | `prevalence_thresholds` |
| LT-FH++ | Same people; age/sex/cohort-specific CIP | `thresholds_from_cip(..., case_mode="pin")` |
| ADuLT | Proband only; same personalised CIP | same helper, role `o` rows only |
| Base PA-FGRS | Lifetime case interval; censored-control mixture | `prevalence_thresholds` + `use_mixture=True` |

LT-FH is
[Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6);
LT-FH++ is
[Pedersen et al. 2022](https://doi.org/10.1016/j.ajhg.2022.01.009);
ADuLT is
[Pedersen et al. 2023](https://doi.org/10.1038/s41467-023-41210-z);
PA-FGRS is
[Dybdahl Krebs et al. 2024](https://doi.org/10.1016/j.ajhg.2024.09.009).

## 1. Simulate a cohort you can check

The simulator draws jointly normal liabilities, thresholds them at prevalence
$K$, and returns ready-to-estimate `Family` objects plus the true genetic
liability $g$. `use_age=False` is classic LT-FH: one threshold, no onset.

```python
from ltpred import simulate_under_LTM_single, estimate_liability

h2, K, n_fam = 0.5, 0.05, 800
sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=h2, pop_prev=K, n_sim=n_fam,
    use_age=False, seed=1,
)
status = sim.status["o"].astype(float)   # proband 0/1
true_g = sim.genetic
```

Each family is a proband (`o`) plus mother, father and one sibling. Roles are
always relative to the proband; see the
[role grammar](data-preparation.md#role-grammar).

## 2. Estimate, then read `est`, `se` and `var`

```python
pa = estimate_liability(sim.families, h2=h2)   # PA is the single-trait default
score = pa.genetic                             # == pa.est["genetic"]
```

`h2` must be **liability-scale**. `score` is aligned to `pa.pids`, not to the
input row order.

Two uncertainty fields are easy to mix up:

- `pa.se["genetic"]` is the **estimator's** numerical error. PA is
  deterministic, so this is exactly 0. Zero Monte-Carlo error is not zero
  approximation error.
- `pa.var["genetic"]` is the **posterior** variance $\mathrm{Var}(a_i \mid D_F)$.
  An affected relative shrinks it; it does not shrink if you draw more Gibbs
  samples.

On this seed an affected proband scored $+0.947$ with posterior variance
$0.271$; an unaffected one scored $-0.119$ with variance $0.430$.

## 3. Why not just use case/control?

```python
import numpy as np

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])

r_cc = corr(status, true_g)
r_pa = corr(score, true_g)
(r_pa / r_cc) ** 2    # squared-corr effective-N proxy, not a GWAS NCP ratio
```

Here that was $0.342$ versus $0.419$, a $1.50\times$ proxy. Family
history is information about $a_i$ that the 0/1 label throws away. The
proxy is useful for ranking; it is not the causal-SNP noncentrality ratio
in [RESULTS.md](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).

## 4. The proband's own diagnosis is optional

Role `o` is an observation you may include or drop. Include it when the
score is a **GWAS phenotype constructed from that diagnosis**. Omit it, or
give it bounds $(-\infty, \infty)$, when the same diagnosis is later the outcome
you will predict — otherwise $D_F$ contains the answer.

```python
from ltpred import Family

relatives_only = [
    Family(fam.fam_id, [m for m in fam.members if m.role != "o"])
    for fam in sim.families
]
rel = estimate_liability(relatives_only, h2=h2)
corr(rel.genetic, true_g)    # 0.268 on this seed: weaker, no leakage
```

## 5. ADuLT is the same construction without relatives

Keep only role `o`. With a **single lifetime threshold**, ADuLT is a
monotone map of case/control status, so it cannot beat the 0/1 label
(both correlations were $0.342$ here). ADuLT earns its keep when the
proband's threshold is personalised by age, sex and cohort.

```python
adult_fams = [
    Family(fam.fam_id, [m for m in fam.members if m.role == "o"])
    for fam in sim.families
]
adult = estimate_liability(adult_fams, h2=h2)
```

## 6. Cross-check PA with Gibbs on a subset

On classic rectangles PA and Gibbs posterior means agree closely. Gibbs is
the sampler you want for a Monte-Carlo SE, for multiple traits, or when a
pedigree is unusual. It does not implement the PA-FGRS mixture.

```python
gibbs = estimate_liability(
    sim.families[:80], h2=h2, method="gibbs",
    tol=0.03, n_sim=8_000, burn_in=400, seed=1,
)
corr(score[:80], gibbs.genetic)     # 0.9997 here
np.median(gibbs.se["genetic"])      # Monte-Carlo SE of the mean, not var
```

## 7. Bring your own table

Production input is a long table: one row per observed person, grouped by
`fam_id`. Rebuild the same simulated families from columns to see the
plumbing. `prevalence_thresholds` is classic LT-FH; it is **not** the
logistic tutorial helper.

```python
from ltpred import prevalence_thresholds, families_from_columns

fam_id, role, st = [], [], []
for i, fam in enumerate(sim.families):
    for m in fam.members:
        fam_id.append(f"fam{i}")
        role.append(m.role)
        st.append(int(np.isfinite(m.lower)))   # 1 = case

lower, upper = prevalence_thresholds(np.array(st), pop_prev=K)
families = families_from_columns(fam_id, role, lower, upper)
score = estimate_liability(families, h2=h2).genetic
```

Replace the loop with your CSV. Missing `fam_id` values are rejected (they
cannot group rows). Duplicate roles inside a family (two rows both `s1`)
are rejected; number repeated siblings `s1`, `s2`.

!!! warning "The logistic `age_thresholds` helper is tutorial-only"

    For register data call `thresholds_from_cip` once per sex × birth-year
    × ancestry stratum with a **population** CIP, not a biobank case fraction.
    See [data preparation](data-preparation.md#a-real-register-data-recipe)
    and [CIP estimation](cip-estimation.md).

## 8. Arbitrary pedigrees: the kinship path

When relatives do not fit the role grammar (cousins, inbreeding, messy
second parents), build the additive relationship matrix from parent
pointers.

```python
from ltpred import kinship_from_pedigree, estimate_liability_from_kinship

ids = ["o", "m", "f", "s1"]
_, A = kinship_from_pedigree(
    ids,
    father=["f", None, None, "f"],
    mother=["m", None, None, "m"],
)
# lower/upper have shape (n_families, 4) in `ids` order
est, se, var = estimate_liability_from_kinship(A, lower, upper, h2=h2, target=0)
```

On this nuclear pedigree the two paths agreed to a few $10^{-4}$. Pearson–
Aitken is a sequential two-moment approximation: a change of fold order is
not Gibbs error. Use the kinship path when the recorded parents disagree
with a role-grammar convention (in particular, same-side half-sibs).

## 9. Getting $h^2$ onto the liability scale

The estimator conditions on $h^2$. Passing an observed-scale value as if
it were liability-scale mis-calibrates the score (ranking is more robust
than the scale; see RESULTS §12).

A quick relative-pair check is Falconer's route: twice the parent–offspring
tetrachoric. On this seed that was $2 \times 0.237 = 0.474$ against truth
$0.5$. The Lee conversion of an observed-scale estimate $h^2_{\mathrm{obs}}$
in a population sample with prevalence $K$ is

$$
h^2_{\mathrm{liab}}
  = h^2_{\mathrm{obs}} \cdot \frac{K(1-K)}{\phi(T)^2},
  \qquad
  T = \Phi^{-1}(1-K),
$$

which at $K = 0.05$ sends $0.20$ to $0.893$. If the study over-samples
cases, pass that fraction as `prop_cases`.

```python
from ltpred import tetrachoric, observed_to_liability_h2

po = tetrachoric(sim.status["o"].astype(int), sim.status["m"].astype(int))
2 * po.rho
float(observed_to_liability_h2(0.20, pop_prev=K))   # 0.893 at K = 0.05
```

Prefer an external liability-scale estimate when you have one.

## 10. Fitting $h^2$ from the family data — only under a sampling contract

`fit_heritability` is optional. It assumes **independent, non-overlapping**
families, either unascertained (`sampling="population"`) or selected with
known positive inclusion probabilities (`sampling="ipw"`). A `pid` that
appears in two different `fam_id`s is rejected: that is the usual register
extraction, valid for **scoring**, invalid for **fitting**.

```python
from ltpred import fit_heritability

fit = fit_heritability(
    sim.families, n_iter=250, burn_in=80, seed=1, sampling="population",
)
fit.h2, fit.h2_se
```

On this 800-family, $K=0.05$ draw the point estimate was $0.363$ against
truth $0.5$, while `h2_se` was $0.006$. That standard error is a
**within-dataset Monte-Carlo diagnostic**. It is not a sampling interval.
With a few hundred nuclear families at this prevalence the across-cohort SD
is much larger. Use an external $h^2$ unless you have met the contract in
[Inference](inference.md).

!!! danger "Do not fit on case-enriched extracts without IPW"

    Unguarded ascertainment pins $\hat{h}^2$ at 1, even when the true value
    is 0. Proband-ascertained designs cannot be reweighted at all.

## 11. Age, CIP, and what “LT-FH++” actually requires

`simulate_under_LTM_single(..., use_age=True)` plus `age_thresholds` is an
**age-only tutorial**. Full LT-FH++ needs a CIP that varies with sex and
birth cohort as well, estimated from a representative source:

```python
from ltpred import thresholds_from_cip, families_from_columns, estimate_liability

# one stratum shown; loop over sex × birth-year × ancestry in production
lower, upper, K_i, K_pop = thresholds_from_cip(
    status, age, cip_ages, cip_values, k_pop=lifetime_K, case_mode="pin",
)
families = families_from_columns(
    fam_id, role, lower, upper, pid=pid, K_i=K_i, K_pop=K_pop,
)
res = estimate_liability(families, h2=h2)          # LT-FH++ if relatives are present
# res = estimate_liability(proband_only, h2=h2)    # ADuLT if they are not
```

`case_mode="pin"` is the LT-FH++ / ADuLT encoding. Pin only when you believe
onset is the CIP inverse of liability; under partial onset dependence the
pin over-conditions. Most of the onset information survives the interval
$[T(\mathrm{onset}), \infty)$.

Competing death needs an Aalen–Johansen CIP, not a Kaplan–Meier that treats
death as censoring. Details: [CIP estimation](cip-estimation.md).

## 12. PA-FGRS, briefly

Base PA-FGRS (Krebs et al. 2024) is not “turn on PA”. Observed cases use the
**lifetime** interval $[\Phi^{-1}(1-K_{\mathrm{pop}}), \infty)$. Age enters a
censored control through the mixture weight $K_i$, via `use_mixture=True`.
The helper `pa_thresholds` is an age-dependent **variant**, not that paper's
case encoding and not PA-FGRS_ADT. Gibbs has no mixture.

```python
res = estimate_liability(
    families, h2=h2, use_mixture=True,     # needs K_i / K_pop on censored controls
)
```

## 13. Handing the score to a GWAS

1. Join `res.pids` to the genotyped sample IDs. Do not assume input-row order.
2. Residualize for sex, birth cohort, ancestry PCs, batch, and other design
   covariates **before** association, or put them in the GWAS mixed model.
3. Prefer a mixed-model association method if related targets remain
   ([estimation](estimation.md#using-the-estimate-in-a-gwas)).
4. The score is a quantitative phenotype. ltpred does not run the GWAS.

Combining the family-history score with a PGS is a **downstream** model
([Hujoel et al. 2022](https://doi.org/10.1016/j.xgen.2022.100152);
[Dybdahl Krebs et al. 2026](https://doi.org/10.1016/j.ajhg.2025.11.016)).

## 14. Mistakes this tour is designed to prevent

- Passing **observed-scale** $h^2$ as `h2=`.
- Using `age_thresholds` (one logistic curve) and calling the analysis LT-FH++.
- Conditioning on role `o` and then using the score to **predict** that
  same diagnosis.
- Fitting $h^2$ on overlapping register pedigrees or on a case/control
  extract without `sampling="ipw"` and valid weights.
- Treating `res.se` as posterior uncertainty, or `res.var` as a standard
  error of the mean.
- Equating `pa_thresholds` with published PA-FGRS.
- Quoting an independent-SNP simulation NCP ratio as a real-LD GWAS gain.

## Where to go next

| | |
|---|---|
| One-page copy-paste | [Quickstart](quickstart.md) |
| Which model to run | [Choose a method](guide.md) |
| Roles, CIPs, real tables | [Data preparation](data-preparation.md) |
| Engines, scaling, GWAS export | [Estimation](estimation.md) |
| Fitting $h^2$ / A+C+M | [Inference](inference.md) |
| Assumptions and pre-flight list | [Assumptions & checklist](assumptions.md) |
| Estimand and Algorithms G, P, M | [Algorithm](algorithm.md) |
| Simulation evidence | [RESULTS.md](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md) |
