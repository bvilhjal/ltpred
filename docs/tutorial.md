# Tutorial

A lab in five questions. Run the blocks in order: each one uses only names
the earlier blocks defined. Copy the page into a script, or paste one block
at a time. `tests/test_tutorial.py` re-runs the blocks and checks the
printouts.

Teach steps 1–4 as one hour (one disease, known truth). Step 5 is a second
hour (two traits). The [method guide](guide.md) is what you open when a
student brings their own table. Contracts this lab simplifies are at the end.

| step | question | the printout should show |
|---|---|---|
| 1 | Can we build a population whose genetic liability we know? | variance of `genetic` near 0.5, and not everyone diagnosed by 70 |
| 2 | Kaplan–Meier or Aalen–Johansen when people die? | KM stays near the no-death curve; AJ drops |
| 3 | Does a GWAS score recover that liability? | correlation about 0.55; most of Var(g) is still posterior uncertainty |
| 4 | What if the proband's own diagnosis is hidden? | correlation falls to about 0.31 |
| 5 | Can one fit recover two traits? | each estimate within about 2 SE of its truth |

## Step 1 — Simulate a population cohort

**Question.** If we know the genetic liability, can we still see a realistic
diagnosis register?

`simulate_pedigree` builds the parent pointers. `simulate_register_liabilities`
draws **one** liability for the whole population and turns it into diagnoses.
A person who sits in several pedigrees therefore has one status and one
genetic value.

```python
import numpy as np

from ltpred import simulate_pedigree, simulate_register_liabilities

H2 = 0.5                                      # liability-scale heritability
CIP_K, CIP_MID, CIP_SLOPE = 0.10, 60.0, 1.0 / 8.0
EVAL_AGE = 70.0                               # everyone is followed to here
INDEX_AGE = 40.0                              # the prospective cut in step 4

# the generating cumulative-incidence curve, on a 1-year age grid
AGE_GRID = np.arange(0, 121, 1.0)
TRUE_CIP = CIP_K / (1.0 + np.exp((CIP_MID - AGE_GRID) * CIP_SLOPE))

ids, father, mother = simulate_pedigree(
    np.random.default_rng(20260921), n_founder_pairs=100, gens=2)

cohort = simulate_register_liabilities(
    np.random.default_rng(20260921), ids, father, mother,
    h2=H2, cip_ages=AGE_GRID, cip_values=TRUE_CIP, eval_age=EVAL_AGE)

print(f"{len(ids)} people, {int(cohort.status.sum())} diagnosed by age "
      f"{EVAL_AGE:.0f} ({cohort.status.mean():.1%})")
print(f"var(true genetic liability) = {cohort.genetic.var():.4f}  (target {H2})")
print(f"birth years: {sorted(set(cohort.birth_time.tolist()))}")
```

```text
984 people, 81 diagnosed by age 70 (8.2%)
var(true genetic liability) = 0.5083  (target 0.5)
birth years: [1920.0, 1950.0, 1980.0]
```

**Answer.** The variance of `genetic` is the heritability we asked for.
The generating curve at age 70 is 7.8%, not the 10% lifetime prevalence,
and 81 diagnoses out of 984 is one draw from that curve. That is why step 2
estimates a curve instead of plugging in 0.10. `cohort.genetic` is the truth
later steps correlate against. The register columns are `status`, `age`, and
`birth_time`.

## Step 2 — Estimate the incidence curve

**Question.** When death competes with diagnosis, which estimator moves?

These 50,000 records are a **new** sample, not the 984 people above. A CIP
from 81 events is too noisy to teach the difference. `simulate_followup_records`
draws from the same logistic curve so the estimate can be checked.

First, nobody dies:

```python
from ltpred import kaplan_meier_cip, simulate_followup_records

plain = simulate_followup_records(
    np.random.default_rng(7), 50_000,
    pop_prev=CIP_K, mid_point=CIP_MID, slope=CIP_SLOPE)

km = kaplan_meier_cip(plain.age_entry, plain.age_exit, plain.event == 1)
km_at_70 = float(np.interp(70.0, km.ages, km.values))
worst = float(np.max(np.abs(np.interp(AGE_GRID, km.ages, km.values) - TRUE_CIP)))
print(f"{km.n_entered:,} records; worst error on the age grid {worst:.4f}")
print(f"KM(70) = {km_at_70:.4f}   true = {float(np.interp(70.0, AGE_GRID, TRUE_CIP)):.4f}")
```

```text
49,999 records; worst error on the age grid 0.0032
KM(70) = 0.0757   true = 0.0777
```

Now death competes. This is the comparison the step exists for:

```python
from ltpred import aalen_johansen_cip

mort = simulate_followup_records(
    np.random.default_rng(7), 50_000,
    pop_prev=CIP_K, mid_point=CIP_MID, slope=CIP_SLOPE, mortality=True)

aj = aalen_johansen_cip(mort.age_entry, mort.age_exit, mort.event)
km_m = kaplan_meier_cip(mort.age_entry, mort.age_exit, mort.event == 1)
print(f"{int((mort.event == 2).sum()):,} died before diagnosis")
print(f"Aalen-Johansen(70) = {float(np.interp(70.0, aj.ages, aj.values)):.4f}")
print(f"Kaplan-Meier(70)   = {float(np.interp(70.0, km_m.ages, km_m.values)):.4f}"
      f"   (was {km_at_70:.4f} with no deaths)")
```

```text
21,066 died before diagnosis
Aalen-Johansen(70) = 0.0626
Kaplan-Meier(70)   = 0.0761   (was 0.0757 with no deaths)
```

**Answer.** Two different quantities, not two estimators of one curve.
Kaplan–Meier, treating death as censoring, stays at 0.076, next to the
generating curve at 70 (0.078): that is the net risk if nobody died.
Aalen–Johansen is the proportion actually diagnosed by 70 when death can
come first, 0.063. Do not pass that lower curve into step 3 in place of
`TRUE_CIP`; step 3's bounds were built from the net curve. See
[CIP estimation](cip-estimation.md).

## Step 3 — Score the population

**Question.** If the proband's own diagnosis is included, how close is the
score to the genetic liability from step 1?

`estimate_liabilities` walks each proband's pedigree and returns one score.

```python
from ltpred import estimate_liabilities

scores = estimate_liabilities(
    cohort.ids, cohort.father, cohort.mother,
    probands=cohort.ids,
    status=cohort.status.astype(int), age=cohort.age,
    use="gwas",                          # the proband's own diagnosis is used
    cip_ages=AGE_GRID, cip_values=TRUE_CIP, k_pop=CIP_K,
    h2=H2)

est = np.asarray(scores.est)
print(f"corr(score, true genetic liability) = {np.corrcoef(est, cohort.genetic)[0, 1]:.4f}")
print(f"score sd = {est.std():.4f}   mean posterior variance = {np.asarray(scores.var).mean():.4f}")
print(f"median relatives conditioned on = {np.median(scores.n_relatives):.0f}")
print(f"records with an unresolved parent = {scores.frac_records_with_unresolved_parents:.4f}")
```

```text
corr(score, true genetic liability) = 0.5502
score sd = 0.3898   mean posterior variance = 0.3558
median relatives conditioned on = 23
records with an unresolved parent = 0.0000
```

**Answer.** The correlation, 0.55, is with the true genetic liability, which
the scorer was not given. It is not a disease-risk correlation. Square the
score standard deviation and add the mean posterior variance:
0.390² + 0.356 = 0.508, the variance of `genetic` in step 1. The data
identified about 0.15 of that variance; about 0.36 is still posterior
uncertainty, so the score is not a measurement. The 984 scores are
overlapping pedigrees, not 984 independent people. Unresolved parents are 0
because every parent id was in the table. This step passes the generating
net curve back in. A real analysis passes a stratified estimate
(`strata`, `cip_by_stratum`).

## Step 4 — Score prospectively, without the proband's own diagnosis

**Question.** How much of step 3 was just the proband's own diagnosis?

`use="prediction"` hides that diagnosis. It needs a calendar: `birth_time`
for every person, `index_time` for every proband. Those two arrays have
different lengths.

```python
at_risk = cohort.onset > INDEX_AGE          # still undiagnosed at the cut
index_time = cohort.birth_time + INDEX_AGE

predicted = estimate_liabilities(
    cohort.ids, cohort.father, cohort.mother,
    probands=[p for p, keep in zip(cohort.ids, at_risk) if keep],
    status=cohort.status.astype(int), age=cohort.age,
    use="prediction",
    cip_ages=AGE_GRID, cip_values=TRUE_CIP, k_pop=CIP_K, h2=H2,
    birth_time=cohort.birth_time, index_time=index_time[at_risk])

pred_est = np.asarray(predicted.est)
print(f"{int(at_risk.sum())} of {len(ids)} probands are disease-free at age {INDEX_AGE:.0f}")
print(f"corr(score, truth) = {np.corrcoef(pred_est, cohort.genetic[at_risk])[0, 1]:.4f}"
      f"   score sd = {pred_est.std():.4f}")
print(f"proband states: {sorted(set(map(str, predicted.proband_state)))}")
```

```text
974 of 984 probands are disease-free at age 40
corr(score, truth) = 0.3069   score sd = 0.2134
proband states: ['disease_free_and_followed']
```

**Answer.** The drop from 0.55 to 0.31 is still a correlation with genetic
liability, now without the proband's own diagnosis. It is not the accuracy
of predicting who becomes a case after 40. Restricting `probands` to people
undiagnosed at 40 is part of that question: scoring someone already diagnosed
and calling it prospective is leakage, and the driver warns. `proband_state`
is that check.

## Step 5 — Two traits at once

**Question.** On a second cohort, can one fit recover both heritabilities,
the genetic correlation, and the residual correlation?

This cohort is not the register from step 1. `simulate_under_LTM_multi`
builds nuclear families with two binary traits. `fit_pairwise_multi` fits
them together. Read `inference_status` before reading a standard error.

```python
from ltpred import fit_pairwise_multi, simulate_under_LTM_multi

multi = simulate_under_LTM_multi(n_families=3000, seed=1)
truth = multi.truth

joint = fit_pairwise_multi(
    multi.families, components=("A", "C", "M"),
    sampling="population", phen_names=list(multi.phen_names))

emp = np.corrcoef(multi.liabilities[:, :, 0].ravel(),
                  multi.liabilities[:, :, 1].ravel())[0, 1]
print(f"simulated trait correlation {emp:.4f} (target "
      f"{truth['target_trait_corr']:.4f})")
for name, fitted, se, target in (
        ("h2 trait 1", joint.h2[0], joint.se["h2"][0], truth["h2"][0]),
        ("h2 trait 2", joint.h2[1], joint.se["h2"][1], truth["h2"][1]),
        ("rg", joint.rg[0, 1], joint.se["rg"][0, 1], truth["rg"]),
        ("residual re", joint.re[0, 1], joint.se["re"][0, 1],
         truth["residual_re"])):
    print(f"  {name:11s} fitted {fitted:+.4f} ± {se:.4f}   truth {target:+.2f}"
          f"   ({abs(fitted - target) / se:.2f} SE)")
print(f"inference status: {joint.inference_status}")
```

```text
simulated trait correlation 0.1551 (target 0.1511)
  h2 trait 1  fitted +0.4314 ± 0.0500   truth +0.35   (1.63 SE)
  h2 trait 2  fitted +0.4137 ± 0.0355   truth +0.40   (0.39 SE)
  rg          fitted +0.5435 ± 0.0760   truth +0.50   (0.57 SE)
  residual re fitted -0.4432 ± 0.1651   truth -0.35   (0.56 SE)
inference status: interior_cluster_sandwich
```

**Answer.** The first line checks the simulator. Trait 1's heritability is
the noisy draw: 0.43 against 0.35, 1.63 standard errors. The other three sit
closer. That is one cohort of 3,000 families, not a calibration study.
`interior_cluster_sandwich` means those standard errors were produced; a
boundary fit withholds them. The replicated fits are in
`benchmarks/RESULTS.md` §33.

## What this lab is not

Do not hand a student a real register and tell them to change the filenames.
Each shortcut below is a real decision, with the page that carries it.

- **One curve for everyone.** Steps 3 and 4 pass `TRUE_CIP`. Real LT-FH++
  stratifies by sex and birth year.
  [CIP estimation](cip-estimation.md),
  [data preparation](data-preparation.md).
- **`h² = 0.5` was given.** It is the liability-scale heritability of this
  disease, it has no default, and fitting it needs a declared sampling design.
  [Which h²?](data-preparation.md#getting-heritability-on-the-liability-scale),
  [inference](inference.md).
- **The generating curve was fed back in.** Passing a wrong curve is a measured
  failure mode. [Assumptions](assumptions.md).
- **The sample is unascertained.** Biobanks selected on case status are not.
  [Ascertainment](inference.md#ascertained-samples).
- **The score is not a risk.** It is a posterior mean liability. A GWAS joins
  it on the returned proband ids and residualises the design covariates.
  [Estimation](estimation.md), [vignette](vignette.md).

## Where to go next

| you want to… | see |
|---|---|
| which function to call on a student's own table | [Choose a method](guide.md) |
| six hand-typed rows | [Quickstart](quickstart.md) |
| the full contract | [Vignette](vignette.md) |
| the checklist before a real analysis | [Assumptions](assumptions.md) |
