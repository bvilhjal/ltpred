# Tutorial

A complete, runnable analysis on **simulated** data, start to finish: build a
population cohort with known truth, estimate the incidence curve it needs, score
every proband, and check the scores against that truth. About 30 lines of code
and a few seconds of runtime.

Every block below runs in order — copy the page into a script, or run the blocks
one at a time in a REPL, and each will work because each only uses names the
earlier blocks defined. `tests/test_tutorial.py` executes the page's blocks on
every commit, so the code cannot rot away from the prose.

This page is deliberately narrow. It states *what to type* and *what you should
see*; it does not restate the modelling contracts. Everything it simplifies is
listed in [what this tutorial skips](#what-this-tutorial-skips) at the end, with
a link to the page that carries the full caveat. If you need to know whether a
step is valid for *your* data, that is the [vignette](vignette.md).

## Step 1 — Simulate a population cohort

ltpred ships the generators, so you do not need real register data to learn the
pipeline. `simulate_pedigree` builds a multi-generation parent-pointer table;
`simulate_register_liabilities` draws **one** liability field over the whole
pedigree and turns it into diagnosis records.

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

`cohort` carries the register columns (`ids`, `father`, `mother`, `status`,
`age`, `birth_time`) **and** the truth (`genetic`, `onset`, `residual_var`), so
every later step can be scored rather than merely run. The 8.2% diagnosis rate
tracks the generating curve at age 70 (7.8%), not the 10% lifetime
prevalence — that gap is
the whole reason step 2 exists.

The pedigree is drawn once for the population, not per proband, so a person who
appears in several probands' pedigrees has one status and one genetic value.

## Step 2 — Estimate the incidence curve

Bounds need a cumulative-incidence curve. With follow-up records — entry age,
exit age, and an event code — ltpred estimates one. `simulate_followup_records`
makes records from a **known** logistic curve so the estimate can be checked.

First without competing mortality:

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

Now add a competing risk of death. This is the case that separates the two
estimators:

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

Read that last line carefully, because it is the trap. Kaplan–Meier barely moved
when 21,066 people died: treating death as ordinary censoring reproduces the
no-death curve. It is not broken — it consistently estimates *its own* estimand,
the risk you would see if nobody died. But a dead person cannot be diagnosed, so
the quantity a registry actually measures is lower, and Aalen–Johansen is the
estimator that targets it. See [CIP estimation](cip-estimation.md).

## Step 3 — Score the population

`estimate_liabilities` is the register driver: it takes the parent-pointer table
plus per-person diagnosis records, walks each proband's pedigree, and returns one
genetic-liability score per proband.

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

Three things to notice. The score correlates 0.55 with the truth it never saw.
The posterior **variance** is returned per proband and is almost as large as the
score's own spread — most of these people have few informative relatives, so the
score is genuinely uncertain and should not be used as if it were a measurement.
And `frac_records_with_unresolved_parents` is a data-quality diagnostic you
should check on real data, where a parent id may point at someone outside the
extract.

Here the generating curve is passed straight in. On real data you would pass the
curve estimated in step 2, stratified by sex and birth cohort —
`strata` plus `cip_by_stratum` — because one curve for everyone is exactly the
approximation step 2 exists to remove.

## Step 4 — Score prospectively, without the proband's own diagnosis

`use="gwas"` builds a GWAS phenotype and includes the proband's own diagnosis.
`use="prediction"` hides it, so the score can be used to predict *future* onset
in people who are still disease-free. That changes the observation set, so it
needs a calendar: `birth_time` for every person and `index_time` for every
proband.

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

The correlation drops from 0.55 to 0.31 — not a defect, the price of the
question. Step 3 was allowed to see each proband's own diagnosis; step 4 is not,
so it has only relatives to work with. Restricting `probands` to the at-risk set
is not optional either: scoring someone already diagnosed and calling the result
prospective is leakage, and the driver warns if you do it. `proband_state` tells
you how each proband was classified.

Note `index_time` is aligned to **probands**, while `birth_time` is aligned to
**ids** — they are different lengths, and passing the wrong one raises.

## Step 5 — Two traits at once

For two binary traits, `simulate_under_LTM_multi` builds families with
per-trait bounds and `fit_pairwise_multi` fits heritabilities, the genetic
correlation and the residual environmental correlation together.

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

The first line is a check on the *simulator*: the realised correlation matches
the parameter it was built from. The rest are checks on the *fitter*, and every
one lands within about two standard errors of its target — which is what one
cohort of 3,000 families should give, and is not evidence of calibration. Always
read `inference_status` before quoting a standard error: fits that reach a
covariance boundary withhold them. The replicated evidence — 1,800 fits across
nine scenarios, heritability bias below 0.004 where the model is correctly
specified — is in `benchmarks/RESULTS.md` §33.

## What this tutorial skips

Each of these is a real modelling decision, deliberately left out so the page
stays runnable. Follow the link before applying any of it to real data.

!!! warning "This is a demonstration, not an analysis template"

    **One incidence curve for everyone.** Steps 3 and 4 pass a single
    `TRUE_CIP`. Real LT-FH++ needs sex- and birth-cohort-stratified curves via
    `strata` and `cip_by_stratum`, because incidence differs enough between
    strata that a pooled curve biases the bounds.
    → [CIP estimation](cip-estimation.md), [data preparation](data-preparation.md)

    **`h2` was given, not chosen.** The tutorial hard-codes `h2 = 0.5` and the
    simulator obeys it, so recovery is unsurprising. On real data `h2` is the
    liability-scale heritability *of that disease*, it has no default, and
    pedigree/twin and SNP estimates differ. Fitting it from family data requires
    a declared sampling design.
    → [data preparation, "Which h²?"](data-preparation.md#getting-heritability-on-the-liability-scale),
    [inference](inference.md)

    **The generating curve was fed back in.** Step 3 uses `TRUE_CIP` rather than
    the curve estimated in step 2, to keep the two steps separable. Passing a
    misspecified curve is a real failure mode with a measured effect.
    → [assumptions](assumptions.md)

    **Nothing is ascertained.** The cohort is an unascertained population sample
    (`sampling="population"`). Biobanks are usually selected on observed status,
    which changes what the fitters estimate.
    → [inference, ascertainment](inference.md#ascertained-samples)

    **The score is not a risk probability.** It is a posterior mean genetic
    liability on a standardised scale. Turning it into absolute risk needs the
    future-risk calculation, and using it in a GWAS needs residualisation for
    design covariates and an explicit join on the returned proband ids.
    → [estimation](estimation.md), [vignette](vignette.md)

    **One simulation illustrates an API.** Steps 1–5 are single seeded runs.
    Every quantitative claim ltpred makes is replicated across seeds in
    `benchmarks/RESULTS.md`, and the prose elsewhere links to it rather than
    repeating numbers.

## Where to go next

| you want to… | see |
|---|---|
| the full contract: three uses, `h²`, pedigrees, CIPs, family history | [Vignette](vignette.md) |
| the smallest possible hand-typed example | [Quickstart](quickstart.md) |
| prepare your own register or family-table data | [Data preparation](data-preparation.md) |
| estimate a stratified CIP from real follow-up records | [CIP estimation](cip-estimation.md) |
| choose an inference engine, or scale to biobank size | [Estimation](estimation.md) |
| fit `h²`, correlations or A/C/M components properly | [Inference](inference.md) |
| the real-data checklist before you trust a score | [Assumptions & checklist](assumptions.md) |
| a function signature | [API reference](api.md) |
