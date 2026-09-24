---
title: Vignette
description: How to run ltpred — three uses, then h², pedigree, CIP, family history.
---

# Vignette: how to run ltpred

ltpred estimates each proband's posterior mean genetic liability

$$
\mu_i = \mathbb{E}\bigl[a_i \mid D_F,\, A,\, h^2,\, K(\cdot)\bigr]
\tag{1}
$$

and writes it as the `genetic` column. That column is not a polygenic
score (PGS) and not an absolute risk. $D_F$ is the family
*observation model* (who was recorded, and as what interval or mixture
on liability), $A$ is the additive relationship matrix, $h^2$ is
liability-scale heritability, and $K(\cdot)$ is the prevalence or
cumulative incidence proportion (CIP). What you *do* with $\mu_i$ —
and whether you need it at all — depends on the use. Pearson–Aitken
(PA) is the single-trait default engine
([Aitken 1935](https://doi.org/10.1017/S0013091500008063);
[Mendell & Elston 1974](https://pubmed.ncbi.nlm.nih.gov/4813384/));
Gibbs is the truncated-normal sampler.

## Three uses

One `estimate_liability` call serves two of them, and what you put in for
the proband decides which: their own diagnosis in gives a GWAS phenotype
*of* that diagnosis, left out it gives a prediction *of* it from family
history. That is an input decision, made in step 3, not a later toggle on
the estimator. `examples/vignette.py` shows how to leave it out. The third
use may never call the estimator at all.

The published method names describe the *observation model*, not a
different genetic model. Three of the four name only that; PA-FGRS is
the exception, because its published specification also fixes the
engine.

- **LT-FH** (liability-threshold family history) uses one lifetime
  prevalence $K$ and relatives' statuses
  ([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6)).
- **LT-FH++** replaces that single $K$ with a person-specific CIP
  $K(t;\text{sex},\text{cohort})$ for the proband and relatives
  ([Pedersen et al. 2022](https://doi.org/10.1016/j.ajhg.2022.01.009)).
- **ADuLT** (age-dependent liability threshold) is the same
  personalised construction with no relatives
  ([Pedersen et al. 2023](https://doi.org/10.1038/s41467-023-41210-z)).
- **PA-FGRS** (Pearson–Aitken family genetic risk score) couples the PA
  engine to lifetime case intervals and an age-censored-control mixture
  ([Dybdahl Krebs et al. 2024](https://doi.org/10.1016/j.ajhg.2024.09.009)).
  It is not the register-standardised family genetic risk score (FGRS)
  of [Kendler et al. (2021)](https://doi.org/10.1001/jamapsychiatry.2021.0336).

Pick a row of Table 1 before building $D_F$.

**Table 1.** The three uses. *Steps* are the stages of Figure 1 and
Table 2 below — 0 heritability, 1 pedigree, 2 CIP, 3 family history,
4 `estimate_liability`, 5 downstream. Use III can stop at 0 and/or 2;
I and II need the estimator.

| Use | Question | Steps | Own status |
|---|---|---|---|
| **I** | Risk prediction from family history, optionally with a PGS | 0–5 | **out** of $D_F$ |
| **II** | Enhance the association signal in a genome-wide association study (GWAS) | 0–5 | **in** |
| **III** | Disease relationships and aetiology | **0 and/or 2** | — |

**Use I.** $\mu_i$ is a predictor of the diagnosis, so that diagnosis
must not leak into it. A PGS, if you have one, is a **downstream**
combination in the target population
([Hujoel et al. 2022](https://doi.org/10.1016/j.xgen.2022.100152);
[Dybdahl Krebs et al. 2026](https://doi.org/10.1016/j.ajhg.2025.11.016)),
not an ltpred call.

**Use II.** $\mu_i$ is a quantitative GWAS phenotype *of* that diagnosis,
which is why the proband's own status belongs in $D_F$
([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6)).
ADuLT skips relatives: all of step 1, and the relatives' rows in step 3.

**Use III.** Liability-scale $h^2$ and the genetic correlation $r_g$
constrain genetic architecture and how two diseases relate; the CIP is
the age/sex/cohort pattern of incidence. Pedigree, family history and
$\mu_i$ are all optional. ltpred does not fit $r_g$ from family data —
you supply it as `genetic_corrmat` — and fitting $h^2$ from the *same*
families is a different contract ([Inference](inference.md)).

## The pipeline at a glance

Figure 1 draws equation (1) as a flow: the four inputs assembled along
two independent tracks, the three uses as exits. Table 2 names the call
at each step.

[![ltpred pipeline. Two independent tracks feed one estimator. Left track, population quantities: step 0 liability-scale heritability and covariances, step 2 the CIP or one lifetime prevalence K; use III, disease relationships and aetiology, exits here. Right track, this sample: step 1 the pedigree A, step 3 the family-history records. The tracks converge into the family covariance Sigma, which has a unit diagonal, and into the observation intervals D_F. Step 4, estimate_liability, returns mu. Use I, risk prediction, exits with own status out of D_F; use II, a GWAS phenotype, exits with own status in.](assets/pipeline.svg)](assets/pipeline.svg)

**Figure 1.** How the inputs assemble, and where you can stop. The left
track is the population model and can be the whole analysis (use III).
The right track is this sample. Step 4 is the first call that conditions
on all four inputs. Open the figure in a new tab for a full-size view.

**Table 2.** What to do, in order. Function names only; the dependencies
are in Figure 1. Skip steps that Table 1 says the use does not need.

| Step | You supply | ltpred |
|---|---|---|
| — | Case definition, who the probands are | not software |
| 0 | Liability-scale $h^2$; optional $c^2$, $m^2$; two-trait $r_g$ | `observed_to_liability_h2`, `tetrachoric`; optionally `fit_heritability` |
| 1 | Who is related to whom | roles, or `kinship_from_pedigree` / `extract_pedigree` |
| 2 | Prevalence or CIP $K(\cdot)$ (cumulative incidence) | `prevalence_thresholds` or `thresholds_from_cip` |
| 3 | Status and ages (family history) | `families_from_columns` |
| 4 | — | `estimate_liability` |
| 5 | People to score (I) or genotype (II) | join `pids`; GWAS / PGS combination is elsewhere |

The step numbers are the order in which you assemble equation (1).
Steps 0–3 can be prepared in parallel except for three real
dependencies:

1. **Heritability scales the pedigree.** You can build $A$ without a
   heritability, but you cannot form the family covariance
   $\Sigma = h^2A + c^2C + m^2M + e^2I$, $e^2 = 1-h^2-c^2-m^2$, without
   both 0 and 1. With no shared-environment components that is just
   $h^2A + (1-h^2)I$. Note the unit diagonal either way: the residual
   absorbs whatever the kernels take, which is what keeps
   $\Phi^{-1}(1-K)$ a prevalence threshold. The weights that
   then project the family onto $a_i$ are the best linear unbiased prediction
   (BLUP) / selection-index weights
   ([Henderson 1975](https://doi.org/10.2307/2529430);
   [derivation](algorithm.md#connection-to-selection-index-and-blup)).
   ADuLT is the $1\times 1$ case.
2. **CIP before family history.** Status and age are a register table
   until $T_i=\Phi^{-1}(1-K_i)$ turns them into intervals on liability.
   That is why 3 follows 2.
3. **Pedigree before family history.** Rows of $D_F$ are grouped by
   `fam_id` / roles (or kinship column order). That is why 3 follows 1.

Lee's map
([Lee et al. 2011](https://doi.org/10.1016/j.ajhg.2011.02.002))
is the only backward dependency — noted between steps 0 and 2 in Figure 1:
converting an observed-scale $h^2$ uses the same population $K$ as the
thresholds.

## How much it buys you

Against a raw 0/1 phenotype, classic LT-FH on registry pedigrees gains a
squared-correlation effective-$N$ proxy of $2.233 \pm 0.127\times$ under
population sampling, but only $1.242 \pm 0.018\times$ at a 10% observed
case fraction, $1.107 \pm 0.008\times$ at 25%, and $1.081 \pm 0.003\times$
at 50%
([RESULTS §10](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).
The gain is largest exactly where cases are rare. Adding age and cohort
personalisation on top is a further $1.02$–$1.05\times$ across the same
grid, and that increment grows rather than shrinks with case enrichment.
Its larger role is elsewhere: the CIP is what keeps cohort differences in
follow-up from confounding the score (section 3 and
[RESULTS §13](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#13-cohort-confounding-bench_confoundingpy)).

## How to read the code on this page

Every block below is a **fragment**: it shows the call, not a runnable
program. Lower-case names (`status`, `age`, `cip_ages`, `fam_id`,
`role`, `pid`) are your own columns. For a page whose blocks all run in
sequence on one simulated cohort, with the truth known and every quoted
output checked by a test, see the **[tutorial](tutorial.md)**. Two
listings do run end to end:

- [`examples/vignette.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/vignette.py)
  — this pipeline on simulated data (`pip install -e ".[fast]"`, then
  `python examples/vignette.py`);
- the [Quickstart](quickstart.md) — six hand-written rows, start to
  finish.

Quoted figures come from that script at seed 1, $n=800$ nuclear
families, true $h^2=0.5$, $K=0.05$. Three blocks need a larger cohort
and say so where they appear: the tetrachoric convergence check in
section 0 (25,000 families), the use-I risk figures in section 5 (10
replicates of 4,000), and the age-censored comparison at the end
(20,000). `tests/test_vignette_numbers.py` re-runs the script and fails
if the figures it checks have drifted; numbers quoted from RESULTS are not
re-run. The figures check the API; they are not a benchmark — for measured behaviour see
[RESULTS](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
The cohort behind them is

```python
from ltpred import simulate_under_LTM_single

h2, K, n_fam = 0.5, 0.05, 800
sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=h2, pop_prev=K, n_sim=n_fam,
    use_age=False, seed=1,
)
```

`use_age=False` makes it classic LT-FH under the liability-threshold
model: one $K$, not a CIP. On real data, delete this block and start
from your own table.

Register-route fragments use a different cohort from the role grammar above:
unique population ids, a 0.10-horizon incidence curve, and `h2=0.5`. The
[tutorial](tutorial.md) builds that cohort, scores it, and checks the score
against `cohort.genetic`. The blocks here stay fragments.

## Before software

Fix a case definition, follow-up, and a family-history source that you are
willing to defend
([checklist](assumptions.md#real-data-checklist)). Decide who the
**probands** are (usually the genotyped people), and pick a row of
Table 1 — uses I and II disagree on whether the proband's own diagnosis
enters $D_F$. Use III may not need a pedigree at all.

Your input is one row per **observed person**, in long format:

| `fam_id` | `pid` | `role` | `status` | `age` |
|---|---|---|---|---|
| F1 | P1 | `o` | 1 | 48 |
| F1 | M1 | `m` | 0 | 71 |
| F1 | D1 | `f` | 1 | 55 |
| F2 | P2 | `o` | 0 | 36 |
| F2 | M2 | `m` | 1 | 62 |

`age` is age of onset for cases and age at last follow-up for controls.
Steps 2 and 3 turn the last two columns into `lower`/`upper` bounds;
`families_from_columns` never sees `status` or `age` itself.

## 0. Heritability and covariances

For uses I and II, `estimate_liability` **conditions** on $h^2$. Pass a
**liability-scale** value. An observed-scale number mis-calibrates
$\hat{\mu}_i$ (ranking is more robust than the scale). For use III, $h^2$
and $r_g$ can *be* the result.

How much the scale matters is measured. At a true $h^2$ of 0.5 and
$K=0.05$, assuming 0.2 leaves the ranking essentially untouched
(correlation with the truth $0.429 \pm 0.004$ against $0.431 \pm 0.004$
when correctly specified) but sweeps the calibration slope to
$2.240 \pm 0.015$; assuming 0.8 drops the slope to $0.678 \pm 0.006$
([RESULTS §12](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).
Rank-only analyses — a GWAS phenotype, a top-decile flag — tolerate a
wrong $h^2$; anything that reads the scale does not. Re-run step 4 at
$h^2 \pm 0.1$ and report both the rank correlation between the two runs
(it should stay near 1) and the ratio of their score standard deviations
(it will not).

Prefer an external estimate. A first-degree check is Falconer's
$h^2 \approx 2\rho_{\mathrm{tet}}$ from the **parent–offspring**
tetrachoric correlation
([Falconer 1965](https://doi.org/10.1111/j.1469-1809.1965.tb00500.x)).
The sib tetrachoric is not interchangeable: full sibs also share the
sibship kernel, so twice the sib correlation estimates $h^2+2c^2$ —
inflated by exactly twice the $c^2$ you might fit below. Both routes are
inflated further by assortative mating and by parent–offspring
environmental transmission.

The Lee map from an observed-scale estimate in a **population**
sample
([Lee et al. 2011](https://doi.org/10.1016/j.ajhg.2011.02.002))
is

$$
h^2_{\mathrm{liab}}
  = h^2_{\mathrm{obs}} \cdot \frac{K(1-K)}{\varphi(T)^2},
  \qquad
  T=\Phi^{-1}(1-K).
\tag{2}
$$

$K$ here is the same population prevalence as step 2 (the CIP plateau, or
the single lifetime $K$ of classic LT-FH). If the GWAS over-samples cases,
pass that fraction as `prop_cases`.

```python
from ltpred.tetrachoric import tetrachoric
from ltpred import observed_to_liability_h2

po = tetrachoric(status_o, status_m)          # parent-offspring statuses
po.rho, po.se                                 # always read the SE
observed_to_liability_h2(0.20, pop_prev=0.05) # -> 0.893 at K = 0.05
```

**Read the standard error, and check the sample size first.** A
tetrachoric at $K=0.05$ is carried by the *case–case* cell, and the
800-family cohort above holds just one such parent–offspring pair (the
median across seeds is five). It returns
$\rho_{\mathrm{tet}}=-0.082 \pm 0.188$, so
$2\rho_{\mathrm{tet}}=-0.164 \pm 0.376$ against a truth of $0.5$: not
evidence against the model, just no evidence at all. Across cohorts of
this size the estimate is centred on the right value but has a standard
deviation of $0.13$ — half of what it is estimating — and comes out
negative about one run in fifteen. The same estimator on 25,000 families
returns $0.247 \pm 0.023$, so $2\rho_{\mathrm{tet}}=0.494 \pm 0.046$.
Falconer's route works; it just does not work on a few hundred
rare-disease families, and a close hit there is luck.

The sib tetrachoric on those 25,000 families is $0.268 \pm 0.023$ —
close only because this simulator has no sibship kernel to inflate it.

Optional sibship $c^2$ and couple $m^2$ go on the single-trait estimator
as `c2` / `m2` ($h^2+c^2+m^2\le 1$). Two traits add $r_g$ here and
require Gibbs in step 4: pass vector `h2`, `genetic_corrmat`, and
`full_corrmat`.

Fitting $h^2$ or $A{+}C{+}M$ (additive genetic, sibship, and couple
shared-environment components) from the **same** families is a
different contract on three counts.

1. *Sampling.* Independent, non-overlapping pedigrees under
   `sampling="population"`, or inverse-probability weighting
   (`sampling="ipw"`) with known positive inclusion probabilities. A
   person identifier (`pid`) in two family identifiers (`fam_id`) is
   rejected. Unguarded ascertainment pins $\hat{h}^2=1$ even when the
   truth is 0.
2. *Bounds.* `fit_heritability` needs **one common case/control
   threshold per trait**. The personalised or onset-pinned bounds built
   in steps 2–3 are rejected outright; fit from `prevalence_thresholds`
   bounds, or bring an external $h^2$.
3. *Scale.* Even when the contract holds, a few hundred families is not
   much data. On the 800 simulated families above, `fit_heritability`
   with a short schedule (`n_iter=250, burn_in=80`) returns
   $\hat h^2=0.469$ against a truth of $0.5$, with a within-dataset
   Monte-Carlo standard error of $0.011$ — that number is a fixed-point
   diagnostic, not a sampling interval, and the default schedule
   (1500/500) moves the estimate to $0.443 \pm 0.007$ at the same seed. Across cohorts of
   this size the spread is an order of magnitude larger. Use
   `bootstrap_fit` for a family-cluster interval.

Details: [Inference](inference.md).

### Joint heritability and genetic/environmental correlation

For two or more binary traits, `fit_pairwise_multi` fits the covariance
components together using observed-pair probabilities. It is deterministic;
there is no Gibbs burn-in or Monte-Carlo trace. `simulate_under_LTM_multi`
generates a **separate** population cohort of 3,000 nuclear families with two
traits at prevalences 0.10 and 0.20, and reports the generating design in
`.truth`:

```python
from ltpred import fit_pairwise_multi, simulate_under_LTM_multi

multi = simulate_under_LTM_multi(n_families=3000, seed=1)
joint = fit_pairwise_multi(
    multi.families, components=("A", "C", "M"),
    sampling="population", phen_names=list(multi.phen_names),
)
joint.h2                      # per-trait liability-scale heritability
joint.rg[0, 1]                # genetic correlation
joint.re[0, 1]                # within-person residual environmental correlation
joint.correlations["C"][0, 1] # shared full-sibship correlation
joint.correlations["M"][0, 1] # couple correlation
joint.inference_status        # inspect before interpreting uncertainty
joint.se["h2"], joint.se["rg"][0, 1], joint.se["re"][0, 1]
```

The simulation has $h^2=(0.35,0.40)$, $r_g=0.50$ and residual $r_e=-0.35$;
`multi.truth` echoes them. [Tutorial step 5](tutorial.md#step-5-two-traits-at-once)
prints this fit against its truth with standard errors. A single cohort
illustrates the API, not its calibration:
the replicated recovery evidence is
[RESULTS §33](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
For real data, prepare one row per person with
one bound per trait ([input recipe](data-preparation.md#preparing-multiple-traits-for-covariance-fitting)).
Thresholds may differ between traits, but not between people for a given
trait. A missing diagnosis is `(-inf, inf)`, not a control; missingness must
preserve the pair distributions. Use the population/positive-IPW sampling
contract above, with weights aligned to the returned family order.

Here `A` is additive genetics, `C` full sibship and `M` couple resemblance;
residual `E` is always included. **`re` refers only to `E`.**
`joint.rp - joint.genetic_cov` includes all non-genetic covariance and differs
from `joint.env_cov` when `C` or `M` is fitted. The default
`components=("A",)` fits A+E; omitting shared components can change the
genetic/residual attribution. Merely adding them does not solve that problem:
the observed relationship contrasts must identify every fitted component.

Unlike `fit_heritability`'s Monte-Carlo `h2_se` above, these SEs quantify
asymptotic **sampling**
uncertainty, clustering all pairs from the same family. They treat thresholds
and weights as fixed. Any covariance boundary withholds all normal SEs
(`NaN`); a correlation with negligible component variance is also undefined.
One example run is not a calibration study; the
[replicated evidence, including boundary counts, is in RESULTS §33](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).

This fit can be the endpoint for use III. Multi-trait **scoring** in step 4
still supports A+E only: do not feed a C/M fit into it by folding shared
covariance into `full_corrmat`. See [Inference](inference.md#joint-heritability-and-cross-trait-correlations)
for the model and full result contract.

## 1. Pedigree

You need $A$, the additive relationships among the people whose records
enter $D_F$. Two routes.

**Role grammar** (nuclear and common extended families). Each label is
relative to the proband: `o` (optional own status), `m`/`f`, `s1`/`s2`,
grandparents `mgm`/`pgf`, half-sibs `mhs1`/`phs1`, avuncular `mau1`/`pau1`,
children `c1.1`. Number repeats. Same-side half-sibs are treated as full
sibs of each other — use the kinship path if that is false
([role grammar](data-preparation.md#role-grammar)).

**Parent pointers**, for cousins, inbreeding, or messy second parents.
Extract first, then build the matrix, so the rows follow the extracted
order:

```python
from ltpred import build_parent_graph, extract_pedigree, kinship_from_pedigree

# ids / father / mother are your population trio records
graph = build_parent_graph(ids, father, mother)
ped = extract_pedigree(graph, proband_id, max_degree=2)   # one proband
_, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
```

**How deep?** First-degree relatives carry most of the signal, and
more distant ones add a small, real amount. In the register pipeline,
correlation with the true genetic value was $0.567 \pm 0.029$ at degree 3
against $0.522 \pm 0.031$ at degree 1, a paired gain of $+0.045$ (95% CI
$+0.020$ to $+0.071$;
[RESULTS §21](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#21-end-to-end-register-pipeline-bench_register_pipelinepy)),
and all relatives to third degree beat the named-role subset by $+0.071$
([RESULTS §20](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#20-pedigree-inference-from-trio-records-bench_pedigree_inferencepy)).
The calibration grid's structure contrast is not a depth contrast: it
swaps one sibling for four grandparents
([RESULTS §12](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md#12-score-calibration-bench_calibrationpy)).
`estimate_liabilities` defaults to `max_degree=3`.

Scoring may use overlapping extracted pedigrees (one per proband). Fitting
$h^2$ in step 0 may not. ADuLT has no relatives: skip this step.

## 2. Cumulative incidence proportion (CIP) or lifetime prevalence

The CIP is the fraction of people in a population stratum who are
diagnosed by a given age,
$K(t; s, b)=\Pr(\text{diagnosed by age }t\mid\text{sex }s,\text{birth year }b)$.
For use III that curve *is* the result, and the estimator is the last
call you make:

```python
from ltpred import aalen_johansen_cip

curve = aalen_johansen_cip(age_entry, age_exit, event)  # 0 censor, 1 case, 2 death
curve.ages, curve.values, curve.se                      # report stratum by stratum
```

Use a **population** curve, stratified by sex, birth year, and ancestry
where incidence differs — not the logistic demo helper and not a biobank
case fraction. For competing death use Aalen–Johansen, not a
Kaplan–Meier that treats death as censoring: the LT-FH++ construction
uses Aalen–Johansen with death and emigration as competing events
([Pedersen et al. 2022](https://doi.org/10.1016/j.ajhg.2022.01.009);
details in [CIP estimation](cip-estimation.md)).
Cover every analysed age; pass `k_pop` unless the last CIP value is a
defensible lifetime prevalence.

This is not cosmetic for use II. Under a prevalence trend of $3\times$
per 30 years, cohort-specific CIPs hold genomic inflation at
$\lambda = 0.922 \pm 0.057$, while the same family-history phenotype
built on a single lifetime $K$ inflates to $10.275 \pm 0.078$ — worse
than the raw 0/1 label at $4.616 \pm 0.220$
([RESULTS §13](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).
A family-history phenotype concentrates a cohort trend rather than
diluting it.

For uses I and II the curve is an input to the thresholds. Classic LT-FH
uses one $T=\Phi^{-1}(1-K)$ instead. LT-FH++ and ADuLT use a
person-specific CIP

$$
T_i = \Phi^{-1}\bigl(1 - K(t_i; s_i, b_i)\bigr),
\tag{3}
$$

with $t_i$ the age of onset for a case and the age at last follow-up for a
control.

```python
from ltpred import prevalence_thresholds, thresholds_from_cip

# classic LT-FH: one K
lower, upper = prevalence_thresholds(status, pop_prev=K)

# LT-FH++ / ADuLT: one call per stratum, scattered back into row order
lower, upper, K_i, K_pop = thresholds_from_cip(
    status, age, cip_ages, cip_values, k_pop=lifetime_K, case_mode="pin",
)
```

With more than one stratum you call this once per (sex, birth-cohort,
ancestry) group and write each result into the rows of that group. The
full loop, including `K_i`/`K_pop`, is the
[register-data recipe](data-preparation.md#a-real-register-data-recipe).

`case_mode="pin"` is the LT-FH++ encoding: a case becomes a point mass at
$T_i$. Pin only when onset really is the CIP inverse of liability.
Encoding the case as the interval $[T_i,\infty)$ instead keeps most of
the onset information. Base PA-FGRS uses the **lifetime** case interval
$[\Phi^{-1}(1-K_{\mathrm{pop}}),\infty)$ and puts age into a censored
control's mixture weight $K_i$
([Dybdahl Krebs et al. 2024](https://doi.org/10.1016/j.ajhg.2024.09.009)).

## 3. Family-history records

One row per observed person: `fam_id`, `role` (or a kinship column order),
`status`, `age`. Missing `fam_id` and duplicate roles in a family are
rejected. Join these records to the pedigree from step 1; the thresholds
from step 2 are the `lower` / `upper` columns.

Families need not have the same relatives. `estimate_liability` groups
families by structure — identical role sets share one covariance — so a
family with only a mother and one with four sibs can go in the same
list. A relative of known relationship but unknown status is a
$(-\infty,\infty)$ row: it contributes nothing and does no harm. Only the
array path needs a fixed column set, where an absent relative is padded
the same way ([Estimation](estimation.md#scaling-to-large-cohorts)).

Include role `o` for **use II** ($\mu_i$ is a GWAS phenotype of that
diagnosis). For **use I**, give `o` bounds of $(-\infty,\infty)$ rather
than dropping the row: `res.pids` is read off the role-`o` record and
uses `fam_id` only when no member has a pid. If relatives have pids but the
proband does not, estimation raises to prevent silently changing
your join key. ADuLT with `o` unbound has nothing left to condition on.
Use III may skip this step.

```python
from ltpred import families_from_columns

families = families_from_columns(
    fam_id, role, lower, upper, pid=pid,
    K_i=K_i, K_pop=K_pop,   # only for use_mixture=True
)
```

## 4. Estimate the score

This is the ltpred run. Everything above is input.

```python
from ltpred import estimate_liability

res = estimate_liability(families, h2=h2)          # PA, single trait
# res = estimate_liability(families, h2=h2, method="gibbs", seed=1)
# res = estimate_liability(families, h2=h2, use_mixture=True)
score = res.genetic                                # aligned to res.pids
```

- `res.se["genetic"]` is Monte-Carlo error in $\hat{\mu}_i$ — exactly 0
  under PA, which is deterministic. Zero means no Monte-Carlo error, not no
  approximation error: PA folds coordinates sequentially and keeps two
  moments.
- `res.var["genetic"]` is $\mathrm{Var}(a_i\mid D_F)$, which does not
  shrink with more draws.

PA is the default for one trait. Use Gibbs for a Monte-Carlo standard
error (SE), multiple traits, or an unusual no-mixture pedigree. The
PA-FGRS mixture is PA-only. The kinship entry point is
`estimate_liability_from_kinship(A, lower, upper, h2=h2, target=0)`,
which returns a bare `(est, se, var)` tuple rather than a result object.

With relatives and a personalised CIP this is LT-FH++; with only role `o`
it is ADuLT; with one lifetime $T$ and relatives it is classic LT-FH.

**Sizing the run.** On the reference machine at 4 Numba threads the
object path scores 208,000–221,000 families/s under PA and 503–512/s
under Gibbs, so 500,000 probands is a few seconds under PA and roughly
17 minutes under Gibbs
([RESULTS §2](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)).
Past a few million families, `Family` construction rather than the PA
arithmetic dominates: switch to `estimate_liability_pa_arrays`, which
takes already-aligned `(n_families, k)` bound arrays, and set the thread
count with `ltpred.set_num_threads(n)`
([Estimation](estimation.md#scaling-to-large-cohorts)).

### Population trio-register route

For unique population `ids`, the public driver combines pedigree extraction,
row alignment, CIP conversion and scoring. Choose a single supplied curve
with `cip_ages`, `cip_values`, `k_pop`, or use `strata` plus a mapping of each
label to `(cip_ages, cip_values, k_pop)`; do not mix the two routes.

```python
from ltpred import estimate_liabilities

common = dict(
    ids=ids, father=father, mother=mother, status=status, age=age,
    h2=h2, max_degree=1, strata=strata, cip_by_stratum=cip_by_stratum,
)
gwas_scores = estimate_liabilities(probands=gwas_ids, use="gwas", **common)
prediction_scores = estimate_liabilities(
    probands=prediction_ids, use="prediction", birth_time=birth_time,
    index_time=index_time, **common,
)
score = prediction_scores.est       # aligned to prediction_scores.probands
posterior_var = prediction_scores.var
```

`birth_time` is aligned to population `ids`; `index_time` is aligned to
`prediction_ids`. Both use one numeric calendar scale whose unit matches
`age`, for example calendar years and attained years. At index time $t$, a
relative born at $b_j$ is censored at attained age $t-b_j$, **not** at the
proband's attained age. A post-index diagnosis becomes a control censored
at that relative-specific age; follow-up ending earlier is kept at its
recorded end. A person born at or after the landmark is uninformative.
The proband's observation is always uninformative under `use="prediction"`.
That makes the prediction estimand $\mathbb{E}[g \mid \text{relatives'
records at the landmark}]$ — it is not additionally conditioned on the
proband being disease-free at the landmark; the two agree on ranking within
an age and family structure but differ in level across ages, since
surviving to an older age disease-free is evidence of lower liability.

The driver currently supports **single-trait, additive-only, pinned-onset
LT-FH++ with deterministic PA**. Use the lower-level APIs for C/M kernels,
Gibbs, interval cases or PA-FGRS mixtures. It returns `PopulationScores`,
not `LiabilityResult`: there is no `genetic` property or Monte-Carlo `se`
column. `est` and `var` are PA mean and posterior-variance approximations.
Its payoff and throughput evidence (RESULTS §§20–21, regenerated 2026-09-21):
degree-3 corr(est, true g) 0.567 ± 0.029 vs degree-1 0.522 ± 0.031,
prospective familywise-censored AUC 0.654 ± 0.021, and 987 probands/s
(v0.7.1, rerun 2026-09-23).

Check `n_relatives` (non-proband members within the chosen degree),
`n_conditioned` (informative diagnosis bounds, including own status for
GWAS), `n_closure_only` (extra structural ancestors) and `degree_max`.
Two further fields guard the inputs rather than the score:
`frac_records_with_unresolved_parents` is the fraction of records with a non-null parent
reference that matched no id — a zero resolved share raises, a share above
half warns, because that pattern is a broken join more often than a register
boundary — and, under prediction, `proband_state` names each proband
`disease_free_and_followed`, `prevalent_case` or `exited_before_index` at
their landmark, warning on prevalent cases: only the first belongs in a
prospective evaluation.
In the runnable six-person example in
[`examples/vignette.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/vignette.py),
`max_degree=1` deliberately selects two parents and a sibling; two maternal
grandparents are retained only for exact kinship. Prediction has **three
relatives, two closure-only ancestors and three conditioned records**;
GWAS has four conditioned records. Changing only post-index proband/mother
records and closure-only diagnoses leaves the prediction mean and variance
exactly unchanged. This is an API/leakage check using a toy CIP, not clinical
calibration or new performance evidence.

Those object-path timings are grouped role families. They omit per-proband
register extraction, kinship, and CIP alignment, so do not extrapolate them
to `estimate_liabilities`.

### Did it work?

The model implies checks you can run on the returned object. The law of
total variance is the useful one: conditioning splits the genetic
variance into what the family explained and what it did not, and the two
must add back to $h^2$.

```python
import numpy as np

mu, v = res.genetic, res.var["genetic"]
# these are per-proband, aligned to res.pids — not the per-row `status` column.
# The subset lines up with `mu` only if every family carries an `o` row; role
# Join on res.pids; retain an uninformative o row when using personal IDs.
own = np.asarray(status)[np.asarray(role) == "o"]

assert len(res.pids) == len(families)                           # one score per proband
np.testing.assert_allclose(mu.var() + v.mean(), h2, atol=0.02)  # law of total variance
assert v.max() <= h2 + 1e-8            # conditioning cannot add genetic variance
assert abs(mu.mean()) < 0.05           # mu is a deviation from the population mean
assert mu[own == 1].mean() > mu[own == 0].mean()
```

The variance identity holds in expectation on a correctly specified,
population-sampled cohort, so give it a tolerance rather than an equality;
`atol=0.02` is comfortable at $n=800$. The `v.max()` bound assumes a
non-inbred proband, where $\mathrm{Var}(a_i)=h^2$ — with inbreeding
$A_{ii}>1$ and the ceiling is $h^2A_{ii}$.

On the cohort above: $\mathrm{Var}(\hat\mu)=0.081$ plus a mean posterior
variance of $0.415$ gives $0.496$ against $h^2=0.5$; $\hat\mu$ has mean
$-0.008$ and standard deviation $0.285$; cases average $+1.05$ and
controls $-0.06$.

The sum holds however much information the families carry, so a sum far
from $h^2$ points to a misspecified prevalence or CIP, ascertainment, or
an approximation error, not to missing relatives. It tracks the *assumed*
$h^2$, so it cannot detect a wrong one. A mean far from 0 says the assumed
prevalence disagrees with the observed case rate; that check holds only
under population sampling.

## 5. What you do with the score

**Use I (prediction).** Join `res.pids` to the people whose risk you
want. Do not put the predicted diagnosis into $D_F$. A PGS, if you have
one, is combined **after** this step — a target-population prediction
model, not something ltpred fits
([Hujoel et al. 2022](https://doi.org/10.1016/j.xgen.2022.100152);
[Dybdahl Krebs et al. 2026](https://doi.org/10.1016/j.ajhg.2025.11.016)).

On the liability scale $\mu_i$ is in population standard-deviation
units, not a probability. For **use I only** — own status out of $D_F$,
so the proband's residual is independent of the family — the model's
implied risk is

```python
import numpy as np
from scipy.stats import norm

T = norm.isf(K)                      # or the person's own T_i from step 2
risk = norm.sf((T - res.genetic) / np.sqrt(res.var["genetic"] + 1 - h2))
```

The reasoning is that $\ell_i = a_i + e_i$ with $\mathrm{Var}(e_i)=1-h^2$,
and with own status out of $D_F$ the residual $e_i$ is independent of the
family, so $\ell_i \mid D_F$ is centred on $\mu_i$ with variance
$\mathrm{Var}(a_i \mid D_F) + 1 - h^2$. Two conditions come with that: it
assumes no shared-environment components ($c^2=m^2=0$, or $e_i$ is coupled
to the relatives), and it treats $a_i \mid D_F$ as Gaussian, which is the
same two-moment approximation PA makes.

At $h^2=0.5$, $K=0.05$, under the generating model, it is calibrated
overall and in the top decile. Over
10 replicates of 4,000 relatives-only families the predicted rate is
$0.0501 \pm 0.0001$ against an observed $0.0504 \pm 0.0014$ (a gap of
0.2 standard errors); the top decile is $0.1192 \pm 0.0006$ predicted
against $0.1175 \pm 0.0073$ observed (0.2 SE); the bottom decile is
$0.0401$ predicted against $0.0420 \pm 0.0013$ observed (1.5 SE). The
bottom decile's *predicted* rate has no replicate-to-replicate spread at
all: with three relatives at $K=0.05$, 86% of families share the single
lowest-risk configuration — no affected relative — so the lowest 400 are
an arbitrary subset of one stratum rather than a tail. The script breaks
such ties with a stable sort (simulation order), so every platform picks
the same families. A *single* replicate can look off
by two or three standard errors in the tail, so do not read one run as a
bias. The formula is **not** valid for use II, where the proband's own
status is already in $D_F$.

**Use II (GWAS).** Join `res.pids` to genotyped IDs. Residualize for sex,
cohort, ancestry principal components (PCs), and batch, or put them in a
mixed model. Prefer an association method that handles relatedness if
related targets remain
([Zhuang et al. 2022](https://doi.org/10.1093/bioinformatics/btac459)).
ltpred does not run the GWAS.

**Use III** does not need this step. The quantities of interest were $h^2$,
$r_g$, and/or $K(\cdot)$ in steps 0 and 2.

## What the run produced

```python
from ltpred import estimate_liability
import numpy as np

pa = estimate_liability(sim.families, h2=h2)
status, true_g = sim.status["o"].astype(float), sim.genetic

np.corrcoef(status, true_g)[0, 1]       # 0.353
np.corrcoef(pa.genetic, true_g)[0, 1]   # 0.426
```

**Table 3.** Correlation with the simulated true genetic value $a_i$, on
the seed-1 cohort defined above. The 0/1 label is the baseline a GWAS
would otherwise use.

| Score | Correlation with $a_i$ |
|---|---|
| Proband 0/1 status (baseline) | 0.353 |
| Own status only (no ages; ADuLT's degenerate case) | 0.353 |
| Relatives only, uninformative `o` | 0.288 |
| Classic LT-FH via PA (`o` plus `m`, `f`, `s1`) | 0.426 |

0.426 against 0.353 is a squared-correlation gain of $1.46\times$, the
effective-$N$ proxy the script prints. Two probands show the shrinkage
behind it: an affected one gets $\hat{\mu}_i=+0.947$ with posterior
variance $0.271$, an unaffected one $-0.119$ with variance $0.430$.
One cohort of 800 is a noisy read on that ratio — treat RESULTS §10,
not this line, as the measurement.

ADuLT ties the baseline for a structural reason, not by coincidence:
with no relatives and a single lifetime $T$ it is a monotone relabelling
of the 0/1 status, so any correlation is identical by construction.

Three identity checks. PA against Gibbs on 80 families:
$\mathrm{Corr}=0.9997$, at a median Gibbs Monte-Carlo SE of $0.0066$.
The same families rebuilt from columns reproduce the role-grammar PA
scores exactly (maximum difference 0). Scored through
`kinship_from_pedigree` they agree to $1.1\times10^{-3}$ — PA's
sequential fold order differs between the two row layouts, so that is
approximation error, not round-off and not Monte-Carlo noise.

The script's last block reruns the same design with `use_age=True`,
where a person counts as a case only once onset precedes their current
age. Simulated proband ages are young (median 37, against the incidence
curve's mid-point of 60), so about 89% of would-be proband cases become
censored controls: the observed case rate falls from 0.045 to 0.0050, so the block uses **20,000** families to leave 99
observed cases behind the own-status baseline — at $n=800$ it would rest
on one to five, and the ratios below would be noise. The
$\mathrm{Corr}=0.251$ there is **not** comparable with the 0.426 above.

The block also carries a control that is easy to omit and easy to
misread without. On that censored cohort the proband's own 0/1 label
reaches 0.139; classic one-$K$ LT-FH on the *same* rows, using no age
information at all, already reaches 0.245; the age-aware encoding, given
the simulator's true incidence curve, then reaches 0.251. So family history does nearly all the work
($3.11\times$ on the squared-correlation proxy) and the age term adds
$1.05\times$ on top — inside the $1.02$–$1.05\times$ quoted above and
in RESULTS §10. Credit the gain to the right input.

## Coming from LTFHPlus or LTFGRS

`estimate_liability` exists in all three packages with incompatible
signatures. Role labels are unchanged; the input object and the
uncertainty column are not.

| R | ltpred |
|---|---|
| `estimate_liability(.tbl = df, …)` | `families_from_columns(...)`, **then** `estimate_liability(families, h2=…)` |
| `fam_id =` / `fid =`, `pid =`, `role =` column-name arguments | pass the columns themselves |
| `$genetic_est` | `res.genetic` |
| LTFGRS PA `$var` (and its square root) | `res.var["genetic"]` — **not** `res.se`, which is 0 under PA because PA is deterministic |
| `future::plan(multisession, workers = n)` | `ltpred.set_num_threads(n)` |

ltpred ships no igraph-style pedigree object and no plotting utilities.

## Where to go next

| you want to… | see |
|---|---|
| copy-paste one page and run it | [Quickstart](quickstart.md) |
| decide which model to run | [Choose a method](guide.md) |
| handle roles, trios, real tables | [Data preparation](data-preparation.md) |
| estimate a CIP — Kaplan–Meier, Aalen–Johansen (use III may stop here) | [CIP estimation](cip-estimation.md) |
| pick an engine, scale up, export to a GWAS | [Estimation](estimation.md) |
| fit $h^2$ or $A{+}C{+}M$ | [Inference](inference.md) |
| run the pre-flight list | [Assumptions & checklist](assumptions.md) |
| see measured accuracy, speed and failure modes | [Benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md) |
| read the maths behind equation (1) | [Algorithm](algorithm.md) |
| read the methods note (same I/II/III) | [Technical report](report.md) |
