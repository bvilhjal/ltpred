# Numerical checks on simulated families

This page explains the numerical checks in
[`examples/validation.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/validation.py).
It is a companion to the [model](algorithm.md), not a second getting-started guide.
Run the complete script from a source checkout:

```bash
python examples/validation.py
```

## Cohorts and scope

Quoted figures come from that script at seed 1, $n=800$ nuclear
families, true $h^2=0.5$, $K=0.05$. Three blocks need a larger cohort
and say so where they appear: the tetrachoric convergence check in
the heritability section (25,000 families), the family-history risk figures (10
replicates of 4,000), and the age-censored comparison at the end
(20,000). `tests/test_validation_numbers.py` re-runs the script and fails
if the checked figures drift. Historical benchmark campaigns are not re-run
by these tests. These examples check numerical behaviour under the simulator;
for replicated evidence see
[RESULTS](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
The cohort behind them is

```python
from ltpred import simulate_under_LTM_single, estimate_liability

h2, K, n_fam = 0.5, 0.05, 800
sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=h2, pop_prev=K, n_sim=n_fam,
    use_age=False, seed=1,
)
```

`use_age=False` makes it classic LT-FH under the liability-threshold
model: one $K$, not a CIP. The [tutorial](tutorial.md) is the sequential worked
example; this page explains additional checks from the script.

## Heritability checks

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
\tag{1}
$$

$K$ here is the same population prevalence as the threshold model (the CIP plateau, or
the single lifetime $K$ of classic LT-FH). If the GWAS over-samples cases,
pass that fraction as `prop_cases`.

```python
from ltpred.tetrachoric import tetrachoric
from ltpred import observed_to_liability_h2

po = tetrachoric(sim.status["o"], sim.status["m"]) # parent-offspring statuses
po.rho, po.se                                 # always read the SE
observed_to_liability_h2(0.20, pop_prev=0.05) # -> 0.893 at K = 0.05
```

**Read the standard error, and check the sample size first.** A
tetrachoric at $K=0.05$ is carried by the *case–case* cell, and the
800-family cohort above holds just one such parent–offspring pair. It returns
$\rho_{\mathrm{tet}}=-0.082 \pm 0.188$, so
$2\rho_{\mathrm{tet}}=-0.164 \pm 0.376$ against a truth of $0.5$: not
evidence against the model, just little precision. The same estimator on 25,000 families
returns $0.247 \pm 0.023$, so $2\rho_{\mathrm{tet}}=0.494 \pm 0.046$.
The larger cohort gives a more precise estimate; neither single cohort
establishes estimator calibration.

The sib tetrachoric on those 25,000 families is $0.268 \pm 0.023$ —
close only because this simulator has no sibship kernel to inflate it.

Family-data fitting has the [sampling and common-threshold contract](inference.md).
Even when the contract holds, a few hundred families is not
much data. On the 800 simulated families above, `fit_heritability`
with a short schedule (`n_iter=250, burn_in=80`) returns
$\hat h^2=0.469$ against a truth of $0.5$, with a within-dataset
Monte-Carlo standard error of $0.011$ — that number is a fixed-point
diagnostic, not a sampling interval. Use
`bootstrap_fit` for a family-cluster interval.

Details: [Fitting](inference.md).

## Score and variance checks

The model implies checks you can run on the returned object. The law of
total variance is the useful one: conditioning splits the genetic
variance into what the family explained and what it did not, and the two
must add back to $h^2$.

```python
import numpy as np

res = estimate_liability(sim.families, h2=h2)
mu, v = res.genetic, res.var["genetic"]
own = sim.status["o"]  # simulated probands, aligned to res.pids

assert len(res.pids) == len(sim.families)                           # one score per proband
np.testing.assert_allclose(mu.var() + v.mean(), h2, atol=0.02)  # law of total variance
assert v.max() <= h2 + 1e-8            # conditioning cannot add genetic variance
assert abs(mu.mean()) < 0.05           # mu is a deviation from the population mean
assert mu[own == 1].mean() > mu[own == 0].mean()
```

The variance identity holds in expectation on a correctly specified,
population-sampled cohort, so give it a tolerance rather than an equality;
`atol=0.02` is comfortable at $n=800$. The `v.max()` bound applies to this non-inbred, no-mixture model.
On the kinship route an inbred target is standardised, so its prior genetic
variance is $h^2 A_{ii}/[1+h^2(A_{ii}-1)]$, not $h^2 A_{ii}$.
Mixture posteriors need not satisfy the same pointwise variance bound.

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

## Risk calibration under the generating model

On the liability scale $\mu_i$ is in population standard-deviation
units, not a probability. For **family-history-only prediction** — own status out of $D_F$,
so the proband's residual is independent of the family — the model's
implied risk can be approximated as follows. Here `prediction` is the
relatives-only result created in the complete script, not the GWAS result above:

```python
import numpy as np
from scipy.stats import norm

T = norm.isf(K)                      # or the person's own threshold T_i
# prediction is a result computed with the proband bounds uninformative.
risk = norm.sf((T - prediction.genetic) / np.sqrt(prediction.var["genetic"] + 1 - h2))
```

The reasoning is that $\ell_i = a_i + e_i$ with $\mathrm{Var}(e_i)=1-h^2$,
and with own status out of $D_F$ the residual $e_i$ is independent of the
family, so $\ell_i \mid D_F$ is centred on $\mu_i$ with variance
$\mathrm{Var}(a_i \mid D_F) + 1 - h^2$. Two conditions come with that: it
assumes no shared-environment components ($c^2=m^2=0$, or $e_i$ is coupled
to the relatives), and it treats $a_i \mid D_F$ as Gaussian, which is the
same two-moment approximation PA makes.

At $h^2=0.5$, $K=0.05$, under the generating model, the simulation is
consistent with calibration overall and in the top decile. Over
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
bias. The formula is **not** valid for a GWAS phenotype, where the proband's own
status is already in $D_F$.

## What the run produced

```python
from ltpred import estimate_liability
import numpy as np

pa = estimate_liability(sim.families, h2=h2)
status, true_g = sim.status["o"].astype(float), sim.genetic

np.corrcoef(status, true_g)[0, 1]       # 0.353
np.corrcoef(pa.genetic, true_g)[0, 1]   # 0.426
```

Table 1 compares the score with the simulated genetic value.

**Table 1.** Correlation with the simulated true genetic value $a_i$, on
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
$1.05\times$ on top — compare the replicated campaign in
[RESULTS §10](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md). Credit the gain to the right input.

## Calendar and pedigree leakage checks

In the runnable six-person example in
[`examples/validation.py`](https://github.com/bvilhjal/ltpred/blob/main/examples/validation.py),
`max_degree=1` deliberately selects two parents and a sibling; two maternal
grandparents are retained only for exact kinship. Prediction has **three
relatives, two closure-only ancestors and three conditioned records**;
GWAS has four conditioned records. Changing only post-index proband/mother
records and closure-only diagnoses leaves the prediction mean and variance
exactly unchanged. This is an API/leakage check using a toy CIP, not clinical
calibration or new performance evidence.

For the input contract, see [data preparation](data-preparation.md#beyond-the-role-grammar-arbitrary-pedigrees).
For measured accuracy, runtime and memory, use the
[benchmark ledger](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
The source archives, manifests and retained arrays beneath `benchmarks/results/`
are evidence for those measurements; they are not alternative user guides.
