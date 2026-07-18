# Algorithm and model

ltpred implements classic LT-FH, personalised LT-FH++ with family history,
family-free ADuLT, and the PA-FGRS censoring variant. Gibbs and deterministic
Pearson–Aitken are inference engines, not additional model names. This page
describes the models and engines. See [guide.md](guide.md) for usage.

## The liability-threshold model

Each person has an unobserved normally-distributed **liability**. It splits into a
heritable genetic part and an independent environmental part:

```text
l_o = l_g + l_e ,   l_g ~ N(0, h2) ,   l_e ~ N(0, 1 - h2) ,   l_o ~ N(0, 1)
```

`l_g` is the **genetic liability** (variance `h2`, the liability-scale
heritability); `l_o` is the **full liability** (variance 1). A person is a case
when `l_o` exceeds a threshold `T`. With a single population prevalence `K`,
`T = Phi^-1(1 - K)`.

The goal is `E[l_g | data]` for a proband — a graded genetic score — where `data`
is the case/control status (and age) of the proband and their relatives.

## Family covariance

Relatives' additive genetic liabilities are correlated by the **additive genetic
relationship** `A_ij = 2 * phi_ij` (twice the kinship coefficient): `A = 1` for
self, `0.5` for parent/offspring and full sibs, `0.25` for grandparents /
half-sibs / aunts-uncles, etc. Writing `a` for additive genetic liability and `l`
for full liability, the model is the animal-model form

```text
Cov(a_i, a_j) = h2 * A_ij
Cov(l_i, l_j) = h2 * A_ij   (i != j)
Var(l_i)      = 1 ,   Var(a_i) = h2
```

For the target proband, `a_0` is written `g` (variance `h2`) and its own full
liability `l_0` is written `o` (variance 1); `Cov(g, o) = h2`.
`get_relatedness(a, b, h2)` returns `A_ab * h2`, and `construct_covmat(...)`
assembles this small fixed-pedigree relationship matrix, ordering `g`, `o` first
followed by the relatives (`correct_positive_definite` nudges a rounding-singular
matrix back to strict PD).

The role grammar is just a compact way to build `A` for common family shapes. For
**arbitrary pedigrees**, `kinship_from_pedigree(ids, father, mother)` builds `A`
directly from the pedigree by the recursive tabular method (Henderson 1976) —
`A_ii = 1 + F_i` (with `F_i` the inbreeding coefficient) and
`A_ij = 0.5 (A_i,sire_j + A_i,dam_j)` — and `construct_covmat_from_kinship(A, h2,
target)` first assembles the raw covariance `V = h2 A + (1-h2) I`, then divides
`V_ij` by `sqrt(V_ii V_jj)`. The target genetic contribution and its covariances
are put on the same standardised full-liability scale, so its variance is
`h2 A_tt / (1 + h2 (A_tt - 1))`. Thus every observed full
liability retains variance 1, and `Phi^-1(1-K)` retains its prevalence meaning,
even when `A_ii > 1`; for non-inbred pedigrees the scaling is a no-op. This
reproduces the role-grammar covariance entry-for-entry where they overlap and
additionally covers half-sibs of any degree, cousins and inbred pedigrees;
`estimate_liability_from_kinship` runs PA by default or Gibbs on request.

This is an **additive-genetic** model: familial resemblance is entirely genetic
sharing. Shared environment, household/cultural transmission, assortative mating
(parents are taken to be genetically unrelated, `A_mf = 0`), dominance/epistasis
and indirect genetic effects are not represented. Where those contribute, the
estimated "genetic liability" is best read as the additive-model projection of the
family history rather than a pure causal genetic value.

### Adding environmental covariance to improve prediction

The covariance is **modular**, and adding non-genetic components to the
between-relative covariance can improve prediction. Following the classic
variance-components (ACE-type) decomposition, extend the full-liability covariance
with valid shared-environment kernels:

```text
Cov(l_i, l_j) = h2 A_ij + c2 C_ij + m2 M_ij + sum_q u2_q K_q[i,j]
Var(l_i)      = h2 + c2 + m2 + sum_q u2_q + e2 = 1
```

Here `C` is the shipped sibship kernel, `M` is reserved for the shipped
mate/couple kernel, and each optional `K_q` is another symmetric
positive-semidefinite (PSD) sharing kernel. A directional maternal effect is not
`M` and is not generally representable by one symmetric covariance kernel. Each
kernel has unit diagonal in the fitted model, so its variance fraction reduces the
individual residual `e2`; the kernels are not off-diagonal adjustments alone. Two payoffs:

- **A sharper genetic estimate.** Modelling shared-environment resemblance lets
  the estimator attribute it to environment rather than genetics, so the genetic
  liability `g` is not *inflated* by families that cluster for environmental
  reasons — better calibration of the score used for GWAS.
- **Better full-liability / risk prediction.** The extra covariance captures real
  familial resemblance the additive model misses, tightening `E[l_o | family]`.

The genetic target `g` still couples to relatives only through `h2 * A`, so it
remains a *genetic* liability; the environmental terms only change how the
relatives' liabilities are conditioned.

This is a **low-level covariance interface, not yet a high-level
`estimate_liability(..., c2=...)` option**: `construct_covmat` builds only the
additive-genetic `h2 * A` table, and there is no user-facing shared-environment
argument. To use environmental components today, assemble the covariance yourself
(add `c2 * C`, `m2 * M`, etc.) and pass it to a covariance-level entry point —
`rtmvnorm_gibbs`, `pa_algorithm`, or `pa_estimate_batched` — which accept an
arbitrary covariance directly. (Note this is separate from `fit_variance_components`,
which *estimates* an `A + C + M` decomposition but does not yet feed fitted `C` or
`M` back into the liability estimator.)

### Relationship-specific environments and identifiability

The environmental term can be *several* components, one per relationship-specific
sharing pattern — a full-sib rearing environment, a couple/household environment
shared by mates, mother– or father–offspring environments, a cousin environment:

```text
Cov(l_i, l_j) = h2 A_ij + sum_c c2_c K_c[i,j] ,   K_c = K_c' >= 0,
Var(l_i) = h2 + sum_c c2_c + e2 = 1               (K_c[i,i] = 1).
```

`fit_variance_components` estimates a set of components **jointly** (multiple HE
regression, or ML with `method="mcem"`); a joint fit partials out the overlap
between components, whereas fitting each alone double-counts. The shipped bank is
`A` (additive genetic), `C` (full-sib / sibship environment, identified from the
full-sib excess) and `M` (couple / spousal environment, identified from the `A = 0`
mate pairs — the parents and the grandparent couples); `M` captures spousal
resemblance from shared environment *or* assortative mating (Robinson et al. 2017),
which parent data alone cannot separate — it is a descriptive spousal-resemblance
component, not a generative model of assortative mating. The shipped `C` and `M`
components are **equivalence-class partitions** (groups fully sharing one
deviation), which is a convenient sufficient construction for a PSD `K_c`, not a
necessary one. Any component must be symmetric and PSD; the fitter rejects a
kernel that is not (e.g. the naive vertical parent-offspring indicator — see
caution (i)). Adding a valid component is adding a column to the design; e.g.
`fit_variance_components(fams, ("A", "C", "M"))` fits all three at once given a
3-generation pedigree.

**The binding constraint is identifiability, not the estimator.** Each relative
*type* yields a single covariance `h2 A_ij + sum_c c2_c K_c[i,j]`, so the data
constrain the components only through the distinct relationship *contrasts* present:

> the number of jointly-identifiable components = rank of the pair design matrix
> (columns `A, K_1, …`) ≤ the number of distinct relationship types in the pedigrees.

A component is identified only when its sharing pattern is linearly independent of
`A` and of the others: `C` (sib) separates from `h2` via the sib-vs-parent-offspring
contrast — both are `A = 0.5`, but only sibs share `C`, hence the need for full-sib
pairs; a couple term comes from the `A = 0` mate pair; separating maternal from
paternal environment needs the `o–m` and `o–f` covariances to actually differ.
Nuclear families give only ~3 contrasts (sib–sib, parent–offspring, spouse), so at
most `A` plus one or two environments; a *bank* of relationship-specific
environments needs the many relative types (grandparents by lineage, half-sibs,
avuncular, cousins, spouses) that **extended registry pedigrees** provide — the
pedigree-scale analogue of why extended-family designs out-identify the MZ/DZ twin
ACE model. `fit_variance_components` raises on a rank-deficient design; near-collinear
components (e.g. `C` vs dominance) also inflate the SEs even when formally identified.

Two cautions. **(i)** A *symmetric* mother–offspring shared-environment matrix is
not a **maternal effect** in the causal sense (the mother's phenotype/genotype
shaping the offspring's environment); that is a directional path which propagates
through the pedigree and couples to genetic transmission, needing a
structural/latent-variable parameterisation rather than one symmetric matrix.
**(ii)** An environment shared *in proportion to relatedness* is a multiple of `A`
and is absorbed into `h2`; only environments whose pattern **differs** from `A` are
estimable, and the couple term is further confounded with assortative mating.
Extending `_COMPONENT_OFFDIAG` past the shipped `A`/`C`/`M` bank with valid PSD
kernels is the natural next step; the rank-deficiency guard already generalises.

## Connection to selection index and BLUP

`ltpred` is a liability-threshold generalisation of the classical **selection
index / BLUP** problem from quantitative genetics and animal breeding (Hazel 1943;
Henderson 1975): predict an individual's additive genetic value from relatives'
phenotypes and a relationship matrix.

If the relatives' **continuous** liabilities `l_F` were observed, the optimal
(Gaussian) predictor of the proband's additive genetic liability would be the
selection-index / BLUP conditional mean,

```text
E[g | l_F] = Cov(g, l_F) Var(l_F)^-1 l_F ,
```

which weights each relative by its relationship to the proband and its information
content, discounting shared covariance among the relatives so they are not
double-counted. The one-relative case makes the weighting transparent: for a
single relative with relationship `r` and observed liability `l_r`,

```text
Cov(g, l_r) = r * h2 ,   Var(l_r) = 1   =>   E[g | l_r] = r * h2 * l_r ,
```

so a parent or full sib (`r = 0.5`) contributes `0.5 h2 l_r` and a grandparent or
half-sib (`r = 0.25`) contributes `0.25 h2 l_r`.

In disease data the liabilities are **not** observed — we only know each lies in
an interval `C_F` (a case above a threshold, a control below one, an age-of-onset
case pinned at an age-specific threshold, a censored relative right-truncated).
The target is therefore

```text
E[g | l_F in C_F] = Cov(g, l_F) Var(l_F)^-1 E[l_F | l_F in C_F] ,
```

by Gaussian conditioning: **first** infer the latent liabilities implied by
status/age/censoring (`E[l_F | intervals]`), **then** project them onto the
proband's additive genetic liability with the same BLUP weights. The pipeline:

```text
family statuses + ages
     |  thresholds / CIPs
latent liability intervals  C_F
     |  truncated MVN
E[l_F | l_F in C_F]
     |  BLUP / selection-index projection
E[g | family data]
```

So the method is **BLUP-like only after conditioning on latent liabilities**.
Classical BLUP is linear in observed continuous phenotypes; here the binary /
censored observations define truncation intervals, so the exact posterior mean is
**nonlinear** in the data. It reduces exactly to selection-index / BLUP when the
liabilities are observed continuously (or pinned to points). The Gibbs backend
estimates the truncated-normal expectation directly; Pearson–Aitken approximates
the same moment updates deterministically. Viewed this way, `ltpred` is a
fixed-variance **probit / threshold liability model with a pedigree random
effect**, used primarily for *prediction* (of `g`) — the liability estimators
condition on an assumed `h2`, CIP/prevalence model and relationship matrix. The
heritability itself can optionally be **fit** from the same family data by
data augmentation (see [Fitting the covariance](#fitting-the-covariance-heritability)
below); the CIP/prevalence model is always supplied.

## Connection to Sham's liability-threshold risk models

The multivariate liability-threshold model on which `ltpred` rests — relatives'
liabilities jointly multivariate normal with covariance set by the relationship
matrix, and affection status *truncating* those liabilities — is the framework
Pak Sham and the genetic-epidemiology tradition formalised (Sham, *Statistical
Methods in Genetic Epidemiology*, 1998). Evaluating the joint distribution of a
family's liabilities under an observed affection pattern is exactly the
truncated-multivariate-normal problem the two backends address: the Gibbs sampler
draws it, Pearson–Aitken approximates its moments.

The tightest link is **So, Kwan, Cherny & Sham (2011)**, a risk-prediction
framework that combines an individual's **family history** — through the
liability-threshold multivariate normal over relatives — with **known
susceptibility loci**, and already handled the ingredients `ltpred` centres on:
each relative's **current age and follow-up period**, and even **competing risks
of mortality**. That is the same pairing the LT-FH family exploits — (a) a
liability under the threshold model, optionally conditioned on family history,
and (b) personalised age/censoring information — and the same downstream idea of fusing the
family-history liability with molecular predictors (a polygenic score). On the
parameter side, So & Sham's liability-scale heritability work is the tradition
`convert_observed_to_liability_scale` (Lee et al. 2011) belongs to.

The difference is the **estimand and the scale of computation**. So & Sham (2011)
target an individual's **absolute disease risk** for screening; `ltpred` targets
the **posterior mean additive genetic liability of the proband** — a
breeding-value-style score and a GWAS phenotype (the [BLUP framing](#connection-to-selection-index-and-blup)
above). LT-FH++ and ADuLT share personalised age/onset/sex/cohort thresholds.
LT-FH++ conditions on relatives; ADuLT uses only the index person. PA-FGRS adds
its interval-case and censored-control-mixture encoding. Gibbs and
Pearson–Aitken are the inference engines applied to those inputs.

## Thresholds: status, age, onset — and personalisation by sex and birth cohort

Each observed person `i` contributes a truncation of their full liability `l_i` to an
interval whose edge is a **personalised threshold** shared by LT-FH++ and ADuLT
(Pedersen 2022/2023):

```text
T_i = Phi^-1( 1 - K(t ; s_i, b_i) )
```

where `K(t; s, b)` is the population **cumulative incidence proportion (CIP)** — the
fraction of people of sex `s` born in year `b` who are diagnosed by age `t`. Status,
age and demographics map person `i` to an interval:

- **control** at current age `c_i`:  `l_i ∈ (-inf, T_i(c_i)]` — the "lived-through-risk"
  bound: an older disease-free person has cleared a *lower* threshold, i.e. stronger
  evidence of low liability;
- **case** with onset age `a_i`:  liability **pinned** at `T_i(a_i)` (`lower == upper`;
  PA-FGRS instead uses the interval `[T_i(a_i), inf)`). Younger onset ⇒ lower CIP ⇒
  higher threshold ⇒ more extreme liability. The map is invertible
  (`convert_liability_to_aoo` ↔ `convert_age_to_thresh`), so `T_i(a_i)` *equals* the
  case's liability at onset.

Age, sex and birth cohort enter **only through `K(t; s, b)`** — i.e. only through the
interval edge `T_i`, never the covariance `Sigma`. What ltpred ships:

| helper | CIP model | strata | family context |
|---|---|---|---|
| `prevalence_thresholds` | one lifetime prevalence `K` → `T = Phi^-1(1-K)` | none | relatives: classic LT-FH |
| `age_thresholds` | logistic `K / (1 + exp((mid_point - age)·slope))` | age only, single `K` | pinned tutorial encoding; model depends on rows |
| `pa_thresholds` | same logistic demo CIP | age only, single `K` | PA-FGRS interval/mixture inputs; not the PA engine switch |
| `thresholds_from_cip` | an **empirical** CIP curve you supply, **called once per stratum** | sex × birth year × ancestry | relatives: full LT-FH++; proband only: ADuLT |

The same personalised CIP therefore feeds two models: relative observations make
it LT-FH++; their absence makes it ADuLT. A threshold helper cannot decide that
for you.

**"Single-`K`"** is context-dependent in the benchmark labels. Classic LT-FH uses
one lifetime threshold and ignores age. The cohort-blind ablation in
`bench_fh_prediction` instead uses one lifetime-`K` *anchor* for everyone while
still letting the logistic threshold vary with age; it is not classic LT-FH.
Full LT-FH++ uses stratified `K(t; s, b)` so each person receives their own
age-, birth-year- and sex-specific threshold.

**Why personalisation can affect calibration and power.** From the BLUP
decomposition above, `E[g | family] = Cov(g, l_F) Var(l_F)^-1 · E[l_F | intervals]`, the
thresholds `T_i` enter **only** the truncated means `E[l_F | intervals]` — never the BLUP
weights `Cov(g,l_F)Var(l_F)^-1`. Hence:

- changing one common threshold often behaves mainly like a score-scale or
  calibration change in the tested structures, but upper- and lower-truncated
  means respond nonlinearly and need not shift identically;
- stratum-specific threshold errors can also change score ordering because
  different observations receive different conditional means. Two cases with the
  same onset age but different population CIPs need not imply the same liability.

These are empirical tendencies, not an algebraic separation between calibration
and ranking. `bench_fh_prediction` shows both effects in its simulation: on the pedigree it is mostly a mean-score
shift (and a truth-referenced single-`K` error) because the high-weight proband spans
a narrow living cohort, while the replicated own-onset panel isolates the ranking
gain — about 1.66× in the squared-correlation effective-N proxy at the widest
tested cohort span. This is a prediction proxy, not a causal-SNP noncentrality
ratio. Both pedigree variants in that benchmark use PA; the change is in their
bounds, not their inference engine. Pinning (a point mass) is handled exactly by
both estimators.

## Inference engine 1: Gibbs sampler

`E[l_g | data]` is the mean of the family covariance's multivariate normal
truncated to the per-person intervals — a truncated MVN with no closed form for
more than a couple of members. `rtmvnorm_gibbs` samples it by sweeping one
coordinate at a time, drawing each from its **conditional** normal restricted to
its interval (inverse-CDF sampling; Kotecha & Djurić 1999):

```text
x_j  <-  mu_j + sd_j * Phi^-1( U( Phi((a_j - mu_j)/sd_j), Phi((b_j - mu_j)/sd_j) ) )
mu_j  =  P[:, j] . x         (conditional mean)
```

`P[:, j] = Sigma[-j,-j]^-1 Sigma[-j, j]` (conditional-regression coefficients)
and `sd_j = sqrt(Sigma[jj] - P[:,j].Sigma[:,j])` depend only on `Sigma`, so they
are precomputed once. Pinned coordinates (`a_j == b_j`) are held fixed. The
posterior means of `g` (and `o`) are the sample averages.

**Convergence.** The sampler is re-run, accumulating draws, until the
**batch-means** Monte-Carlo standard error of every requested estimate falls
below `tol` (`batch_means`, R's `batchmeans::bmmat`).

**Performance.** The inner sweep is Numba-JIT'd. Families with the same role
sequence share one covariance, so the estimator groups them and runs the group in
one `prange`-parallel kernel that accumulates the mean and the batch-means SE
**online** from streaming batch summaries (running sums and sums-of-squares) — no
full `(n_sim × n_out)` sample array, so the Monte-Carlo-SE memory is `O(families)`
regardless of `n_sim`, and each family seeds its own RNG so results are
deterministic regardless of thread scheduling. Without Numba the identical code
runs serially in pure Python.

## Inference engine 2: Pearson–Aitken

The **Pearson–Aitken selection formula** gives, in closed form, how a
jointly-Gaussian vector's mean and covariance change when one component's marginal
is *selected* (truncated). If component `i` moves from `N(m_i, v_i)` to a selected
mean/variance `(m*, v*)`, every component updates by a rank-1 correction:

```text
mean_j  +=  (Sigma_ji / v_i) * (m* - m_i)
cov_jk  +=  (Sigma_ji Sigma_ik / v_i^2) * (v* - v_i)
```

`pa_algorithm` places the target genetic liability first and folds the observed
members in one at a time (last to first). For each, `(m*, v*)` are the
truncated-normal moments on its interval (`tnorm_moments`: `_tnorm_mean` /
`_tnorm_var`, with `v* = 0` for a pinned point mass — exact conditioning). Reading
the target's updated mean gives `E[l_g | data]` and its variance the posterior
variance — **deterministically, with no Monte-Carlo error**.

After one truncation the selected distribution is no longer, in general,
multivariate normal. Pearson–Aitken keeps only the updated first two moments and
proceeds as if the remaining variables were Gaussian with those moments. Hence it
is **exact for a single truncation** (and for exact Gaussian conditioning on
point-pinned variables), but an approximation for multiple interval observations —
the standard sequential-selection approximation, which matches the Gibbs posterior
to corr ≥ 0.997 on realistic families while running 315–510× faster in the
controlled 10-thread benchmark. Same grouping / `prange` structure as the Gibbs
path.

### PA-FGRS censored-control mixture (optional)

`tnorm_mixture_conditional` extends the truncated moments for **age-censored
controls**: someone unaffected only up to their current follow-up is a mixture of
a true control and a not-yet-onset future case. With the individual cumulative
incidence `K_i` and lifetime prevalence `K_pop`, the selected moments become

```text
thr_pop   = Phi^-1(1 - K_pop)          # lifetime threshold
Phi_below = Phi((thr_pop - mu) / sd)
mix       = Phi_below / (Phi_below + (1 - Phi_below) * (K_pop - K_i) / K_pop)
mean*     = mix * mean(below thr_pop) + (1 - mix) * mean(above thr_pop)
```

(with the matching two-component variance), following PA-FGRS supp. eqs. S3–S5.
The split is the **lifetime** threshold, not the passed `upper`: age enters only
through the mixture weight via `K_i`, and `upper` merely flags a censored control
(finite) versus an observed case (`+inf`), so passing either the lifetime bound or
an age-specific `Phi^-1(1 - K_i)` gives the same result.
Enabled via `use_mixture=True`; off, PA reduces to the plain truncated-moment
sweep.

## Fitting the covariance (heritability)

Both estimators above *condition* on a known `h2`. `fit_heritability` instead
**fits** it — estimating the liability-scale heritability from the case/control
(and age-of-onset) statuses of relatives — with a Gibbs sampler modelled on
bipred's joint effect/parameter loop (sample the latents, then re-estimate the
covariance parameters each sweep). It treats the latent liabilities as missing
data and alternates:

1. **Augment** — one persistent truncated-MVN sweep per family under the current
   covariance `Sigma(h2) = (1-h2) I + h2 A` (`A` the additive relationship matrix
   over the observed relatives), holding pinned cases (`gibbs_advance`).
2. **Update** — a damped moment step for `h2`: a Haseman–Elston regression of the
   sampled liability cross-products on relatedness, pooled over all related pairs
   in all families,

   ```text
   h2_hat = sum_pairs A_ij * l_i l_j / sum_pairs A_ij^2 ,
   h2     <- (1 - damp) * h2 + damp * h2_hat .
   ```

Because the liabilities are drawn conditional on each family's observed intervals,
pooling their cross-products reconstructs the model covariance, so the chain
settles at the `h2` consistent with the observed familial resemblance — a
threshold-model variance-component estimate from pedigree affection data (in the
Sorensen–Gianola / Bayesian animal-model tradition; the moment-with-damping update
is the bipred-style analogue of a full conjugate step). In the repository simulation
benchmark, bias across `h2 = 0.3–0.8` was small relative to the across-dataset SD,
but larger than the within-fit Monte-Carlo error in some settings. The reported `h2_se` is the
*within-dataset* Monte-Carlo error; sampling variability across datasets is larger
and depends on the number and informativeness of the families (bootstrap over
families for that). Identifiability comes entirely from the *between-relative*
covariance, so relatives are required (lone probands carry no information).

**Seeding.** Unlike the estimator's kernel, where each family seeds its own RNG,
the augmentation kernel is a single `prange` over families, and Numba's random
state belongs to whichever worker thread picks a family up — so seeding the calling
thread could not by itself pin the result. `gibbs_advance` instead draws its
uniforms from a seeded NumPy generator *before* entering the kernel and passes them
in, so a seeded fit reproduces exactly whatever the thread count and scheduling.
The generator is thread-local (concurrent seeded fits do not disturb each other)
and the uniforms are chunked, capping the temporary at a few MiB no matter how many
families or sweeps. `seed` must be an integer in `[0, 2^32 - 1]`; anything else
raises rather than being silently coerced.

`fit_variance_components` extends the same augment-then-regress machinery to
several components at once — a **multiple** Haseman–Elston regression. Each sweep
draws the liabilities from the full family truncated-MVN under
`Sigma = sum_c h2_c K_c + e2 I` and updates all proportions jointly,

```text
[h2_c] = (X'X)^-1 X'y ,   X[pair, c] = K_c[i,j] ,   y[pair] = l_i l_j ,
```

with the same cross-sweep damping. Fitting additive `A` together with the
environment components works because they load on *different* relationship
contrasts — `A` is pinned by the parent-offspring / grandparent / avuncular
relatednesses, `C` by the full-sib excess, `M` by the resemblance between the
genetically-unrelated mates — so each needs its identifying pairs (`C` full-sib
pairs, `M` mate pairs); the design is otherwise rank-deficient and the fit raises.
Simulation benchmarks recover `A`, `A+C` and `A+M` with small bias relative to
their across-dataset SD. Note the two
shared-environment components differ in how they bias `A` if omitted: ignoring a
real `C` inflates `A` (sibs share both, so sib resemblance is over-credited to
genes), whereas ignoring `M` biases `A` **much less** — mates have `A = 0`, so
they carry little direct weight in the additive regression (a residual can remain
from imputing under the misspecified model). In `bench_couple_env` (3000
families, true `a² = 0.4`) the same shared-environment variance biases the
additive-only `Â` by +0.04, +0.11, +0.16 as `s²` runs 0.1 → 0.2 → 0.3 when it is
`C`, but only +0.00, +0.01, +0.04 when it is `M`. So `M` is worth fitting for its
own sake (quantifying/testing spousal resemblance) rather than to de-bias `h²`. A
**dominance** component is deliberately not offered: in the current role-based
design its sharing pattern is not separated reliably from additive and sibship
components. Identification would require an explicit dominance relationship
kernel and relationship contrasts linearly independent of `A` and `C`. MZ/DZ
twin observations can contribute such contrasts within a richer design, but MZ/DZ
pairs alone cannot identify `A`, `C`, and `D` simultaneously.

With `method="mcem"`, the M-step instead optimises the Gaussian likelihood of the
imputed liabilities. This implementation is an **approximate finite-iteration
Monte-Carlo EM-style procedure**: it uses a fixed damping coefficient, averages
post-burn-in iterates, and does not stop on an observed-likelihood convergence
criterion. Its OPG/BHHH information SE and GHK log-likelihood/AIC therefore retain
both Monte-Carlo and finite-iteration error. Check stability across seeds and
iteration settings and use family resampling for sampling uncertainty when the
independent-cluster assumptions hold.

## Multiple traits

For `n` genetically/environmentally correlated traits the covariance is
phenotype-major: same-trait blocks use `A_ij * h2_p`; cross-trait blocks scale the
relationship `A_ij` by the genetic covariance `rho_g[p,q] * sqrt(h2_p h2_q)`, and
the same individual's full liabilities across traits correlate by
`full_corrmat[p,q]`. The Gibbs sampler then returns the genetic/full liability of
each trait (`estimate_liability_multi`). This lets a well-powered trait sharpen
the estimate for a correlated, under-powered one.

These inputs are jointly constrained. Let `D = diag(sqrt(h2))`,
`G = D genetic_corrmat D`, and `E = full_corrmat - G`. Both `G` and `E` must be
positive semi-definite, while the two supplied correlation matrices must be
symmetric with unit diagonal. The full pedigree covariance is
`G ⊗ A + E ⊗ I`. The constructor rejects an incoherent decomposition rather than
silently applying a positive-definite correction to a different model.

### Fitting the genetic correlation

`fit_genetic_correlation` estimates `rho_g` between traits from family data — the
multi-trait analogue of `fit_heritability`, a **cross-trait** Haseman–Elston
regression. Each member carries one case/control interval per trait. Each sweep
draws the members' `P`-trait liabilities from the full truncated-MVN under the
current parameters, then regresses the sampled cross-products on the additive
relationship `A`:

```text
same trait, diff relatives:  h2_p    = sum A_ij l_ip l_jp / sum A_ij^2
diff trait, diff relatives:  G[p,q]  = sum A_ij (l_ip l_jq + l_iq l_jp) / (2 sum A_ij^2)
same individual, diff trait: rp[p,q] = mean_i l_ip l_iq              (phenotypic corr)
```

so the genetic correlation is `rg[p,q] = G[p,q] / sqrt(h2_p h2_q)`. The
cross-relative, cross-trait resemblance carries the genetic covariance because only
the genetic part transmits by relatedness, so `E[l_ip l_jq] = A_ij G[p,q]` for
`i != j` — the within-individual environmental covariance drops out. The phenotypic
correlation then splits into genetic and environmental covariances, `rp = G + E`,
so the **environmental correlation** `re[p,q] = (rp[p,q] - G[p,q]) / sqrt(e2_p e2_q)`
(`e2 = 1 - h2`) is returned alongside `rg`.

The moment step is unconstrained, so its raw `G` and `E = rp - G` need not be
positive semi-definite — especially with few families or a large `|rg|`. Each sweep
therefore projects both onto the convex set of correlation matrices (the PSD cone
intersected with the unit-diagonal constraint) with the fitted variances held
fixed (an eigenvalue projection, then a shrink towards the identity
that leaves the diagonal alone), and the chain carries the two projected
*covariance* states rather than ratios. The reported estimates are the post-burn-in
averages of those states — averaging covariances is convex, so `G_est` and `E_est`
are PSD too — with `rg`, `re` and `rp` all derived from that same pair. So the
returned object is one coherent model: `rp == genetic_cov + env_cov` exactly, and
every correlation it reports comes from a single PSD fit rather than from
separately averaged ratios. In the repository benchmarks it was approximately
unbiased near the null (no
spurious `rg` when traits are genetically independent but
phenotypically correlated), with mild attenuation at large `|rg|` (the bounded
ratio estimator); use an iid-family cluster bootstrap for sampling uncertainty when
its assumptions hold. This is the pedigree-scale analogue
of bivariate GREML / cross-trait LD-score regression.

### Genetic factor structure (common-factor model)

With more than a handful of traits, the genetic correlation matrix `r_g` is itself a
structured object worth summarising. `fit_genetic_factor` fits a **common-factor
model** to it,

```text
r_g ≈ Λ Λ' + Ψ ,     Ψ = diag(uniquenesses),
```

where `Λ` is a `P × m` matrix of factor loadings and `Ψ` the trait-specific genetic
residuals — the pedigree-scale analogue of the **Genomic SEM** common-factor model
(Grotzinger et al. 2019), which fits the same structure to an LD-score-regression
genetic covariance. For `m = 1` it answers a concrete question: does a *single*
latent genetic factor — one general axis of shared genetic liability — reproduce all
the pairwise `r_g`, or do the traits split into several genetic dimensions?

The fit is **MINRES** (minimum-residual) factor analysis: choose `Λ` to minimise the
sum of squared **off-diagonal** residuals of `r_g − Λ Λ'`,

```text
minimise  sum_{p≠q} w_pq (r_g[p,q] − (Λ Λ')[p,q])^2 ,   then  Ψ_p = 1 − (Λ Λ')_pp .
```

The diagonal is *excluded* from the objective and absorbed afterwards by the
uniquenesses, so the factors are pinned by the **cross-trait correlations** — the
shared signal — not by each trait's own heritable variance. (This is the factor-
analytic counterpart of what the `C`/`M` environment components do in the
variance-component fit: model the off-diagonal resemblance, leave the diagonal to a
residual.) Loadings come back on the correlation scale — a covariance input is
standardised first — so `communality_p = sum_k Λ_pk^2` is the fraction of trait `p`'s
*genetic* variance explained by the common factor(s). The optimisation is warm-started
from the top-`m` eigenvectors of `r_g` (principal factors) and polished by L-BFGS-B
with the analytic gradient `−2 (W∘R_res) Λ`; an optional weight matrix `W` (e.g.
`1/se²` of each `r_g`) gives a diagonally-weighted (DWLS) fit.

Fit is read off the **off-diagonal residuals**: `srmr` is their standardised
root-mean-square, and `prop_explained` is the fraction of off-diagonal structure
captured. Values around 0.05–0.08 are informal descriptive heuristics, not a
calibrated factor-number test. The optimizer constrains each communality
`sum_k Λ_pk²` to `[0, 1]`; a value at 1 is a Heywood boundary. A single factor
requires `P ≥ 3` traits, and in
general the usual parameter count requires
`df = ½((P − m)² − (P + m)) ≥ 0`. At `P = 3, m = 1`, `df = 0`, but sign and
communality constraints can still prevent an exact representation. As with the
`r_g` estimate itself the loadings carry no inference of their own — bootstrap the whole
`fit_genetic_correlation → fit_genetic_factor` pipeline over families for uncertainty,
since the within-dataset `se` understates it. `benchmarks/bench_genetic_factor.py`
recovers planted loadings end-to-end and shows `srmr` rising when a one-factor model
is fit to two-factor data.

## Background and references

The method sits in a long quantitative-genetics lineage. **Threshold models** for
binary/categorical traits — mapping a continuous latent liability through a
threshold — go back to Wright, Dempster & Lerner (1950), Falconer (1965) and
Gianola (1982). Their continuous-trait analogue is **selection-index / BLUP**
prediction of additive genetic value from relatives' phenotypes and a relationship
matrix (Hazel 1943; Henderson 1975; the animal model, and its genomic-relationship
extensions, VanRaden 2008). **LT-FH** (Hujoel 2020) adapts the threshold model to
case-control GWAS by using the posterior mean genetic liability as the phenotype;
**LT-FH++** (Pedersen 2022) adds personalised age-, birth-year- and sex-dependent
prevalence to family-history liability; **ADuLT** (Pedersen 2023) uses the same
personalised construction without family history as an alternative to time-to-event
GWAS; and **PA-FGRS** (Krebs 2024) gives a deterministic
Pearson–Aitken approximation for large, age-censored genealogies.

Core methods:

- Hujoel et al. 2020, *Nat Genet* — LT-FH.
- Pedersen et al. 2022, *AJHG* — LT-FH++ (flexible pedigrees, age, sex).
- Pedersen et al. 2023, *Nat Commun* — ADuLT (family-free personalised threshold).
- Krebs et al. 2024, *AJHG* — PA-FGRS (Pearson–Aitken family genetic risk scores).

Threshold-model background:

- Dempster & Lerner 1950, *Genetics* — heritability of threshold characters.
- Falconer 1965, *Ann. Hum. Genet.* — liability to disease from incidence in relatives.
- Gianola 1982, *J. Anim. Sci.* — threshold characters in animal breeding.

Selection index / BLUP background:

- Hazel 1943, *Genetics* — the genetic basis for constructing selection indexes.
- Henderson 1975, *Biometrics* — best linear unbiased estimation/prediction (BLUP).
- VanRaden 2008, *J. Dairy Sci.* — genomic relationship matrices for prediction.

Liability-threshold risk models (Sham and colleagues):

- Sham 1998, *Statistical Methods in Genetic Epidemiology* (Oxford) — the
  multivariate liability-threshold model in genetic epidemiology.
- So, Kwan, Cherny & Sham 2011, *AJHG* — risk prediction from family history and
  known susceptibility loci, with age, follow-up and competing mortality risks.

Numerics:

- Pearson 1903 / Aitken 1935 — the selection formula for conditioning a Gaussian.
- Tallis 1961, *JRSS B* — moments of the truncated multivariate normal.
- Kotecha & Djurić 1999 — Gibbs sampling for truncated multivariate normals.
- Genz & Bretz 2009, *Springer* — computation of multivariate normal probabilities
  (the GHK simulator behind the MCEM log-likelihood).
- Lee et al. 2011, *AJHG* — observed-to-liability-scale heritability.

Multi-trait / genetic factor structure:

- Grotzinger et al. 2019, *Nat. Hum. Behav.* — Genomic SEM: structural equation
  models (including the common-factor model) fit to a multi-trait genetic
  covariance — the model `fit_genetic_factor` ports to the pedigree scale.
