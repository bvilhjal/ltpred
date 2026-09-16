# Algorithm and model

This page records the estimand, the observation models, and the two
algorithms that compute the score. Which steps to run, and which of
three uses they serve, is in the [vignette](vignette.md) and
[guide.md](guide.md). The typeset companion is the
[methods note](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf).

Write the names LT-FH, LT-FH++, ADuLT and PA-FGRS for *observation
models*: they choose the set `D_F` that is conditioned on. They are
not different genetic models, and they are not inference engines.
Gibbs and Pearson–Aitken (PA) are the engines. Both apply to LT-FH,
LT-FH++ and ADuLT. The PA-FGRS censoring mixture is PA-only.

The argument is in this order: the liability-threshold model and the
estimand; the family covariance; the BLUP identity; then the two
engines, written as algorithms. Sections after the engines
(environment kernels, thresholds, fitting) may be skipped on a first
reading.

## The liability-threshold model

Let each person carry an unobserved liability. Write `l_g` for its
additive genetic part and `l_e` for an independent environmental
part. On the usual unit-liability scale,

```text
l_o  =  l_g + l_e ,
l_g  ~  N(0, h²) ,     l_e  ~  N(0, 1 − h²) ,     l_o  ~  N(0, 1).     (1)
```

Here `h²` is the liability-scale heritability. A person is a case
when `l_o` exceeds a threshold `T`. With a single population
prevalence `K`,

```text
T  =  Φ⁻¹(1 − K).                                                       (2)
```

The later sections write `a` or `g` for a designated proband's
additive genetic liability (so `Var(a) = h²`) and
`ℓ` or `o` for a full liability (so `Var(ℓ) = 1`).
Those are the same two coordinates as in (1).

## The estimand

Let `i` be a designated proband, `A` the additive relationship
matrix of the people whose records are used, and `K( · )` the
prevalence or cumulative-incidence model that turns status and age
into interval endpoints or mixture weights. Write `ℓ_F` for the
vector of full liabilities on those people, and `D_F` for the
complete observation model encoded by the records. The target is

```text
μ_i  =  E[ a_i  |  D_F, A, h², K(·) ].                                  (3)
```

For LT-FH, LT-FH++ and ADuLT, `D_F` is the rectangle
`ℓ_F ∈ C_F`, one interval per person. The PA-FGRS censoring
mixture is a mixture of truncated laws and is not, in general, one
rectangle.

If `ℓ_F` were observed continuously, (3) would be the selection
index / animal-model BLUP

```text
E[a_i | ℓ_F]  =  Cov(a_i, ℓ_F)  Var(ℓ_F)⁻¹  ℓ_F.                       (4)
```

A single relative with relationship `r` contributes `r h² ℓ_r`.
Disease records do not give `ℓ_F`. They give `ℓ_F ∈ C_F`.
Under joint normality,

```text
E[a_i | ℓ_F ∈ C_F]  =  Cov(a_i, ℓ_F)  Var(ℓ_F)⁻¹  E[ℓ_F | ℓ_F ∈ C_F]. (5)
```

The BLUP weights are unchanged; only the right-hand side is replaced
by a truncated-normal mean. Hence the map from the binary (or
censored) records to `μ_i` is nonlinear, and reduces to (4) only
when every interval collapses to a point. Gibbs estimates the
truncated-normal mean in (5) by sampling. PA approximates the same
moment updates deterministically. Both return an estimate of (3).

Two remarks, both easy to get wrong in an analysis.

1. The proband's own diagnosis is optional. Include it when `μ_i`
   is a GWAS phenotype constructed from that diagnosis. Leave the
   corresponding interval as `(-∞,∞)` when the same
   diagnosis is the outcome you will later predict or classify.
   Conditioning on the answer is leakage.
2. Equation (3) is not a SNP polygenic score. No marker effects
   enter. Combining `μ_i` with a PGS is a downstream model.

Those remarks split two uses of `μ_i`. A third use never needs
`μ_i`. The [vignette](vignette.md) writes them as Table 1; the
methods note as its table of three uses.

- **I. Risk prediction** from family history (own status *out*). An
  optional PGS is combined afterwards, not inside (3).
- **II. A quantitative GWAS phenotype** (own status *in*). ADuLT is
  the no-relative special case.
- **III. Architecture, relationships, aetiology.** Liability-scale
  `h²` and `r_g`, and the CIP `K(·)`, can stand alone. Pedigree
  scoring is optional. Fitting `h²` from the same families is a
  different contract ([Inference](inference.md)).

The [BLUP section](#connection-to-selection-index-and-blup) records
the same identity in the language of the selection index. The
[Sham section](#connection-to-shams-liability-threshold-risk-models)
places the truncated-MVN problem in the genetic-epidemiology
tradition.

## Family covariance

Relatives' additive genetic liabilities are correlated by the additive
genetic relationship `A_ij = 2φ_ij` (twice the kinship
coefficient): `A_ii = 1` for a non-inbred person, `A_ij = 1/2`
for parent–offspring and full sibs, `A_ij = 1/4` for grandparents,
half-sibs and avuncular pairs, and so on. The animal-model covariances
are

```text
Cov(a_i, a_j)  =  h² A_ij ,
Cov(ℓ_i, ℓ_j)  =  h² A_ij     (i ≠ j) ,
Var(ℓ_i)       =  1 ,     Var(a_i)  =  h².                              (6)
```

For the designated proband, write `g` for `a_0` and `o` for
`ℓ_0`. Then `Var(g) = h²`, `Var(o) = 1`,
and `Cov(g, o) = h²`.
`get_relatedness(a, b, h2)` returns `A_ab * h2`, and `construct_covmat_single(...)`
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
That high-level kinship API accepts `lower`/`upper`, caller-supplied
`c2`/`c_kernel` and `m2`/`m_kernel` components, and, with Pearson–Aitken,
`use_mixture=True` plus per-member `K_i`/`K_pop` for the PA-FGRS
censored-control mixture. With kernels it assembles
`V = h2 A + c2 C + m2 M + (1-h2-c2-m2) I` before the same standardisation.
`C` and `M` are not inferred from `A`: identical additive relatedness can
describe different environmental relationships. Gibbs still has no mixture
implementation.
For discovering the relatives from population trio records, `ltpred.pedigree`
implements graph-based relative extraction in the style of [Pedersen et al.
(2025)](https://doi.org/10.3389/fgene.2025.1708315): `build_parent_graph`
indexes the records, `extract_pedigree` visits every person within
`max_degree` relationship-degrees of a proband (full-sibling edges make the
distance equal the standard relationship degree) and closes on all recorded
ancestors, so the extracted pedigree's kinship is exact for every member pair.
`Pedigree.closure_only` marks ancestors outside the requested observation
degree. The public register driver keeps their diagnosis bounds uninformative
by default while retaining them in `A`.
Its kinship is the exact tabular one (inbreeding-aware), not the paper's
path-counting approximation. The
[LTFGRS](https://emilmip.github.io/LTFGRS/) R package (Pedersen et al.) —
LT-FH++, PA-FGRS and Kendler's FGRS with unified data preparation — consumes
the same extraction downstream.

The default is an **additive-genetic** model: familial resemblance is entirely
genetic sharing. Shared environment, household/cultural transmission, assortative mating
(parents are taken to be genetically unrelated, `A_mf = 0`), dominance/epistasis
and indirect genetic effects are not represented. Where those contribute, the
estimated "genetic liability" is best read as the additive-model projection of the
family history rather than a pure causal genetic value.

### Adding environmental covariance to improve prediction

The covariance is modular. Adding a valid shared-environment kernel
changes the conditional law of `ℓ_F` and therefore the right-hand
side of (5). Following the classic variance-components (ACE-type)
decomposition, extend the full-liability covariance with valid
shared-environment kernels:

```text
Cov(l_i, l_j) = h2 A_ij + c2 C_ij + m2 M_ij + sum_q u2_q K_q[i,j]
Var(l_i)      = h2 + c2 + m2 + sum_q u2_q + e2 = 1
```

Here `C` is the shipped sibship kernel, `M` is reserved for the shipped
mate/couple kernel, and each optional `K_q` is another symmetric
positive-semidefinite (PSD) sharing kernel. A directional maternal effect is not
`M` and is not generally representable by one symmetric covariance kernel. Each
kernel has unit diagonal in the fitted model, so its variance fraction reduces the
individual residual `e2`; the kernels are not off-diagonal adjustments alone.
Two consequences follow.

- **A sharper genetic estimate.** Modelling shared-environment resemblance lets
  the estimator attribute it to environment rather than genetics, so the genetic
  liability `g` is not *inflated* by families that cluster for environmental
  reasons — better calibration of the score used for GWAS.
- **Better full-liability / risk prediction.** The extra covariance captures real
  familial resemblance the additive model misses, tightening `E[l_o | family]`.

The genetic target `g` still couples to relatives only through `h2 * A`, so it
remains a *genetic* liability; the environmental terms only change how the
relatives' liabilities are conditioned.

The sibship (`C`) and couple (`M`) components are **wired into the
estimator**: `construct_covmat_single(..., c2=..., m2=...)` adds them to the
relatives' covariance (off-diagonals only; the residual environmental variance
absorbs them, so full liabilities keep unit variance and the genetic target
stays coupled through `h2 * A` only). The single-trait role/object and array
front doors accept them: `estimate_liability` with scalar `h2`,
`estimate_liability_pa_arrays` and `estimate_liability_gibbs_arrays` all take
`c2`/`m2` arguments, so the `A + C + M` decomposition fitted by
`fit_variance_components` can be fed straight back into single-trait liability
estimation. The high-level multi-trait route rejects nonzero `c2`/`m2` until an
explicit cross-trait component covariance is defined. The arbitrary-kinship
route accepts the same proportions only with aligned, caller-supplied `C` and
`M` kernels; it never guesses relationship classes from `A`.
Validated in `benchmarks/bench_shared_env.py` panel (c): wiring recalibrates the
genetic estimate (slope 0.93 -> 0.99) and sharpens the full-liability
prediction. Other user-defined kernels still go through the covariance-level
entry points (`rtmvnorm_gibbs`, `pa_algorithm`, `pa_estimate_batched`), which
accept an arbitrary covariance directly.

`fit_variance_components` estimates a set of components **jointly** (multiple HE
regression; a Monte-Carlo EM likelihood route, `fit_variance_components_mcem`,
lives in `research/advanced_fitting.py`); a joint fit partials out the overlap
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
`fit_variance_components(fams, ("A", "C", "M"), sampling="population")` fits
all three at once given a 3-generation pedigree.

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

### Research-only covariance models

Directional genetic-nurture and sex-limited covariance constructors are not
part of the installed package; they live in `research/covariance_extensions.py`
and are documented on the [research extensions](research.md) page.

## Connection to selection index and BLUP

Equations (4) and (5) are the classical selection-index / BLUP problem
(Hazel 1943; Henderson 1975), with one change: the phenotypes on the
right-hand side are not observed continuously. They are truncated
liabilities. We record the same identity here in the language of the
selection index, because that is the shortest way to see what the
two engines are computing.

If the relatives' continuous liabilities `ℓ_F` were observed, the
optimal Gaussian predictor of the proband's additive genetic liability
would be (4). Each relative is weighted by its relationship to the
proband and by its residual information after the other relatives have
been accounted for, so shared covariance is not double-counted. For a
single relative with relationship `r` and observed liability
`ℓ_r`,

```text
Cov(g, ℓ_r)  =  r h² ,     Var(ℓ_r)  =  1
          ⇒     E[g | ℓ_r]  =  r h² ℓ_r.                                (7)
```

A parent or full sib (`r = 1/2`) contributes `½ h² ℓ_r`;
a grandparent or half-sib (`r = 1/4`) contributes `¼ h² ℓ_r`.

In disease data we know only that `ℓ_F` lies in a rectangle
`C_F` (a case above a threshold, a control below one, an onset case
pinned at an age-specific threshold, a censored relative
right-truncated). Gaussian conditioning then gives (5): first form
the truncated-normal mean `E[ℓ_F | ℓ_F ∈ C_F]`,
then apply the same BLUP weights. The pipeline is

```text
family statuses and ages
        |   K(·)  →  thresholds / mixture weights
rectangle (or mixture)  D_F
        |   truncated MVN, Algorithm G or P
E[ℓ_F | D_F]
        |   BLUP / selection-index projection  (4)
μ_i  =  E[a_i | D_F].
```

The rectangle `C_F` may include the proband's own interval when the
score is a GWAS phenotype constructed from that diagnosis (use II).
It must exclude, or leave unbounded, the proband's interval when the
same diagnosis is the outcome of a prospective evaluation (use I);
otherwise `D_F` contains the answer (remark 1 above). Use III can
stop before this pipeline.

Hence the method is BLUP-like only after the latent liabilities have
been replaced by their truncated means. Classical BLUP is linear in
observed continuous phenotypes. Here the exact posterior mean is
nonlinear in the binary or censored records, and reduces to (4) when
every interval is a point. The estimators condition on an assumed
`h²`, a CIP or prevalence model, and a relationship matrix. The
heritability itself can optionally be fit from the same family data
by data augmentation, but only for independent, non-overlapping
families under the declared population or known-probability IPW
sampling contracts (see
[Fitting the covariance](#fitting-the-covariance-heritability)
below). The CIP model is always supplied.

## Connection to Sham's liability-threshold risk models

The multivariate liability-threshold model on which `ltpred` rests — relatives'
liabilities jointly multivariate normal with covariance set by the relationship
matrix, and affection status *truncating* those liabilities — is the framework
Pak Sham and the genetic-epidemiology tradition formalised (Sham, *Statistics
in Human Genetics*, 1998). Evaluating the joint distribution of a
family's liabilities under an observed affection pattern is exactly the
truncated-multivariate-normal problem Algorithms G and P address: G
draws it, P approximates its moments.

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
`observed_to_liability_h2` (Lee et al. 2011) belongs to.

The difference is the **estimand and the scale of computation**. So & Sham (2011)
target an individual's **absolute disease risk** for screening; `ltpred` targets
the **posterior mean additive genetic liability of the proband** — a
breeding-value-style score that is either a relatives-only predictor
(use I) or a GWAS phenotype (use II; the [BLUP framing](#connection-to-selection-index-and-blup)
above). Use III may never form that score.
LT-FH++ and ADuLT share personalised age/onset/sex/cohort thresholds.
LT-FH++ conditions on relatives; ADuLT uses only the index person. Base PA-FGRS
uses lifetime-threshold intervals for observed cases and an age-censored-control
mixture with Pearson–Aitken. Gibbs and Pearson–Aitken can both be applied to the
LT-FH/LT-FH++/ADuLT bounds; ltpred's PA-FGRS mixture has no Gibbs implementation.

## Expected correlation between a PGS and the family-history score

Under a **conditionally independent measurement model**, the expected
correlation between a polygenic score (PGS) and the family-history (LT-FH)
score is pinned by three quantities: the two prediction accuracies and the
heritability decomposition.

**Setup.** Let `g` be the proband's additive genetic liability, standardised
to `Var(g) = 1` (everything below is a correlation, hence scale-free), and
split it into the SNP-captured part and the rest:

```text
g = s + u,   s ⊥ u,   Var(s) = p := h²_SNP / h²_total,   Var(u) = 1 − p,
```

so `Corr(s, g) = √p`. The PGS has prediction accuracy `a = Corr(PGS, s) =
√R²_pgs` against the SNP-captured part; the family-history score has
`b = Corr(FH, g) = √R²_fh` against the full genetic value (for LT-FH the
prediction R² comes from [bench_accuracy](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)). Both
scores are taken as standardised and jointly Gaussian, so
`E[PGS | g] = a√p · g` and `E[FH | g] = b · g`.

**Key assumption (conditional independence).** Given `g`, the two scores'
errors are independent: PGS ⊥ FH | `g`. This is an additional measurement-model
assumption, not a consequence of using independent cohorts. Non-overlapping
training and family samples remove an obvious shared-sample noise source, but
they do not rule out residual covariance after conditioning on the scalar `g`
(for example from ancestry, assortative mating, indirect genetic effects,
selection, or genetic structure not captured by that scalar).

**Derivation** (law of total covariance):

```text
Cov(PGS, FH) = E[Cov(PGS, FH | g)] + Cov(E[PGS | g], E[FH | g])
             = 0                     + Cov(a√p · g, b · g)
             = a · b · √p.
```

All variables standardised, so the population correlation is

```text
Corr(PGS, FH) = a · b · √p = √( R²_pgs · R²_fh · h²_SNP / h²_total ).
```

Limits: `h²_SNP = h²_total` gives `√(R²_pgs · R²_fh)`; a perfect family proxy
(`b → 1`) gives `√(R²_pgs · h²_SNP / h²_total)`. The correlation is bounded
above by each `√R²` and by `√(h²_SNP / h²_total)`.

**Why "weakly correlated yet complementary".** Two scores can each be
genuinely predictive of `g` while correlating weakly with *each other*,
because each is mostly noise relative to `g` — e.g. `R²_pgs = 0.05`,
`R²_fh = 0.15`, `h²_SNP/h²_total = 0.7` gives `Corr ≈ 0.07`. And since the
errors are independent, combining them pays: with
`c_g = Corr(PGS, g) = a√p` and `ρ = a · b · √p`, the joint multiple
correlation is

```text
R²_joint = (c_g² + b² − 2 c_g b ρ) / (1 − ρ²)   >   max(c_g², b²)
```

whenever the second score carries independent signal (at `p = 1` this reduces
to `(a² + b² − 2a²b²)/(1 − a²b²)`). This is exactly the empirical picture of
Hujoel et al. (2022, *Cell Genomics*) — PGS and family history combine with
large gains despite a small mutual correlation — and of Dybdahl Krebs et al.
(2026, *AJHG*), who report PA-FGRS and PGS weakly correlated yet
complementary, consistent with both being "noisy measures of additive
genetic liability" — i.e. the formula above.

**Caveats.** When conditional independence fails, the first term in the law of
total covariance is a residual contribution:
`delta = E[Cov(PGS, FH | g)]`. For standardised scores the correlation becomes
`a*b*sqrt(p) + delta`; `delta` may be positive or negative. Sample or family
overlap is one possible source, but neither overlap nor non-overlap determines
its sign or proves it is zero. Conditioning FH on the proband's diagnosis can
also induce residual dependence under selection, phenotype-informed training
or gene-environment correlation, so it needs justification for the study at
hand. Large non-Gaussian effects weaken the Gaussian conditioning step, though
the moment identity survives for scores linear in `s`/`g` with independent
errors.

The correlation identity is verified inside the independent-SNP simulation
(`bench_pgs_comparison.py`, §28 of RESULTS.md: observed 0.2009 ± 0.0046 vs
theory 0.2003 ± 0.0050). Its independent-SNP association panel uses a lightweight marginal score
statistic, not a real-LD mixed model. The joint model is evaluated by two-fold
cross-fitting and improves held-out R² from 0.236 (PGS) and 0.170 (LT-FH) to
0.338; that is an empirical complementarity check, not a separate numerical
verification of the closed-form joint-R² identity.

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
- **case** with onset age `a_i`: two LT-FH++ encodings exist, and LTFHPlus's
  `prepare_LTFHPlus_input` emits either. The **pin** sets `lower == upper ==
  T_i(a_i)`: under threshold-crossing onset the liability is fixed and the
  threshold falls with age, so onset happens exactly when `T_i(t)` reaches `l_i`
  and the map is invertible (`convert_liability_to_aoo` ↔
  `convert_age_to_thresh`). This is LTFHPlus's `use_fixed_case_thr = TRUE` and
  ltpred's default (`age_thresholds`, `thresholds_from_cip(case_mode="pin")`,
  the register driver). The **interval** `[T_i(a_i), inf)` records only that
  the liability had crossed the onset threshold; it is LTFHPlus's default
  (`use_fixed_case_thr = FALSE`) and ltpred's `case_mode="interval"` /
  `pa_thresholds`. Younger onset ⇒ lower CIP ⇒ higher threshold ⇒ more extreme
  liability under either. Base PA-FGRS is a third encoding: the lifetime
  interval `[T_pop, inf)`, `T_pop = Φ⁻¹(1 − K_pop)`, with age entering only
  through the censored-control mixture. `[T_i(a_i), inf)` under the PA engine is
  therefore an LT-FH++ interval encoding, not base PA-FGRS.

What pinning *assumes* is worth stating, since the benchmarks show the case
encoding (pinned vs lifetime interval vs age-specific interval) dominates
calibration. A pinned case's liability is a deterministic function of its onset
age — zero conditional variance at `T_i(a_i)` — with the stratum's CIP curve
hard-coded as the liability–onset map, error-free onset dates and homogeneous
severity (two cases with the same stratum and onset age carry identical
liability). That identity is calibrated only under threshold-crossing onset
(RESULTS.md §16: pinned slope 0.98–1.00). When onset only *tends* to track
liability (Gaussian copula ρ = 0.6), the same pin over-conditions (slope 0.92
under heavy censoring). On the same families, replacing the pin with
`[T(onset), ∞)` keeps almost all of the age-of-onset increment except at high
prevalence, where the pin adds another 0.01–0.02 in the squared-correlation
proxy (RESULTS.md §3). Most of what onset contributes is the lower bound, not
the point mass. When onset timing is uncertain or recorded only as a window,
an exact likelihood must integrate over the recording bin under an explicit
onset model. The lifetime or age-specific liability interval is a conservative
partial-information encoding: the lifetime interval conditions only on being
affected, and the age-specific interval leaves liability free above its edge. The
[real-data checklist](assumptions.md#real-data-checklist) already asks for the
encoding choice to be recorded.

For **LT-FH++ and ADuLT**, age, sex and birth cohort enter **only through
`K(t; s, b)` and its interval edge `T_i`**, never the covariance `Sigma`. In base
PA-FGRS, a control's age/stratum instead enters the mixture through `K_i`; the case
threshold remains the lifetime threshold. What ltpred ships:

| helper | CIP model | strata | family context |
|---|---|---|---|
| `prevalence_thresholds` | one lifetime prevalence `K` → `T = Phi^-1(1-K)` | none | relatives: classic LT-FH |
| `age_thresholds` | logistic `K / (1 + exp((mid_point - age)·slope))` | age only, single `K` | pinned tutorial encoding; model depends on rows |
| `pa_thresholds` | same logistic demo CIP | age only, single `K` | age-dependent interval/mixture variant; not base PA-FGRS, exact PA-FGRS_ADT, or the PA engine switch |
| `thresholds_from_cip` | an **empirical** CIP curve you supply, **called once per stratum** | sex × birth year × ancestry | `case_mode="pin"`: LT-FH++ with relatives or ADuLT without; `"interval"`: age-dependent PA-FGRS-style variant, not exact PA-FGRS_ADT |

The same personalised CIP therefore feeds two models: relative observations make
it LT-FH++; their absence makes it ADuLT. A threshold helper cannot decide that
for you.

For paper-faithful base PA-FGRS, use lifetime case/control bounds from
`prevalence_thresholds`, attach the control-specific `K_i` and `K_pop` calculated
from the appropriate CIP, and run PA with `use_mixture=True`. In that model age
enters the control mixture weight; it does not change an observed case's threshold.

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

There is no closed form for `E[ℓ_F | ℓ_F ∈ C_F]`
once several intervals are live. Algorithm G samples the truncated
multivariate normal by coordinate-wise inverse-CDF draws
(Kotecha & Djurić 1999). The public estimator is
`gibbs_estimate_batched`; the low-level chain `rtmvnorm_gibbs` draws
the full vector and is never collapsed.

Write `Σ` for the family covariance, `Q = Σ⁻¹` for
its precision, and

```text
sd_j   =  (1 / Q_jj)^{1/2} ,
P_ij   =  −Q_ij / Q_jj     (i ≠ j) ,     P_jj  =  0.                    (8)
```

The `j`th conditional, given the rest of the current state `x`, is
then `N(μ_j, sd_j²)` restricted to
`(a_j, b_j)`, with `μ_j = P[:, j] · x`. Both `P` and
`sd` depend only on `Σ`, so they are computed once
per structure. `Σ` must be strictly positive definite.

**Algorithm G** (truncated-MVN Gibbs).

**Input.** A covariance `Σ` shared by a group of families;
per-family intervals `[a, b]`; the coordinates whose posterior
means are required; `n_sim`, burn-in, batch size, and
tolerance.

**Output.** For each family, an estimate of (5), the posterior
variance of the target, and a batch-means Monte-Carlo SE.

**G1.** [Group.] Partition the families by ordered role set. For each
group, form (8) once.

**G2.** [Collapse.] Let `U` be the coordinates that are untruncated
in every family of the group (the genetic rows on the public path).
If `U` is nonempty, reduce the sweep to the complement `y` and
store the Gaussian map `W` and residual variances of `U` given
`y`. The genetic mean is then `E[g | y]`; the
reported posterior variance is
`Var(E[g | y]) + Var(g | y)`.

**G3.** [Initialise.] For each family, set `x` to a feasible point
of the rectangle (the same construction LTFHPlus uses). Hold a
coordinate with `a_j = b_j` fixed.

**G4.** [Sweep.] For `k = -B, …, n_sim−1` and for
each free coordinate `j`, draw

```text
x_j  ←  μ_j + sd_j  Φ⁻¹( U( Φ((a_j−μ_j)/sd_j), Φ((b_j−μ_j)/sd_j) ) ).  (9)
```

Far-tail intervals cannot be drawn on the probability scale
(`Φ` underflows near `|z| ≈ 38.5`); those use a
log-scale Rayleigh construction.

**G5.** [Accumulate.] After burn-in, stream the target: a kept
coordinate contributes `x_j`; a collapsed coordinate contributes
`(Wy)_u`. Update running sums, sums of squares, and batch-mean
summaries. Do not store the `(n_sim × n_out)`
draw array. Monte-Carlo-SE memory is therefore `O(F)`.

**G6.** [Stop.] If every requested batch-means SE
(`batch_means`, R's `batchmeans::bmmat`) is below `tol`, halt.
Otherwise draw another block of `n_sim` and accumulate.
Families that have already met `tol` are dropped from later rounds.

Each family carries its own RNG seed, so the draw does not depend on
which worker picked it up. The inner loop is Numba-JIT'd and
`prange`-parallel across families in a group; without Numba the
identical code runs serially.

## Inference engine 2: Pearson–Aitken

The Pearson–Aitken selection formula gives, in closed form, how a
jointly Gaussian vector's mean and covariance change when one
component's marginal is *selected* (truncated). If component `i`
moves from `N(m_i, v_i)` to selected moments
`(m*, v*)`, every remaining component updates by a
rank-1 correction:

```text
m_j     ←  m_j + (Σ_ji / v_i) (m* − m_i) ,
Σ_jk    ←  Σ_jk + (Σ_ji Σ_ik / v_i²) (v* − v_i).                        (10)
```

Aitken's source is his multivariate-normal selection note, not his
generalized-least-squares paper
([Aitken 1935](https://doi.org/10.1017/S0013091500008063)).
Mendell & Elston applied (10) sequentially to multifactorial
threshold traits
([1974, *Biometrics*](https://pubmed.ncbi.nlm.nih.gov/4813384/)).

**Algorithm P** (Pearson–Aitken sequential selection).

**Input.** A covariance `Σ`; per-family intervals; a target
coordinate `t` (usually `g`).

**Output.** An approximation to `E[t | C_F]` and to
`Var(t | C_F)`, with no Monte-Carlo error.

**P1.** [Order.] Place the target first. Canonicalise the remaining
roles to a sorted name order and realign the bounds. This makes the
result a function of the pedigree shape, not of input row order. It
is a reproducibility choice, not a claim of smaller approximation
error.

**P2.** [Reduce.] Marginalize unobserved non-target coordinates. Jointly
condition on all pins using Gaussian conditioning, checking singular support,
and center the remaining bounds. This step excludes the censoring mixture,
whose sequential observation semantics remain unchanged.

**P3.** [Fold.] For each remaining interval coordinate `i`, last to first,
replace its marginal `N(m_i, v_i)` by the truncated-normal
moments `(m*, v*)` of its interval, and apply (10) to the
remaining mean and covariance.

**P4.** [Target.] Apply the target's own interval to the updated
`N(m_0, v_0)`. For the genetic target the interval is
`(-∞, ∞)` and this step is a no-op. For `out="full"` it
is `E[ℓ_o | own interval and relatives]`, the same
estimand Algorithm G uses.

**P5.** [Read.] The target's updated mean is the estimate of (5);
its updated variance is the reported posterior variance.

After one truncation the selected law is no longer, in general,
multivariate normal. Algorithm P keeps only the updated first two
moments and proceeds as if the remaining variables were Gaussian
with those moments. After exact pin conditioning, it is exact with zero or
one remaining interval, and a sequential two-moment approximation thereafter.
Conditioning pins first can change earlier PA outputs without changing the
specified model. It does not guarantee an improvement for every rectangle.
For additive nuclear families, `method="quadrature"` instead integrates at most
two parental factors; the [methods report](https://github.com/bvilhjal/ltpred/blob/main/report/ltpred_methods.pdf)
derives the factorization, both target moments, numerical diagnostics, and
the selected-relationship and pairwise-fitting reductions.

In the archived benchmark snapshots
([RESULTS §§1, 2 and 30](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)),
on separately observed family-member intervals **without the censoring
mixture**, PA and Gibbs posterior means agreed closely while PA ran orders of
magnitude faster than grouped Gibbs at matched threads. A locked comparison
to R LTFHPlus 2.2.0 and LTFGRS 1.0.1 on the same classic LT-FH families
shows both ltpred engines reproducing the R Gibbs scores and ltpred PA
reproducing LTFGRS PA; that section also gives the same-algorithm fold times
at four threads and at one, per-family times, and isolated-process peak RSS.
PA versus LTFHPlus is a different algorithm, not a faster Gibbs. The PA-only
mixture was not part of either comparison. The same grouping / `prange`
structure as Algorithm G applies.

On the `benchmarks/bench_pa_robustness.py` stress pedigrees the
spread across fold orders was a median < 0.12% and p95 < 3.4% of
the between-proband score SD. No fold order is more exact than
another.

### Base PA-FGRS: lifetime cases and censored controls

An observed case contributes the lifetime interval
`[Φ⁻¹(1-K_pop), ∞)`. An age-censored control
is not one truncated law. It is a mixture of a lifetime control and
a not-yet-onset future case. Algorithm P folds that person by
replacing the ordinary truncated moments in **P3** with the
two-component moments below (`_tnorm_mixture` in
`ltpred.pearson_aitken`; PA-FGRS supp. eqs. S3–S5).

**Algorithm M** (censored-control mixture moments).

**Input.** Current conditional mean and variance `(m, v)` of the
coordinate being folded; current CIP `K_i`; lifetime prevalence
`K_pop`.

**Output.** Selected moments `(m*, v*)` for use in (10).

**M1.** [Split.] Let `T_pop = Φ⁻¹(1-K_pop)`
be the lifetime threshold, and write
`Φ_below = Φ((T_pop − m) / √v)`.

**M2.** [Weight.]

```text
π  =  Φ_below  /  ( Φ_below + (1 − Φ_below) (K_pop − K_i)/K_pop ).     (11)
```

**M3.** [Mix.] Set `m*` and `v*` to the two-component
mean and variance of the below-threshold and above-threshold
truncated normals, with weights `π` and `1-π`.

The split in **M1** is the lifetime threshold, not the passed
`upper`. Age enters only through `K_i` in (11). The bound `upper`
merely flags a censored control (finite) versus an observed case
(`+∞`), so passing either the lifetime bound or an
age-specific `Φ⁻¹(1-K_i)` gives the same result. Enabled via
`use_mixture=True`; off, Algorithm P reduces to the plain
truncated-moment sweep.

The factor `(K_pop-K_i)/K_pop` is the CIP
probability that a future case has not yet onset. That is an
onset-timing assumption: among people who will eventually be cases,
time-to-onset is independent of liability, and censoring is
non-informative given the stratum. If higher liability advances
onset, the not-yet-onset component is only approximately the
above-threshold tail **M3** assigns it.

That independence assumption is now a generative arm
(`simulate_under_LTM_single(..., onset_model="liability_dependent")`,
default ρ = 0.6), not only a caveat. In RESULTS.md §16 four of ten
paired Δcorr 95% CIs exclude zero, but the largest shift is only −0.00034 (95% CI ± 0.00011);
the ranking effect is statistically detectable and practically negligible. The mixture still always
lowers the calibration slope. What breaks is the *pin*, not the
mixture: under partial onset dependence the lifetime interval sits
between the crossing and stochastic extremes (MID slope 1.13),
while pinning over-conditions (0.92). There is still no Gibbs
implementation of the mixture, so this is a PA-only check.

## Fitting the covariance (heritability)

Both estimators above *condition* on a known `h2`. `fit_heritability` instead
**fits** it — estimating the liability-scale heritability from the case/control
(and age-of-onset) statuses of relatives — with a Gibbs sampler modelled on
bipred's joint effect/parameter loop (sample the latents, then re-estimate the
covariance parameters each sweep). It treats the latent liabilities as missing
data and alternates:

**Sampling requirement.** The cross-product reconstruction below assumes
independent, non-overlapping families sampled from the population observation
model encoded by the bounds and prevalence, either directly or reconstructed
with valid IPW. Unmodelled case/control or family-history ascertainment changes
the latent cross-products and can severely bias the fit. Under selection with
unknown, misspecified, or zero inclusion probabilities, supply an externally
estimated population-scale `h2` or fit an explicit ascertainment model; neither
more Gibbs iterations nor a different optimiser repairs a missing selection
likelihood.

Two sampling modes are accepted. `sampling="population"` is the unascertained
contract, and is **screened for gross marginal inconsistency**: the thresholds
imply `K = 1 - Phi(T)`, each role's case count is `Binomial(n_fam, K)` under that
contract, and a gross departure raises. Passing this role-wise screen does not
certify the joint family-pattern distribution. `sampling="ipw"` takes per-family
`weights = 1 / P(family sampled)` and re-mixes the per-family moment
contributions to population proportions — sound because the augmentation of a
*given* family with *given* statuses is already the correct conditional law, and
it is the **mix** that selection corrupts. It requires a known, strictly positive
inclusion probability for every complete observed family-pattern stratum.
Proband-ascertained designs have zero probability for unaffected probands and
cannot be reweighted. The weighted marginal screen can falsify some bad weight
sets; it is not a general positivity test.
Omitting `sampling` still warns.

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
is the bipred-style analogue of a full conjugate step). In the repository
benchmark of unascertained simulated families, bias across `h2 = 0.2–0.8` was
small relative to the across-dataset SD, but larger than the within-fit
Monte-Carlo error in some settings. The reported `h2_se` is the
*within-dataset* Monte-Carlo error; sampling variability across datasets is
larger and depends on the number and informativeness of the families. An
iid-family cluster bootstrap can estimate that sampling variability when
families are independent and population-sampled, but it does not remove
ascertainment bias. Identifiability comes entirely from the *between-relative*
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
additive-only `Â` by +0.047, +0.101, +0.156 as `s²` runs 0.1 → 0.2 → 0.3 when it is
`C`, but only −0.005, +0.025, +0.034 when it is `M`. So `M` is worth fitting for its
own sake (quantifying/testing spousal resemblance) rather than to de-bias `h²`. A
**dominance** component is deliberately not offered: in the current role-based
design its sharing pattern is not separated reliably from additive and sibship
components. Identification would require an explicit dominance relationship
kernel and relationship contrasts linearly independent of `A` and `C`. MZ/DZ
twin observations can contribute such contrasts within a richer design, but MZ/DZ
pairs alone cannot identify `A`, `C`, and `D` simultaneously.

In `research/advanced_fitting.py`'s `fit_variance_components_mcem`, the M-step
instead optimises the Gaussian likelihood of the
imputed liabilities. This implementation is an **approximate finite-iteration
Monte-Carlo EM-style procedure**: it uses a fixed damping coefficient, averages
post-burn-in iterates, and does not stop on an observed-likelihood convergence
criterion. Its OPG/BHHH information SE and GHK log-likelihood/AIC therefore retain
both Monte-Carlo and finite-iteration error. Check stability across seeds and
iteration settings and use family resampling for sampling uncertainty when the
independent-cluster assumptions hold. This research MCEM route uses the same
unconditional population observation model: changing the fitting algorithm does
not account for ascertainment. Likewise, a family bootstrap propagates sampling
variation under the fitted design; it does not correct selection bias and is
invalid when nominal family clusters overlap.

## Multiple traits

For `n` genetically/environmentally correlated traits the covariance is
phenotype-major: same-trait blocks use `A_ij * h2_p`; cross-trait blocks scale the
relationship `A_ij` by the genetic covariance `rho_g[p,q] * sqrt(h2_p h2_q)`, and
the same individual's full liabilities across traits correlate by
`full_corrmat[p,q]`. The Gibbs sampler then returns the genetic/full liability of
each trait (the multi-trait path of `estimate_liability`). This lets a
well-powered trait sharpen
the estimate for a correlated, under-powered one.

These inputs are jointly constrained. Let `D = diag(sqrt(h2))`,
`G = D genetic_corrmat D`, and `E = full_corrmat - G`. Both `G` and `E` must be
positive semi-definite, while the two supplied correlation matrices must be
symmetric with unit diagonal. The observed full-liability block of the pedigree
covariance is `G ⊗ A + E ⊗ I`; the appended target genetic coordinate carries
`Var(g^p) = h2_p` and same-person cross-trait `Cov(g^p, g^q) = G[p,q]`, with no
`E` contribution. The constructor rejects an incoherent decomposition rather than
silently applying a positive-definite correction to a different model.

### Fitting multi-trait covariances

The installed, opt-in `fit_pairwise_multi` jointly estimates heritabilities,
genetic correlation and residual environmental correlation from binary family
records. It maximises a composite likelihood of observed pairs with PSD trait
covariance matrices, optionally including full-sibship C and couple M.
Identifiability is checked using observed contrasts before fitting. Sampling
SEs use family-cluster scores and a delta method; covariance boundaries
withhold normal SEs. See [Inference](inference.md#joint-heritability-and-cross-trait-correlations)
for the model, common-threshold/population-IPW contracts and uncertainty limits.
The multi-trait scorer above remains A+E only.

The HE genetic-correlation, onset-age-decay and common-factor prototypes remain
checkout-only in `research/advanced_fitting.py`; their separate assumptions
and limitations are documented under [research extensions](research.md).

## Background and references

The method sits in a long quantitative-genetics lineage. **Threshold models** for
binary/categorical traits — mapping a continuous latent liability through a
threshold — go back to Wright (1934), Dempster & Lerner (1950), Falconer (1965) and
Gianola (1982). Their continuous-trait analogue is **selection-index / BLUP**
prediction of additive genetic value from relatives' phenotypes and a relationship
matrix (Hazel 1943; Henderson 1975; the animal model, and its genomic-relationship
extensions, VanRaden 2008). Liu et al. introduced **GWAX**, coding affected
relatives through a binary proxy phenotype
([2017, *Nat Genet*](https://doi.org/10.1038/ng.3766)).
**LT-FH** ([Hujoel et al. 2020](https://doi.org/10.1038/s41588-020-0613-6))
instead adapts the threshold model to case-control GWAS by using the posterior mean
genetic liability as a configuration-specific phenotype. **LT-FH++**
([Pedersen et al. 2022](https://doi.org/10.1016/j.ajhg.2022.01.009)) adds
personalised age-, birth-year- and sex-dependent prevalence; **ADuLT**
([Pedersen et al. 2023](https://doi.org/10.1038/s41467-023-41210-z)) uses the same
personalised construction without family history as an alternative to time-to-event
GWAS; and **PA-FGRS**
([Dybdahl Krebs et al. 2024](https://doi.org/10.1016/j.ajhg.2024.09.009)) gives a
deterministic Pearson–Aitken approximation for large, age-censored genealogies.

Core methods:

- [Hujoel et al. 2020, *Nat Genet*](https://doi.org/10.1038/s41588-020-0613-6) —
  LT-FH.
- [Pedersen et al. 2022, *AJHG*](https://doi.org/10.1016/j.ajhg.2022.01.009) —
  LT-FH++ (personalised thresholds for first-degree relatives). The simulation
  panels used parents plus zero to two siblings; that was a simulation design, not
  a method limit. ltpred's arbitrary-pedigree path extends beyond the 2022 LT-FH++
  publication.
- [Pedersen et al. 2023, *Nat Commun*](https://doi.org/10.1038/s41467-023-41210-z)
  — ADuLT (family-free personalised threshold).
- [Dybdahl Krebs et al. 2024, *AJHG*](https://doi.org/10.1016/j.ajhg.2024.09.009)
  — PA-FGRS (Pearson–Aitken family genetic risk scores).

Related family-history methods and interpretation:

- [Liu, Erlich & Pickrell 2017, *Nat Genet*](https://doi.org/10.1038/ng.3766) —
  GWAX, the binary proxy-case precursor; it is a comparator, not implemented here.
- [Kendler et al. 2021, *JAMA Psychiatry*](https://doi.org/10.1001/jamapsychiatry.2021.0336)
  — a distinct register-standardised FGRS, also not implemented here.
- [Hujoel et al. 2022, *Cell Genomics*](https://doi.org/10.1016/j.xgen.2022.100152)
  — target-population risk prediction combining a PRS with family history; this is
  downstream of, and separate from, ltpred liability estimation.
- [Dybdahl Krebs et al. 2026, *AJHG*](https://doi.org/10.1016/j.ajhg.2025.11.016)
  — across five psychiatric disorders, PA-FGRS and PGS were weakly correlated yet
  complementary, consistent with both being noisy estimates of the same additive
  genetic liability under the paper's model.
- [Pedersen et al. 2025, *Front Genet*](https://doi.org/10.3389/fgene.2025.1708315)
  — graph-based extraction of arbitrary-degree relatives and kinship matrices from
  population trio records; relevant upstream preprocessing, not PA validation.
- [Pedersen et al., LTFGRS R package](https://emilmip.github.io/LTFGRS/) —
  LT-FH++, PA-FGRS and Kendler's FGRS with unified data preparation on the same
  graph-based extraction.

Threshold-model background:

- Wright 1934, *Genetics* — the underlying-scale threshold analysis of digit
  number in guinea pigs.
- Dempster & Lerner 1950, *Genetics* — heritability of threshold characters.
- Falconer 1965, *Ann. Hum. Genet.* — liability to disease from incidence in relatives.
- Gianola 1982, *J. Anim. Sci.* — threshold characters in animal breeding.

Selection index / BLUP background:

- Hazel 1943, *Genetics* — the genetic basis for constructing selection indexes.
- Henderson 1975, *Biometrics* — best linear unbiased estimation/prediction (BLUP).
- VanRaden 2008, *J. Dairy Sci.* — genomic relationship matrices for prediction.

Liability-threshold risk models (Sham and colleagues):

- Sham 1998, *Statistics in Human Genetics* (Edward Arnold) — the
  multivariate liability-threshold model in genetic epidemiology.
- So, Kwan, Cherny & Sham 2011, *AJHG* — risk prediction from family history and
  known susceptibility loci, with age, follow-up and competing mortality risks.

Numerics:

- Pearson 1903 and
  [Aitken 1935](https://doi.org/10.1017/S0013091500008063) — the selection formula
  for a multivariate-normal population.
- [Mendell & Elston 1974, *Biometrics*](https://pubmed.ncbi.nlm.nih.gov/4813384/) —
  sequential application to multifactorial threshold traits.
- Tallis 1961, *JRSS B* — moments of the truncated multivariate normal.
- Kotecha & Djurić 1999 — Gibbs sampling for truncated multivariate normals.
- Genz & Bretz 2009, *Springer* — computation of multivariate normal probabilities
  (the GHK simulator behind the MCEM log-likelihood).
- Lee et al. 2011, *AJHG* — observed-to-liability-scale heritability.

Association and phenotype-quality cautions:

- [Zhuang et al. 2022, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btac459)
  — related family-history phenotypes, case-control imbalance and tail calibration.
- [Wu et al. 2024, *Nat Genet*](https://doi.org/10.1038/s41588-024-01963-9) —
  survival and participation bias in parental-history Alzheimer GWAX; a
  disease-specific warning, not a universal indictment of family-history models.
- [Cai et al. 2026, *Nat Genet*](https://doi.org/10.1038/s41588-025-02465-y) —
  a Perspective arguing that shallow phenotypes can introduce heritable
  confounding in psychiatric genetics.

Multi-trait / genetic factor structure:

- Grotzinger et al. 2019, *Nat. Hum. Behav.* — Genomic SEM: structural equation
  models (including the common-factor model) fit to a multi-trait genetic
  covariance — the model `fit_genetic_factor` ports to the pedigree scale.
