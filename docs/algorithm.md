# Algorithm and model

This page records the estimand, the observation models, and the two
algorithms that compute the score. Usage is in [guide.md](guide.md).
The typeset companion is the
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
That high-level kinship API accepts `lower`/`upper` and, with Pearson–Aitken,
`use_mixture=True` plus per-member `K_i`/`K_pop` for the PA-FGRS
censored-control mixture. Gibbs still has no mixture implementation.
For discovering the relatives from population trio records, `ltpred.pedigree`
implements graph-based relative extraction in the style of [Pedersen et al.
(2025)](https://doi.org/10.3389/fgene.2025.1708315): `build_parent_graph`
indexes the records, `extract_pedigree` visits every person within
`max_degree` relationship-degrees of a proband (full-sibling edges make the
distance equal the standard relationship degree) and closes on all recorded
ancestors, so the extracted pedigree's kinship is exact for every member pair.
Its kinship is the exact tabular one (inbreeding-aware), not the paper's
path-counting approximation. The
[LTFGRS](https://emilmip.github.io/LTFGRS/) R package (Pedersen et al.) —
LT-FH++, PA-FGRS and Kendler's FGRS with unified data preparation — consumes
the same extraction downstream.

This is an **additive-genetic** model: familial resemblance is entirely genetic
sharing. Shared environment, household/cultural transmission, assortative mating
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
explicit cross-trait component covariance is defined.
Validated in `benchmarks/bench_env_components.py`: wiring recalibrates the
genetic estimate (slope 0.93 -> 0.99) and sharpens the full-liability
prediction. Arbitrary user-supplied kernels still go through the covariance-
level entry points (`rtmvnorm_gibbs`, `pa_algorithm`, `pa_estimate_batched`),
which accept an arbitrary covariance directly.

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

### Direct and indirect (genetic-nurture) effects

Caution (i) above says a symmetric shared-environment matrix is not a maternal
effect, because that is a *directional* path. `construct_covmat_nurture` (in
`research/covariance_extensions.py` — research code, not the supported core)
builds
the directional model instead, from the path structure rather than from
kinship:

```text
A_o = (A_m + A_f)/2 + w_o              # transmission + Mendelian sampling
l_o = A_o + n*(A_m + A_f) + e_o        # own genes, plus parental nurture
l_m = A_m + e_m                        # parents are founders here
```

`n` (`nurture`) scales the **indirect** path: the parents' genotypes shaping the
offspring's environment. The target row `g` remains the proband's **own**
additive value `A_o` -- what a GWAS phenotype should predict -- while the
indirect path contributes to their *liability* without being part of their
direct effect.

Writing `h` for `h2`, the entries are no longer `2*phi * h`:

| pair | additive model | with nurture |
|---|---|---|
| parent - offspring liability | `h/2` | `h/2 + n*h` |
| sib - sib liability | `h/2` | `h/2 + 2*n*h + 2*n^2*h` |
| `Cov(A_o, l_parent)` | `h/2` | `h/2` (unchanged) |
| `Cov(A_o, l_o)` | `h` | `h*(1 + n)` |
| `Cov(A_o, l_sibling)` | `h/2` | `h/2 + n*h` |

Three things follow. **The two familial covariances inflate by different
amounts**, which is what makes `n` identifiable from a nuclear family at all.
**The `g` row changes selectively**: `Cov(A_o, l_parent)` remains `h/2`, but
`Cov(A_o, l_o)` and `Cov(A_o, l_sibling)` increase because the parental
genotypes driving nurture are correlated with `A_o`. This directional,
pair-specific pattern cannot be represented by one symmetric kinship-scaled
component, which is why it needs a separate constructor rather than another
entry in the `A`/`C`/`M` bank. **`n = 0` reproduces
`construct_covmat_single` exactly.**

**The identifiability trap.** The sib-sib inflation `2*n*h + 2*n^2*h` is shared
by every offspring of the couple, so *on sibling covariance alone genetic
nurture is indistinguishable from a sibship environment `C`*. What separates
them is the parent-offspring covariance: `C` leaves it untouched, nurture
raises it by `n*h`. Fitting both from sibs only is not identified, and a study
with no parental phenotypes cannot tell the two apart at all -- it will load
whichever one it is offered.

Offspring liabilities are standardised to unit variance, so thresholds keep
their prevalence meaning; this requires `1 - h - 2*n^2*h - 2*n*h >= 0`, since
the shared nurture term takes variance the residual must give up.

**Fitting `n`.** Unlike the sex-limitation parameters, `n` need not be
supplied. The two moment equations

```text
cov_parent_offspring = h*(1 + 2n)/2
cov_sib_sib          = h*(1 + 2n)^2/2
```

are two equations in two unknowns, and the ratio isolates the indirect path, so
`fit_nurture` (in `research/advanced_fitting.py`) inverts them in closed form
rather than iteratively:

```text
1 + 2n = cov_sib_sib / cov_parent_offspring
h      = 2 * cov_parent_offspring^2 / cov_sib_sib
```

Both inputs are liability-scale covariances between the two relative types; from
binary data obtain them with `ltpred.tetrachoric` rather than from
observed-scale correlations. It is a moment estimator, so it carries no standard
errors and inherits whatever bias the input covariances have. It also reports
`h2_additive_po` and `h2_additive_sib` -- what a nurture-blind additive model
would claim from each relative type alone -- whose **disagreement is the
diagnostic**, and which is zero exactly when `n` is zero. A sibling covariance
*below* the parent-offspring one yields a negative `n` (a contrast effect),
returned rather than clipped.

**Scope.** Nuclear roles only (`m`, `f`, `s...`). Extending to grandparents
means propagating the path model up the pedigree, which stops the parents being
founders -- their liabilities would gain their own nurture terms from the
grandparents. That recursion is not implemented, and silently treating a
grandparent as a founder would understate the covariance, so those roles are
rejected. Like the other covariance constructors, the result is an ordinary
`Covmat` and feeds `pa_algorithm` / `rtmvnorm_gibbs` / `pa_estimate_batched`
directly.

### Sex-limited genetic architecture

Everywhere else in ltpred, sex enters through the **threshold**: a sex-specific
CIP gives each person their own `T`. That is where it belongs for calibration,
but recall the factorisation `g0 = w' mu` with `w = V^-1 c`. The weights `w`
depend on heritability and kinship only — they are threshold-free. So a
sex-specific threshold moves `mu` and can reorder scores through the truncated
means, but it never changes how much weight a relative carries.

`construct_covmat_sex_limited` (in `research/covariance_extensions.py`) puts
sex in `V` instead:

```text
Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) * rg_cross^[sex_i != sex_j]
```

with `h2_i` the heritability of person `i`'s sex. This is the standard
sex-limitation model of the twin/family literature, in two parts:

- **Scalar (quantitative) sex limitation** — `h2_female != h2_male`. The same
  genes act in both sexes, with different variance. A relative of the
  higher-heritability sex is more informative and gets more weight.
- **Qualitative sex limitation** — `rg_cross < 1`. Partly *different* genetic
  architectures between the sexes, so an opposite-sex relative tells you less
  about the proband than a same-sex relative at the same kinship.

Full liabilities keep unit variance, so thresholds retain their prevalence
meaning; only the genetic scale differs by sex. Setting `h2_female == h2_male`
and `rg_cross == 1` reproduces `construct_covmat_single` exactly.

The matrix is positive semi-definite for any `|rg_cross| <= 1`. Writing it as
`D^(1/2) (A o Rg) D^(1/2)`, where `o` is the Hadamard product, `D` holds the
per-person heritabilities and `Rg` has `1` within a sex and `rg_cross` between,
both `A` and `Rg` are PSD, so their Schur product is PSD and the symmetric
scaling preserves it.

The role grammar fixes the sex of parents and grandparents (`m`, `f`, `mgm`,
`mgf`, `pgm`, `pgf`). Siblings, children, half-sibs and aunts/uncles are
ambiguous — `mau`/`pau` covers both aunts and uncles — and must be declared in
`sex=`; the constructor raises rather than defaulting, since a silent default
would impose one sex's heritability on the other. The genetic row `g` follows
the proband `o`.

Two cautions. **(i)** These are *inputs*, not fitted quantities: the constructor
takes `h2_female`, `h2_male` and `rg_cross` and builds the covariance. Nothing
here estimates them, and the identification requirements are real — separating
`rg_cross` from a scalar difference needs opposite-sex relative pairs
(brother–sister, and opposite-sex avuncular or half-sib links) contrasted
against same-sex pairs at matched kinship, which nuclear families supply
sparsely. **(ii)** A sex difference in *observed* prevalence is not by itself
evidence of sex-limited genetics; it is exactly what a sex-specific threshold
already absorbs. Reach for this model when same- and opposite-sex relative
correlations differ **after** the thresholds are personalised.

Because the result is an ordinary `Covmat`, it feeds the covariance-level
entry points (`pa_algorithm`, `rtmvnorm_gibbs`, `pa_estimate_batched`)
directly, the same route documented for any user-supplied kernel. The
role-based `estimate_liability` still takes a scalar `h2`.

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
score is a GWAS phenotype constructed from that diagnosis. It must
exclude, or leave unbounded, the proband's interval when the same
diagnosis is the outcome of a prospective evaluation; otherwise
`D_F` contains the answer (remark 1 above).

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
breeding-value-style score and a GWAS phenotype (the [BLUP framing](#connection-to-selection-index-and-blup)
above). LT-FH++ and ADuLT share personalised age/onset/sex/cohort thresholds.
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
complementary, "consistent with both being noisy estimates of the same
additive genetic liability" — i.e. the formula above.

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

Both identities above are verified numerically to Monte-Carlo precision
(simulated `s`, two noisy linear predictors, four
`(h²_SNP/h²_total, R²_pgs, R²_fh)` settings; correlation and joint `R²`
recovered to ~1e-3).

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
- **case** with onset age `a_i`: LT-FH++ and ADuLT **pin** liability at `T_i(a_i)`
  (`lower == upper`). Younger onset ⇒ lower CIP ⇒ higher threshold ⇒ more extreme
  liability. The map is invertible
  (`convert_liability_to_aoo` ↔ `convert_age_to_thresh`), so `T_i(a_i)` *equals* the
  case's liability at onset. Base PA-FGRS instead uses the lifetime interval
  `[T_pop, inf)`, where `T_pop = Φ⁻¹(1 − K_pop)`. The current `pa_thresholds` and
  `thresholds_from_cip(..., case_mode="interval")` helpers use
  `[T_i(a_i), inf)` and therefore define an age-dependent PA-FGRS-style variant,
  not the paper's base case encoding.

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

**P2.** [Fold.] For each remaining coordinate `i`, last to first,
replace its marginal `N(m_i, v_i)` by the truncated-normal
moments `(m*, v*)` of its interval, and apply (10) to the
remaining mean and covariance.

**P3.** [Pin.] If `a_i = b_i`, set `v* = 0`. This is exact
Gaussian conditioning on a point.

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
with those moments. Hence it is exact for a single interval or a
pin, and a sequential two-moment approximation thereafter.

On separately observed family-member intervals **without the
censoring mixture**, PA and Gibbs posterior-mean estimates had
correlation ≥ 0.997 while PA ran 384–488× faster than grouped Gibbs
in the controlled 4-thread benchmark in this package. A locked
comparison to R LTFHPlus 2.2.0 on the same classic LT-FH families
gave corr(ltpred Gibbs, LTFHPlus) = 0.9999 and corr(PA, LTFHPlus) =
0.9999 (RMSE 0.0041). LTFHPlus is Gibbs-only; public PA is LTFGRS
1.0.1, and ltpred PA matches it at RMSE 0.000087. On that machine
same-algorithm fold times were 6.87× (LTFHPlus Gibbs / ltpred Gibbs;
53.3 vs 7.76 ms/family) and 1178× (LTFGRS PA / ltpred PA; 9.47 vs
0.0083 ms/family). Isolated-process peak RSS (ldpred3 `wait4`
launcher) was 442 MiB (LTFHPlus), 260 MiB (LTFGRS PA) and
~148–165 MiB (ltpred). PA versus LTFHPlus is a different algorithm,
not a faster Gibbs. The PA-only mixture was not part of either
comparison. The same grouping / `prange` structure as Algorithm G
applies.

On the `benchmarks/bench_pa_robustness.py` stress pedigrees the
spread across fold orders was a median < 0.12% and p95 < 3.4% of
the between-proband score SD. No fold order is more exact than
another.

### Base PA-FGRS: lifetime cases and censored controls

An observed case contributes the lifetime interval
`[Φ⁻¹(1-K_pop), ∞)`. An age-censored control
is not one truncated law. It is a mixture of a lifetime control and
a not-yet-onset future case. Algorithm P folds that person by
replacing the ordinary truncated moments in **P2** with the
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
default ρ = 0.6), not only a caveat. In RESULTS.md §16 three of ten
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
benchmark of unascertained simulated families, bias across `h2 = 0.3–0.8` was
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

### Fitting the genetic correlation

`fit_genetic_correlation` (in `research/advanced_fitting.py`) estimates `rho_g`
between traits from family data — the
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

### Onset-age-structured genetic correlation

`fit_genetic_correlation_decay` (also `research/advanced_fitting.py`) generalises
`fit_genetic_correlation` to a genetic
correlation that **decays with the difference in age at onset** between two
relatives (or between two traits). The motivating idea is that genetic liability
need not be one static quantity: the genes driving early- and late-onset forms of
the same trait may overlap only partly, and two traits diagnosed at very different
ages may share fewer genetic drivers than their lifetime correlation suggests. The
genetic covariance between relative `i`'s trait `p` (onset age `a_ip`) and relative
`j`'s trait `q` (onset age `a_jq`) is

```text
Cov(g_i^p, g_j^q) = A_ij * sqrt(h2_p h2_q) * rho_g * K(|a_ip - a_jq| ; lam)
```

where `A_ij` is the additive relationship, `rho_g` the headline genetic
correlation at equal onset age, and `K` a decay kernel with rate scalar `lam` --
the age-difference importance parameter. `lam = 0` gives `K = 1` and recovers the
scalar `fit_genetic_correlation` model exactly (the structured covariance reduces
to the scalar one); larger `lam` makes the shared genetic signal die faster with
onset-age distance. Three kernels are offered: **OU / exponential**
`K(d) = exp(-lam |d|)` (the default), **Gaussian** `K(d) = exp(-(lam d)^2 / 2)`,
and **tent** `K(d) = (1 - lam |d|)+`. The exponential is preferred: it is the
Markovian (Ornstein-Uhlenbeck) covariance, positive-definite for any configuration
of ages, and has the deepest precedent for age/time-varying genetic correlation
(random-regression and character-process models in quantitative genetics, e.g.
Pletcher & Geyer 1999; Jaffrezic & Pletcher 2000; genetic "simplex" models; the
phylogenetic OU model). Within a trait (`p == q`) the same structure models
genetic heterogeneity by onset age; across traits it is the `rho_g` decay above.
With a single shared rate across blocks the covariance is positive-definite by
construction (a Schur/Kronecker sum of PSD terms).

**Estimation is by Monte-Carlo EM, not moments.** The natural cross-trait
Haseman-Elston step (regressing the augmented cross-products `l_ip l_jq` on
`A_ij K`) fails here, for a reason worth understanding: case/control ascertainment
truncates the liabilities, and the resulting inflation of `E[l_ip l_jq]` is itself
age-dependent -- closely related, similar-onset pairs are more often jointly
affected, so their cross-products are inflated most, and that extra, steeply
age-decaying signal is indistinguishable from fast genetic decay. A moment
regression therefore drives `lam` to its bound. The fit instead maximises the
expected complete-data Gaussian log-likelihood: the E-step imputes each family's
liability second moment `M_f = E[x_f x_f' | status, ages, params]` with a
truncated-MVN Gibbs sampler (averaged over `n_draw` draws), and the M-step
maximises `Q = -1/2 sum_f [ log|Sig_f| + tr(Sig_f^-1 M_f) ]` over the
heritabilities, the genetic/environmental covariances and the decay rates by
L-BFGS with the analytic score. Each family's covariance `Sig_f` depends on its
own onset ages, so the E-step loops over families; the M-step is batched.

**Identifiability is the limiting factor, and it is worth being honest about.**
The amplitude (`rho_g`) and the rate (`lam`) trade off along a likelihood ridge --
a strong correlation that decays fast can mimic a weak one that decays slowly --
and the cross-trait genetic signal competes with the environmental correlation.
The model is identifiable *in principle* (the cross-relative cross-trait
covariance `A G K` is purely genetic here, since environment is not shared across
relatives), but only **data-rich** designs pin it down: the repository kill-test
(`benchmarks/bench_aod_decay.py`) recovers both `rho_g` and `lam` well at
`n_fam ~ 2500` (`r_g ~ 0.51-0.53`, `lam ~ 0.041` vs true 0.5 / 0.04; `lam ~ 0.001`
under the scalar null, and `r_g ~ 0.005` at the `r_g = 0` null), but at
`n_fam ~ 1200` both run high (`~0.62` / `~0.064`) -- the ridge makes the model
**data-hungry**, converging only as `n` grows into the thousands with several
dozen EM iterations, and the across-replicate SD of `r_g` is ~0.11-0.18 even at
`n_fam = 2500`, so single estimates carry wide intervals. With few families or
little onset-age spread within relative pairs the estimates are noisy and
ridge-dominated; use the scalar model there.

Two robustness caveats matter for application (`benchmarks/bench_aod_decay_robustness.py`).
The fitted amplitude is **robust to the kernel shape** (fitting OU to Gaussian-decay
data still gives `r_g ~ 0.55` vs true 0.5), so the OU default is not a fragile
choice. But the model is **fragile to unmodelled shared family environment**: it
has no cross-relative environmental component, so a family-level environmental
correlation is attributed to genetics -- inflating `h2` and thereby *attenuating*
`r_g = G / sqrt(h2_0 h2_1)` (`r_g ~ 0.33` vs true 0.5 at `c2 = 0.10`). For traits
with real household effects this is the binding limitation.

**Options that address these limits.** `shared_lambda` ties every block's decay
rate to a single scalar -- fewer parameters, a guaranteed-PSD covariance, and less
amplitude-decay ridge; it is the recommended default unless there is reason to let
the rates differ. `shared_env` adds the shared-family environmental component `C`
directly to the model (an onset-age **ACE decomposition**: genetics `A`, shared
environment `C`, unique environment `E`), so household environment is estimated
rather than absorbed into genetics -- it recovers `c2` and stops the `h2`
inflation. But note the honest caveat: the *cross-trait* genetic-vs-shared-env
separation is itself hard (both produce cross-trait familial covariance, one
scaling with relatedness and age, the other constant), so `shared_env` can
overestimate `r_g` at moderate `n`; it is a variance-attribution tool, not a free
`r_g` fix. `n_starts` re-runs the EM from perturbed inits and keeps the best
objective (the likelihood is multi-modal at small `n`), and the result carries a
`converged` flag plus the `negq` objective trace. The analytic gradient of every
block (genetic, environmental, decay, shared-environment) is pinned against
finite differences in `research/tests/test_decay.py`.

### Genetic factor structure (common-factor model)

With more than a handful of traits, the genetic correlation matrix `r_g` is itself a
structured object worth summarising. `fit_genetic_factor` (in
`research/advanced_fitting.py`) fits a **common-factor
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
