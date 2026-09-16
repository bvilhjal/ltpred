# Research extensions

The `research/` package holds unsupported, checkout-only code: it is not
installed with ltpred, it is not part of the public API, and its interfaces may
change without notice. It is exercised by its own tests (`research/tests`, run
by a CI job) and imported by five benchmark scripts, so it stays importable from
a source checkout as `research.<module>` with the repository root on `sys.path`.
The [`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
inventories the modules and the [roadmap](ROADMAP.md#graduation-rule) states
when a capability graduates into `ltpred` proper.

This page documents the models behind those modules. The supported core is
described in [Algorithm & model](algorithm.md); these sections were moved out of
that page so that it describes only what the installed package does.

## Covariance extensions (`research/covariance_extensions.py`)

### Direct and indirect (genetic-nurture) effects

Caution (i) in the algorithm page's
[environmental-covariance section](algorithm.md#adding-environmental-covariance-to-improve-prediction)
says a symmetric shared-environment matrix is not a maternal
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
but the predictor factorises as `g0 = w' mu`, with `mu` the vector of
folded truncated means and `w = V^-1 c` the BLUP weights. The weights `w`
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

## Multi-trait fitters (`research/advanced_fitting.py`)

The supported [multi-trait scorer](algorithm.md#multiple-traits) takes
covariance parameters as inputs. For joint common-threshold fitting, use the
installed [`fit_pairwise_multi`](inference.md#joint-heritability-and-cross-trait-correlations)
API, including optional sibship/couple covariance and family-cluster sampling
SEs. The alternative prototypes below remain uninstalled, checkout-only code;
their moment and EM routes carry the caveats stated in each section.

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
under the scalar null, and `r_g ~ 0.009` at the `r_g = 0` null), but at
`n_fam ~ 1000` both run high (`~0.58` / `~0.063`) -- the ridge makes the model
**data-hungry**, converging only as `n` grows into the thousands with several
dozen EM iterations, and the across-replicate SD of `r_g` is ~0.11-0.18 even at
`n_fam = 2500`, so single estimates carry wide intervals. With few families or
little onset-age spread within relative pairs the estimates are noisy and
ridge-dominated; use the scalar model there.

Two robustness caveats matter for application (`benchmarks/bench_aod_decay.py`
panel (d), `--robustness`).
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
since the within-dataset `se` understates it. `benchmarks/bench_genetic_correlation.py`
panel (c)
recovers planted loadings end-to-end and shows `srmr` rising when a one-factor model
is fit to two-factor data.
