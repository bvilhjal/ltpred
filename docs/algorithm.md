# Algorithm and model

ltpred implements the **liability-threshold model conditioned on family history**
(LT-FH++) and its deterministic cousin (PA-FGRS). This page describes the model
and the two estimators. See [guide.md](guide.md) for usage.

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

Relatives' genetic liabilities are correlated by the fraction of DNA they share.
For a vector of family members `[g, o, relative_1, ...]` the covariance entries
are

```text
Cov(a, b) = shared_DNA(a, b) * h2        (off-diagonal genetic sharing)
Var(l_g)  = h2 ,   Var(l_o) = Var(relative) = 1
```

`shared_DNA` is 1 for self, 1 for `g`↔`o`, 0.5 for parent/offspring and full
sibs, 0.25 for grandparents / half-sibs / aunts-uncles, etc.
`get_relatedness(a, b, h2)` returns `shared_DNA * h2`, and
`construct_covmat(...)` assembles the matrix, ordering `g`, `o` first followed by
the relatives. A near-singular matrix (from relatedness rounding) is nudged back
to positive-definite by `correct_positive_definite`.

This is an **additive-genetic** model: familial resemblance is entirely genetic
sharing. Shared environment, household/cultural transmission, assortative mating
(parents are taken to be genetically unrelated), dominance/epistasis and indirect
genetic effects are not represented. Where those contribute, the estimated
"genetic liability" is best read as the additive-model projection of the family
history rather than a pure causal genetic value.

## Thresholds: status, age and onset

Each observed person contributes a truncation of their liability:

- **classic LT-FH** (`prevalence_thresholds`): case `(T, inf)`, control
  `(-inf, T)`.
- **LT-FH++ / ADuLT** (`age_thresholds`): the cumulative incidence rises with age
  along a logistic curve,

  ```text
  CIP(age) = K / (1 + exp((mid_point - age) * slope))
  thresh(age) = Phi^-1(1 - CIP(age))
  ```

  A **case** is *pinned* at `thresh(age_of_onset)` (`lower == upper`): younger
  onset ⇒ lower incidence ⇒ higher threshold ⇒ more extreme liability — the
  age-of-onset map. A **control** is `(-inf, thresh(current_age))`: surviving
  disease-free to an older age is stronger evidence of low liability. The map is
  invertible (`convert_liability_to_aoo` ↔ `convert_age_to_thresh`), so
  `thresh(onset)` equals the case's liability at onset.

Pinning (a point mass) is handled exactly by both estimators.

## Estimator 1: Gibbs sampler (LT-FH++)

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
**online** — no full `(n_sim × n_out)` sample array, and each family seeds its own
RNG so results are deterministic regardless of thread scheduling. Without Numba
the identical code runs serially in pure Python.

## Estimator 2: Pearson–Aitken (PA-FGRS)

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

This is **exact for a single truncation**; with several it is the standard
sequential-selection approximation, which matches the Gibbs posterior to
corr ≥ 0.997 on realistic families while running 100–360× faster. Same grouping /
`prange` structure as the Gibbs path.

### Censored-control mixture (optional)

`tnorm_mixture_conditional` extends the truncated moments for **age-censored
controls**: someone unaffected only up to their current follow-up is a mixture of
a true control and a not-yet-onset future case. With the individual cumulative
incidence `K_i` and lifetime prevalence `K_pop`, the selected moments become

```text
mix   = Phi_below / (Phi_below + (1 - Phi_below) * (K_pop - K_i) / K_pop)
mean* = mix * mean(below upper) + (1 - mix) * mean(above upper)
```

(with the matching two-component variance), following PA-FGRS supp. eqs. S3–S5.
Enabled via `use_mixture=True`; off, PA reduces to the plain truncated-moment
sweep.

## Multiple traits

For `n` genetically/environmentally correlated traits the covariance is
phenotype-major: same-trait blocks use `shared_DNA * h2_p`; cross-trait blocks
scale relatedness by the genetic covariance `rho_g[p,q] * sqrt(h2_p h2_q)`, and
the same individual's full liabilities across traits correlate by
`full_corrmat[p,q]`. The Gibbs sampler then returns the genetic/full liability of
each trait (`estimate_liability_multi`). This lets a well-powered trait sharpen
the estimate for a correlated, under-powered one.

## References

- Hujoel et al. 2020, *Nat Genet* — LT-FH.
- Pedersen et al. 2022, *AJHG* — LT-FH++ (flexible pedigrees, age, sex).
- Pedersen et al. 2023, *Nat Commun* — ADuLT (age-dependent liability threshold).
- Krebs et al. 2024, *AJHG* — PA-FGRS (Pearson–Aitken family genetic risk scores).
- Kotecha & Djurić 1999 — Gibbs sampling for truncated multivariate normals.
- Lee et al. 2011, *AJHG* — observed-to-liability-scale heritability.
