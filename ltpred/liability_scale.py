"""Liability-scale transformations: heritability, genetic correlation, and
per-SNP liability r².

**The probit model IS the liability-threshold model**: the probit latent
variable with unit residual variance is the liability, with the threshold set
by the prevalence. Everything in this module follows from that identity.

**Per-SNP liability r² needs no transformation.** On the liability (probit)
scale, a SNP with minor-allele frequency ``f`` and effect ``beta`` explains
``Var(x * beta) = 2 f (1 - f) beta^2`` of liability variance -- an identity,
not an approximation (:func:`probit_liability_r2`). A probit GWAS z-statistic
estimates ``beta * sqrt(N * 2 f (1 - f))``, so liability r² can be recovered
directly from summary statistics as ``z^2 / N`` in the small-effect limit
(:func:`liability_r2_from_z`).

**The Lee transformations are bridges to other scales.** When heritability or
genetic correlation is estimated on the observed case-control scale (OLS on
the 0/1 outcome, e.g. GREML/LDSC output), the factors of Lee et al. (2011,
*AJHG*) and Lee et al. (2012) convert between that scale and the liability
scale, using the population prevalence ``K`` (and, for ascertained
case/control studies, the sample case proportion ``P``) with
``z = phi(Phi^-1(1 - K))``:

* ``h²_liab  = h²_obs  * [K(1-K)/z²] * [K(1-K)/(P(1-P))]``   (Lee et al. 2011;
  the second factor applies only to ascertained studies)

The existing helper :func:`ltpred.thresholds.convert_observed_to_liability_scale`
is the forward h² bridge; :func:`observed_to_liability_h2` here matches it and
:func:`liability_to_observed_h2` is its inverse. The Lee et al. (2012) genetic-
correlation bridges are :func:`observed_to_liability_rg` and
:func:`liability_to_observed_rg`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["observed_to_liability_h2", "liability_to_observed_h2",
           "observed_to_liability_gencov", "liability_to_observed_gencov",
           "observed_to_liability_rg",
           "probit_liability_r2", "liability_r2_from_z"]


def _z_density(pop_prev):
    """z = phi(Phi^-1(1 - K)), the normal density at the liability threshold."""
    pop_prev = np.asarray(pop_prev, dtype=float)
    if np.any((pop_prev <= 0) | (pop_prev >= 1)):
        raise ValueError("pop_prev must lie in (0, 1)")
    t = ndtri(1.0 - pop_prev)
    return pop_prev, np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def _ascertainment(pop_prev, prop_cases):
    """The K(1-K)/(P(1-P)) ascertainment factor (1 when prop_cases is None)."""
    if prop_cases is None:
        return 1.0
    pop_prev = np.asarray(pop_prev, dtype=float)
    prop_cases = np.asarray(prop_cases, dtype=float)
    if np.any((prop_cases <= 0) | (prop_cases >= 1)):
        raise ValueError("prop_cases must lie in (0, 1)")
    return pop_prev * (1.0 - pop_prev) / (prop_cases * (1.0 - prop_cases))


def observed_to_liability_h2(obs_h2, pop_prev, prop_cases=None):
    """Observed-scale h² -> liability scale (Lee et al. 2011).

    Multiplies by ``K(1-K)/z²`` with ``z = phi(Phi^-1(1-K))``, plus the
    ascertainment factor ``K(1-K)/(P(1-P))`` when the study over-samples cases.
    Numerically identical to
    :func:`ltpred.thresholds.convert_observed_to_liability_scale`; provided
    here so both directions live together. Vectorised over the inputs.
    """
    pop_prev, z = _z_density(pop_prev)
    factor = pop_prev * (1.0 - pop_prev) / (z * z)
    return np.asarray(obs_h2, dtype=float) * factor * _ascertainment(
        pop_prev, prop_cases)


def liability_to_observed_h2(liab_h2, pop_prev, prop_cases=None):
    """Liability-scale h² -> observed scale (inverse of
    :func:`observed_to_liability_h2`)."""
    pop_prev, z = _z_density(pop_prev)
    factor = (z * z) / (pop_prev * (1.0 - pop_prev))
    return np.asarray(liab_h2, dtype=float) * factor / _ascertainment(
        pop_prev, prop_cases)


def probit_liability_r2(beta, maf, *, fraction=False):
    """Liability-scale variance explained by one SNP: ``2 f (1-f) beta²``.

    An identity on the liability (probit) scale: the SNP contributes
    ``Var(x * beta)`` to the liability, and the liability's residual variance
    is 1 by definition of the probit/threshold model. ``beta`` is the
    per-allele effect on the probit scale and ``maf`` the minor-allele
    frequency (variance-standardised genotype ``x`` is NOT required -- the
    formula uses the allele-count variance ``2 f (1-f)``). With
    ``fraction=True`` returns the McKelvey-Zavoina coefficient of
    determination ``var(x beta) / (1 + var(x beta))`` (Lee et al. 2012,
    *Genet Epidemiol*, eq. 9) -- the fraction of *total* liability variance,
    for summing/comparing across architectures. Vectorised.
    """
    beta = np.asarray(beta, dtype=float)
    maf = np.asarray(maf, dtype=float)
    if np.any((maf < 0) | (maf > 0.5)):
        raise ValueError("maf must lie in [0, 0.5]")
    r2 = 2.0 * maf * (1.0 - maf) * beta * beta
    if fraction:
        r2 = r2 / (1.0 + r2)
    return r2


def liability_r2_from_z(z, n, pop_prev, prop_cases=None):
    """Liability-scale r² per SNP from a GWAS z-statistic (Lee & Wray 2013).

    For a marginal case-control GWAS statistic ``z_j``, the liability-scale
    variance explained is, in the small-effect limit,

        r²_liab,j ~ (z_j² / N) * c,   c = [K(1-K)/z_K²] * [K(1-K)/(P(1-P))],

    with ``K`` the population prevalence, ``P`` the sample case proportion
    (ascertained studies), and ``z_K = phi(Phi^-1(1-K))``. This is Lee & Wray
    (2013, *PLoS ONE* 8:e71494, eq. 2-3; their non-centrality
    ``E[chi²] - 1 ~ N * r²_liab * z_K² P(1-P) / (K(1-K))²``), and the factor
    ``c`` is exactly the Lee et al. (2011) master factor used by
    :func:`observed_to_liability_h2`. Without ``prop_cases`` the
    ascertainment factor drops out (population samples). ``n`` is the per-SNP
    sample size (scalar or per-SNP array).
    """
    z = np.asarray(z, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    pop_prev, z_k = _z_density(pop_prev)
    factor = pop_prev * (1.0 - pop_prev) / (z_k * z_k)
    factor = factor * _ascertainment(pop_prev, prop_cases)
    return z * z / n * factor


# --- Lee et al. (2012) genetic covariance / correlation bridges -------------


def _master_factor(pop_prev, prop_cases):
    """c = K²(1-K)² / (z² P(1-P)) -- the full per-trait h² factor (Lee 2011
    eq. 23); equals K(1-K)/z² when prop_cases is None."""
    pop_prev, z = _z_density(pop_prev)
    f = pop_prev * (1.0 - pop_prev) / (z * z)
    return f * _ascertainment(pop_prev, prop_cases)


def observed_to_liability_gencov(gencov_obs, pop_prev1, pop_prev2,
                                 prop_cases1=None, prop_cases2=None):
    """Observed-scale genetic covariance -> liability scale (Lee et al. 2012).

    Multiplies by the geometric mean of the two per-trait master factors,
    ``sqrt(c1 * c2)`` (Lee et al. 2012, *Bioinformatics* 28:2540-2542, sec.
    2.2; the LDSC ``gencov_obs_to_liab`` convention). Each disease takes its
    own prevalence (and sample case proportion for ascertained studies). The
    genetic *correlation* needs no transformation -- the factors cancel in
    the ratio; see :func:`observed_to_liability_rg`.
    """
    c1 = _master_factor(pop_prev1, prop_cases1)
    c2 = _master_factor(pop_prev2, prop_cases2)
    return np.asarray(gencov_obs, dtype=float) * np.sqrt(c1 * c2)


def liability_to_observed_gencov(gencov_liab, pop_prev1, pop_prev2,
                                 prop_cases1=None, prop_cases2=None):
    """Inverse of :func:`observed_to_liability_gencov`."""
    c1 = _master_factor(pop_prev1, prop_cases1)
    c2 = _master_factor(pop_prev2, prop_cases2)
    return np.asarray(gencov_liab, dtype=float) / np.sqrt(c1 * c2)


def observed_to_liability_rg(rg_obs, pop_prev1=None, pop_prev2=None,
                             prop_cases1=None, prop_cases2=None):
    """Genetic correlation across scales: the identity (scale-invariant).

    Lee et al. (2012, sec. 2.2) show the linear scale factors enter the
    genetic variances and the covariance identically, so they cancel in the
    ratio: **the genetic correlation is the same on the observed and
    liability scales**, approximately, even under ascertainment (LDSC wiki:
    "there is no notion of observed or liability scale genetic correlation";
    van Rheenen et al. 2019). Provided for API symmetry -- it validates the
    prevalences if given and returns ``rg_obs`` unchanged.
    """
    if pop_prev1 is not None:
        _z_density(pop_prev1)
    if pop_prev2 is not None:
        _z_density(pop_prev2)
    return np.asarray(rg_obs, dtype=float)

