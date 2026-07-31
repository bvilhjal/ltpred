"""Liability-scale transformations: heritability and probit-scale SNP variance.

**The probit model IS the liability-threshold model**: the probit latent
variable with unit residual variance is the liability, with the threshold set
by the prevalence. Everything in this module follows from that identity.

**Two probit scales must not be conflated.** A probit coefficient fixes the
residual variance to one. For a SNP with minor-allele frequency ``f`` and effect
``beta``, ``2 f (1-f) beta^2`` is genetic variance in those residual-variance
units. It is not yet a fraction of total latent variance; for a single predictor
that fraction is ``q / (1 + q)``. :func:`probit_liability_r2` retains its
historical name but documents this distinction explicitly.

A squared GWAS z-statistic contains one unit of expected null sampling noise.
:func:`liability_r2_from_z` subtracts that null contribution by default; set
``subtract_null=False`` only when the raw second moment is deliberately wanted.

**The Lee transformation is a bridge to the observed scale.** When
heritability is estimated on the observed case-control scale (OLS on
the 0/1 outcome, e.g. GREML/LDSC output), the factor of Lee et al. (2011,
*AJHG*) converts between that scale and the liability
scale, using the population prevalence ``K`` (and, for ascertained
case/control studies, the sample case proportion ``P``) with
``z = phi(Phi^-1(1 - K))``:

* ``h²_liab  = h²_obs  * [K(1-K)/z²] * [K(1-K)/(P(1-P))]``   (Lee et al. 2011;
  the second factor applies only to ascertained studies)

:func:`observed_to_liability_h2` is the forward h² bridge and
:func:`liability_to_observed_h2` its inverse.
"""
from __future__ import annotations

import numpy as np

from ._mathfun import norm_ppf

__all__ = ["observed_to_liability_h2", "liability_to_observed_h2",
           "probit_liability_r2", "liability_r2_from_z"]


def _z_density(pop_prev):
    """z = phi(Phi^-1(1 - K)), the normal density at the liability threshold."""
    pop_prev = np.asarray(pop_prev, dtype=float)
    if np.any((pop_prev <= 0) | (pop_prev >= 1)):
        raise ValueError("pop_prev must lie in (0, 1)")
    t = norm_ppf(1.0 - pop_prev)
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


def _master_factor(pop_prev, prop_cases):
    """c = K²(1-K)² / (z² P(1-P)) -- the full per-trait h² factor (Lee 2011
    eq. 23); equals K(1-K)/z² when prop_cases is None."""
    pop_prev, z = _z_density(pop_prev)
    f = pop_prev * (1.0 - pop_prev) / (z * z)
    return f * _ascertainment(pop_prev, prop_cases)


def observed_to_liability_h2(obs_h2, pop_prev, prop_cases=None):
    """Observed-scale h² -> liability scale (Lee et al. 2011).

    Multiplies by ``K(1-K)/z²`` with ``z = phi(Phi^-1(1-K))``, plus the
    ascertainment factor ``K(1-K)/(P(1-P))`` when the study over-samples cases.
    Vectorised over the inputs.
    """
    return np.asarray(obs_h2, dtype=float) * _master_factor(pop_prev, prop_cases)


def liability_to_observed_h2(liab_h2, pop_prev, prop_cases=None):
    """Liability-scale h² -> observed scale (inverse of
    :func:`observed_to_liability_h2`)."""
    pop_prev, z = _z_density(pop_prev)
    factor = (z * z) / (pop_prev * (1.0 - pop_prev))
    return np.asarray(liab_h2, dtype=float) * factor / _ascertainment(
        pop_prev, prop_cases)


def probit_liability_r2(beta, maf, *, fraction=False):
    """Residual-scale probit genetic variance: ``2 f (1-f) beta²``.

    Despite the compatibility name, the default is **not a total-liability
    R²**. It is ``q = Var(x * beta)`` in units where probit residual variance
    is one. ``beta`` is the
    per-allele effect on the probit scale and ``maf`` the minor-allele
    frequency (variance-standardised genotype ``x`` is NOT required -- the
    formula uses the allele-count variance ``2 f (1-f)``). With
    ``fraction=True`` returns the single-predictor McKelvey-Zavoina fraction
    ``q / (1 + q)`` (Lee et al. 2012, *Genet Epidemiol*, eq. 9). Do not sum
    those elementwise fractions across SNPs: sum the default ``q`` values first,
    then transform the total once. Vectorised.
    """
    beta = np.asarray(beta, dtype=float)
    maf = np.asarray(maf, dtype=float)
    if np.any((maf < 0) | (maf > 0.5)):
        raise ValueError("maf must lie in [0, 0.5]")
    r2 = 2.0 * maf * (1.0 - maf) * beta * beta
    if fraction:
        r2 = r2 / (1.0 + r2)
    return r2


def liability_r2_from_z(z, n, pop_prev, prop_cases=None, *,
                        subtract_null=True):
    """Residual-scale liability-variance signal from a GWAS z-statistic.

    For a marginal case-control GWAS statistic ``z_j``, the liability-scale
    variance explained is, in the small-effect limit,

        q_liab,j ~ ((z_j² - 1) / N) * c,

    with ``K`` the population prevalence, ``P`` the sample case proportion
    (ascertained studies), and ``z_K = phi(Phi^-1(1-K))``. This is Lee & Wray
    (2013, *PLoS ONE* 8:e71494, eq. 2-3; their non-centrality
    ``E[chi²] - 1 ~ N * q_liab * z_K² P(1-P) / (K(1-K))²``), and the factor
    ``c`` is exactly the Lee et al. (2011) master factor used by
    :func:`observed_to_liability_h2`. Without ``prop_cases`` the
    ascertainment factor drops out (population samples). ``n`` is the per-SNP
    sample size (scalar or per-SNP array). Null subtraction is unbiased in
    expectation but permits negative per-SNP estimates; aggregate before
    interpreting. Marginal per-SNP signals may be summed only across independent
    or suitably LD-pruned variants; otherwise use an LD-aware aggregation to
    avoid counting tagged signal repeatedly. Set ``subtract_null=False`` to
    recover the historical raw second-moment calculation ``z² / N``.
    """
    z = np.asarray(z, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if not isinstance(subtract_null, (bool, np.bool_)):
        raise TypeError("subtract_null must be bool")
    signal = z * z - (1.0 if subtract_null else 0.0)
    return signal / n * _master_factor(pop_prev, prop_cases)
