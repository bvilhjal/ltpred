"""Observed/liability heritability and probit variance transformations.

The Lee factor converts observed case/control heritability to liability scale
using population prevalence ``K`` and, for ascertained studies, sample case
fraction ``P``. The inverse transformation divides by the same factor.

For SNP effect ``beta`` and allele frequency ``f``, probit residual-scale
variance is ``q = 2 f (1-f) beta^2``; its single-predictor fraction of total
latent variance is ``q/(1+q)``. GWAS z-based variance subtracts one unit of
null noise by default.
"""
from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from ._mathfun import norm_ppf

__all__ = ["observed_to_liability_h2", "liability_to_observed_h2",
           "probit_liability_r2", "liability_r2_from_z"]


def _z_density(pop_prev):
    """Validate K and return ``(K, z)`` with ``z = phi(Phi^-1(1 - K))``, the
    normal density at the liability threshold."""
    pop_prev = np.asarray(pop_prev, dtype=float)
    if np.any(~np.isfinite(pop_prev) | (pop_prev <= 0) | (pop_prev >= 1)):
        raise ValueError("pop_prev must be finite and lie in (0, 1)")
    t = -norm_ppf(pop_prev)        # -ppf(p) avoids the 1-p tail cancellation
    return pop_prev, np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def _ascertainment(pop_prev, prop_cases):
    """The K(1-K)/(P(1-P)) ascertainment factor (1 when prop_cases is None)."""
    if prop_cases is None:
        return 1.0
    pop_prev = np.asarray(pop_prev, dtype=float)
    prop_cases = np.asarray(prop_cases, dtype=float)
    if np.any(~np.isfinite(prop_cases) | (prop_cases <= 0) | (prop_cases >= 1)):
        raise ValueError("prop_cases must be finite and lie in (0, 1)")
    return pop_prev * (1.0 - pop_prev) / (prop_cases * (1.0 - prop_cases))


def _master_factor(pop_prev, prop_cases):
    """c = K²(1-K)² / (z² P(1-P)) -- the full per-trait h² factor (Lee 2011
    eq. 23); equals K(1-K)/z² when prop_cases is None."""
    pop_prev, z = _z_density(pop_prev)
    f = pop_prev * (1.0 - pop_prev) / (z * z)
    return f * _ascertainment(pop_prev, prop_cases)


def observed_to_liability_h2(obs_h2: ArrayLike, pop_prev: ArrayLike,
                             prop_cases: ArrayLike | None = None) -> np.ndarray | np.floating:
    """Observed-scale h² -> liability scale (Lee et al. 2011).

    Multiplies ``obs_h2`` by ``K(1-K)/z²`` with ``z = phi(Phi^-1(1-K))``, plus
    the ascertainment factor ``K(1-K)/(P(1-P))`` when the study over-samples
    cases. ``pop_prev`` is the population prevalence ``K`` and ``prop_cases``
    the sample case fraction ``P`` (``None`` for an unascertained sample, which
    drops the ascertainment factor); both must be finite and lie in (0, 1), or
    ``ValueError`` is raised. Vectorised over the inputs.
    """
    return np.asarray(obs_h2, dtype=float) * _master_factor(pop_prev, prop_cases)


def liability_to_observed_h2(liab_h2: ArrayLike, pop_prev: ArrayLike,
                             prop_cases: ArrayLike | None = None) -> np.ndarray | np.floating:
    """Liability-scale h² -> observed scale (Lee et al. 2011).

    The inverse of `observed_to_liability_h2`: divides ``liab_h2`` by the same
    factor, with ``pop_prev`` (``K``) and ``prop_cases`` (``P``, or ``None``)
    under the same rules.
    """
    # Exactly 1 / _master_factor: one definition of the Lee factor serves
    # both directions (differs from the recomputed ratio by <= 2 ulp).
    return np.asarray(liab_h2, dtype=float) / _master_factor(
        pop_prev, prop_cases)


def probit_liability_r2(beta: ArrayLike, maf: ArrayLike, *,
                        fraction: bool = False) -> np.ndarray | np.floating:
    """Return probit residual-scale variance ``2 f (1-f) beta²``.

    With ``fraction=True``, return the single-predictor fraction ``q/(1+q)``.
    Sum residual-scale variances across independent SNPs before applying that
    nonlinear transformation, not the fractions themselves.
    """
    beta = np.asarray(beta, dtype=float)
    maf = np.asarray(maf, dtype=float)
    if np.any((maf < 0.0) | (maf > 1.0)):
        raise ValueError("maf must lie in [0, 1] (allele frequency; 2f(1-f) is symmetric)")
    r2 = 2.0 * maf * (1.0 - maf) * beta * beta
    if fraction:
        r2 = r2 / (1.0 + r2)
    return r2


def liability_r2_from_z(z: ArrayLike, n: ArrayLike, pop_prev: ArrayLike,
                        prop_cases: ArrayLike | None = None, *,
                        subtract_null: bool = True) -> np.ndarray | np.floating:
    """Estimate small-effect liability variance from GWAS z statistics.

    Returns ``((z² - 1) / n) * c`` by default, where ``c`` is the same Lee
    factor used by `observed_to_liability_h2`; ``subtract_null=False`` omits
    the unit null term. Estimates can be negative per SNP and assume a
    calibrated null. Sum only independent or LD-pruned signals, or use an
    LD-aware aggregate.
    """
    z = np.asarray(z, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if not isinstance(subtract_null, (bool, np.bool_)):
        raise TypeError("subtract_null must be bool")
    signal = z * z - (1.0 if subtract_null else 0.0)
    return signal / n * _master_factor(pop_prev, prop_cases)
