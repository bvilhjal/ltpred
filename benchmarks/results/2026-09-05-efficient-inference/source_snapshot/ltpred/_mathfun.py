"""Normal CDF / quantile primitives.

Two flavours live here because they serve two callers:

* ``_norm_cdf`` / ``_norm_ppf`` are **scalar, branch-only** functions built on
  ``math.erfc`` and Wichura's algorithm AS 241 (Applied Statistics, 1988). They
  contain nothing Numba cannot compile, so the Gibbs kernel calls them directly
  and produces identical numbers with or without the JIT. AS 241 is accurate to
  ~1e-16, matching R's ``qnorm`` (which uses the same algorithm), so the port's
  thresholds line up with LTFHPlus.
* ``norm_cdf`` / ``norm_ppf`` are the **vectorised** versions used by the
  public threshold helpers; they defer to SciPy's ``ndtr`` / ``ndtri`` for speed
  and to stay exact on arrays. ``+/-inf`` inputs map to the obvious 1 / 0 and
  ``+/-inf`` limits.

Keeping the kernel on the scalar pair (rather than SciPy) is what lets the
sampler run inside ``numba.njit`` at all.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr, ndtri

from ._numba import _jit

__all__ = ["norm_cdf", "norm_ppf", "_norm_cdf", "_norm_ppf"]

_SQRT1_2 = 0.7071067811865476  # 1 / sqrt(2)


@_jit
def _norm_cdf(x):
    """Standard-normal CDF of a scalar; ``0.5 * erfc(-x / sqrt(2))``.

    ``math.erfc`` handles ``+/-inf`` (erfc(-inf)=2, erfc(inf)=0), so bounds of
    ``+/-inf`` return 1.0 / 0.0 without special-casing."""
    return 0.5 * math.erfc(-x * _SQRT1_2)


@_jit
def _norm_ppf(p):
    """Standard-normal quantile of a scalar via Wichura's AS 241 (PPND16).

    ``p <= 0`` returns ``-inf`` and ``p >= 1`` returns ``+inf``. Elsewhere the
    absolute error is below ~1e-16, so it agrees with R's ``qnorm`` to double
    precision."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = ((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
                   67265.770927008700853) * r + 45921.953931549871457) * r +
                 13731.693765509461125) * r + 1971.5909503065514427) * r +
               133.14166789178437745) * r + 3.387132872796366608
        den = ((((((5226.495278852854561 * r + 28729.085735721942674) * r +
                   39307.89580009271061) * r + 21213.794301586595867) * r +
                 5394.1960214247511077) * r + 687.1870074920579083) * r +
               42.313330701600911252) * r + 1.0
        return q * num / den

    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = ((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
                   0.24178072517745061177) * r + 1.27045825245236838258) * r +
                 3.64784832476320460504) * r + 5.7694972214606914055) * r +
               4.6303378461565452959) * r + 1.42343711074968357734
        den = ((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
                   0.0151986665636164571966) * r + 0.14810397642748007459) * r +
                 0.68976733498510000455) * r + 1.6763848301838038494) * r +
               2.05319162663775882187) * r + 1.0
        val = num / den
    else:
        r -= 5.0
        num = ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                   0.0012426609473880784386) * r + 0.026532189526576123093) * r +
                 0.29656057182850489123) * r + 1.7848265399172913358) * r +
               5.4637849111641143699) * r + 6.6579046435011037772
        den = ((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
                   1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
                 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
               0.59983220655588793769) * r + 1.0
        val = num / den
    return -val if q < 0.0 else val


def norm_cdf(x):
    """Vectorised standard-normal CDF (SciPy ``ndtr``); accepts scalars or arrays."""
    return ndtr(np.asarray(x, dtype=float))


def norm_ppf(p):
    """Vectorised standard-normal quantile (SciPy ``ndtri``); accepts scalars or arrays."""
    return ndtri(np.asarray(p, dtype=float))
