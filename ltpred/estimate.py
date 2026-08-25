"""Liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

Public estimation APIs. Internals live in :mod:`ltpred._estimate_core`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._own_status import _families_for_own_status
from ._estimate_core import LiabilityResult, batch_means, _resolve_method
from ._estimate_engines import (
    _estimate_liability_pa, _estimate_liability_single,
    _estimate_liability_multi,
)
from ._estimate_arrays import (
    estimate_liability_pa_arrays, estimate_liability_gibbs_arrays,
)
from ._estimate_kinship import estimate_liability_from_kinship
from ._estimate_group import (
    _estimate_group, _base_seeds,
    _assert_nonempty_families, _check_unique_roles, _group_by_structure,
)
from ._estimate_role_arrays import _align_to_cov

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
           "estimate_liability_from_kinship"]

def estimate_liability(families: Sequence, h2: ArrayLike = 0.5, *,
                       method: str | None = None,
                       out: str | Sequence[str] = ("genetic",),
                       tol: float = 0.01, use_mixture: bool = False,
                       genetic_corrmat: ArrayLike | None = None,
                       full_corrmat: ArrayLike | None = None,
                       phen_names: Sequence[str] | None = None,
                       n_sim: int = 100_000, burn_in: int = 1000,
                       seed: int | None = None, max_rounds: int = 100,
                       dtype: object = np.float64, c2: float | None = None,
                       m2: float | None = None,
                       own_status: Literal["in", "out"] = "in") -> LiabilityResult:
    """Estimate conditional liabilities, dispatching on method and trait count.

    Bounds and relative rows distinguish LT-FH, LT-FH++, ADuLT. PA-FGRS also
    requires ``K_i``/``K_pop`` and ``use_mixture=True`` and is implemented only by
    the Pearson-Aitken engine.

    ``method`` selects the inference engine; the **default** (``None``) picks the
    deterministic **Pearson-Aitken (PA)** estimator for a single trait. Across the
    benchmark's tested no-mixture structures, PA and Gibbs posterior-mean estimates
    had correlation at least ``0.997`` and PA ran 392-518x faster than
    grouped Gibbs at four threads on the no-mixture benchmark grid in this
    package. On a locked comparison to R LTFHPlus 2.2.0 (same families,
    same bounds, LTFHPlus's Gibbs settings) both engines had correlation
    0.9999 with the R Gibbs scores (RMSE ``0.0041`` Gibbs, ``0.0046`` PA).
    LTFHPlus is Gibbs-only; public PA is
    LTFGRS 1.0.1, and ltpred PA matches it at RMSE ``0.000087``.
    Same-algorithm fold times on that lock were 6.79x (LTFHPlus Gibbs /
    ltpred Gibbs) and 1418x (LTFGRS PA / ltpred PA). The dispatcher
    falls back to the **Gibbs**
    sampler for the multi-trait model, which PA does not support.
    That path collapses untruncated genetic coordinates out of the
    sweep and Rao--Blackwellises their posterior means. Pass ``method``
    explicitly to override: ``"pearson-aitken"`` (aliases ``"pa"``, ``"aitken"``;
    single trait only, ``use_mixture`` enables the age-censored-control correction) or
    ``"gibbs"`` (the truncated-MVN sampler; needed for multiple traits or a
    Monte-Carlo SE). The high-level result contains Gibbs estimates or PA
    approximations to posterior means, plus the method-specific uncertainty fields,
    not retained draws; call :func:`~ltpred.gibbs.rtmvnorm_gibbs` directly when
    draws are required. An explicit ``method="pearson-aitken"``
    with a multi-trait request raises.

        Scalar ``h2`` -> single trait; a vector ``h2`` with ``genetic_corrmat`` and
    ``full_corrmat`` -> multi-trait. For single-trait estimation, ``c2``/``m2``
    wire sibship (``C``) and couple (``M``) shared-environment components into
    the family covariance (see
    :func:`ltpred.covariance.construct_covmat_single`; ``h2 + c2 + m2 <= 1``
    required). Nonzero ``c2``/``m2`` are not supported for multi-trait
    estimation. ``dtype=np.float32`` stores the per-family
    liability bounds in single precision (half the memory) - useful at biobank
    scale. For Gibbs, ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``; PA ignores it. The same applies to ``tol``,
    ``n_sim``, ``burn_in`` and ``max_rounds``: they steer the Gibbs sampler's
    convergence loop and are not read on the deterministic PA path.

    ``own_status`` is ``"in"`` (default: the proband's own diagnosis stays in
    ``D_F``, the LT-FH GWAS / use II path) or ``"out"`` (use I: rewrite role
    ``o`` to ``(-inf, inf)`` without dropping the row, so ``pids`` still come
    from that record). Other spellings raise. A family that is only role
    ``o`` with ``own_status="out"`` has nothing to condition on and raises,
    matching an empty family. Truncated ``o`` with the default ``"in"`` is
    use II, not a mistake - there is no warning. Dropping the ``o`` row also
    unbinds, but then ``pids`` silently falls back to ``fam_id``.

    For use I the implied risk is
    ``Phi-bar((T - mu) / sqrt(Var(a|D_F) + 1 - h2))``, i.e.
    ``norm.sf((T - res.genetic) / np.sqrt(res.var["genetic"] + 1 - h2))``.
    That formula is not valid for use II, where own status is already in
    ``D_F``."""
    families = _families_for_own_status(families, own_status)
    if (np.ndim(h2) > 0 and np.size(h2) == 1 and genetic_corrmat is None
            and full_corrmat is None):
        h2 = float(np.ravel(h2)[0])   # a length-1 h2 is a scalar request
    is_multi = np.ndim(h2) > 0 or genetic_corrmat is not None or full_corrmat is not None

    # default: PA (single trait), Gibbs (multi, PA can't)
    method_name = _resolve_method(method, "gibbs" if is_multi else "pearson-aitken")
    if method_name == "pearson-aitken":
        if is_multi:
            raise NotImplementedError(
                "Pearson-Aitken estimation is single-trait; use method='gibbs' "
                "for the multi-trait model.")
        return _estimate_liability_pa(families, h2=h2, out=out,
                                      use_mixture=use_mixture, dtype=dtype,
                                      c2=c2, m2=m2)

    if use_mixture:
        raise ValueError(
            "use_mixture=True is only supported by Pearson-Aitken; the Gibbs "
            "estimator does not implement the censored-control mixture")

    if not is_multi:
        return _estimate_liability_single(families, h2=h2, out=out, tol=tol,
                                          n_sim=n_sim, burn_in=burn_in, seed=seed,
                                          max_rounds=max_rounds, dtype=dtype,
                                          c2=c2, m2=m2)
    if genetic_corrmat is None or full_corrmat is None:
        raise ValueError("multi-trait estimation needs genetic_corrmat and full_corrmat")
    if ((c2 is not None and np.any(np.asarray(c2, dtype=float) != 0.0))
            or (m2 is not None and np.any(np.asarray(m2, dtype=float) != 0.0))):
        raise NotImplementedError(
            "c2/m2 shared-environment components are not supported for "
            "multi-trait liability estimation")
    return _estimate_liability_multi(families, h2_vec=h2,
                                     genetic_corrmat=genetic_corrmat,
                                     full_corrmat=full_corrmat, phen_names=phen_names,
                                     out=out, tol=tol, n_sim=n_sim, burn_in=burn_in,
                                     seed=seed, max_rounds=max_rounds, dtype=dtype)
