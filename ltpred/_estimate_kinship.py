"""Kinship-based liability estimator with own_status support."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .gibbs import as_bounds
from ._validation import validate_bounds, validate_mixture_inputs, validate_own_status
from .covariance import construct_covmat_from_kinship, correct_positive_definite
from .pearson_aitken import pa_estimate_batched
from ._estimate_core import _single_out, _resolve_method
from ._estimate_group import _base_seeds, _estimate_group, _warn_if_corrected

def estimate_liability_from_kinship(A: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                                    h2: float = 0.5, target: int = 0,
                                    out: str = "genetic", tol: float = 0.01,
                                    n_sim: int = 100_000, burn_in: int = 1000,
                                    seed: int | None = None, max_rounds: int = 100,
                                    method: str | None = None,
                                    K_i: ArrayLike | None = None,
                                    K_pop: ArrayLike | None = None,
                                    use_mixture: bool = False,
                                    own_status: Literal["in", "out"] = "in"
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a target individual's liability from an **arbitrary pedigree**.

    The kinship-based counterpart of the array estimators: instead of the fixed role
    grammar you pass the additive relationship matrix ``A``
    (``n\u00d7n``, e.g. from :func:`~ltpred.covariance.kinship_from_pedigree`) shared by a
    batch of families, and the per-individual truncation bounds. ``lower``/``upper``
    are ``(n_families, n)`` (one column per pedigree member, in ``A`` order); the
    genetic-liability row for ``target`` is added internally and left unbounded.
    Inbred members are standardised to unit marginal full-liability variance, and
    the target's genetic contribution is scaled by its raw liability SD. Thus the
    supplied standard-normal bounds retain their prevalence interpretation when
    diagonal entries of ``A`` exceed one.

    ``out`` selects ``"genetic"`` (the target's genetic liability — the usual
    family-history GWAS phenotype) or ``"full"`` (``E[l_o | own interval and
    relatives]`` on both engines). ``method=None`` uses the
    deterministic Pearson-Aitken engine, matching the main single-trait default;
    pass ``method="gibbs"`` for reference sampling. Returns ``(est, se, var)``:
    ``se`` is the Monte-Carlo SE of ``est`` (exactly zero under PA, which is
    deterministic — that means *no sampling error*, not no approximation error),
    and ``var`` is the posterior (conditional) variance of the target liability
    on both engines.
    ``use_mixture=True`` with per-member ``K_i``/``K_pop`` (same shape as
    ``lower``) runs the PA-FGRS censored-control mixture; Gibbs does not
    implement it. Each array has length ``n_families``. The covariance is built by
    :func:`~ltpred.covariance.construct_covmat_from_kinship`, so results match the
    role-based estimator whenever the pedigree encodes the same relationships — but
    this also handles half-sibs of any degree, cousins, and inbred pedigrees.
    For Gibbs, ``seed`` must be a non-boolean integer in ``[0, 2**32 - 1]`` or
    ``None``; the PA branch ignores it.

    ``own_status="out"`` unbinds the ``target`` person's full-liability column
    (the kinship analogue of role ``o``) without dropping it. A one-person
    pedigree with ``own_status="out"`` raises: nothing remains to condition
    on. ``"in"`` (default) is use II."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be a square (n, n) relationship matrix")
    lower = np.atleast_2d(as_bounds(lower))
    upper = np.atleast_2d(as_bounds(upper))
    validate_bounds(lower, upper, context="kinship estimator bounds")
    if lower.shape[1] != n or upper.shape[1] != n:
        raise ValueError(f"lower/upper must have {n} columns (one per pedigree member)")
    own_status = validate_own_status(own_status)
    if own_status == "out":
        tgt_col = int(target)
        if n == 1:
            raise ValueError(
                "own_status='out' with a one-person pedigree (only the target) "
                "leaves nothing to condition on; the estimate would be the "
                "prior mean 0, indistinguishable from a real score. Include "
                "relatives, or drop own_status='out'.")
        lower = np.array(lower, copy=True)
        upper = np.array(upper, copy=True)
        lower[:, tgt_col] = -np.inf
        upper[:, tgt_col] = np.inf
        if K_i is not None:
            K_i = np.array(K_i, copy=True, dtype=float)
            K_i[:, tgt_col] = np.nan
        if K_pop is not None:
            K_pop = np.array(K_pop, copy=True, dtype=float)
            K_pop[:, tgt_col] = np.nan
    out_coord = _single_out(out)
    method_name = _resolve_method(method, "pearson-aitken")
    if use_mixture and method_name == "gibbs":
        raise ValueError(
            "use_mixture=True is only supported by Pearson-Aitken; the Gibbs "
            "estimator does not implement the censored-control mixture")

    cov_obj = construct_covmat_from_kinship(A, h2=h2, target=target, add_ind=True)
    cov, n_corrections = correct_positive_definite(cov_obj.matrix)
    _warn_if_corrected(n_corrections, "liability estimation")
    # prepend the unbounded genetic-liability (g) coordinate
    F = lower.shape[0]
    neg = np.full((F, 1), -np.inf, dtype=lower.dtype)
    pos = np.full((F, 1), np.inf, dtype=upper.dtype)
    lo = np.ascontiguousarray(np.concatenate([neg, lower], axis=1))
    hi = np.ascontiguousarray(np.concatenate([pos, upper], axis=1))
    tgt = 0 if out_coord == 0 else 1 + int(target)     # g row, or the target's o row

    if method_name == "pearson-aitken":
        if not use_mixture:
            est, var = pa_estimate_batched(cov, lo, hi, target=tgt)
        else:
            K_i, K_pop = validate_mixture_inputs(
                K_i, K_pop, expected_shape=lower.shape, require_pair=True,
                lower=lower, upper=upper,
                context="kinship estimator mixture inputs")
            nan_g = np.full((F, 1), np.nan, dtype=as_bounds(K_i).dtype)
            ki = np.ascontiguousarray(np.concatenate([nan_g, as_bounds(K_i)], axis=1))
            kp = np.ascontiguousarray(np.concatenate(
                [np.full((F, 1), np.nan, dtype=as_bounds(K_pop).dtype),
                 as_bounds(K_pop)], axis=1))
            est, var = pa_estimate_batched(cov, lo, hi, target=tgt,
                                           K_is=ki, K_pops=kp)
        # PA is deterministic: no Monte-Carlo error, so se is exactly zero.
        return est, np.zeros_like(est), var

    seeds = _base_seeds(seed, F, max_rounds)
    est, se, var = _estimate_group(cov, [tgt], lo, hi, seeds, tol, n_sim,
                                   burn_in, max_rounds)
    return est[:, 0], se[:, 0], var[:, 0]
