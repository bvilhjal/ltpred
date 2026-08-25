"""Liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

Public estimation APIs. Internals live in :mod:`ltpred._estimate_core`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .gibbs import as_bounds
from ._validation import validate_bounds, validate_mixture_inputs, validate_own_status
from .covariance import construct_covmat_from_kinship, correct_positive_definite
from .pearson_aitken import pa_estimate_batched
from ._estimate_core import (
    LiabilityResult, batch_means,
    _single_out, _pa_from_role_arrays, _gibbs_from_role_arrays, _base_seeds,
    _estimate_group, _estimate_liability_pa, _estimate_liability_single,
    _estimate_liability_multi, _resolve_method, _warn_if_corrected,
)

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
           "estimate_liability_from_kinship"]

def _unbounded_like(value):
    """``(-inf, inf)`` with the same shape as a member's bound."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return -np.inf, np.inf
    return np.full(arr.shape, -np.inf), np.full(arr.shape, np.inf)


def _raise_unbound_proband_only(ids, n_total, what):
    """Refuse use I with nothing left to condition on (ADuLT + out)."""
    shown = ", ".join(repr(x) for x in ids[:5])
    more = "" if len(ids) <= 5 else f", ... (+{len(ids) - 5} more)"
    raise ValueError(
        f"{len(ids)} of {n_total} {what} have only the proband with "
        f"own_status='out' ({shown}{more}); with nothing to condition on "
        "the estimate would be the prior mean 0, indistinguishable from a "
        "real score. Include relatives, or drop those families.")


def _families_for_own_status(families, own_status):
    """Copy-unbind role ``o`` when ``own_status='out'``; keep the row."""
    own_status = validate_own_status(own_status)
    if own_status == "in":
        return families
    out, only_o = [], []
    for fam in families:
        roles = [m.role for m in fam.members]
        if roles and all(r == "o" for r in roles):
            only_o.append(fam.fam_id)
        members = []
        for m in fam.members:
            if m.role == "o":
                lo, hi = _unbounded_like(m.lower)
                m = replace(m, lower=lo, upper=hi, K_i=None, K_pop=None)
            members.append(m)
        out.append(replace(fam, members=members))
    if only_o:
        _raise_unbound_proband_only(only_o, len(families), "families")
    return out


def _unbind_o_column(roles, lower, upper, own_status, K_i=None, K_pop=None):
    """Rewrite the role-``o`` column to ``(-inf, inf)`` without dropping it."""
    own_status = validate_own_status(own_status)
    if own_status == "in":
        return lower, upper, K_i, K_pop
    roles = list(roles)
    if "o" not in roles:
        return lower, upper, K_i, K_pop
    if all(r == "o" for r in roles):
        raise ValueError(
            "own_status='out' with only role 'o' leaves nothing to condition "
            "on; the estimate would be the prior mean 0, indistinguishable "
            "from a real score. Include relative columns, or drop "
            "own_status='out'.")
    j = roles.index("o")
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    lower[:, j] = -np.inf
    upper[:, j] = np.inf
    if K_i is not None:
        K_i = np.array(K_i, copy=True, dtype=float)
        K_i[:, j] = np.nan
    if K_pop is not None:
        K_pop = np.array(K_pop, copy=True, dtype=float)
        K_pop[:, j] = np.nan
    return lower, upper, K_i, K_pop


def estimate_liability_pa_arrays(roles: Sequence[str], lower: ArrayLike,
                                 upper: ArrayLike, h2: float = 0.5,
                                 out: str = "genetic",
                                 K_i: ArrayLike | None = None,
                                 K_pop: ArrayLike | None = None,
                                 use_mixture: bool = False,
                                 c2: float | None = None, m2: float | None = None,
                                 own_status: Literal["in", "out"] = "in"
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Array-level Pearson-Aitken estimator — skips ``Family``/``Member`` objects.

    The production fast path for many same-structure probands: ``roles`` is the
    shared list of member roles (``o`` and relatives; ``g`` is added), and ``lower``
    / ``upper`` are ``(n_families, len(roles))`` bounds aligned to ``roles`` (build
        them straight from your columns, e.g. with a threshold helper). The covariance
    is built once (with the ``c2``/``m2`` sibship and couple shared-environment
    components, ``h2 + c2 + m2 <= 1``). ``out`` is ``"genetic"`` (target ``g``) or
    ``"full"`` (``E[l_o | own interval and relatives]``). ``use_mixture`` with
    ``K_i``/``K_pop`` (same shape) enables the
    censored-control mixture. ``own_status=\"out\"`` unbinds the ``o`` column
    if present (use I) without dropping it; ``\"in\"`` (default) is use II.
    Only role ``o`` with ``own_status=\"out\"`` raises. Returns PA
    sequential-moment approximations ``(est, var)`` of length ``n_families``."""
    coord = _single_out(out)
    lower, upper, K_i, K_pop = _unbind_o_column(
        roles, lower, upper, own_status, K_i=K_i, K_pop=K_pop)
    est, var = _pa_from_role_arrays(
        roles, lower, upper, h2, [coord], K_i=K_i, K_pop=K_pop,
        use_mixture=use_mixture, c2=c2, m2=m2)
    return est[coord], var[coord]
